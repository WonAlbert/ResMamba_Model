from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from resmamba_signal_model.data.packing import pack_valid_tokens
from resmamba_signal_model.models.adapters import SharedTaskAdapter, TaskAdapter, apply_task_adapters
from resmamba_signal_model.models.backbone import HybridEncoder
from resmamba_signal_model.models.decoder import AttentionPooling, SharedDecoder
from resmamba_signal_model.models.domain import DomainDiscriminator, GradientReversal
from resmamba_signal_model.models.heads import (
    EmitterHead,
    ImputationHead,
    ModulationHead,
    PredictionHead,
    PrototypeClusteringHead,
    RecognitionHeads,
    TASK_HEAD_REGISTRY,
    apply_emitter_dataset_mask,
    remap_task_head_checkpoints,
    register_task as register_task_head,
)
from resmamba_signal_model.models.physics import project_patch_energy, restore_absolute_log_power, sequence_physics
from resmamba_signal_model.models.prototypes import DEVICE_NAMESPACE, CONTENT_NAMESPACE, PrototypeRegistry
from resmamba_signal_model.models.revin import RevIN, RevINStats, clip_normalized
from resmamba_signal_model.models.signal_adapter import SignalAdapterRegistry, SignalSpec
from resmamba_signal_model.models.task_interface import (
    DEFAULT_TASKS,
    TaskSpec,
    UniversalTaskInterface,
    default_task_spec,
)
from resmamba_signal_model.training.task_catalog import BUILTIN_HEAD_ATTR, KIND_MASK_MODE, normalize_kind
from resmamba_signal_model.models.tokenizer import TimeFreqTokenizer, TimeFreqTokenizerConfig
from resmamba_signal_model.models.varlen import (
    apply_truncation_aug,
    assert_min_length,
    chunk_starts,
    overlap_weights,
    pad_iq_list,
    patches_to_iq,
    patchify_iq,
    random_train_chunk,
)


def _ensure_min_visible(visible: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
    none = (visible.sum(dim=1) == 0) & (patch_mask.sum(dim=1) > 0)
    if not none.any():
        return visible
    visible = visible.clone()
    first = patch_mask.to(dtype=torch.long).argmax(dim=1)
    visible[none, first[none]] = True
    return visible


@dataclass
class SignalModelConfig:
    d_model: int = 640
    encoder_mamba_layers: int = 5
    encoder_transformer_layers: int = 1
    decoder_mamba_layers: int = 1
    query_decoder_dim: int = 320
    legacy_decoder_reconstruction: bool = False
    use_legacy_generation_heads: bool = False
    force_unified_generation: bool = False
    legacy_target_energy_projection: bool = False
    mamba_d_state: int = 64
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_headdim: int = 64
    mamba_ngroups: int = 1
    mamba_chunk_size: int = 256
    scan_direction: str = "bidirectional"
    require_mamba_kernel: bool = True
    allow_fallback_mamba: bool = False
    share_bidirectional_weights: bool = False
    norm_type: str = "rmsnorm"
    drop_path: float = 0.0
    attn_num_heads: int | None = None
    attn_ffn_expand: int = 2
    attn_window: int = 1024
    dropout: float = 0.1
    patch_size: int = 16
    mask_ratio: float = 0.5
    stem_channels: int = 64
    freq_bands: int = 8
    physics_bias: bool = True
    physics_project: bool = True
    revin_std_min: float = 1.0e-2
    revin_clip: float = 8.0
    revin_scale_mode: str = "joint_energy"
    revin_winsorize_top_frac: float = 0.01
    revin_peak_papr_clip: float = 16.0
    revin_affine: bool = False
    revin_shared_affine: bool = True
    mae_mask_probs: dict[str, float] = field(
        default_factory=lambda: {"random": 1.0 / 3.0, "contiguous": 1.0 / 3.0, "mixed": 1.0 / 3.0}
    )
    phase_plugin: bool = False
    encode_visible_only: bool = True
    sequence_packing: bool = True
    l_min: int = 16
    chunk_len: int = 8192
    chunk_overlap: float = 0.125
    p_trunc: float = 0.3
    num_datasets: int = 32
    num_mod_classes: int = 256
    num_emitters: int = 512
    num_prototypes: int = 32
    use_dataset_bias: bool = False
    low_rank_prototype: bool = True
    prototype_rank: int = 64
    build_task_heads: bool = False
    build_task_interface: bool = False
    build_prototype_registry: bool = False
    build_ema_teacher: bool = False
    ema_momentum: float = 0.996
    grl_invariant_views: tuple[str, ...] = ("semantic",)
    train_encoder: bool = True
    train_decoder: bool = True
    train_heads: bool = True
    uti_rank: int = 64
    uti_metadata_dim: int = 8
    uti_legacy_mode: bool = False
    use_specialist_views: bool = True
    domain_prompt_size: int = 6
    adapter_down_dim: int = 64
    num_task_types: int = 8
    build_adapters: bool = False
    build_shared_adapter: bool = False
    task_names: tuple[str, ...] = DEFAULT_TASKS
    task_kinds: dict[str, str] = field(default_factory=dict)
    task_specs: dict[str, dict[str, Any]] = field(default_factory=dict)
    tokenizer: TimeFreqTokenizerConfig = field(default_factory=TimeFreqTokenizerConfig)

    def __post_init__(self) -> None:
        if int(self.decoder_mamba_layers) != 1:
            raise ValueError("decoder_mamba_layers 必须为 1")
        if isinstance(self.chunk_overlap, (int, float)) and self.chunk_overlap > 1:
            self.chunk_overlap = float(self.chunk_overlap) / float(self.chunk_len)
        self.tokenizer.d_model = self.d_model
        self.tokenizer.patch_size = self.patch_size
        self.tokenizer.stem_channels = self.stem_channels
        self.tokenizer.freq_bands = self.freq_bands
        self.tokenizer.dropout = self.dropout
        self.tokenizer.physics_bias = self.physics_bias
        self.tokenizer.l_min = self.l_min
        tok_phase = bool(getattr(self.tokenizer, "phase_plugin", False))
        self.phase_plugin = bool(self.phase_plugin or tok_phase)
        self.tokenizer.phase_plugin = self.phase_plugin
        if self.attn_num_heads is None:
            for heads in (8, 4, 2, 1):
                if self.d_model % heads == 0:
                    self.attn_num_heads = heads
                    break

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SignalModelConfig":
        names = {item.name for item in fields(cls)}
        payload = {k: v for k, v in data.items() if k in names and k != "tokenizer"}
        if "task_names" in payload and not isinstance(payload["task_names"], tuple):
            payload["task_names"] = tuple(str(x) for x in payload["task_names"])
        if "task_kinds" in payload and payload["task_kinds"] is not None:
            payload["task_kinds"] = {str(k): str(v) for k, v in dict(payload["task_kinds"]).items()}
        if "grl_invariant_views" in payload and not isinstance(payload["grl_invariant_views"], tuple):
            payload["grl_invariant_views"] = tuple(str(x) for x in payload["grl_invariant_views"])
        if "chunk_overlap_ratio" in data and "chunk_overlap" not in payload:
            payload["chunk_overlap"] = float(data["chunk_overlap_ratio"])
        if "mae_mask_probs" in payload:
            raw_probs = payload["mae_mask_probs"]
            names = ("random", "contiguous", "mixed")
            if isinstance(raw_probs, (list, tuple)) and len(raw_probs) == 3:
                payload["mae_mask_probs"] = {name: float(val) for name, val in zip(names, raw_probs)}
            elif isinstance(raw_probs, dict):
                payload["mae_mask_probs"] = {str(k): float(v) for k, v in raw_probs.items()}
        tok = data.get("tokenizer") or {}
        if isinstance(tok, dict):
            tok_names = {item.name for item in fields(TimeFreqTokenizerConfig)}
            payload["tokenizer"] = TimeFreqTokenizerConfig(**{k: v for k, v in tok.items() if k in tok_names})
        return cls(**payload)

    @property
    def chunk_overlap_samples(self) -> int:
        return max(0, int(self.chunk_len * float(self.chunk_overlap)))


class SignalFoundationModel(nn.Module):
    def __init__(self, cfg: SignalModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        heads = int(cfg.attn_num_heads or 8)
        self.input_adapters = SignalAdapterRegistry()
        self.revin = RevIN(
            num_channels=2,
            std_min=float(cfg.revin_std_min),
            clip=float(cfg.revin_clip),
            affine=bool(cfg.revin_affine),
            scale_mode=str(cfg.revin_scale_mode),
            winsorize_top_frac=float(cfg.revin_winsorize_top_frac),
            peak_papr_clip=float(cfg.revin_peak_papr_clip),
            shared_affine=bool(cfg.revin_shared_affine),
        )
        self.tokenizer = TimeFreqTokenizer(cfg.tokenizer)
        mamba_kwargs = dict(
            d_state=cfg.mamba_d_state,
            d_conv=cfg.mamba_d_conv,
            expand=cfg.mamba_expand,
            headdim=cfg.mamba_headdim,
            dropout=cfg.dropout,
            ngroups=cfg.mamba_ngroups,
            chunk_size=cfg.mamba_chunk_size,
            scan_direction=cfg.scan_direction,
            share_bidirectional_weights=cfg.share_bidirectional_weights,
            require_mamba_kernel=cfg.require_mamba_kernel,
            allow_fallback_mamba=cfg.allow_fallback_mamba,
            norm_type=cfg.norm_type,
        )
        self.encoder = HybridEncoder(
            d_model=cfg.d_model,
            encoder_mamba_layers=cfg.encoder_mamba_layers,
            encoder_transformer_layers=cfg.encoder_transformer_layers,
            attn_num_heads=heads,
            attn_ffn_expand=cfg.attn_ffn_expand,
            attn_window=cfg.attn_window,
            drop_path=cfg.drop_path,
            **mamba_kwargs,
        )
        self.decoder = SharedDecoder(
            cfg.d_model,
            cfg.patch_size,
            decoder_mamba_layers=cfg.decoder_mamba_layers,
            attn_num_heads=heads,
            dropout=cfg.dropout,
            sequence_packing=cfg.sequence_packing,
            query_dim=cfg.query_decoder_dim,
            condition_dim=cfg.uti_rank,
            legacy_reconstruction=cfg.legacy_decoder_reconstruction,
            **{k: v for k, v in mamba_kwargs.items() if k != "dropout"},
        )
        self.domain_disc = DomainDiscriminator(cfg.d_model, num_datasets=max(2, cfg.num_datasets))
        self.grl = GradientReversal()
        self.chunk_pool = AttentionPooling(cfg.d_model, num_heads=min(4, heads))
        # 分类身份：encoder token AttnPool（与 decoder ReprHead 的重建 z 分离）
        self.encoder_pool = AttentionPooling(cfg.d_model, num_heads=min(4, heads))
        self.encoder_repr_norm = nn.LayerNorm(cfg.d_model)
        self.task_interface: UniversalTaskInterface | None = None
        self.task_adapters: nn.ModuleDict | None = None
        self.shared_adapter: SharedTaskAdapter | None = None
        self.modulation_head: ModulationHead | None = None
        self.emitter_head: EmitterHead | None = None
        self.clustering_head: PrototypeClusteringHead | None = None
        self.prediction_head: PredictionHead | None = None
        self.imputation_head: ImputationHead | None = None
        self.recognition_heads: RecognitionHeads | None = None
        self.extra_task_heads = nn.ModuleDict()
        self.z_linear_probes = nn.ModuleDict()
        self.prototype_registry: PrototypeRegistry | None = None
        self.peft = None
        self.truncate_backward = False
        self.skip_recon = False
        self._skip_recon = False
        self._active_task: str | None = None
        self._negcos_temperature: float | None = None
        task_names = tuple(cfg.task_names) or DEFAULT_TASKS
        need_uti = bool(cfg.build_task_heads or cfg.build_task_interface)
        if need_uti:
            task_specs: dict[str, TaskSpec | dict[str, Any]] = {}
            for name in task_names:
                if name in cfg.task_specs:
                    task_specs[name] = cfg.task_specs[name]
                else:
                    task_specs[name] = default_task_spec(name, cfg.task_kinds.get(name))
            self.task_interface = UniversalTaskInterface(
                cfg.d_model,
                rank=cfg.uti_rank,
                num_task_types=max(int(cfg.num_task_types), len(task_names) + 4),
                dropout=cfg.dropout,
                task_names=task_names,
                task_specs=task_specs,
                metadata_dim=cfg.uti_metadata_dim,
                legacy_mode=cfg.uti_legacy_mode,
                use_specialist_views=cfg.use_specialist_views,
                domain_prompt_size=int(getattr(cfg, "domain_prompt_size", 6) or 6),
                num_datasets=max(int(cfg.num_datasets), 2),
            )
            if cfg.build_task_heads:
                for name in task_names:
                    self._attach_task_head(name, self._make_task_head(name))
                    kind = self.task_kind(name)
                    if kind in ("classification", "emitter"):
                        n_cls = int(cfg.num_emitters if kind == "emitter" else cfg.num_mod_classes)
                        self.z_linear_probes[name] = nn.Linear(cfg.d_model, n_cls)
        if cfg.build_adapters:
            self.task_adapters = nn.ModuleDict(
                {name: TaskAdapter(cfg.d_model, down_dim=cfg.adapter_down_dim) for name in task_names}
            )
        if cfg.build_shared_adapter:
            self.shared_adapter = SharedTaskAdapter(cfg.d_model, down_dim=cfg.adapter_down_dim)
        if cfg.build_prototype_registry or cfg.build_task_heads:
            self.prototype_registry = PrototypeRegistry(
                int(cfg.d_model),
                num_prototypes=int(cfg.num_prototypes),
            )
        self.apply_train_flags(cfg.train_encoder, cfg.train_decoder, cfg.train_heads)

    def apply_train_flags(self, train_encoder: bool, train_decoder: bool, train_heads: bool) -> None:
        for module in (self.input_adapters, self.revin, self.tokenizer, self.encoder):
            for param in module.parameters():
                param.requires_grad = train_encoder
        for module in (self.encoder_pool, self.encoder_repr_norm):
            for param in module.parameters():
                param.requires_grad = train_encoder
        for param in self.decoder.parameters():
            param.requires_grad = train_decoder
        for param in self.domain_disc.parameters():
            param.requires_grad = train_decoder
        for param in self.chunk_pool.parameters():
            param.requires_grad = train_decoder
        if self.task_interface is not None:
            for param in self.task_interface.parameters():
                param.requires_grad = train_heads
        for head in self.iter_task_heads():
            for param in head.parameters():
                param.requires_grad = train_heads
        for param in self.z_linear_probes.parameters():
            param.requires_grad = train_heads
        if self.task_adapters is not None:
            for param in self.task_adapters.parameters():
                param.requires_grad = train_heads
        if self.shared_adapter is not None:
            for param in self.shared_adapter.parameters():
                param.requires_grad = train_heads
        if self.prototype_registry is not None:
            for param in self.prototype_registry.parameters():
                param.requires_grad = train_heads
        if self.recognition_heads is not None:
            for param in self.recognition_heads.parameters():
                param.requires_grad = train_heads

    def sample_mae_mask_strategy(self) -> str:
        probs = dict(getattr(self.cfg, "mae_mask_probs", None) or {})
        if not probs:
            choices = ("random", "contiguous", "mixed")
            return choices[int(torch.randint(0, len(choices), (1,)).item())]
        names = list(probs.keys())
        weights = torch.tensor([float(probs[n]) for n in names], dtype=torch.float32)
        weights = weights / weights.sum().clamp_min(1.0e-8)
        idx = int(torch.multinomial(weights, 1).item())
        return str(names[idx])

    def task_kind(self, name: str) -> str:
        kinds = getattr(self.cfg, "task_kinds", None) or {}
        if name in kinds:
            return normalize_kind(kinds[name])
        if name in ("modulation",):
            return "classification"
        if name in ("emitter", "clustering", "prediction", "imputation"):
            return name
        return "classification"

    def _make_task_head(self, name: str, head_cls: type[nn.Module] | None = None, **head_kwargs: Any) -> nn.Module:
        kind = self.task_kind(name)
        cls = head_cls or TASK_HEAD_REGISTRY.get(name)
        if cls is None:
            cls = {
                "classification": ModulationHead,
                "emitter": EmitterHead,
                "clustering": PrototypeClusteringHead,
                "prediction": PredictionHead,
                "imputation": ImputationHead,
            }[kind]
        kwargs = dict(head_kwargs)
        kwargs.setdefault("d_model", self.cfg.d_model)
        head_rank = bool(getattr(self.cfg, "low_rank_prototype", False))
        prototype_rank = int(getattr(self.cfg, "prototype_rank", 64) or 64)
        if cls is ModulationHead:
            kwargs.setdefault("num_mod_classes", self.cfg.num_mod_classes)
            kwargs.setdefault("dropout", self.cfg.dropout)
            kwargs.setdefault("num_datasets", self.cfg.num_datasets)
            kwargs.setdefault("use_dataset_bias", self.cfg.use_dataset_bias)
            if name != "modulation":
                kwargs.setdefault("logits_key", f"{name}_logits")
            if head_rank:
                kwargs.setdefault("low_rank_prototype", True)
                kwargs.setdefault("prototype_rank", prototype_rank)
        elif cls is EmitterHead:
            kwargs.setdefault("num_emitters", self.cfg.num_emitters)
            kwargs.setdefault("dropout", self.cfg.dropout)
            if head_rank:
                kwargs.setdefault("low_rank_prototype", True)
                kwargs.setdefault("prototype_rank", prototype_rank)
        elif cls is PrototypeClusteringHead:
            kwargs.setdefault("proj_dim", max(64, self.cfg.d_model // 2))
            kwargs.setdefault("num_prototypes", self.cfg.num_prototypes)
            if head_rank:
                kwargs.setdefault("low_rank_prototype", True)
                kwargs.setdefault("prototype_rank", prototype_rank)
        elif cls in (PredictionHead, ImputationHead):
            kwargs.setdefault("patch_size", self.cfg.patch_size)
            kwargs.setdefault("dropout", self.cfg.dropout)
            if head_rank:
                kwargs.setdefault("low_rank_prototype", True)
                kwargs.setdefault("prototype_rank", prototype_rank)
        return cls(**kwargs)

    def _attach_task_head(self, name: str, head: nn.Module) -> nn.Module:
        attr = BUILTIN_HEAD_ATTR.get(name)
        if attr:
            setattr(self, attr, head)
        else:
            self.extra_task_heads[name] = head
        return head

    def iter_task_heads(self) -> list[nn.Module]:
        heads: list[nn.Module] = []
        for attr in ("modulation_head", "emitter_head", "clustering_head", "prediction_head", "imputation_head"):
            module = getattr(self, attr, None)
            if isinstance(module, nn.Module):
                heads.append(module)
        heads.extend(list(self.extra_task_heads.values()))
        return heads

    def get_task_head(self, task: str) -> nn.Module | None:
        attr = BUILTIN_HEAD_ATTR.get(task)
        if attr:
            head = getattr(self, attr, None)
            if head is not None:
                return head
        return self.extra_task_heads[task] if task in self.extra_task_heads else None

    def register_task(
        self,
        name: str,
        head_cls: type[nn.Module] | None = None,
        *,
        task_spec: TaskSpec | dict[str, Any] | None = None,
        **head_kwargs: Any,
    ) -> nn.Module:
        """冻结骨干时只训 UTI 新行 + TaskAdapter + 新头。"""
        register_task_head(name, head_cls or TASK_HEAD_REGISTRY.get(name, ModulationHead))
        if self.task_interface is None:
            raise RuntimeError("register_task 需要 build_task_heads=True")
        kinds = dict(getattr(self.cfg, "task_kinds", None) or {})
        if name not in kinds and head_cls is not None:
            kinds[name] = {
                ModulationHead: "classification",
                EmitterHead: "emitter",
                PrototypeClusteringHead: "clustering",
                PredictionHead: "prediction",
                ImputationHead: "imputation",
            }.get(head_cls, "classification")
            self.cfg.task_kinds = kinds
        if task_spec is None:
            task_spec = default_task_spec(name, kinds.get(name))
        self.task_interface.add_task(name, task_spec)
        resolved_spec = self.task_interface.task_specs[name]
        specs = dict(getattr(self.cfg, "task_specs", None) or {})
        specs[name] = {
            "name": resolved_spec.name,
            "family": resolved_spec.family,
            "readout": resolved_spec.readout,
            "view": resolved_spec.view,
            "modality": resolved_spec.modality,
            "metadata": dict(resolved_spec.metadata),
        }
        self.cfg.task_specs = specs
        names = tuple(self.cfg.task_names) if self.cfg.task_names else ()
        if name not in names:
            self.cfg.task_names = names + (name,)
        head = self._make_task_head(name, head_cls, **head_kwargs)
        self._attach_task_head(name, head)
        if self.task_adapters is not None and name not in self.task_adapters:
            self.task_adapters[name] = TaskAdapter(self.cfg.d_model, down_dim=self.cfg.adapter_down_dim)
        return head

    def set_active_task(self, task: str | None) -> None:
        self._active_task = task
        handle = getattr(self, "peft", None)
        if handle is not None:
            handle.set_active_task(self, task)
        else:
            from resmamba_signal_model.models.peft import MultiTaskLoRALinear

            for module in self.modules():
                if isinstance(module, MultiTaskLoRALinear):
                    module.set_active_task(task)

    def _invariant_view_names(self) -> tuple[str, ...]:
        task = self._active_task
        if task and self.task_interface is not None:
            try:
                spec = self.task_interface.resolve_spec(task)
            except KeyError:
                spec = None
            if spec is not None and spec.invariant_views:
                return tuple(spec.invariant_views)
            if task == "modulation":
                return ("semantic",)
            if task == "emitter":
                return ()
            if task == "clustering":
                return ("semantic",)
        return tuple(getattr(self.cfg, "grl_invariant_views", ("semantic",)) or ())

    def _domain_logits(self, z_general: torch.Tensor, representations: dict[str, torch.Tensor]) -> torch.Tensor:
        """GRL 只约束声明不变的低秩视图，梯度不进入 z_general。"""
        names = [name for name in self._invariant_view_names() if f"z_{name}" in representations]
        adapters = getattr(self.task_interface, "view_adapters", None) if self.task_interface is not None else None
        feats = []
        if adapters is not None:
            for name in names:
                if name not in adapters:
                    continue
                feats.append(adapters[name](z_general.detach()))
        if not feats:
            return self.domain_disc(z_general.detach())
        stacked = feats[0] if len(feats) == 1 else torch.stack(feats, dim=0).mean(dim=0)
        return self.domain_disc(self.grl(stacked))

    def _attach_uti_readouts(
        self,
        out: dict[str, Any],
        *,
        task: str = "pretrain",
        allow_dataset_condition: bool = True,
    ) -> dict[str, Any]:
        if self.task_interface is None:
            return out
        h_enc = out.get("h_enc", out.get("h_general", out.get("patch_h")))
        if h_enc is None:
            return out
        if h_enc.dim() == 3 and h_enc.shape[1] == out["patch_mask"].shape[1] + 1:
            h_enc = h_enc[:, 1:]
        z_enc = out.get("z_enc", out.get("z_general", out["z"]))
        patch_mask = out["patch_mask"]
        visible = out.get("visible")
        enc_mask = (visible & patch_mask) if visible is not None else patch_mask
        view_pairs = self.task_interface.build_views(z_enc, h_enc, patch_mask=enc_mask)
        n_tokens = patch_mask.shape[1]
        pos = torch.linspace(0.0, 1.0, n_tokens, device=h_enc.device, dtype=h_enc.dtype)
        pos = pos.view(1, n_tokens, 1).expand(h_enc.shape[0], -1, -1)
        target_coord = out.get("target_mask", torch.zeros_like(patch_mask))
        query_coords = torch.cat([pos, target_coord.to(dtype=h_enc.dtype).unsqueeze(-1)], dim=-1)
        spec = default_task_spec(task)
        metadata = out.get("task_metadata")
        if not allow_dataset_condition and isinstance(metadata, dict):
            metadata = {k: v for k, v in metadata.items() if k != "dataset_id"}
        features = self.task_interface(
            z_enc,
            h_enc,
            enc_mask,
            spec,
            recon_norm=out.get("recon_norm"),
            views=view_pairs,
            metadata=metadata,
            query_coords=query_coords,
        )
        out["uti_pooled"] = features.pooled
        out["uti_tokens"] = features.tokens
        out["uti_query"] = features.query
        out["task_pooled"] = features.pooled
        out["task_tokens"] = features.tokens
        out["task_query"] = features.query
        return out

    def load_weights(self, state: dict[str, Any], *, strict: bool = False) -> Any:
        state = remap_task_head_checkpoints(dict(state))
        if not strict:
            own = self.state_dict()
            filtered: dict[str, Any] = {}
            for key, value in state.items():
                current = own.get(key)
                if (
                    current is not None
                    and torch.is_tensor(value)
                    and torch.is_tensor(current)
                    and tuple(current.shape) != tuple(value.shape)
                ):
                    continue
                filtered[key] = value
            state = filtered
        return self.load_state_dict(state, strict=strict)

    def load_emitter_dataset_class_mask(
        self,
        rfdata_root: str | Any = None,
        mask: torch.Tensor | None = None,
    ) -> None:
        """个体头按数据集掩码分类，避免 ADSB 类干扰 WiSig。"""
        head = getattr(self, "emitter_head", None)
        if head is None or not hasattr(head, "set_dataset_class_mask"):
            return
        if mask is None:
            from resmamba_signal_model.training.emitter_labels import build_emitter_dataset_class_mask

            mask = build_emitter_dataset_class_mask(
                rfdata_root,
                num_emitters=int(self.cfg.num_emitters),
                num_datasets=int(self.cfg.num_datasets),
            )
        head.set_dataset_class_mask(mask)

    def register_signal_adapter(
        self,
        spec: SignalSpec | dict[str, Any],
        adapter: nn.Module | None = None,
    ) -> nn.Module:
        """注册新模态的小型通道适配器，不修改 tokenizer/backbone。"""
        return self.input_adapters.register(spec, adapter)

    def _adapt_signal_input(
        self,
        values: torch.Tensor | list[torch.Tensor],
        signal_spec: SignalSpec | dict[str, Any] | list[Any] | tuple[Any, ...] | None,
        channel_mask: torch.Tensor | list[torch.Tensor] | None,
    ) -> torch.Tensor | list[torch.Tensor]:
        if isinstance(values, list):
            adapted: list[torch.Tensor] = []
            for idx, value in enumerate(values):
                spec_i = (
                    signal_spec[idx]
                    if isinstance(signal_spec, (list, tuple))
                    else signal_spec
                )
                if isinstance(channel_mask, list):
                    channel_mask_i = channel_mask[idx]
                elif isinstance(channel_mask, torch.Tensor) and channel_mask.ndim >= 2:
                    channel_mask_i = channel_mask[idx]
                else:
                    channel_mask_i = channel_mask
                value_b = value.unsqueeze(0) if value.ndim == 2 else value
                projected = self.input_adapters(value_b, spec_i, channel_mask_i)
                adapted.append(projected.squeeze(0) if value.ndim == 2 else projected)
            return adapted
        return self.input_adapters(values, signal_spec, channel_mask)

    def _resolve_modality_id(
        self,
        batch_size: int,
        *,
        device: torch.device,
        modality_id: torch.Tensor | str | int | list[Any] | tuple[Any, ...] | None = None,
        signal_spec: SignalSpec | dict[str, Any] | list[Any] | tuple[Any, ...] | None = None,
    ) -> torch.Tensor:
        from resmamba_signal_model.models.physics import MODALITY_GENERIC
        from resmamba_signal_model.models.task_interface import MODALITY_TO_ID

        default_id = int(MODALITY_TO_ID.get("rf", MODALITY_GENERIC))
        if modality_id is not None:
            if torch.is_tensor(modality_id):
                ids = modality_id.long().reshape(-1)
                if ids.numel() == 1 and batch_size > 1:
                    ids = ids.expand(batch_size)
                return ids[:batch_size].to(device=device)
            values = list(modality_id) if isinstance(modality_id, (list, tuple)) else [modality_id]
            mapped: list[int] = []
            for item in values:
                if torch.is_tensor(item):
                    mapped.append(int(item.reshape(-1)[0].item()) if item.numel() else default_id)
                elif isinstance(item, (int, bool)):
                    mapped.append(int(item))
                else:
                    name = str(item).strip().lower() if item is not None else "rf"
                    mapped.append(int(MODALITY_TO_ID.get(name, MODALITY_GENERIC)))
            if len(mapped) == 1 and batch_size > 1:
                mapped = mapped * batch_size
            if len(mapped) < batch_size:
                mapped.extend([default_id] * (batch_size - len(mapped)))
            return torch.tensor(mapped[:batch_size], device=device, dtype=torch.long)
        if signal_spec is not None:
            if isinstance(signal_spec, (list, tuple)):
                names = [
                    str(getattr(item, "modality_id", None) or (item.get("modality_id") if isinstance(item, dict) else "rf"))
                    for item in signal_spec
                ]
            elif isinstance(signal_spec, dict):
                names = [str(signal_spec.get("modality_id", signal_spec.get("modality", "rf")))] * batch_size
            else:
                names = [str(getattr(signal_spec, "modality_id", "rf"))] * batch_size
            ids = [MODALITY_TO_ID.get(name.lower(), MODALITY_GENERIC) for name in names[:batch_size]]
            if len(ids) < batch_size:
                ids.extend([MODALITY_TO_ID.get("rf", MODALITY_GENERIC)] * (batch_size - len(ids)))
            return torch.tensor(ids, device=device, dtype=torch.long)
        return torch.full((batch_size,), MODALITY_TO_ID.get("rf", MODALITY_GENERIC), device=device, dtype=torch.long)

    def _resolve_complex_pair(
        self,
        batch_size: int,
        *,
        device: torch.device,
        complex_pair: torch.Tensor | bool | None = None,
        signal_spec: SignalSpec | dict[str, Any] | list[Any] | tuple[Any, ...] | None = None,
    ) -> torch.Tensor:
        if isinstance(complex_pair, torch.Tensor):
            flag = complex_pair.bool().reshape(-1)
            if flag.numel() == 1 and batch_size > 1:
                flag = flag.expand(batch_size)
            return flag.to(device=device)
        if complex_pair is not None:
            return torch.full((batch_size,), bool(complex_pair), device=device, dtype=torch.bool)
        if signal_spec is not None:
            if isinstance(signal_spec, (list, tuple)):
                flags = [bool(getattr(item, "complex_pairs", None) or (item.get("complex_pairs") if isinstance(item, dict) else True)) for item in signal_spec]
            elif isinstance(signal_spec, dict):
                flags = [bool(signal_spec.get("complex_pairs", True))] * batch_size
            else:
                flags = [bool(getattr(signal_spec, "complex_pairs", True))] * batch_size
            if len(flags) < batch_size:
                flags.extend([True] * (batch_size - len(flags)))
            return torch.tensor(flags[:batch_size], device=device, dtype=torch.bool)
        return torch.ones(batch_size, device=device, dtype=torch.bool)

    def make_mae_mask(self, patch_mask: torch.Tensor, mask_ratio: float | None = None) -> torch.Tensor:
        ratio = self.cfg.mask_ratio if mask_ratio is None else mask_ratio
        rand = torch.rand(patch_mask.shape, device=patch_mask.device).masked_fill(~patch_mask, 2.0)
        num_valid = patch_mask.sum(dim=-1).clamp_min(1)
        num_mask = (num_valid.float() * ratio).round().long().clamp(min=1)
        mae_mask = torch.zeros_like(patch_mask)
        for b in range(patch_mask.shape[0]):
            n_valid = int(num_valid[b].item())
            count = min(int(num_mask[b].item()), max(0, n_valid - 1))
            if count > 0:
                mae_mask[b, rand[b].argsort()[:count]] = True
        return mae_mask & patch_mask

    def make_span_mask(self, patch_mask: torch.Tensor, mask_ratio: float | None = None, max_span: int = 8) -> torch.Tensor:
        ratio = self.cfg.mask_ratio if mask_ratio is None else mask_ratio
        span_mask = torch.zeros_like(patch_mask)
        for i in range(patch_mask.shape[0]):
            valid_idx = patch_mask[i].nonzero(as_tuple=False).flatten()
            n_valid = int(valid_idx.numel())
            if n_valid <= 1:
                continue
            n_mask = max(1, int(round(n_valid * ratio)))
            filled = 0
            while filled < n_mask:
                span = int(torch.randint(1, min(max_span, n_valid) + 1, ()).item())
                start_k = int(torch.randint(0, n_valid, ()).item())
                end_k = min(n_valid, start_k + span)
                span_mask[i, valid_idx[start_k:end_k]] = True
                filled = int(span_mask[i, patch_mask[i]].sum().item())
                if filled >= n_valid - 1:
                    break
        return span_mask & patch_mask

    def make_suffix_mask(self, patch_mask: torch.Tensor, mask_ratio: float | None = None) -> torch.Tensor:
        ratio = 0.25 if mask_ratio is None else mask_ratio
        suffix = torch.zeros_like(patch_mask)
        for i in range(patch_mask.shape[0]):
            valid_idx = patch_mask[i].nonzero(as_tuple=False).flatten()
            n_valid = int(valid_idx.numel())
            if n_valid <= 1:
                continue
            n_mask = max(1, int(round(n_valid * ratio)))
            n_mask = min(n_mask, n_valid - 1)
            suffix[i, valid_idx[-n_mask:]] = True
        return suffix & patch_mask

    def _mask_for_mode(
        self,
        patch_mask: torch.Tensor,
        mask_mode: str | None,
        *,
        mae_strategy: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        zeros = torch.zeros_like(patch_mask)
        if mask_mode in (None, "none"):
            return zeros, zeros, zeros, "none"
        if mask_mode == "suffix":
            return zeros, self.make_suffix_mask(patch_mask), zeros, "suffix"
        if mask_mode == "span":
            return zeros, zeros, self.make_span_mask(patch_mask), "span"
        if mae_strategy is not None:
            strategy = str(mae_strategy).strip().lower()
        elif mask_mode in ("random", "contiguous", "mixed"):
            strategy = mask_mode
        else:
            strategy = "mixed"
        if strategy == "random":
            mae = self.make_mae_mask(patch_mask)
            return mae, zeros, zeros, "random"
        if strategy == "contiguous":
            span = self.make_span_mask(patch_mask)
            return span, zeros, span, "contiguous"
        mae = self.make_mae_mask(patch_mask)
        span = self.make_span_mask(patch_mask)
        mixed = (mae | span) & patch_mask
        visible = patch_mask & ~mixed
        if (visible.sum(dim=1) == 0).any():
            first = patch_mask.to(dtype=torch.long).argmax(dim=1)
            for b in range(patch_mask.shape[0]):
                if visible[b].sum() == 0:
                    mixed[b, first[b]] = False
        mixed = mixed & patch_mask
        return mae & mixed, zeros, span & mixed, "mixed"

    def _observed_sample_mask(
        self,
        sample_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        target_samples = target_mask.repeat_interleave(self.cfg.patch_size, dim=1)
        target_samples = target_samples[:, : sample_mask.shape[1]]
        if target_samples.shape[1] < sample_mask.shape[1]:
            target_samples = F.pad(target_samples, (0, sample_mask.shape[1] - target_samples.shape[1]))
        return sample_mask & ~target_samples

    def _prepare_iq(
        self,
        iq: torch.Tensor | list[torch.Tensor],
        sample_mask: torch.Tensor | None,
        *,
        apply_aug: bool,
        allow_chunk: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(iq, list):
            iq, sample_mask, _lengths = pad_iq_list(iq, self.cfg.l_min)
        else:
            if sample_mask is None:
                sample_mask = torch.ones(iq.shape[0], iq.shape[-1], dtype=torch.bool, device=iq.device)
            for length in sample_mask.sum(dim=1).tolist():
                assert_min_length(int(length), self.cfg.l_min)
        if apply_aug and self.training and self.cfg.p_trunc > 0:
            iq, sample_mask = apply_truncation_aug(iq, sample_mask, p_trunc=self.cfg.p_trunc, l_min=self.cfg.l_min)
        if allow_chunk and self.training and iq.shape[-1] > self.cfg.chunk_len:
            iq, sample_mask = random_train_chunk(iq, sample_mask, self.cfg.chunk_len)
        return iq, sample_mask

    def _encode_tokens(self, tokens: torch.Tensor, visible: torch.Tensor, patch_mask: torch.Tensor) -> tuple[torch.Tensor, bool]:
        visible = _ensure_min_visible(visible, patch_mask)
        if self.cfg.encode_visible_only:
            if int(visible.sum()) == 0:
                return tokens.new_zeros(1, 0, tokens.shape[-1]), True
            packed, cu_seqlens, seq_idx = pack_valid_tokens(tokens, visible)
            if packed.shape[1] == 0:
                return packed, True
            return self.encoder(packed, seq_idx=seq_idx, cu_seqlens=cu_seqlens), True
        if self.cfg.sequence_packing:
            packed, cu_seqlens, seq_idx = pack_valid_tokens(tokens, patch_mask)
            return self.encoder(packed, seq_idx=seq_idx, cu_seqlens=cu_seqlens), True
        return self.encoder(tokens, key_padding_mask=~patch_mask), False

    def _pool_encoder_identity(
        self,
        tokens: torch.Tensor,
        patch_mask: torch.Tensor,
        *,
        h_vis: torch.Tensor | None = None,
        packed_enc: bool | None = None,
        visible: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encoder 读出 ``(z_enc, h_enc)``，供分类 / VICReg / UTI。

        有 MAE 可见编码 ``h_vis`` 时直接 scatter 后只在 ``visible`` 上池化，
        避免二次全序列 encode；无 ``h_vis`` 时才全序列 encode。
        """
        if h_vis is not None and packed_enc is not None and visible is not None:
            h_enc = self.decoder.scatter_encoder(h_vis, visible, patch_mask, packed=packed_enc)
            pool_mask = visible & patch_mask
        else:
            h_all, packed_all = self._encode_tokens(tokens, patch_mask, patch_mask)
            h_enc = self.decoder.scatter_encoder(h_all, patch_mask, patch_mask, packed=packed_all)
            pool_mask = patch_mask
        h_enc = h_enc.masked_fill(~pool_mask.unsqueeze(-1), 0.0)
        z_enc = self.encoder_pool(h_enc, key_padding_mask=~pool_mask)
        z_enc = self.encoder_repr_norm(z_enc.float()).to(dtype=h_enc.dtype)
        return z_enc, h_enc

    def _refresh_encoder_identity(self, out: dict[str, Any]) -> dict[str, Any]:
        """truncate_backward 后在可微路径上重算 encoder 读出，使 ``encoder_pool`` 可训。"""
        h_enc = out.get("h_enc")
        patch_mask = out.get("patch_mask")
        if h_enc is None or patch_mask is None:
            return out
        visible = out.get("visible")
        pool_mask = (visible & patch_mask) if visible is not None else patch_mask
        z_enc = self.encoder_pool(h_enc, key_padding_mask=~pool_mask)
        z_enc = self.encoder_repr_norm(z_enc.float()).to(dtype=h_enc.dtype)
        out["z_enc"] = z_enc
        out["z_general"] = z_enc
        out["z"] = z_enc
        return out

    def _denorm_recon(
        self,
        recon_norm: torch.Tensor,
        orig_patches: torch.Tensor,
        stats: RevINStats,
        orig_length: int,
        project_mask: torch.Tensor,
        sample_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        wave_norm = patches_to_iq(recon_norm, orig_length)
        wave = self.revin.denormalize(wave_norm, stats)
        if sample_mask is not None:
            wave = wave.masked_fill(~sample_mask.unsqueeze(1), 0.0)
        recon = patchify_iq(wave, sample_mask, self.cfg.patch_size)[0]
        n = min(recon.shape[1], orig_patches.shape[1], recon_norm.shape[1])
        recon = recon[:, :n]
        orig = orig_patches[:, :n]
        if self.cfg.physics_project:
            recon = project_patch_energy(recon, orig, mask=project_mask[:, :n])
        return recon

    def _core_forward(
        self,
        iq: torch.Tensor,
        sample_mask: torch.Tensor,
        *,
        mask_mode: str | None,
        dataset_id: torch.Tensor | None,
        task_context: torch.Tensor | None = None,
        modality_id: torch.Tensor | None = None,
        complex_pair: torch.Tensor | bool | None = True,
        mae_strategy: str | None = None,
    ) -> dict[str, Any]:
        orig_patches, orig_patch_mask = patchify_iq(iq, sample_mask, self.cfg.patch_size)
        mae_mask, suffix_mask, span_mask, mask_strategy = self._mask_for_mode(
            orig_patch_mask,
            mask_mode,
            mae_strategy=mae_strategy,
        )
        target_mask = (mae_mask | suffix_mask | span_mask) & orig_patch_mask
        observed_sample_mask = self._observed_sample_mask(sample_mask, target_mask)
        iq_observed = iq.masked_fill(~observed_sample_mask.unsqueeze(1), 0.0)
        iq_norm, stats = self.revin.normalize(
            iq_observed,
            sample_mask,
            observed_mask=observed_sample_mask,
        )
        log_scale = None if stats.amp_aux is None else stats.amp_aux.log_scale
        amp_aux_vec = self.revin.amp_aux_vector(stats)
        # observed-only std 在静默背景上会塌缩；clip 只限制幅度，不把目标写进 stats。
        target_norm_wave = clip_normalized(
            self.revin.apply_stats(iq, stats, sample_mask),
            self.cfg.revin_clip,
        )
        iq_norm = clip_normalized(iq_norm, self.cfg.revin_clip)
        norm_patches, _ = patchify_iq(target_norm_wave, sample_mask, self.cfg.patch_size)
        batch_size = int(iq.shape[0])
        if modality_id is None:
            modality_id = self._resolve_modality_id(batch_size, device=iq.device)
        if complex_pair is None:
            complex_pair = self._resolve_complex_pair(batch_size, device=iq.device)
        context_phys = sequence_physics(
            iq_norm.detach(),
            observed_sample_mask,
            modality_id=modality_id,
            complex_pair=complex_pair,
        )
        global_phys_target = sequence_physics(
            target_norm_wave.detach(),
            sample_mask,
            modality_id=modality_id,
            complex_pair=complex_pair,
            log_scale=log_scale,
        )
        tok = self.tokenizer(
            iq_norm,
            sample_mask,
            modality_id=modality_id,
            complex_pair=complex_pair,
        )
        tokens = tok["tokens"]
        patch_mask = tok["patch_mask"]
        n = min(tokens.shape[1], orig_patches.shape[1], patch_mask.shape[1], norm_patches.shape[1])
        tokens = tokens[:, :n]
        patch_mask = patch_mask[:, :n] & orig_patch_mask[:, :n]
        orig_patches = orig_patches[:, :n]
        norm_patches = norm_patches[:, :n]
        phys = tok["patch_physics"][:, :n]
        phys_mask = tok.get("physics_mask")
        if phys_mask is not None:
            phys_mask = phys_mask[:, :n]
        # Tokenizer physics_proj 用相对量；Decoder FiLM / query 读出还原绝对 log_power。
        phys_dec = restore_absolute_log_power(phys, log_scale)
        context_phys = restore_absolute_log_power(context_phys, log_scale)
        mae_mask = mae_mask[:, :n] & patch_mask
        suffix_mask = suffix_mask[:, :n] & patch_mask
        span_mask = span_mask[:, :n] & patch_mask
        target_mask = (mae_mask | suffix_mask | span_mask) & patch_mask
        visible = _ensure_min_visible(patch_mask & ~target_mask, patch_mask)
        target_mask = patch_mask & ~visible
        mae_mask = mae_mask & target_mask
        suffix_mask = suffix_mask & target_mask
        span_mask = span_mask & target_mask
        h_vis, packed_enc = self._encode_tokens(tokens, visible, patch_mask)
        z_enc, h_enc = self._pool_encoder_identity(
            tokens,
            patch_mask,
            h_vis=h_vis,
            packed_enc=packed_enc,
            visible=visible,
        )
        dec = self.decoder(
            h_vis,
            tokens,
            patch_mask,
            visible,
            phys_dec,
            packed_encoder=packed_enc,
            sequence_packing=self.cfg.sequence_packing,
            skip_recon=bool(getattr(self, "_skip_recon", False)),
            target_mask=target_mask,
            context_physics=context_phys,
            task_context=task_context,
            physics_mask=phys_mask,
            amp_aux=amp_aux_vec,
        )
        skip_recon = dec["recon_norm"] is None
        if skip_recon:
            recon_norm = torch.zeros_like(norm_patches)
            recon = torch.zeros_like(orig_patches)
        else:
            recon_norm = dec["recon_norm"][:, :n]
            if self.cfg.legacy_target_energy_projection:
                project_mask = target_mask if target_mask.any() else patch_mask
            else:
                project_mask = visible
            recon = self._denorm_recon(
                recon_norm, orig_patches, stats, int(iq.shape[-1]), project_mask, sample_mask=sample_mask
            )
        n = min(n, recon.shape[1], recon_norm.shape[1], dec["patch_h"].shape[1], h_enc.shape[1])
        patch_h = dec["patch_h"][:, :n]
        h_enc = h_enc[:, :n]
        z_recon = dec["z"]
        representations: dict[str, torch.Tensor] = {
            "z_general": z_enc,
            "h_general": h_enc,
            "z_enc": z_enc,
            "h_enc": h_enc,
            "z_recon": z_recon,
            "h_recon": patch_h,
        }
        # MAE 时 h_enc 仅 visible 有有效状态；UTI 视图池化与之对齐。
        enc_pool_mask = (visible & patch_mask)[:, :n]
        if self.task_interface is not None:
            for view_name, (view_z, view_h) in self.task_interface.build_views(
                z_enc, h_enc, patch_mask=enc_pool_mask
            ).items():
                if view_name == "general":
                    continue
                representations[f"z_{view_name}"] = view_z
                representations[f"h_{view_name}"] = view_h
        domain_logits = self._domain_logits(z_enc, representations)
        out: dict[str, Any] = {
            **tok,
            "tokens": tokens,
            "patch_mask": patch_mask,
            "mae_mask": mae_mask[:, :n],
            "suffix_mask": suffix_mask[:, :n],
            "span_mask": span_mask[:, :n],
            "target_mask": target_mask[:, :n],
            "recon_mask": target_mask[:, :n],
            "visible": visible[:, :n],
            "observed_sample_mask": observed_sample_mask,
            "recon_norm": recon_norm[:, :n],
            "patch_targets_norm": norm_patches[:, :n],
            "mae_pred": recon[:, :n],
            "patch_targets": orig_patches[:, :n],
            "z": z_enc,
            "z_enc": z_enc,
            "z_general": z_enc,
            "z_recon": z_recon,
            "h_enc": h_enc,
            "h_general": h_enc,
            "h_recon": patch_h,
            "h_dec": dec["h_dec"],
            "h_full": dec["h_full"],
            "patch_h": patch_h,
            "query_h": dec.get("query_h"),
            "global_phys_pred": dec["global_phys_pred"],
            "global_phys_target": global_phys_target,
            "context_physics": context_phys,
            "domain_logits": domain_logits,
            "n_tokens": patch_mask.sum(dim=1),
            "revin_stats": stats,
            "amp_aux": amp_aux_vec,
            "mask_strategy": mask_strategy,
            "iq_length": int(iq.shape[-1]),
        }
        if dataset_id is not None:
            out["dataset_id"] = dataset_id
        for key, value in representations.items():
            out.setdefault(key, value)
        return out

    def _forward_chunked(
        self,
        iq: torch.Tensor,
        sample_mask: torch.Tensor,
        dataset_id: torch.Tensor | None,
        *,
        mask_mode: str | None,
        task_context: torch.Tensor | None = None,
        modality_id: torch.Tensor | None = None,
        complex_pair: torch.Tensor | bool | None = True,
        mae_strategy: str | None = None,
    ) -> dict[str, Any]:
        length = int(iq.shape[-1])
        overlap = self.cfg.chunk_overlap_samples
        starts = chunk_starts(length, self.cfg.chunk_len, overlap)
        zs: list[torch.Tensor] = []
        recon_wave = iq.new_zeros(iq.shape)
        weight = iq.new_zeros(iq.shape[0], iq.shape[-1])
        w_full = overlap_weights(self.cfg.chunk_len, overlap, iq.device, iq.dtype)
        last: dict[str, Any] | None = None
        for start in starts:
            sl = slice(start, min(length, start + self.cfg.chunk_len))
            if not sample_mask[:, sl].any():
                continue
            chunk_len = sl.stop - sl.start
            out = self._core_forward(
                iq[:, :, sl],
                sample_mask[:, sl],
                mask_mode=mask_mode,
                dataset_id=dataset_id,
                task_context=task_context,
                modality_id=modality_id,
                complex_pair=complex_pair,
                mae_strategy=mae_strategy,
            )
            if not out["patch_mask"].any():
                continue
            zs.append(out["z"])
            if not bool(getattr(self, "_skip_recon", False)):
                wave = patches_to_iq(out["mae_pred"], chunk_len)
                ww = w_full[:chunk_len]
                recon_wave[:, :, sl] = recon_wave[:, :, sl] + wave * ww.view(1, 1, -1)
                weight[:, sl] = weight[:, sl] + ww.view(1, -1)
            last = out
        if not zs:
            return self._core_forward(
                iq[:, :, : self.cfg.chunk_len],
                sample_mask[:, : self.cfg.chunk_len],
                mask_mode=mask_mode,
                dataset_id=dataset_id,
                task_context=task_context,
                modality_id=modality_id,
                complex_pair=complex_pair,
                mae_strategy=mae_strategy,
            )
        assert last is not None
        stacked = torch.stack(zs, dim=1)
        z = self.chunk_pool(stacked, key_padding_mask=None)
        last["z"] = F.normalize(z.float(), dim=-1).to(dtype=iq.dtype)
        last["chunk_z"] = stacked
        if bool(getattr(self, "_skip_recon", False)):
            last["recon_wave"] = recon_wave
        else:
            last["recon_wave"] = recon_wave / weight.unsqueeze(1).clamp_min(1.0e-8)
        return last

    @staticmethod
    def _pad_cat_tensors(tensors: list[torch.Tensor], *, dim: int = 1) -> torch.Tensor:
        if not tensors:
            raise ValueError("tensors 不能为空")
        if len(tensors) == 1:
            return tensors[0]
        max_len = max(t.shape[dim] for t in tensors)
        padded: list[torch.Tensor] = []
        for tensor in tensors:
            gap = max_len - tensor.shape[dim]
            if gap <= 0:
                padded.append(tensor)
                continue
            pad_spec = [0, 0] * (tensor.ndim - dim - 1) + [0, gap]
            padded.append(F.pad(tensor, pad_spec))
        return torch.cat(padded, dim=0)

    def _forward_single(
        self,
        iq: torch.Tensor,
        sample_mask: torch.Tensor,
        dataset_id: torch.Tensor | None,
        *,
        mask_mode: str | None,
        is_train: bool,
        task_context: torch.Tensor | None = None,
        modality_id: torch.Tensor | None = None,
        complex_pair: torch.Tensor | bool | None = True,
        mae_strategy: str | None = None,
    ) -> dict[str, Any]:
        length = int(sample_mask.sum().item())
        if (not is_train) and length > self.cfg.chunk_len:
            return self._forward_chunked(
                iq,
                sample_mask,
                dataset_id,
                mask_mode=mask_mode,
                task_context=task_context,
                modality_id=modality_id,
                complex_pair=complex_pair,
                mae_strategy=mae_strategy,
            )
        return self._core_forward(
            iq,
            sample_mask,
            mask_mode=mask_mode,
            dataset_id=dataset_id,
            task_context=task_context,
            modality_id=modality_id,
            complex_pair=complex_pair,
            mae_strategy=mae_strategy,
        )

    def _forward_heterogeneous_batch(
        self,
        iq: torch.Tensor,
        sample_mask: torch.Tensor,
        dataset_id: torch.Tensor | None,
        *,
        mask_mode: str | None,
        is_train: bool,
        mae_strategy: str | None = None,
    ) -> dict[str, Any]:
        batch_size = iq.shape[0]
        outs: list[dict[str, Any]] = []
        for i in range(batch_size):
            ds = dataset_id[i : i + 1] if dataset_id is not None else None
            outs.append(
                self._forward_single(
                    iq[i : i + 1],
                    sample_mask[i : i + 1],
                    ds,
                    mask_mode=mask_mode,
                    is_train=is_train,
                    mae_strategy=mae_strategy,
                )
            )
        merged: dict[str, Any] = {}
        keys = outs[0].keys()
        for key in keys:
            val0 = outs[0][key]
            if isinstance(val0, torch.Tensor):
                vals = [o[key] for o in outs]
                if val0.ndim == 0:
                    merged[key] = torch.stack(vals)
                elif val0.shape[0] == 1:
                    if val0.ndim >= 2 and any(v.shape[1:] != val0.shape[1:] for v in vals[1:]):
                        merged[key] = self._pad_cat_tensors(vals, dim=1)
                    else:
                        merged[key] = torch.cat(vals, dim=0)
                else:
                    merged[key] = val0
            elif key == "revin_stats":
                merged[key] = val0
            else:
                merged[key] = val0
        return merged

    def _run_backbone(
        self,
        iq_t: torch.Tensor,
        mask_t: torch.Tensor,
        dataset_id: torch.Tensor | None,
        *,
        mask_mode: str | None,
        is_train: bool,
        task_context: torch.Tensor | None = None,
        modality_id: torch.Tensor | None = None,
        complex_pair: torch.Tensor | bool | None = True,
        mae_strategy: str | None = None,
    ) -> dict[str, Any]:
        if (not is_train) and int(mask_t.sum(dim=1).max().item()) > self.cfg.chunk_len:
            return self._forward_chunked(
                iq_t,
                mask_t,
                dataset_id,
                mask_mode=mask_mode,
                task_context=task_context,
                modality_id=modality_id,
                complex_pair=complex_pair,
                mae_strategy=mae_strategy,
            )
        return self._core_forward(
            iq_t,
            mask_t,
            mask_mode=mask_mode,
            dataset_id=dataset_id,
            task_context=task_context,
            modality_id=modality_id,
            complex_pair=complex_pair,
            mae_strategy=mae_strategy,
        )

    @staticmethod
    def _detach_output(out: dict[str, Any]) -> dict[str, Any]:
        detached: dict[str, Any] = {}
        for key, value in out.items():
            if torch.is_tensor(value):
                detached[key] = value.detach()
            elif isinstance(value, RevINStats):
                amp = value.amp_aux
                detached_amp = None
                if amp is not None:
                    detached_amp = type(amp)(
                        log_scale=amp.log_scale.detach(),
                        log_peak=amp.log_peak.detach(),
                        papr_preclip=amp.papr_preclip.detach(),
                        scale_gap=amp.scale_gap.detach(),
                    )
                detached[key] = RevINStats(
                    mean=value.mean.detach(),
                    std=value.std.detach(),
                    amp_aux=detached_amp,
                )
            else:
                detached[key] = value
        return detached

    def encode_backbone(
        self,
        iq: torch.Tensor | list[torch.Tensor] | dict[str, Any] | None = None,
        sample_mask: torch.Tensor | None = None,
        *,
        batch: dict[str, Any] | None = None,
        mask_mode: str | None = None,
        dataset_id: torch.Tensor | None = None,
        signal_spec: SignalSpec | dict[str, Any] | None = None,
        channel_mask: torch.Tensor | list[torch.Tensor] | None = None,
        training: bool | None = None,
        skip_recon: bool = False,
        apply_aug: bool | None = None,
    ) -> dict[str, Any]:
        if isinstance(iq, dict):
            batch = iq
            iq = None
        if batch is not None:
            iq = batch.get("iq", batch.get("values"))
            sample_mask = batch.get("sample_mask", sample_mask)
            dataset_id = batch.get("dataset_id", dataset_id)
            signal_spec = batch.get("signal_spec", signal_spec)
            channel_mask = batch.get("channel_mask", channel_mask)
        if iq is None:
            raise ValueError("encode_backbone 需要 iq 或 batch")
        is_train = self.training if training is None else training
        if apply_aug is None:
            apply_aug = bool(is_train)
        prev_skip = self._skip_recon
        self._skip_recon = bool(skip_recon)
        try:
            iq = self._adapt_signal_input(iq, signal_spec, channel_mask)
            iq_t, mask_t = self._prepare_iq(iq, sample_mask, apply_aug=apply_aug, allow_chunk=True)
            modality_id = self._resolve_modality_id(
                iq_t.shape[0],
                device=iq_t.device,
                modality_id=(batch or {}).get("modality_id") if batch is not None else None,
                signal_spec=signal_spec,
            )
            complex_pair = self._resolve_complex_pair(
                iq_t.shape[0],
                device=iq_t.device,
                complex_pair=(batch or {}).get("complex_pair") if batch is not None else None,
                signal_spec=signal_spec,
            )
            return self._run_backbone(
                iq_t,
                mask_t,
                dataset_id,
                mask_mode=mask_mode,
                is_train=is_train,
                modality_id=modality_id,
                complex_pair=complex_pair,
            )
        finally:
            self._skip_recon = prev_skip

    def _should_use_generation_head(self, kind: str, head: nn.Module | None) -> bool:
        """有 Prediction/Imputation 头时默认走 head，避免 stage2 冻 decoder 后变成零预测。

        ``use_legacy_generation_heads=true`` 仍强制走头；
        仅当头缺失或 ``force_unified_generation=true`` 时回退 decoder ``recon_norm``。
        """
        if kind not in ("prediction", "imputation") or head is None:
            return False
        if bool(getattr(self.cfg, "use_legacy_generation_heads", False)):
            return True
        return not bool(getattr(self.cfg, "force_unified_generation", False))

    def forward_tasks(
        self,
        backbone_out: dict[str, Any],
        task: str,
        *,
        dataset_id: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        out = dict(backbone_out)
        if dataset_id is None:
            dataset_id = out.get("dataset_id")
        self.set_active_task(task)
        if bool(getattr(self, "truncate_backward", False)):
            out = self._refresh_encoder_identity(out)
        patch_h = out.get("patch_h", out.get("h_dec"))
        h_enc = out.get("h_enc", out.get("h_general", patch_h))
        z_enc = out.get("z_enc", out.get("z_general", out.get("z")))
        if self.task_interface is not None and h_enc is not None:
            if h_enc.dim() == 3 and h_enc.shape[1] == out["patch_mask"].shape[1] + 1:
                h_enc = h_enc[:, 1:]
            if patch_h is not None and patch_h.dim() == 3 and patch_h.shape[1] == out["patch_mask"].shape[1] + 1:
                patch_h = patch_h[:, 1:]
            # truncate_backward：在 detach 的 h_enc 上重算视图，view_adapters / encoder_pool 可训。
            view_mask = out.get("visible", out["patch_mask"])
            if view_mask is not None:
                view_mask = view_mask & out["patch_mask"]
            else:
                view_mask = out["patch_mask"]
            view_pairs = self.task_interface.build_views(
                z_enc, h_enc, patch_mask=view_mask
            )
            for view_name, (view_z, view_h) in view_pairs.items():
                if view_name == "general":
                    continue
                out[f"z_{view_name}"] = view_z
                out[f"h_{view_name}"] = view_h
            n_tokens = out["patch_mask"].shape[1]
            pos_dev = h_enc.device
            pos_dtype = h_enc.dtype
            pos = torch.linspace(0.0, 1.0, n_tokens, device=pos_dev, dtype=pos_dtype)
            pos = pos.view(1, n_tokens, 1).expand(h_enc.shape[0], -1, -1)
            target_coord = out.get("target_mask", torch.zeros_like(out["patch_mask"]))
            query_coords = torch.cat([pos, target_coord.to(dtype=pos_dtype).unsqueeze(-1)], dim=-1)
            task_metadata = out.get("task_metadata")
            if dataset_id is not None:
                if isinstance(task_metadata, dict):
                    task_metadata = {**task_metadata, "dataset_id": dataset_id}
                elif task_metadata is None:
                    task_metadata = {"dataset_id": dataset_id}
            features = self.task_interface(
                z_enc,
                h_enc,
                view_mask,
                task,
                recon_norm=out.get("recon_norm"),
                views=view_pairs,
                metadata=task_metadata,
                query_coords=query_coords,
            )
            features = apply_task_adapters(
                features,
                task=task,
                adapters=self.task_adapters,
                shared=self.shared_adapter,
            )
            out["task_pooled"] = features.pooled
            out["task_tokens"] = features.tokens
            out["task_query"] = features.query
            out["task_readout"] = features.readout
            head = self.get_task_head(task)
            kind = self.task_kind(task)
            unified_generation = kind in ("prediction", "imputation") and not self._should_use_generation_head(
                kind, head
            )
            if unified_generation:
                pred = out["recon_norm"]
                out["pred_patches"] = pred
                if kind == "prediction":
                    out["prediction_patches"] = pred
                else:
                    out["imputation_patches"] = pred
            elif head is not None:
                if kind == "classification" or task == "modulation":
                    out.update(head(features, dataset_id=dataset_id))
                elif kind == "emitter" or task == "emitter":
                    out.update(head(features, dataset_id=dataset_id))
                elif kind == "clustering":
                    ns = DEVICE_NAMESPACE if task == "emitter" else CONTENT_NAMESPACE
                    out.update(
                        head(
                            features,
                            registry=self.prototype_registry,
                            namespace=ns,
                            temperature=self._negcos_temperature,
                        )
                    )
                elif kind == "imputation":
                    out.update(head(features, span_mask=out.get("span_mask")))
                else:
                    out.update(head(features))
                if self.prototype_registry is not None and kind in ("classification", "emitter"):
                    pooled = out.get("task_pooled", features.pooled)
                    logits = out.get("task_logits", out.get("modulation_logits", out.get("emitter_logits")))
                    ns = DEVICE_NAMESPACE if kind == "emitter" or task == "emitter" else CONTENT_NAMESPACE
                    if pooled is not None and logits is not None and pooled.shape[-1] == self.prototype_registry.dim:
                        tau = float(self._negcos_temperature or 0.1)
                        out.update(self.prototype_registry.score(ns, pooled, logits, temperature=tau))
            if kind in ("prediction", "imputation") and "pred_patches" in out:
                pred = out["pred_patches"]
                n = min(pred.shape[1], out["patch_targets"].shape[1], out["patch_mask"].shape[1])
                pred = pred[:, :n]
                out["pred_patches"] = pred
                out["recon_norm"] = pred
                if unified_generation:
                    continue_denorm = False
                else:
                    continue_denorm = True
                stats = out.get("revin_stats")
                length = int(out["iq_length"]) if "iq_length" in out else n * self.cfg.patch_size
                if self.cfg.legacy_target_energy_projection:
                    project_mask = out.get("target_mask", out["patch_mask"])[:, :n]
                else:
                    project_mask = out.get("visible", out["patch_mask"])[:, :n]
                if continue_denorm and isinstance(stats, RevINStats):
                    out["mae_pred"] = self._denorm_recon(
                        pred,
                        out["patch_targets"][:, :n],
                        stats,
                        length,
                        project_mask,
                    )
                elif continue_denorm:
                    out["mae_pred"] = pred
        else:
            if task in ("modulation", "emitter", "recognition") and self.recognition_heads is not None:
                head_mode = "emitter" if task == "emitter" else "modulation"
                out.update(self.recognition_heads(out["z"], dataset_id=dataset_id, heads=head_mode))
            if task == "clustering" and self.clustering_head is not None:
                out.update(
                    self.clustering_head(
                        out["z"],
                        registry=self.prototype_registry,
                        namespace=CONTENT_NAMESPACE,
                        temperature=self._negcos_temperature,
                    )
                )
        kind = self.task_kind(task)
        if task in self.z_linear_probes and kind in ("classification", "emitter"):
            z_feat = out.get("z_enc", out.get("z_general", out["z"]))
            z_feat = F.normalize(z_feat.float(), dim=-1).to(dtype=z_feat.dtype)
            probe = self.z_linear_probes[task](z_feat)
            if kind == "emitter" or task == "emitter":
                mask = getattr(self.emitter_head, "dataset_class_mask", None) if self.emitter_head is not None else None
                probe = apply_emitter_dataset_mask(probe, dataset_id, mask)
            out["z_probe_logits"] = probe
        return out

    def forward(
        self,
        iq: torch.Tensor | list[torch.Tensor] | dict[str, Any] | None = None,
        sample_mask: torch.Tensor | None = None,
        *,
        batch: dict[str, Any] | None = None,
        mode: str = "pretrain",
        task: str | None = None,
        mask_mode: str | None = None,
        dataset_id: torch.Tensor | None = None,
        signal_spec: SignalSpec | dict[str, Any] | None = None,
        channel_mask: torch.Tensor | list[torch.Tensor] | None = None,
        task_metadata: torch.Tensor | dict[str, Any] | None = None,
        training: bool | None = None,
    ) -> dict[str, Any]:
        if isinstance(iq, dict):
            batch = iq
            iq = None
        if batch is not None:
            iq = batch.get("iq", batch.get("values"))
            sample_mask = batch.get("sample_mask", sample_mask)
            dataset_id = batch.get("dataset_id", dataset_id)
            signal_spec = batch.get("signal_spec", signal_spec)
            channel_mask = batch.get("channel_mask", channel_mask)
            task_metadata = batch.get("task_metadata", task_metadata)
        if iq is None:
            raise ValueError("forward 需要 iq 或 batch")

        if mode in ("task", "downstream") and task:
            kind = self.task_kind(task)
            default_mask = KIND_MASK_MODE.get(kind, "none")
            if mask_mode is None:
                mask_mode = default_mask
        elif mode in ("pretrain", "mae"):
            mask_mode = mask_mode or "mae"
        elif mode == "encode":
            mask_mode = mask_mode or "none"

        is_train = self.training if training is None else training
        apply_aug = bool(is_train) and mode in ("pretrain", "mae", "task", "downstream")
        task_mode = mode in ("task", "downstream")
        mae_strategy = None
        if mask_mode in ("random", "contiguous", "mixed"):
            mae_strategy = mask_mode
        elif mask_mode == "mae":
            mae_strategy = self.sample_mae_mask_strategy()
        generation_task = bool(task and self.task_kind(task) in ("prediction", "imputation"))
        skip_recon = bool(
            getattr(self, "skip_recon", False)
            and task_mode
            and not generation_task
        )
        truncate = bool(getattr(self, "truncate_backward", False) and task_mode)
        self._skip_recon = skip_recon
        if task:
            self.set_active_task(task)
        elif mode in ("pretrain", "mae"):
            self.set_active_task(None)
        iq = self._adapt_signal_input(iq, signal_spec, channel_mask)
        iq_t, mask_t = self._prepare_iq(iq, sample_mask, apply_aug=apply_aug, allow_chunk=True)
        modality_id = self._resolve_modality_id(
            iq_t.shape[0],
            device=iq_t.device,
            modality_id=(batch or {}).get("modality_id") if batch is not None else None,
            signal_spec=signal_spec,
        )
        complex_pair = self._resolve_complex_pair(
            iq_t.shape[0],
            device=iq_t.device,
            complex_pair=(batch or {}).get("complex_pair") if batch is not None else None,
            signal_spec=signal_spec,
        )
        task_context = None
        if self.task_interface is not None:
            if task is not None:
                task_context = self.task_interface.condition_vector(
                    task,
                    iq_t.shape[0],
                    device=iq_t.device,
                    dtype=iq_t.dtype,
                    metadata=task_metadata,
                    dataset_id=dataset_id,
                )
            elif mode in ("pretrain", "mae"):
                # 任务无关预训练：Decoder/UTI 条件不注入 dataset_id，避免重建偷域。
                pretrain_meta = task_metadata
                if isinstance(pretrain_meta, dict):
                    pretrain_meta = {k: v for k, v in pretrain_meta.items() if k != "dataset_id"}
                task_context = self.task_interface.condition_vector(
                    default_task_spec("pretrain"),
                    iq_t.shape[0],
                    device=iq_t.device,
                    dtype=iq_t.dtype,
                    metadata=pretrain_meta,
                    dataset_id=None,
                )
        if truncate:
            with torch.no_grad():
                out = self._run_backbone(
                    iq_t,
                    mask_t,
                    dataset_id,
                    mask_mode=mask_mode,
                    is_train=is_train,
                    task_context=task_context,
                    modality_id=modality_id,
                    complex_pair=complex_pair,
                    mae_strategy=mae_strategy,
                )
            out = self._detach_output(out)
        else:
            out = self._run_backbone(
                iq_t,
                mask_t,
                dataset_id,
                mask_mode=mask_mode,
                is_train=is_train,
                task_context=task_context,
                modality_id=modality_id,
                complex_pair=complex_pair,
                mae_strategy=mae_strategy,
            )
        if task_metadata is not None:
            out["task_metadata"] = task_metadata

        if mode in ("pretrain", "mae") and self.task_interface is not None:
            out = self._attach_uti_readouts(out, task="pretrain", allow_dataset_condition=False)
        if task_mode and task:
            out = self.forward_tasks(out, task, dataset_id=dataset_id)
        return out


def build_model_from_dict(data: dict[str, Any], **overrides: Any) -> SignalFoundationModel:
    payload = dict(data.get("model", data))
    payload.update(overrides)
    return SignalFoundationModel(SignalModelConfig.from_dict(payload))
