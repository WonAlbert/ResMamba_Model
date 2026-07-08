from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from resmamba_signal_model.data.packing import build_cu_seqlens, build_seq_idx, pad_sequence_list, segment_slices
from resmamba_signal_model.models.heads import PrototypeClusteringHead, RecognitionHeads
from resmamba_signal_model.models.mamba_backbone import Mamba2Stack
from resmamba_signal_model.models.resmamba_backbone import ResMambaStack
from resmamba_signal_model.models.tokenizer import MultiScaleResNetTokenizer, MultiScaleTokenizerConfig


@dataclass
class ResMambaSignalConfig:
    d_model: int = 256
    encoder_layers: int = 4
    decoder_layers: int = 1
    space_layers: int = 1
    mamba_d_state: int = 64
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_headdim: int = 64
    mamba_ngroups: int = 1
    mamba_chunk_size: int = 256
    dropout: float = 0.1
    patch_size: int = 8
    mask_ratio: float = 0.5
    max_tokens: int = 512
    num_datasets: int = 16
    num_mod_classes: int = 64
    num_emitters: int = 256
    num_prototypes: int = 128
    sequence_packing: bool = False
    tokenizer: MultiScaleTokenizerConfig = field(default_factory=MultiScaleTokenizerConfig)

    def __post_init__(self) -> None:
        self.tokenizer.d_model = self.d_model
        self.tokenizer.patch_size = self.patch_size
        self.tokenizer.max_tokens = self.max_tokens
        self.tokenizer.num_datasets = self.num_datasets
        self.tokenizer.dropout = self.dropout


class ResMambaSignalModel(nn.Module):
    space_names = ("mod_specific", "emitter_specific", "long_context_shared", "cross_domain_shared")

    def __init__(self, cfg: ResMambaSignalConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = MultiScaleResNetTokenizer(cfg.tokenizer)
        stack_kwargs = dict(d_model=cfg.d_model, d_state=cfg.mamba_d_state, d_conv=cfg.mamba_d_conv, expand=cfg.mamba_expand, headdim=cfg.mamba_headdim, dropout=cfg.dropout, ngroups=cfg.mamba_ngroups, chunk_size=cfg.mamba_chunk_size)
        self.encoder = ResMambaStack(cfg.encoder_layers, **stack_kwargs)
        self.decoder = Mamba2Stack(cfg.decoder_layers, **stack_kwargs)
        self.space_projs = nn.ModuleDict({name: nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.d_model)) for name in self.space_names})
        self.space_backbones = nn.ModuleDict({name: ResMambaStack(cfg.space_layers, **stack_kwargs) for name in self.space_names})
        self.space_pool = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model))
        self.mod_fuse = nn.Sequential(nn.LayerNorm(cfg.d_model * 3), nn.Linear(cfg.d_model * 3, cfg.d_model), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.d_model, cfg.d_model))
        self.emitter_fuse = nn.Sequential(nn.LayerNorm(cfg.d_model * 3), nn.Linear(cfg.d_model * 3, cfg.d_model), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.d_model, cfg.d_model))
        self.shared_fuse = nn.Sequential(nn.LayerNorm(cfg.d_model * 2), nn.Linear(cfg.d_model * 2, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.d_model))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.iq_mae_out = nn.Linear(cfg.d_model, 2 * cfg.patch_size)
        self.physical_out = nn.Linear(cfg.d_model, 5)
        self.future_out = nn.Linear(cfg.d_model, cfg.d_model)
        self.recognition_heads = RecognitionHeads(cfg.d_model, cfg.num_mod_classes, cfg.num_emitters, cfg.dropout, num_datasets=cfg.num_datasets + 1)
        self.clustering_head = PrototypeClusteringHead(cfg.d_model, proj_dim=max(64, cfg.d_model // 2), num_prototypes=cfg.num_prototypes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    @staticmethod
    def _uses_packed_batch(batch: dict[str, Any]) -> bool:
        iq = batch.get("iq")
        return isinstance(iq, list)

    def _sample_tokenizer_kwargs(self, batch: dict[str, Any], index: int) -> dict[str, torch.Tensor | None]:
        iq = batch["iq"]
        if isinstance(iq, list):
            sample_iq = iq[index].unsqueeze(0)
            sample_mask = None
        else:
            sample_iq = iq[index : index + 1]
            sample_mask = batch.get("sample_mask")
            sample_mask = sample_mask[index : index + 1] if sample_mask is not None else None
        kwargs: dict[str, torch.Tensor | None] = {"iq": sample_iq, "sample_mask": sample_mask}
        for key in ("dataset_id", "task_type_id"):
            value = batch.get(key)
            if value is not None:
                kwargs[key] = value[index : index + 1]
        return kwargs

    def _tokenizer_inputs(self, batch: dict[str, Any]) -> dict[str, torch.Tensor | None]:
        if self._uses_packed_batch(batch):
            raise TypeError("packed batch 请使用 _tokenize_samples")
        return {"iq": batch["iq"], "sample_mask": batch.get("sample_mask"), "dataset_id": batch.get("dataset_id"), "task_type_id": batch.get("task_type_id")}

    def _batch_variable_iq(self, batch: dict[str, Any]) -> dict[str, torch.Tensor | None]:
        iq_list = batch["iq"]
        device = iq_list[0].device
        dtype = iq_list[0].dtype
        max_len = max(int(tensor.shape[-1]) for tensor in iq_list)
        iq = torch.zeros(len(iq_list), 2, max_len, device=device, dtype=dtype)
        sample_mask = torch.zeros(len(iq_list), max_len, dtype=torch.bool, device=device)
        for i, tensor in enumerate(iq_list):
            length = int(tensor.shape[-1])
            iq[i, :, :length] = tensor
            sample_mask[i, :length] = True
        out: dict[str, torch.Tensor | None] = {"iq": iq, "sample_mask": sample_mask}
        for key in ("dataset_id", "task_type_id"):
            value = batch.get(key)
            if value is not None:
                out[key] = value
        return out

    def _split_tokenizer_batch(self, tok: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        batch_size = tok["tokens"].shape[0]
        special = self.tokenizer.num_special_tokens
        samples: list[dict[str, torch.Tensor]] = []
        for i in range(batch_size):
            num_patches = int(tok["patch_mask"][i].sum().item())
            num_tokens = special + num_patches
            samples.append({
                "tokens": tok["tokens"][i : i + 1, :num_tokens],
                "token_mask": tok["token_mask"][i : i + 1, :num_tokens],
                "patch_mask": tok["patch_mask"][i : i + 1, :num_patches],
                "iq_patch_targets": tok["iq_patch_targets"][i : i + 1, :num_patches],
                "physical_stats": tok["physical_stats"][i : i + 1],
                "patch_offset": tok["patch_offset"],
                "physical_token_index": tok["physical_token_index"],
            })
        return samples

    def _tokenize_samples(self, batch: dict[str, Any]) -> list[dict[str, torch.Tensor]]:
        if isinstance(batch["iq"], list):
            return self._split_tokenizer_batch(self.tokenizer(**self._batch_variable_iq(batch)))
        batch_size = batch["iq"].shape[0]
        return [self.tokenizer(**self._sample_tokenizer_kwargs(batch, i)) for i in range(batch_size)]

    def _pack_token_samples(self, samples: list[dict[str, torch.Tensor]]) -> dict[str, Any]:
        lengths = [int(sample["tokens"].shape[1]) for sample in samples]
        device = samples[0]["tokens"].device
        cu_seqlens = build_cu_seqlens(lengths, device=device)
        seq_idx = build_seq_idx(lengths, device=device)
        tokens = torch.cat([sample["tokens"] for sample in samples], dim=1)
        return {
            "tokens": tokens,
            "seq_idx": seq_idx,
            "cu_seqlens": cu_seqlens,
            "token_lengths": lengths,
            "per_sample": samples,
            "patch_offset": int(samples[0]["patch_offset"]),
            "physical_token_index": int(samples[0]["physical_token_index"]),
        }

    def _apply_mae_mask_packed(self, tokens: torch.Tensor, packed: dict[str, Any], mae_masks: list[torch.Tensor]) -> torch.Tensor:
        tokens = tokens.clone()
        patch_offset = int(packed["patch_offset"])
        for (start, _end), sample, mae_mask in zip(segment_slices(packed["cu_seqlens"]), packed["per_sample"], mae_masks):
            num_patches = int(sample["patch_mask"].shape[1])
            limit = min(num_patches, mae_mask.shape[0], tokens.shape[1] - patch_offset - start)
            if limit <= 0:
                continue
            seg_start = start + patch_offset
            masked = self.mask_token.expand(1, limit, -1)
            orig = tokens[:, seg_start : seg_start + limit]
            tokens[:, seg_start : seg_start + limit] = torch.where(mae_mask[:limit].view(1, limit, 1), masked, orig)
        return tokens

    def make_mae_mask(self, patch_mask: torch.Tensor, mask_ratio: float | None = None) -> torch.Tensor:
        ratio = self.cfg.mask_ratio if mask_ratio is None else mask_ratio
        rand = torch.rand(patch_mask.shape, device=patch_mask.device).masked_fill(~patch_mask, 2.0)
        num_valid = patch_mask.sum(dim=-1).clamp_min(1)
        num_mask = (num_valid.float() * ratio).round().long().clamp_min(1)
        mae_mask = torch.zeros_like(patch_mask)
        for b in range(patch_mask.shape[0]):
            mae_mask[b, rand[b].argsort()[: num_mask[b]]] = True
        return mae_mask & patch_mask

    def _stack_patch_tensors(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        padded, _mask = pad_sequence_list(tensors, pad=0.0)
        return padded

    def _encode_packed(self, batch: dict[str, Any], *, samples: list[dict[str, torch.Tensor]] | None = None, mae_masks: list[torch.Tensor] | None = None) -> dict[str, Any]:
        if samples is None:
            samples = self._tokenize_samples(batch)
        packed = self._pack_token_samples(samples)
        tokens = packed["tokens"]
        if mae_masks is not None:
            tokens = self._apply_mae_mask_packed(tokens, packed, mae_masks)
        pack_kwargs = {"seq_idx": packed["seq_idx"], "cu_seqlens": packed["cu_seqlens"]}
        hidden = self.encoder(tokens, **pack_kwargs)
        cls_rows = [hidden[0, start] for start, _end in segment_slices(packed["cu_seqlens"])]
        patch_masks = [sample["patch_mask"].squeeze(0) for sample in samples]
        iq_patch_targets = [sample["iq_patch_targets"].squeeze(0) for sample in samples]
        physical_stats = torch.stack([sample["physical_stats"].squeeze(0) for sample in samples], dim=0)
        return {
            **packed,
            "hidden": hidden,
            "cls": torch.stack(cls_rows, dim=0),
            "patch_mask": self._stack_patch_tensors(patch_masks),
            "iq_patch_targets": self._stack_patch_tensors(iq_patch_targets),
            "physical_stats": physical_stats,
            "token_mask": None,
        }

    def encode(self, batch: dict[str, Any], *, mae_mask: torch.Tensor | None = None, mae_masks: list[torch.Tensor] | None = None, token_samples: list[dict[str, torch.Tensor]] | None = None) -> dict[str, Any]:
        if self.cfg.sequence_packing and self._uses_packed_batch(batch):
            if mae_mask is not None:
                raise ValueError("packed 模式下请传 mae_masks 列表，而非 mae_mask 张量")
            return self._encode_packed(batch, samples=token_samples, mae_masks=mae_masks)
        tok = self.tokenizer(**self._tokenizer_inputs(batch))
        tokens = tok["tokens"]
        patch_offset = int(tok["patch_offset"])
        if mae_mask is not None:
            tokens = tokens.clone()
            limit = min(mae_mask.shape[1], tokens.shape[1] - patch_offset)
            tokens[:, patch_offset : patch_offset + limit] = torch.where(
                mae_mask[:, :limit].unsqueeze(-1),
                self.mask_token.expand(tokens.shape[0], limit, -1),
                tokens[:, patch_offset : patch_offset + limit],
            )
        hidden = self.encoder(tokens, key_padding_mask=~tok["token_mask"])
        return {**tok, "hidden": hidden, "cls": hidden[:, 0]}

    def _space_outputs(self, enc: dict[str, Any]) -> dict[str, torch.Tensor]:
        if enc.get("seq_idx") is not None:
            pack_kwargs = {"seq_idx": enc["seq_idx"], "cu_seqlens": enc["cu_seqlens"]}
            spaces: dict[str, torch.Tensor] = {}
            for name in self.space_names:
                tokens = self.space_projs[name](enc["hidden"])
                hidden = self.space_backbones[name](tokens, **pack_kwargs)
                cls_rows = [hidden[0, start] for start, _end in segment_slices(enc["cu_seqlens"])]
                spaces[name] = self.space_pool(torch.stack(cls_rows, dim=0))
            return spaces

        key_padding_mask = ~enc["token_mask"]
        spaces = {}
        for name in self.space_names:
            tokens = self.space_projs[name](enc["hidden"])
            hidden = self.space_backbones[name](tokens, key_padding_mask=key_padding_mask)
            spaces[name] = self.space_pool(hidden[:, 0])
        return spaces

    def route_representations(self, spaces: dict[str, torch.Tensor], task_type_id: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        mod = self.mod_fuse(torch.cat([spaces["mod_specific"], spaces["long_context_shared"], spaces["cross_domain_shared"]], dim=-1))
        emitter = self.emitter_fuse(torch.cat([spaces["emitter_specific"], spaces["long_context_shared"], spaces["cross_domain_shared"]], dim=-1))
        cluster = self.shared_fuse(torch.cat([spaces["long_context_shared"], spaces["cross_domain_shared"]], dim=-1))
        routed = mod
        if task_type_id is not None:
            task = task_type_id.long().clamp(0, 1).view(-1, 1)
            routed = torch.where(task.eq(1), emitter, mod)
        return {"modulation_repr": mod, "emitter_repr": emitter, "cluster_repr": cluster, "routed_repr": routed}

    def _decode_mae(self, batch: dict[str, Any], enc: dict[str, Any], mae_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if enc.get("seq_idx") is not None:
            return self._decode_mae_packed(batch, enc, mae_mask)
        dec = self.decoder(enc["hidden"], key_padding_mask=~enc["token_mask"])
        patch_offset = int(enc["patch_offset"])
        num_patches = mae_mask.shape[1]
        patch_h = dec[:, patch_offset : patch_offset + num_patches]
        iq_pred = self.iq_mae_out(patch_h).view(enc["hidden"].shape[0], num_patches, 2, self.cfg.patch_size)
        future_pred = self.future_out(patch_h[:, :-1]) if num_patches > 1 else None
        future_target = enc["hidden"][:, patch_offset + 1 : patch_offset + num_patches].detach() if num_patches > 1 else None
        future_mask = enc["patch_mask"][:, :-1] & enc["patch_mask"][:, 1:] if num_patches > 1 else None
        return {
            "mae_pred": iq_pred,
            "mae_mask": mae_mask,
            "patch_targets": enc["iq_patch_targets"],
            "physical_pred": self.physical_out(enc["hidden"][:, enc["physical_token_index"]]),
            "physical_targets": enc["physical_stats"],
            "future_pred": future_pred,
            "future_targets": future_target,
            "future_mask": future_mask,
        }

    def _decode_mae_packed(self, batch: dict[str, Any], enc: dict[str, Any], mae_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        dec = self.decoder(enc["hidden"], seq_idx=enc["seq_idx"], cu_seqlens=enc["cu_seqlens"])
        patch_offset = int(enc["patch_offset"])
        batch_size = mae_mask.shape[0]
        iq_preds: list[torch.Tensor] = []
        physical_preds: list[torch.Tensor] = []
        future_preds: list[torch.Tensor] = []
        future_targets: list[torch.Tensor] = []
        future_masks: list[torch.Tensor] = []
        for (start, _end), sample in zip(segment_slices(enc["cu_seqlens"]), enc["per_sample"]):
            num_patches = int(sample["patch_mask"].shape[1])
            patch_h = dec[:, start + patch_offset : start + patch_offset + num_patches]
            iq_preds.append(self.iq_mae_out(patch_h).view(num_patches, 2, self.cfg.patch_size))
            physical_preds.append(self.physical_out(dec[0, start + enc["physical_token_index"]]))
            if num_patches > 1:
                future_preds.append(self.future_out(patch_h[:, :-1]).squeeze(0))
                future_targets.append(enc["hidden"][:, start + patch_offset + 1 : start + patch_offset + num_patches].detach().squeeze(0))
                patch_mask = sample["patch_mask"].squeeze(0)
                future_masks.append(patch_mask[:-1] & patch_mask[1:])
        iq_pred = self._stack_patch_tensors(iq_preds)
        future_pred = self._stack_patch_tensors(future_preds) if future_preds else None
        future_target = self._stack_patch_tensors(future_targets) if future_targets else None
        future_mask, _ = pad_sequence_list(future_masks, pad=False) if future_masks else (None, None)
        return {
            "mae_pred": iq_pred,
            "mae_mask": mae_mask,
            "patch_targets": enc["iq_patch_targets"],
            "physical_pred": torch.stack(physical_preds, dim=0),
            "physical_targets": enc["physical_stats"],
            "future_pred": future_pred,
            "future_targets": future_target,
            "future_mask": future_mask,
        }

    def _encode_for_mae(
        self,
        batch: dict[str, Any],
        *,
        packed_batch: bool,
        token_samples: list[dict[str, torch.Tensor]] | None = None,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        if packed_batch:
            samples = token_samples if token_samples is not None else self._tokenize_samples(batch)
            mae_masks = [self.make_mae_mask(sample["patch_mask"]).squeeze(0) for sample in samples]
            mae_mask = self._stack_patch_tensors(mae_masks)
            enc = self.encode(batch, mae_masks=mae_masks, token_samples=samples)
            return enc, mae_mask
        with torch.no_grad():
            preview = self.tokenizer(**self._tokenizer_inputs(batch))
        mae_mask = self.make_mae_mask(preview["patch_mask"])
        enc = self.encode(batch, mae_mask=mae_mask)
        return enc, mae_mask

    def forward(self, batch: dict[str, Any], *, mode: str = "mae", task: str = "modulation") -> dict[str, torch.Tensor]:
        packed_batch = self.cfg.sequence_packing and self._uses_packed_batch(batch)
        if mode == "task" and task == "prediction":
            enc, mae_mask = self._encode_for_mae(batch, packed_batch=packed_batch)
            return self._decode_mae(batch, enc, mae_mask)
        if mode == "mae":
            enc, mae_mask = self._encode_for_mae(batch, packed_batch=packed_batch)
            spaces = self._space_outputs(enc)
            routed = self.route_representations(spaces, batch.get("task_type_id"))
            return {**enc, "spaces": spaces, **routed, **self._decode_mae(batch, enc, mae_mask)}
        enc = self.encode(batch)
        spaces = self._space_outputs(enc)
        routed = self.route_representations(spaces, batch.get("task_type_id"))
        out = {**enc, "spaces": spaces, **routed}
        if mode == "encode":
            return out
        if task in ("modulation", "emitter", "recognition"):
            h = routed["emitter_repr"] if task == "emitter" else routed["modulation_repr"]
            head_mode = "emitter" if task == "emitter" else "modulation"
            out.update(self.recognition_heads(h, dataset_id=batch.get("dataset_id"), heads=head_mode))
        if task == "clustering":
            out.update(self.clustering_head(routed["cluster_repr"]))
        return out
