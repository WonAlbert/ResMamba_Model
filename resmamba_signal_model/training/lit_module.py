from __future__ import annotations

import logging
import math
import time
from typing import Any

import torch
import torch.nn.functional as F

from resmamba_signal_model.models.model import SignalFoundationModel, SignalModelConfig
from resmamba_signal_model.models.prototypes import PretrainPrototypeDisk, clustering_registry_namespace
from resmamba_signal_model.models.peft import TOKENIZER_LAST_PREFIXES
from resmamba_signal_model.models.task_interface import SOURCE_TO_TASK
from resmamba_signal_model.training.task_catalog import resolve_task_catalog
from resmamba_signal_model.training.data_module import (
    PRETRAIN_DISC_LABEL_KEYS,
    merge_source_batches,
    pretrain_collate_firewall,
)
from resmamba_signal_model.data.chronos_sampling import apply_iq_mixup, parse_chronos_sampling_cfg
from resmamba_signal_model.training.checkpointing import (
    aggregate_classification_geomean,
    aggregate_multitask_geomean,
    aggregate_specialist_geomean,
    aggregate_val_monitor,
    prune_inactive_task_val_metrics,
)
from resmamba_signal_model.training.clustering_labels import load_dataset_id_names, resolve_cluster_eval_labels
from resmamba_signal_model.training.freeze import iter_head_param_prefixes
from resmamba_signal_model.training.logging_utils import format_val_epoch_metrics, should_log_loss_part
from resmamba_signal_model.training.losses import (
    _first_present,
    downstream_task_loss,
    negcos_temperature,
    reconstruction_monitor_loss,
    resolve_pretrain_supcon_labels,
    weighted_pretrain_loss,
)
from resmamba_signal_model.training.lr_schedule import (
    apply_param_group_lr_scales,
    as_torch_lr_scheduler,
    build_lr_scheduler,
    resolve_lr_schedule,
    scale_lr_for_grad_accum,
    total_optimizer_steps_from_cfg,
    use_cosine_lr_decay,
)
from resmamba_signal_model.training.mix import resolve_mix_strategy
from resmamba_signal_model.training.metrics import (
    classification_epoch_scores,
    clustering_epoch_scores,
    dataset_display_name,
    masked_patch_mae,
    masked_patch_mse,
    openset_detection_metrics,
    reconstruction_epoch_scores,
    reconstruction_eval_pair,
    ssim_iq,
    ssim_iq_accumulate,
)
from resmamba_signal_model.training.mix import DynamicRatioScheduler
from resmamba_signal_model.training.emitter_labels import (
    GlobalEmitterLabelMap,
    build_global_emitter_label_map,
    global_emitter_labels,
)
from resmamba_signal_model.training.modulation_labels import (
    CompactModulationLabelMap,
    build_compact_modulation_label_map,
    remap_modulation_labels,
)
from resmamba_signal_model.training.compact_labels import compact_task_labels_enabled

try:
    import lightning as L
except ImportError:  # pragma: no cover
    L = None  # type: ignore[misc, assignment]


def _source_key_from_batch(batch: dict[str, Any]) -> str | None:
    name = batch.get("source_name")
    if isinstance(name, str) and name:
        return name
    if isinstance(name, (list, tuple)) and name:
        first = name[0]
        if isinstance(first, bytes):
            first = first.decode()
        text = str(first)
        return text or None
    return None


def _as_source_map(batch: Any, *, default_name: str | None = None) -> dict[str, dict[str, Any]]:
    """把 CombinedLoader 输出解析成 {source_name: batch}。"""
    if isinstance(batch, (tuple, list)) and batch and isinstance(batch[0], dict):
        batch = batch[0]
    if isinstance(batch, dict) and ("iq" in batch or "values" in batch):
        key = _source_key_from_batch(batch) or default_name or "default"
        return {key: batch}
    if isinstance(batch, dict):
        out: dict[str, dict[str, Any]] = {}
        for key, value in batch.items():
            if value is None:
                continue
            if isinstance(value, dict) and ("iq" in value or "values" in value):
                out[str(key)] = value
        if out:
            return out
    raise TypeError(f"无法解析 CombinedLoader batch: {type(batch)}")


def _resolve_batch_stem(batch: dict[str, Any]) -> str | None:
    """从 collate 后的 batch 取当前同质 H5 stem（预训练 step 诊断用）。"""
    stems = batch.get("moe_route_stem")
    if isinstance(stems, str) and stems:
        return stems
    if isinstance(stems, (list, tuple)) and stems:
        first = stems[0]
        if isinstance(first, bytes):
            first = first.decode()
        return str(first) if first is not None and str(first) else None
    source = batch.get("source_name")
    if isinstance(source, str) and source and source != "pretrain":
        return source
    return None


def _per_source_mse(outputs: dict[str, torch.Tensor], source_names: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    pred = outputs.get("recon_norm", outputs["mae_pred"])
    target = outputs.get("patch_targets_norm", outputs["patch_targets"])
    mask = outputs.get("mae_mask", outputs.get("patch_mask"))
    n_tokens = outputs["n_tokens"]
    losses: dict[str, float] = {}
    tokens: dict[str, float] = {}
    n = min(len(source_names), int(pred.shape[0]), int(mask.shape[0]), int(n_tokens.shape[0]))
    for i in range(n):
        name = source_names[i]
        m = mask[i]
        n_tok = float(n_tokens[i].detach())
        if m.any():
            length = min(int(pred.shape[1]), int(target.shape[1]), int(m.shape[0]))
            diff = pred[i, :length][m[:length]].float() - target[i, :length][m[:length]].float()
            value = float(diff.square().mean().detach())
        else:
            value = 0.0
        losses[name] = losses.get(name, 0.0) + value * n_tok
        tokens[name] = tokens.get(name, 0.0) + n_tok
    return losses, tokens


def sanitize_nonfinite_grads(module: torch.nn.Module) -> int:
    """把非有限梯度清零，避免 ``clip_grad_norm_`` 的 NaN 系数污染全部参数。"""
    n_bad = 0
    for param in module.parameters():
        grad = param.grad
        if grad is None or torch.isfinite(grad).all():
            continue
        param.grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        n_bad += 1
    return n_bad


if L is None:  # pragma: no cover
    _Base = object
else:
    _Base = L.LightningModule


class SignalLitModule(_Base):
    def __init__(
        self,
        model: SignalFoundationModel,
        train_cfg: dict[str, Any],
        *,
        stage: str = "pretrain",
        mix: DynamicRatioScheduler | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.train_cfg = train_cfg
        self.stage = stage
        self.mix = mix
        self.loss_weights = dict(
            train_cfg.get("loss_weights")
            or {"mae": 1.0, "physical": 0.2, "readout": 0.1, "structure": 0.2, "structure_phase": 0.15}
        )
        self._last_step_t = time.perf_counter()
        self.catalog = resolve_task_catalog(train_cfg)
        self._dataset_id_names: dict[int, str] | None = None
        self._last_val_report: dict[str, Any] = {}
        self._ema_box: dict[str, Any] = {"teacher": None, "state": None, "distill": None}
        self._frozen_prototypes: dict[str, torch.Tensor] = {}
        self._val_absorb_embeds: list[torch.Tensor] = []
        self._val_absorb_scores: list[torch.Tensor] = []
        self._val_openset: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]] = []
        self._emitter_offset_lookup: torch.Tensor | None = None
        self._ld_model_offset_lookup: torch.Tensor | None = None
        self._tx_modulation_offset_lookup: torch.Tensor | None = None
        self._tx_modulation_canonical_lookup: torch.Tensor | None = None
        self._intrapulse_compact_lookup: torch.Tensor | None = None
        self._warned_detached_loss = False
        self._mse_log_ema: float | None = None
        self._mae_log_ema: float | None = None
        self._recon_mse_log_ema: float | None = None
        self.pretrain_disc: PretrainPrototypeDisk | None = None
        if stage == "pretrain" and float(self.loss_weights.get("proto_swav", 0.0) or 0.0) > 0.0:
            disc_cfg = train_cfg.get("pretrain_disc") if isinstance(train_cfg.get("pretrain_disc"), dict) else {}
            d_model = int(getattr(model.cfg, "d_model", 64))
            self.pretrain_disc = PretrainPrototypeDisk(
                d_model,
                proj_dim=int(disc_cfg.get("proj_dim", 128) or 128),
                num_prototypes=int(disc_cfg.get("num_prototypes", 32) or 32),
                temperature=float(disc_cfg.get("temperature", 0.1) or 0.1),
                view_dropout=float(disc_cfg.get("view_dropout", 0.1) or 0.1),
            )
        self._total_log_ema: float | None = None
        self._total_train_ema: float | None = None
        self._stem_total_ema: dict[str, float] = {}
        self._mse_stem_ema: dict[str, float] = {}
        self._mae_stem_ema: dict[str, float] = {}
        self._recon_mse_stem_ema: dict[str, float] = {}
        self._recon_mse_stem_baseline: dict[str, float] = {}
        self._recon_mse_stem_updates: dict[str, int] = {}
        self._adaptive_mae_unlocked = bool(getattr(self.model.cfg, "adaptive_mae_mask", False))
        self._init_compact_label_lookups()
        self._reset_val_buffers()

    def _init_compact_label_lookups(self) -> None:
        """阶段二/三：调制/个体用下游紧凑标签空间。"""
        if not compact_task_labels_enabled(self.train_cfg, stage=self.stage):
            return
        root = self.train_cfg.get("rfdata_root")
        if not root:
            return
        compact_emitter = self.train_cfg.get("compact_emitter")
        if isinstance(compact_emitter, dict) and compact_emitter.get("offsets"):
            offsets = {int(k): int(v) for k, v in dict(compact_emitter["offsets"]).items()}
            counts_raw = compact_emitter.get("class_counts") or {}
            counts = {int(k): int(v) for k, v in dict(counts_raw).items()} if counts_raw else None
            names_raw = compact_emitter.get("datasets") or {}
            names = {int(k): str(v) for k, v in dict(names_raw).items()}
            emitter_map = GlobalEmitterLabelMap(
                offsets=offsets,
                dataset_names=names,
                num_emitters=int(compact_emitter.get("num_emitters") or self.model.cfg.num_emitters),
                class_counts=counts,
            )
            self._emitter_offset_lookup = emitter_map.offset_lookup()
        else:
            try:
                emitter_map = build_global_emitter_label_map(root, train_cfg=self.train_cfg)
                self._emitter_offset_lookup = emitter_map.offset_lookup()
            except (FileNotFoundError, KeyError, OSError):
                self._emitter_offset_lookup = None

        self._load_tx_modulation_label_lookups(root)
        self._intrapulse_compact_lookup = self._load_compact_modulation_lookup(
            root, key="compact_intrapulse", fallback_from_cfg=False
        )
        self._ld_model_offset_lookup = self._load_ld_model_offset_lookup(root)

    def _load_ld_model_offset_lookup(self, root: str) -> torch.Tensor | None:
        payload = self.train_cfg.get("compact_ld_model")
        if isinstance(payload, dict) and payload.get("offsets"):
            offsets = {int(k): int(v) for k, v in dict(payload["offsets"]).items()}
            counts_raw = payload.get("class_counts") or {}
            counts = {int(k): int(v) for k, v in dict(counts_raw).items()} if counts_raw else None
            names_raw = payload.get("datasets") or {}
            names = {int(k): str(v) for k, v in dict(names_raw).items()}
            label_map = GlobalEmitterLabelMap(
                offsets=offsets,
                dataset_names=names,
                num_emitters=int(payload.get("num_emitters") or 0),
                class_counts=counts,
            )
            return label_map.offset_lookup()
        try:
            from resmamba_signal_model.training.emitter_labels import build_global_radar_model_label_map

            label_map = build_global_radar_model_label_map(root, train_cfg=self.train_cfg)
            if int(label_map.num_emitters) <= 0:
                return None
            return label_map.offset_lookup()
        except (FileNotFoundError, KeyError, OSError):
            return None

    def _load_tx_modulation_label_lookups(self, root: str) -> None:
        from resmamba_signal_model.training.compact_labels import tx_modulation_map_from_train_cfg
        from resmamba_signal_model.training.modulation_labels import build_global_comm_modulation_label_map

        tx_map, canonical = tx_modulation_map_from_train_cfg(self.train_cfg)
        if tx_map is None:
            try:
                tx_map, canonical = build_global_comm_modulation_label_map(root, train_cfg=self.train_cfg)
            except (FileNotFoundError, KeyError, OSError):
                return
        if int(tx_map.num_emitters) <= 0:
            return
        self._tx_modulation_offset_lookup = tx_map.offset_lookup()
        if canonical is not None:
            self._tx_modulation_canonical_lookup = canonical.lookup()

    def _load_compact_modulation_lookup(
        self,
        root: str,
        *,
        key: str,
        fallback_from_cfg: bool,
    ) -> torch.Tensor | None:
        payload = self.train_cfg.get(key)
        if isinstance(payload, dict) and payload.get("old_to_new"):
            old_to_new = {int(k): int(v) for k, v in dict(payload["old_to_new"]).items()}
            mod_map = CompactModulationLabelMap(
                old_to_new=old_to_new,
                num_classes=int(payload.get("num_classes") or len(old_to_new)),
                dataset_names=tuple(str(x) for x in (payload.get("datasets") or ())),
            )
            return mod_map.lookup()
        if not fallback_from_cfg:
            return None
        try:
            mod_map = build_compact_modulation_label_map(root, train_cfg=self.train_cfg)
            return mod_map.lookup()
        except (FileNotFoundError, KeyError, OSError):
            return None

    def _modulation_lookup_for_task(self, task: str | None) -> torch.Tensor | None:
        name = str(task or "")
        if name == "ld_intrapulse":
            return self._intrapulse_compact_lookup
        if name in ("tx_modulation", "modulation"):
            return self._tx_modulation_canonical_lookup
        return None

    def _modulation_offset_lookup_for_task(self, task: str | None) -> torch.Tensor | None:
        if str(task or "") in ("tx_modulation", "modulation"):
            return self._tx_modulation_offset_lookup
        return None

    def _remap_emitter_labels(
        self,
        batch: dict[str, Any],
        labels: Any,
    ) -> Any:
        if self._emitter_offset_lookup is None:
            return labels
        local = batch.get("emitter_id")
        dataset_id = batch.get("dataset_id")
        if local is None or dataset_id is None:
            return labels
        lookup = self._emitter_offset_lookup
        if torch.is_tensor(local):
            lookup = lookup.to(device=local.device)
        return global_emitter_labels(dataset_id, local, lookup)

    def _remap_modulation_labels(
        self,
        labels: Any,
        *,
        task: str | None = None,
        batch: dict[str, Any] | None = None,
    ) -> Any:
        if labels is None:
            return labels
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        name = str(task or "")
        if name in ("tx_modulation", "modulation") and self._tx_modulation_offset_lookup is not None:
            canonical = self._tx_modulation_canonical_lookup
            if canonical is None:
                return labels
            from resmamba_signal_model.training.modulation_labels import global_comm_modulation_labels

            dataset_id = batch.get("dataset_id") if batch is not None else None
            if dataset_id is None:
                return labels
            return global_comm_modulation_labels(
                dataset_id,
                labels,
                canonical.to(device=labels.device),
                self._tx_modulation_offset_lookup.to(device=labels.device),
            )
        lookup = self._modulation_lookup_for_task(task)
        if lookup is None:
            return labels
        return remap_modulation_labels(labels, lookup.to(device=labels.device))

    @property
    def ema_teacher(self):
        return self._ema_box.get("teacher")

    def _dann_lambda(self) -> float:
        warmup = int(self.train_cfg.get("grl_warmup_steps", 1000))
        if warmup <= 0:
            return 1.0
        step = int(getattr(self, "global_step", 0) or 0)
        progress = min(1.0, float(step) / float(max(warmup, 1)))
        return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0

    def _task_for_source(self, name: str, batch: dict[str, Any]) -> str:
        explicit = batch.get("task")
        if isinstance(explicit, str) and explicit:
            return explicit
        if name in self.catalog.source_to_task:
            return self.catalog.source_to_task[name]
        if name in SOURCE_TO_TASK:
            return SOURCE_TO_TASK[name]
        if name in self.catalog.by_name:
            return name
        return str(self.train_cfg.get("task") or "modulation")

    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val=None,
        gradient_clip_algorithm=None,
    ):
        n_bad = sanitize_nonfinite_grads(self)
        if n_bad:
            try:
                self.log(
                    "train/nonfinite_grad_tensors",
                    float(n_bad),
                    on_step=True,
                    logger=True,
                    prog_bar=False,
                )
            except Exception:
                pass
        if gradient_clip_val is None or float(gradient_clip_val) <= 0:
            return
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

    def configure_optimizers(self):
        peak_lr = scale_lr_for_grad_accum(
            float(self.train_cfg.get("learning_rate", 5e-5)),
            int(self.train_cfg.get("gradient_accumulation_steps", 1)),
            enabled=bool(self.train_cfg.get("lr_scale_with_grad_accum", True)),
        )
        param_groups = apply_param_group_lr_scales(self._optimizer_param_groups(peak_lr), peak_lr)
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=float(self.train_cfg.get("weight_decay", 0.01)),
        )
        warmup = int(self.train_cfg.get("warmup_steps", 100))
        # 必须用 optimizer step 数；旧写法用 microbatch×epochs，在 accum>1 时余弦几乎不衰减
        total = total_optimizer_steps_from_cfg(self.train_cfg)
        min_ratio = float(self.train_cfg.get("lr_min_ratio", 0.1))
        schedule_name = resolve_lr_schedule(None, self.train_cfg)
        wrapped = build_lr_scheduler(
            optimizer,
            peak_lr=peak_lr,
            warmup_steps=warmup,
            total_optimizer_steps=total,
            lr_min_ratio=min_ratio,
            use_cosine_decay=use_cosine_lr_decay(schedule_name),
        )
        scheduler = as_torch_lr_scheduler(wrapped)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def _optimizer_include_frozen_heads(self) -> bool:
        """``task_schedule`` 按 epoch 切任务时，优化器须在首轮就登记全部任务头/探针。"""
        return self.stage == "stage2" and bool(self.train_cfg.get("task_schedule"))

    def _optimizer_param_groups(self, peak_lr: float) -> list[dict[str, Any]]:
        if self.stage in ("pretrain", "downstream"):
            params = [p for p in self.model.parameters() if p.requires_grad]
            if self.pretrain_disc is not None:
                params.extend(p for p in self.pretrain_disc.parameters() if p.requires_grad)
            return [{"params": params, "lr": peak_lr}]
        ratio = float(
            self.train_cfg.get("loraplus_lr_ratio")
            or (self.train_cfg.get("peft") or {}).get("loraplus_lr_ratio", 16.0)
        )
        heads_lr = float(self.train_cfg.get("heads_lr", peak_lr))
        tok_lr = peak_lr * float(self.train_cfg.get("tokenizer_last_lr_scale", 0.1))
        used: set[int] = set()
        groups: list[dict[str, Any]] = []
        head_prefixes = tuple(f"{name}." for name in iter_head_param_prefixes())
        adapter_prefixes = ("task_adapters.", "shared_adapter.")
        include_all_heads = self._optimizer_include_frozen_heads()

        def take(predicate, lr: float, name: str, *, include_frozen: bool = False) -> None:
            chosen = []
            for n, p in self.model.named_parameters():
                if id(p) in used:
                    continue
                if not include_frozen and not p.requires_grad:
                    continue
                if predicate(n, p):
                    chosen.append(p)
                    used.add(id(p))
            if chosen:
                groups.append({"params": chosen, "lr": lr, "name": name})

        take(lambda n, p: "lora_A" in n, peak_lr, "lora_A")
        take(lambda n, p: "lora_B" in n or (p.ndim == 1 and "lora_" in n), peak_lr * ratio, "lora_B")
        take(lambda n, p: n.startswith(adapter_prefixes) or n.startswith("prototype_registry."), peak_lr, "adapters")
        take(lambda n, p: any(n == prefix or n.startswith(prefix + ".") for prefix in TOKENIZER_LAST_PREFIXES), tok_lr, "tokenizer_last")
        take(
            lambda n, p: n.startswith(head_prefixes) or n.startswith("extra_task_heads."),
            heads_lr,
            "heads",
            include_frozen=include_all_heads,
        )
        take(
            lambda n, p: n.startswith("z_linear_probes."),
            heads_lr,
            "z_probes",
            include_frozen=include_all_heads,
        )
        take(lambda n, p: n.startswith("task_interface."), peak_lr, "uti")
        take(lambda n, p: True, peak_lr, "other")
        if not groups:
            params = [p for p in self.model.parameters() if p.requires_grad]
            if not params:
                raise RuntimeError("当前 stage 没有可训练参数，请检查 freeze 配置")
            return [{"params": params, "lr": peak_lr}]
        return groups

    def _peak_learning_rate(self) -> float:
        return scale_lr_for_grad_accum(
            float(self.train_cfg.get("learning_rate", 5e-5)),
            int(self.train_cfg.get("gradient_accumulation_steps", 1)),
            enabled=bool(self.train_cfg.get("lr_scale_with_grad_accum", True)),
        )

    def _heads_learning_rate(self) -> float:
        peak_lr = self._peak_learning_rate()
        return float(self.train_cfg.get("heads_lr", peak_lr))

    def sync_optimizer_trainable_params(self) -> int:
        """``task_schedule`` 切任务 / resume 后，把新解冻但未登记的头参数补进优化器。"""
        if self.stage != "stage2" or not self.train_cfg.get("task_schedule"):
            return 0
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return 0
        optimizers = getattr(trainer, "optimizers", None)
        if not optimizers:
            return 0
        optimizer = optimizers[0] if isinstance(optimizers, (list, tuple)) else optimizers
        existing = {id(param) for group in optimizer.param_groups for param in group["params"]}
        head_prefixes = tuple(f"{name}." for name in iter_head_param_prefixes())
        heads_lr = self._heads_learning_rate()
        missing_by_lr: dict[float, list[torch.nn.Parameter]] = {}

        def _collect(name: str, param: torch.nn.Parameter, lr: float) -> None:
            if not param.requires_grad or id(param) in existing:
                return
            missing_by_lr.setdefault(float(lr), []).append(param)
            existing.add(id(param))

        for name, param in self.model.named_parameters():
            if name.startswith(head_prefixes) or name.startswith("extra_task_heads.") or name.startswith("z_linear_probes."):
                _collect(name, param, heads_lr)
            elif param.requires_grad:
                _collect(name, param, self._peak_learning_rate())

        added = 0
        for lr, params in missing_by_lr.items():
            if not params:
                continue
            optimizer.add_param_group({"params": params, "lr": lr, "name": "schedule_sync"})
            added += len(params)
        if added:
            tasks = list(self.train_cfg.get("active_train_tasks") or [])
            logging.getLogger("resmamba").warning(
                "optimizer synced %s trainable tensors for task_schedule tasks=%s",
                added,
                tasks,
            )
            print(
                f"optimizer synced {added} trainable tensors for tasks={tasks}",
                flush=True,
            )
        return added

    def _val_source_hint(self, dataloader_idx: int) -> str | None:
        dm = getattr(self, "trainer", None)
        dm = getattr(dm, "datamodule", None) if dm is not None else None
        names = list(getattr(dm, "val_source_names", None) or getattr(dm, "source_names", None) or [])
        if 0 <= dataloader_idx < len(names):
            return str(names[dataloader_idx])
        return None

    def _schedule_temperature(self) -> float:
        total = max(total_optimizer_steps_from_cfg(self.train_cfg), 1)
        progress = float(getattr(self, "global_step", 0) or 0) / float(total)
        tau = negcos_temperature(
            progress,
            tau_max=float(self.train_cfg.get("negcos_tau_max", 0.5)),
            tau_min=float(self.train_cfg.get("negcos_tau_min", 0.05)),
        )
        self.model._negcos_temperature = tau
        for attr in ("ld_clustering_head", "tx_clustering_head", "clustering_head"):
            head = getattr(self.model, attr, None)
            if head is not None and hasattr(head, "temperature"):
                head.temperature = tau
        return tau

    @property
    def distill_teacher(self):
        return self._ema_box.get("distill")

    def _broadcast_module_params(self, module: torch.nn.Module) -> None:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        trainer = getattr(self, "trainer", None)
        if trainer is None or int(getattr(trainer, "world_size", 1) or 1) <= 1:
            return
        for param in module.parameters():
            torch.distributed.broadcast(param.data, src=0)
        for buf in module.buffers():
            torch.distributed.broadcast(buf.data, src=0)

    def _place_shadow_model(self, module: torch.nn.Module) -> torch.nn.Module:
        device = getattr(self, "device", None)
        if device is not None:
            module.to(device)
        self._broadcast_module_params(module)
        return module

    def refresh_continual_teacher(self) -> None:
        need = (
            float(self.train_cfg.get("distill_weight", 0.0) or 0.0) > 0
            or str(self.train_cfg.get("replay_strategy", "class_center")).lower() in ("class_center", "center", "exemplar", "icarl")
        )
        if not need:
            return
        from resmamba_signal_model.models.ema import EMATeacher

        teacher = EMATeacher(self.model, momentum=1.0)
        self._place_shadow_model(teacher)
        self._ema_box["distill"] = teacher
        registry = getattr(self.model, "prototype_registry", None)
        if registry is not None:
            self._frozen_prototypes = {
                str(name): bank.mean.detach().clone() for name, bank in registry.banks.items()
            }

    def build_class_center_replay_memory(
        self,
        replay_tasks: list[str],
        *,
        datamodule: Any | None = None,
    ) -> dict[str, int]:
        """会话切换后为前序任务构建最近类中心 exemplar replay 索引。"""
        from resmamba_signal_model.training.replay_memory import (
            build_replay_memories_for_tasks,
            resolve_replay_strategy,
        )

        if resolve_replay_strategy(self.train_cfg) != "class_center" or not replay_tasks:
            return {}
        if self.distill_teacher is None:
            self.refresh_continual_teacher()
        teacher = self.distill_teacher
        if teacher is None:
            return {}
        dm = datamodule
        if dm is None:
            trainer = getattr(self, "trainer", None)
            dm = getattr(trainer, "datamodule", None) if trainer is not None else None
        if dm is None:
            return {}
        memory = build_replay_memories_for_tasks(
            teacher.model,
            dm,
            replay_tasks,
            train_cfg=self.train_cfg,
            catalog=self.catalog,
            device=getattr(self, "device", None) or next(self.model.parameters()).device,
        )
        if hasattr(dm, "update_replay_memory"):
            dm.update_replay_memory(memory)
        return {str(k): len(v) for k, v in memory.items()}

    def _ema_enabled(self) -> bool:
        explicit = self.train_cfg.get("ema_teacher")
        if explicit is not None:
            return bool(explicit)
        return any(
            float(self.loss_weights.get(name, 0.0) or 0.0) > 0.0
            for name in ("latent",)
        )

    def _ensure_ema_teacher(self) -> None:
        if self.stage != "pretrain" or not self._ema_enabled():
            return
        from resmamba_signal_model.models.ema import EMATeacher

        if self._ema_box["teacher"] is None:
            momentum = float(self.train_cfg.get("ema_momentum", getattr(self.model.cfg, "ema_momentum", 0.996)))
            teacher = EMATeacher(self.model, momentum=momentum)
            saved = self._ema_box.get("state")
            if saved:
                teacher.load_state_dict(saved, strict=False)
                self._ema_box["state"] = None
            self._place_shadow_model(teacher)
            self._ema_box["teacher"] = teacher
        else:
            self._place_shadow_model(self._ema_box["teacher"])

    def _attach_ema_teacher_outputs(self, outputs: dict[str, Any], batch: dict[str, Any]) -> dict[str, Any]:
        teacher = self.ema_teacher
        if teacher is None:
            return outputs
        teacher.eval()
        teacher_out = teacher.forward_unmasked(batch)
        outputs["teacher_z"] = teacher_out.get("z_general", teacher_out["z"])
        outputs["teacher_h"] = teacher_out.get("h_general", teacher_out.get("patch_h"))
        return outputs

    def on_fit_start(self) -> None:
        self._ensure_ema_teacher()
        if bool(self.train_cfg.get("continual")) or self.stage == "continual":
            from resmamba_signal_model.training.continual import apply_continual_freeze

            apply_continual_freeze(self.model)
            self.refresh_continual_teacher()
            registry = getattr(self.model, "prototype_registry", None)
            if registry is not None and not self._frozen_prototypes:
                self._frozen_prototypes = {
                    str(name): bank.mean.detach().clone() for name, bank in registry.banks.items()
                }

    def on_validation_epoch_start(self) -> None:
        self._reset_val_buffers()
        self._last_val_report = {}
        self._ensure_ema_teacher()
        teacher = self.ema_teacher
        if teacher is not None:
            teacher.eval()
            self._place_shadow_model(teacher)

    def on_train_epoch_start(self) -> None:
        teacher = self.ema_teacher
        if teacher is not None:
            teacher.eval()
            self._place_shadow_model(teacher)

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        del outputs, batch, batch_idx
        if self.ema_teacher is not None:
            self.ema_teacher.update(self.model)

    def _pretrain_forward(
        self, batch: Any, *, default_source: str | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        sources = _as_source_map(batch, default_name=default_source)
        if len(sources) == 1:
            merged = next(iter(sources.values()))
        elif bool(self.train_cfg.get("combine_then_pack", True)):
            merged = merge_source_batches(sources)
        else:
            merged = next(iter(sources.values()))
        # 可分损失标签：防火墙前抽出，模型前向仍无标签泄漏
        disc_labels = {
            key: merged[key] for key in PRETRAIN_DISC_LABEL_KEYS if key in merged and torch.is_tensor(merged[key])
        }
        merged = pretrain_collate_firewall(merged)
        chronos = parse_chronos_sampling_cfg(self.train_cfg)
        # z_supcon 与 iq_mixup 互斥：mixup 污染对比标签
        if float(self.loss_weights.get("z_supcon", 0.0) or 0.0) > 0.0:
            chronos["iq_mixup_enabled"] = False
        if self.training and chronos.get("enabled") and chronos.get("iq_mixup_enabled"):
            merged = apply_iq_mixup(
                merged,
                k=int(chronos.get("iq_mixup_k", 2)),
                alpha=float(chronos.get("iq_mixup_alpha", 0.3)),
            )
        self.model.grl.lambd = self._dann_lambda()
        self._schedule_temperature()
        outputs = self.model(merged, mode="pretrain")
        if self.ema_teacher is not None:
            outputs = self._attach_ema_teacher_outputs(outputs, merged)
        if self.pretrain_disc is not None:
            z_enc = outputs.get("z_enc", outputs.get("z_general", outputs.get("z")))
            if z_enc is not None and torch.is_tensor(z_enc):
                outputs.update(self.pretrain_disc(z_enc))
                outputs["proto_swav_temperature"] = float(
                    (self.train_cfg.get("pretrain_disc") or {}).get("temperature", 0.1) or 0.1
                )
        if disc_labels:
            merged.update(disc_labels)
            labels = resolve_pretrain_supcon_labels(merged)
            if labels is not None:
                outputs["supcon_labels"] = labels
        if float(self.loss_weights.get("vicreg", 0.0) or 0.0) > 0.0:
            outputs["vicreg_gamma"] = self.train_cfg.get("vicreg_gamma", "l2_unit")
        if float(self.loss_weights.get("vicreg_token", 0.0) or 0.0) > 0.0:
            outputs["vicreg_gamma_token"] = self.train_cfg.get("vicreg_gamma_token", 1.0)
        if float(self.loss_weights.get("tcl", 0.0) or 0.0) > 0.0:
            outputs["tcl_temperature"] = self.train_cfg.get("tcl_temperature", 0.5)
            outputs["tcl_max_tokens"] = self.train_cfg.get("tcl_max_tokens", 512)
        weights = dict(self.loss_weights)
        weights["domain"] = float(weights.get("domain", 0.05)) * float(self.model.grl.lambd)
        warmup = int(self.train_cfg.get("vicreg_warmup_steps", 0) or 0)
        if warmup > 0 and self.training:
            ramp = min(1.0, float(self.global_step) / float(warmup))
            if float(weights.get("vicreg", 0.0) or 0.0) > 0.0:
                weights["vicreg"] = float(weights["vicreg"]) * ramp
            if float(weights.get("vicreg_token", 0.0) or 0.0) > 0.0:
                weights["vicreg_token"] = float(weights["vicreg_token"]) * ramp
            if float(weights.get("tcl", 0.0) or 0.0) > 0.0:
                weights["tcl"] = float(weights["tcl"]) * ramp
        total_raw, parts = weighted_pretrain_loss(outputs, merged, weights)
        total_train = self._stem_normalize_pretrain_total(total_raw, merged)
        return total_train, parts, {**outputs, "_batch": merged, "_total_raw": total_raw}

    def _active_source_allowlist(self) -> set[str] | None:
        """``task_schedule`` / 单任务过滤：返回允许的 source 名；``None`` 表示不限制。"""
        sources = self.train_cfg.get("active_train_sources")
        allow: set[str] | None = None
        if isinstance(sources, (list, tuple)) and sources:
            allow = {str(x) for x in sources}
        else:
            tasks = self.train_cfg.get("active_train_tasks")
            if isinstance(tasks, (list, tuple)) and tasks:
                from resmamba_signal_model.training.task_schedule import sources_for_tasks

                allow = set(sources_for_tasks(self.train_cfg, [str(t) for t in tasks]))
        if self.training:
            replay = self.train_cfg.get("active_replay_sources")
            if isinstance(replay, (list, tuple)) and replay:
                replay_set = {str(x) for x in replay}
                allow = replay_set if allow is None else allow | replay_set
        return allow

    def _downstream_forward(
        self, batch: Any, *, default_source: str | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        sources = _as_source_map(batch, default_name=default_source)
        allow = self._active_source_allowlist()
        if allow is not None:
            sources = {name: sub for name, sub in sources.items() if str(name) in allow}
        if not sources:
            raise RuntimeError(
                f"下游 batch 无可用源：allow={sorted(allow) if allow is not None else None} "
                f"got={list(_as_source_map(batch, default_name=default_source))}"
            )
        total = None
        parts: dict[str, torch.Tensor] = {}
        last: dict[str, Any] | None = None
        src_losses: dict[str, float] = {}
        src_tokens: dict[str, float] = {}
        packed_all: list[dict[str, Any]] = []
        token_mass = None
        # 分任务：每个 source/task 独立算 loss，再按需汇总；单任务时 total 即该任务 loss。
        per_task_totals: dict[str, torch.Tensor] = {}
        replay_sources = {str(x) for x in (self.train_cfg.get("active_replay_sources") or [])}
        distill_weight = float(self.train_cfg.get("distill_weight", 0.0) or 0.0)
        need_teacher = self.training and self.distill_teacher is not None and distill_weight > 0
        for name, sub in sources.items():
            task = self._task_for_source(name, sub)
            if hasattr(self.model, "set_active_task"):
                self.model.set_active_task(task)
            spec = self.catalog.get(task)
            self._schedule_temperature()
            outputs = self.model(sub, mode="task", task=task)
            if (
                self.training
                and spec is not None
                and spec.kind == "clustering"
                and sub.get("view2") is not None
            ):
                view_batch = dict(sub)
                view_batch["iq"] = sub["view2"]
                view_batch["values"] = sub["view2"]
                view_out = self.model(view_batch, mode="task", task=task)
                if view_out.get("cluster_embedding") is not None:
                    outputs["cluster_embedding_view2"] = view_out["cluster_embedding"]
                if view_out.get("cluster_logits") is not None:
                    outputs["cluster_logits_view2"] = view_out["cluster_logits"]
            is_replay = self.training and str(name) in replay_sources
            if need_teacher:
                with torch.no_grad():
                    t_out = self.distill_teacher.model(sub, mode="task", task=task)
                if distill_weight > 0:
                    teacher_logits = _first_present(
                        t_out, "task_logits", "cluster_logits", "modulation_logits", "emitter_logits"
                    )
                    if teacher_logits is not None:
                        outputs["teacher_logits"] = teacher_logits.detach()
            registry = getattr(self.model, "prototype_registry", None)
            embed = outputs.get("task_pooled", outputs.get("cluster_embedding"))
            assign = outputs.get("cluster_probs")
            if (
                registry is not None
                and self.training
                and spec is not None
                and spec.kind == "clustering"
                and embed is not None
                and int(embed.shape[-1]) == int(registry.dim)
            ):
                ns = clustering_registry_namespace(task)
                bank = registry.bank(ns)
                if assign is not None and int(assign.shape[-1]) == int(bank.num_prototypes):
                    reg_assign = assign
                else:
                    tau = float(getattr(self.model, "_negcos_temperature", None) or negcos_temperature(0.0))
                    reg_assign = bank.cosine_logits(embed, tau).softmax(dim=-1)
                registry.update(ns, embed.detach(), reg_assign.detach())
            if registry is not None and self._frozen_prototypes:
                ns = clustering_registry_namespace(task) if spec is not None and spec.kind == "clustering" else (
                    "emitter" if task in ("emitter", "ld_model") else "modulation"
                )
                if ns in registry.banks and ns in self._frozen_prototypes:
                    outputs["prototype_mean"] = registry.bank(ns).mean
                    outputs["frozen_prototype_mean"] = self._frozen_prototypes[ns]
                    outputs["prototype_count"] = registry.bank(ns).count
            supervised = bool(self.train_cfg.get("supervised_clustering", False))
            label_field = None if spec is None else spec.label_field
            if spec is not None and spec.kind == "clustering" and not supervised:
                label_field = None
            loss, sub_parts = downstream_task_loss(
                outputs,
                sub,
                task,
                emitter_offset_lookup=(
                    self._ld_model_offset_lookup if task == "ld_model" else self._emitter_offset_lookup
                ),
                modulation_compact_lookup=self._modulation_lookup_for_task(task),
                modulation_offset_lookup=self._modulation_offset_lookup_for_task(task),
                modulation_contrastive_weight=float(self.train_cfg.get("modulation_contrastive_weight", 0.25)),
                emitter_contrastive_weight=float(self.train_cfg.get("emitter_contrastive_weight", 0.0)),
                z_contrastive_weight=float(self.train_cfg.get("z_contrastive_weight", 0.0)),
                domain_weight=float(self.loss_weights.get("domain", 0.1)),
                recon_weight=float(self.train_cfg.get("lambda_recon", 0.1)),
                phys_weight=float(self.loss_weights.get("physical", 0.1)),
                task_kind=None if spec is None else spec.kind,
                task_catalog=self.train_cfg,
                label_field=label_field,
                supervised_clustering=supervised,
                distill_weight=distill_weight,
                distill_temperature=float(self.train_cfg.get("distill_temperature", 2.0)),
                distill_confidence=float(self.train_cfg.get("distill_confidence", 0.5)),
                prototype_anchor_weight=float(self.train_cfg.get("prototype_anchor_weight", 0.0)),
                z_probe_weight=float(self.train_cfg.get("z_probe_weight", 1.0)),
                emitter_label_smoothing=float(self.train_cfg.get("emitter_label_smoothing", 0.0)),
                cluster_utilization_weight=float(self.train_cfg.get("cluster_utilization_weight", 0.15)),
                cluster_consistency_weight=float(self.train_cfg.get("cluster_consistency_weight", 1.0)),
                cluster_balance_mix=float(self.train_cfg.get("cluster_balance_mix", 0.7)),
                cluster_sinkhorn_epsilon=float(self.train_cfg.get("cluster_sinkhorn_epsilon", 0.1)),
                cluster_sinkhorn_iters=int(self.train_cfg.get("cluster_sinkhorn_iters", 3)),
            )
            n_tokens = outputs["n_tokens"].sum().clamp_min(1).to(dtype=loss.dtype)
            src_losses[name] = float(loss.detach())
            src_tokens[name] = float(n_tokens.detach())
            # 分量与任务总 loss 都挂在任务名下，避免多源混写同一条曲线。
            if task in per_task_totals:
                # 同任务多源：按 token 加权合成该任务 total（仍与其它任务隔离）
                prev = per_task_totals[task]
                prev_tok = parts[f"{task}/_tokens"]
                merged_tok = prev_tok + n_tokens
                per_task_totals[task] = (prev * prev_tok + loss * n_tokens) / merged_tok.clamp_min(1)
                parts[f"{task}/_tokens"] = merged_tok.detach()
            else:
                per_task_totals[task] = loss
                parts[f"{task}/_tokens"] = n_tokens.detach()
            parts[f"{task}/total"] = per_task_totals[task].detach()
            for key, value in sub_parts.items():
                # 同名分量后写覆盖前写；同任务多源时保留最后一次（监控用），total 已按 token 合成
                parts[f"{task}/{key}"] = value
            if bool(self.train_cfg.get("token_normalized_loss", True)):
                weighted = loss * n_tokens
                total = weighted if total is None else total + weighted
            else:
                total = loss if total is None else total + loss
            token_mass = n_tokens if token_mass is None else token_mass + n_tokens
            last = {**outputs, "_batch": sub, "_task": task, "_source": name}
            packed_all.append(last)
        assert total is not None and last is not None
        if bool(self.train_cfg.get("token_normalized_loss", True)) and token_mass is not None:
            total = total / token_mass.clamp_min(1)
        # 单任务阶段：反传目标就是该任务 loss，不掺其它任务。
        if len(per_task_totals) == 1:
            total = next(iter(per_task_totals.values()))
        # 去掉内部辅助键，避免写入日志
        parts = {k: v for k, v in parts.items() if not k.endswith("/_tokens")}
        last = {**last, "_src_losses": src_losses, "_src_tokens": src_tokens, "_all": packed_all, "_task_losses": dict(per_task_totals)}
        return total, parts, last

    def _forward_and_loss(
        self,
        batch: Any,
        *,
        train: bool,
        default_source: str | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        del train
        if self.stage == "pretrain":
            return self._pretrain_forward(batch, default_source=default_source)
        return self._downstream_forward(batch, default_source=default_source)

    def _log_mix(self, packed: dict[str, Any]) -> None:
        if self.mix is None:
            return
        source_names = packed["_batch"].get("source_name")
        if resolve_mix_strategy(self.train_cfg.get("mix_strategy")) == "token_share":
            if packed.get("_src_losses"):
                self.mix.update(packed["_src_losses"], packed["_src_tokens"])
            elif source_names:
                src_losses, src_tokens = _per_source_mse(packed, source_names)
                self.mix.update(src_losses, src_tokens)
        ratios = self.mix.ratios()
        batch_size = max(1, int(packed["n_tokens"].shape[0]))
        for name, ratio in ratios.items():
            self.log(f"mix/ratio_{name}", ratio, on_step=True, batch_size=batch_size)
            self.log(f"mix/ema_loss_{name}", self.mix.ema[name], on_step=True, batch_size=batch_size)
        shares = self.mix.token_shares(int(self.train_cfg.get("token_budget", 4096)))
        for name, share in shares.items():
            self.log(f"mix/tokens_{name}", float(share), on_step=True, batch_size=batch_size)

    def _ensure_trainable_loss(self, loss: torch.Tensor) -> torch.Tensor:
        if loss.requires_grad:
            return loss
        dummy = None
        for param in self.model.parameters():
            if param.requires_grad:
                dummy = param.reshape(-1)[:1].sum() * 0.0
                break
        if dummy is None:
            raise RuntimeError("当前 stage 没有可训练参数，请检查 freeze 配置")
        if not self._warned_detached_loss:
            logging.getLogger("resmamba").warning(
                "loss 无 grad_fn（常见于标签被紧凑映射成全 -1，或 batch 任务头已冻结）；本 step 跳过有效反传"
            )
            self._warned_detached_loss = True
        return loss + dummy

    def _next_log_ema(self, prev: float | None, value: float, alpha: float) -> float:
        alpha = min(max(float(alpha), 0.0), 0.9999)
        if prev is None:
            return value
        return alpha * prev + (1.0 - alpha) * value

    def _stem_loss_denom(self, stem: str, raw_total: float) -> float:
        """反传 stem 归一化分母：用该库上一时刻 EMA，首次见库时用当前 raw。"""
        prev = self._stem_total_ema.get(stem)
        if prev is None:
            return max(float(raw_total), 1.0e-6)
        return max(float(prev), 1.0e-6)

    def _stem_normalize_pretrain_total(self, total: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
        """按 H5 stem EMA 归一化预训练总 loss，换库时优化目标量级一致。"""
        if not bool(self.train_cfg.get("stem_normalized_loss", False)):
            return total
        stem = _resolve_batch_stem(batch)
        if not stem:
            return total
        raw = float(total.detach())
        denom = self._stem_loss_denom(stem, raw)
        return total / total.new_tensor(denom)

    def _maybe_unlock_adaptive_mae(self) -> None:
        """难库 recon_mse 相对基线下降后，再打开 adaptive_mae_mask。"""
        unlock = self.train_cfg.get("adaptive_mae_unlock")
        if not isinstance(unlock, dict) or not bool(unlock.get("enabled", False)):
            return
        if self._adaptive_mae_unlocked or bool(getattr(self.model.cfg, "adaptive_mae_mask", False)):
            self._adaptive_mae_unlocked = True
            return
        min_steps = int(unlock.get("min_steps", 0) or 0)
        if int(self.global_step) < min_steps:
            return
        hard_stems = [str(s).strip() for s in (unlock.get("hard_stems") or []) if str(s).strip()]
        if not hard_stems:
            return
        rel_drop = float(unlock.get("rel_drop", 0.08))
        min_updates = max(1, int(unlock.get("min_stem_updates", 30) or 30))
        for stem in hard_stems:
            ema = self._recon_mse_stem_ema.get(stem)
            updates = int(self._recon_mse_stem_updates.get(stem, 0))
            if ema is None or updates < min_updates:
                return
            if stem not in self._recon_mse_stem_baseline:
                self._recon_mse_stem_baseline[stem] = float(ema)
        ready = True
        for stem in hard_stems:
            baseline = float(self._recon_mse_stem_baseline[stem])
            ema = float(self._recon_mse_stem_ema[stem])
            target = baseline * (1.0 - rel_drop)
            if ema > target:
                ready = False
                break
        if not ready:
            return
        self.model.cfg.adaptive_mae_mask = True
        self._adaptive_mae_unlocked = True
        # 自适应区间从固定 0.3 附近温和上探，避免立刻跳到 0.75
        if float(getattr(self.model.cfg, "mae_mask_ratio_min", 0.45)) > float(self.model.cfg.mask_ratio):
            self.model.cfg.mae_mask_ratio_min = float(self.model.cfg.mask_ratio)
        print(
            f"[adaptive_mae] unlocked at step={int(self.global_step)} "
            f"hard_stems={hard_stems} baselines={ {s: round(self._recon_mse_stem_baseline[s], 4) for s in hard_stems} } "
            f"emas={ {s: round(self._recon_mse_stem_ema[s], 4) for s in hard_stems} }",
            flush=True,
        )

    def _log_mse_diagnostics(self, parts: dict[str, torch.Tensor], packed: dict[str, Any], *, batch_size: int) -> None:
        """EMA + stem：训练目标分项（mae Smooth L1 / mse）+ 稳定尺子 loss/recon_mse（纯 MSE，对齐 val）。"""
        if parts.get("mae") is not None and torch.is_tensor(parts["mae"]):
            key = "mae"
            stem_ema = self._mae_stem_ema
            ema_attr = "_mae_log_ema"
        elif parts.get("mse") is not None and torch.is_tensor(parts["mse"]):
            key = "mse"
            stem_ema = self._mse_stem_ema
            ema_attr = "_mse_log_ema"
        else:
            key = None
            stem_ema = None
            ema_attr = None

        batch = packed.get("_batch") or {}
        stem = _resolve_batch_stem(batch)
        if not stem:
            source = packed.get("_source") or _source_key_from_batch(batch)
            if isinstance(source, str) and source and source != "pretrain":
                stem = source
        stem_alpha = float(self.train_cfg.get("stem_loss_ema_alpha", 0.99))
        alpha = float(
            self.train_cfg.get(
                "mse_log_ema_alpha",
                self.train_cfg.get("mae_log_ema_alpha", 0.98),
            )
        )

        if key is not None:
            recon = parts[key]
            recon_val = float(recon.detach())
            key_alpha = float(self.train_cfg.get(f"{key}_log_ema_alpha", alpha))
            setattr(self, ema_attr, self._next_log_ema(getattr(self, ema_attr), recon_val, key_alpha))
            self.log(f"loss/{key}", recon, on_step=True, prog_bar=False, batch_size=batch_size)
            self.log(f"loss/{key}_ema", getattr(self, ema_attr), on_step=True, prog_bar=False, batch_size=batch_size)
            if stem:
                safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)
                prev = stem_ema.get(stem)
                stem_ema[stem] = self._next_log_ema(prev, recon_val, stem_alpha)
                recon_norm = recon_val / max(stem_ema[stem], 1.0e-6)
                self.log(f"loss/{key}_norm", recon_norm, on_step=True, prog_bar=True, batch_size=batch_size)
                self.log(f"loss/{key}_stem/{safe}", recon, on_step=True, batch_size=batch_size)

        # 与 val/recon_mse 同口径的纯 MSE，不随训练目标切换而改名
        try:
            pred, target, mask = reconstruction_eval_pair(packed, kind="pretrain")
            recon_mse = float(masked_patch_mse(pred, target, mask).detach())
        except Exception:
            recon_mse = None
        if recon_mse is not None and recon_mse == recon_mse:
            self._recon_mse_log_ema = self._next_log_ema(self._recon_mse_log_ema, recon_mse, alpha)
            self.log("loss/recon_mse", recon_mse, on_step=True, prog_bar=False, batch_size=batch_size)
            self.log(
                "loss/recon_mse_ema",
                self._recon_mse_log_ema,
                on_step=True,
                prog_bar=False,
                batch_size=batch_size,
            )
            if stem:
                safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)
                prev = self._recon_mse_stem_ema.get(stem)
                self._recon_mse_stem_ema[stem] = self._next_log_ema(prev, recon_mse, stem_alpha)
                self._recon_mse_stem_updates[stem] = int(self._recon_mse_stem_updates.get(stem, 0)) + 1
                self.log(
                    f"loss/recon_mse_stem/{safe}",
                    recon_mse,
                    on_step=True,
                    prog_bar=False,
                    batch_size=batch_size,
                )
                self._maybe_unlock_adaptive_mae()

        self.log(
            "mask/adaptive_enabled",
            1.0 if bool(getattr(self.model.cfg, "adaptive_mae_mask", False)) else 0.0,
            on_step=True,
            prog_bar=False,
            batch_size=batch_size,
        )

        n_tok = packed.get("n_tokens")
        if torch.is_tensor(n_tok) and n_tok.numel() > 0 and hasattr(self.model, "resolve_mae_mask_ratios"):
            ratios = self.model.resolve_mae_mask_ratios(n_tok.detach())
            self.log("mask/ratio_mean", float(ratios.mean().item()), on_step=True, batch_size=batch_size)

    def _log_pretrain_step_diagnostics(
        self,
        total: torch.Tensor,
        parts: dict[str, torch.Tensor],
        packed: dict[str, Any],
        *,
        batch_size: int,
        total_train: torch.Tensor | None = None,
    ) -> None:
        """预训练：raw total 写 TB；stem 归一化反传目标看 total_train_ema。"""
        total_val = float(total.detach())
        alpha = float(
            self.train_cfg.get(
                "total_log_ema_alpha",
                self.train_cfg.get(
                    "mse_log_ema_alpha",
                    self.train_cfg.get("mae_log_ema_alpha", 0.98),
                ),
            )
        )
        self._total_log_ema = self._next_log_ema(self._total_log_ema, total_val, alpha)
        self.log("loss/total", total, on_step=True, prog_bar=False, batch_size=batch_size)
        self.log("loss/total_ema", self._total_log_ema, on_step=True, prog_bar=total_train is None, batch_size=batch_size)
        if total_train is not None:
            train_val = float(total_train.detach())
            self._total_train_ema = self._next_log_ema(self._total_train_ema, train_val, alpha)
            self.log("loss/total_train", total_train, on_step=True, prog_bar=False, batch_size=batch_size)
            self.log(
                "loss/total_train_ema",
                self._total_train_ema,
                on_step=True,
                prog_bar=True,
                batch_size=batch_size,
            )

        batch = packed.get("_batch") or {}
        stem = _resolve_batch_stem(batch)
        if stem:
            stem_alpha = float(self.train_cfg.get("stem_loss_ema_alpha", 0.99))
            prev = self._stem_total_ema.get(stem)
            self._stem_total_ema[stem] = self._next_log_ema(prev, total_val, stem_alpha)
            stem_ema = max(self._stem_total_ema[stem], 1.0e-6)
            self.log("loss/stem_norm", total_val / stem_ema, on_step=True, prog_bar=False, batch_size=batch_size)
            safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)
            self.log(f"batch/stem/{safe}", 1.0, on_step=True, prog_bar=False, batch_size=batch_size)

        self._log_mse_diagnostics(parts, packed, batch_size=batch_size)

    def training_step(self, batch: Any, batch_idx: int):
        if self.train_cfg.get("active_eval_only"):
            return None
        total, parts, packed = self._forward_and_loss(batch, train=True)
        total_raw = packed.get("_total_raw", total)
        total = self._ensure_trainable_loss(total)
        now = time.perf_counter()
        dt = max(now - self._last_step_t, 1e-6)
        self._last_step_t = now
        n_tokens = float(packed["n_tokens"].sum().detach())
        batch_size = max(1, int(packed["n_tokens"].shape[0]))
        if self.stage == "pretrain":
            total_train = total if bool(self.train_cfg.get("stem_normalized_loss", False)) else None
            self._log_pretrain_step_diagnostics(
                total_raw,
                parts,
                packed,
                batch_size=batch_size,
                total_train=total_train,
            )
        else:
            self.log("loss/total", total, prog_bar=True, on_step=True, batch_size=batch_size)
        mask_strategy = packed.get("mask_strategy")
        if mask_strategy is not None:
            self.log(
                "mask/is_suffix",
                1.0 if str(mask_strategy) == "suffix" else 0.0,
                on_step=True,
                batch_size=batch_size,
            )
        for name, value in parts.items():
            if not should_log_loss_part(name, self.loss_weights):
                continue
            if self.stage == "pretrain" and name in ("mse", "mae"):
                continue
            log_name = name if name.startswith("loss/") else f"loss/{name}"
            # mae/mse 原始值保留；预训练进度条改看 ema，避免同质 batch 尖峰刷屏
            self.log(log_name, value, on_step=True, prog_bar=False, batch_size=batch_size)
        if self.stage != "pretrain":
            self._log_mse_diagnostics(parts, packed, batch_size=batch_size)
        self.log("grl/lambda", float(getattr(self.model.grl, "lambd", 1.0)), on_step=True, batch_size=batch_size)
        self.log("tokens_per_sec", n_tokens / dt, on_step=True, batch_size=batch_size)
        length = packed["_batch"].get("length")
        if torch.is_tensor(length):
            self.log("seq_len/mean", length.float().mean(), on_step=True, batch_size=batch_size)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True, batch_size=batch_size)
        self._log_mix(packed)
        return total

    def on_train_epoch_end(self) -> None:
        log = logging.getLogger("resmamba")
        metrics = {
            k: float(v)
            for k, v in self.trainer.callback_metrics.items()
            if torch.is_tensor(v) or isinstance(v, (float, int))
        }
        mix = self.mix.ratios() if self.mix is not None else {}
        log.info("epoch=%s metrics=%s mix=%s", int(self.current_epoch), metrics, mix)

    def _dataset_names(self) -> dict[int, str]:
        if self._dataset_id_names is None:
            self._dataset_id_names = load_dataset_id_names(self.train_cfg.get("rfdata_root"))
        return self._dataset_id_names

    def _ensure_dataset_name(self, name: str) -> int:
        """为 stem / source 分配稳定 dataset_id，供分数据集指标聚合。"""
        names = self._dataset_names()
        for did, existing in names.items():
            if existing == name:
                return int(did)
        did = max(names.keys(), default=-1) + 1
        names[did] = name
        return did

    def _resolve_dataset_ids(self, batch: dict[str, Any] | None, n: int) -> torch.Tensor:
        """分数据集指标：优先 ``moe_route_stem``（H5 真名），再 ``dataset_id``，最后 ``source_name``。

        ``label_maps.datasets`` 可能滞后（如 id=10 仍写 open_real_data 而 H5 已是 radar_mod15），
        用 stem 分组可避免验证摘要错名。
        """
        batch = batch or {}

        stems = batch.get("moe_route_stem")
        if isinstance(stems, str) and stems:
            stem_list = [stems] * n
        elif isinstance(stems, (list, tuple)) and stems:
            stem_list = []
            for i in range(n):
                item = stems[i] if i < len(stems) else stems[0]
                if isinstance(item, bytes):
                    item = item.decode()
                stem_list.append(str(item) if item else "")
        else:
            stem_list = []
        if stem_list and any(stem_list):
            ids = [
                self._ensure_dataset_name(stem) if stem else -1
                for stem in stem_list[:n]
            ]
            while len(ids) < n:
                ids.append(ids[-1] if ids else -1)
            return torch.tensor(ids[:n], dtype=torch.long)

        ds_t = self._cpu_1d(batch.get("dataset_id"))
        if ds_t is not None and ds_t.numel() == n and bool((ds_t >= 0).any()):
            return ds_t[:n].long()

        source = batch.get("source_name")
        if isinstance(source, (list, tuple)) and source:
            first = source[0]
            if isinstance(first, bytes):
                first = first.decode()
            fallback = str(first) if first else ""
        elif isinstance(source, str):
            fallback = source
        else:
            fallback = ""
        if fallback:
            did = self._ensure_dataset_name(fallback)
            return torch.full((n,), did, dtype=torch.long)
        return torch.full((n,), -1, dtype=torch.long)

    def _reset_val_buffers(self) -> None:
        self._val_cls: dict[str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
        self._val_cluster: dict[str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
        self._val_recon: dict[str, dict[str, Any]] = {}
        self._val_pretrain_by_ds: dict[str, dict[str, float]] = {}
        self._val_absorb_embeds = []
        self._val_absorb_scores = []
        self._val_openset = []

    def _cpu_1d(self, value: Any) -> torch.Tensor | None:
        if value is None:
            return None
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        return value.detach().reshape(-1).cpu()

    def _collect_cls(self, task: str, pred: torch.Tensor, labels: Any, batch_u: dict[str, Any] | None) -> None:
        pred_t = self._cpu_1d(pred)
        label_t = self._cpu_1d(labels)
        if pred_t is None or label_t is None:
            return
        n = min(pred_t.numel(), label_t.numel())
        pred_t = pred_t[:n].long()
        label_t = label_t[:n].long()
        ds_t = self._resolve_dataset_ids(batch_u, n)
        self._val_cls.setdefault(task, []).append((pred_t, label_t, ds_t))

    def _collect_cluster(self, task: str, pred: torch.Tensor, labels: Any, batch_u: dict[str, Any] | None) -> None:
        pred_t = self._cpu_1d(pred)
        label_t = self._cpu_1d(labels)
        if pred_t is None or label_t is None:
            return
        n = min(pred_t.numel(), label_t.numel())
        pred_t = pred_t[:n].long()
        label_t = label_t[:n].long()
        ds_t = self._resolve_dataset_ids(batch_u, n)
        self._val_cluster.setdefault(task, []).append((pred_t, label_t, ds_t))

    def _recon_bucket(self, task: str) -> dict[str, Any]:
        return self._val_recon.setdefault(
            task,
            {"mse_weighted": 0.0, "mae_weighted": 0.0, "n": 0, "datasets": {}},
        )

    def _add_recon_stats(
        self,
        bucket: dict[str, Any],
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None,
        n: int,
    ) -> None:
        mse = float(masked_patch_mse(pred, target, mask).detach())
        mae = float(masked_patch_mae(pred, target, mask).detach())
        bucket["mse_weighted"] += mse * max(n, 1)
        bucket["mae_weighted"] += mae * max(n, 1)
        bucket["n"] += int(n)

    def _collect_recon(self, task: str, packed: dict[str, Any], batch_u: dict[str, Any]) -> None:
        if packed.get("recon_norm") is None and packed.get("mae_pred") is None and packed.get("pred_patches") is None:
            return
        spec = self.catalog.get(task)
        kind = spec.kind if spec is not None else self.catalog.kind(task) if task else None
        if kind not in ("prediction", "imputation"):
            kind = task if task in ("prediction", "imputation") else kind
        try:
            pred, target, mask = reconstruction_eval_pair(packed, kind=kind)
        except (KeyError, ValueError):
            return
        if pred is None or target is None:
            return
        n = int(pred.shape[0])
        self._add_recon_stats(
            self._recon_bucket(task),
            pred,
            target,
            mask,
            n,
        )
        ds = self._resolve_dataset_ids(batch_u, n)
        names = self._dataset_names()
        for did in ds.unique():
            ident = int(did)
            if ident < 0:
                continue
            select = ds == did
            n_ds = int(select.sum())
            if n_ds == 0:
                continue
            mask_ds = mask[:n][select] if mask is not None else None
            name = dataset_display_name(ident, names)
            ds_bucket = self._recon_bucket(task)["datasets"].setdefault(
                name,
                {"mse_weighted": 0.0, "mae_weighted": 0.0, "n": 0},
            )
            self._add_recon_stats(
                ds_bucket,
                pred[:n][select],
                target[:n][select],
                mask_ds,
                n_ds,
            )

    def _collect_val_item(self, packed: dict[str, Any]) -> None:
        batch_u = packed.get("_batch") or {}
        task = str(packed.get("_task") or self.train_cfg.get("task") or "")
        spec = self.catalog.get(task)
        kind = spec.kind if spec is not None else self.catalog.kind(task) if task else ""
        label_field = None if spec is None else spec.label_field
        logits = None
        for key in ("task_logits", f"{task}_logits", "modulation_logits"):
            if key in packed and packed[key] is not None:
                logits = packed[key]
                break
        if kind == "classification" and logits is not None and int(logits.shape[-1]) > 0:
            if task == "ld_model" and self._ld_model_offset_lookup is not None:
                labels = batch_u.get(label_field or "mod_label_id", batch_u.get("mod_label_id"))
                labels = global_emitter_labels(
                    batch_u.get("dataset_id"),
                    labels,
                    self._ld_model_offset_lookup,
                )
            else:
                labels = batch_u.get(label_field or "canonical_mod_label_id", batch_u.get("mod_label_id"))
                labels = self._remap_modulation_labels(labels, task=task, batch=batch_u)
            self._collect_cls(task or "modulation", logits.argmax(dim=-1), labels, batch_u)
            z_probe = packed.get("z_probe_logits")
            if z_probe is not None and int(z_probe.shape[-1]) > 0:
                self._collect_cls(
                    f"{task or 'modulation'}_z",
                    z_probe.argmax(dim=-1),
                    labels,
                    batch_u,
                )
        if kind == "emitter" or (kind != "classification" and packed.get("emitter_logits") is not None):
            pred_e = packed.get("task_logits")
            if pred_e is None:
                pred_e = packed.get("emitter_logits")
            if pred_e is not None and int(pred_e.shape[-1]) > 0:
                labels_e = batch_u.get(label_field or "global_emitter_id", batch_u.get("emitter_id"))
                labels_e = self._remap_emitter_labels(batch_u, labels_e)
                self._collect_cls(task or "emitter", pred_e.argmax(dim=-1), labels_e, batch_u)
                z_probe = packed.get("z_probe_logits")
                if z_probe is not None and int(z_probe.shape[-1]) > 0:
                    self._collect_cls(
                        f"{task or 'emitter'}_z",
                        z_probe.argmax(dim=-1),
                        labels_e,
                        batch_u,
                    )
        cluster_logits = packed.get("cluster_logits")
        cluster_labels = None
        if spec is not None and spec.kind == "clustering":
            cluster_labels = resolve_cluster_eval_labels(
                batch_u.get("mod_label_id"),
                batch_u.get("emitter_id"),
                batch_u.get("source_label_id"),
            )
        elif kind == "clustering" or task in ("ld_clustering", "tx_clustering"):
            cluster_labels = resolve_cluster_eval_labels(
                batch_u.get("mod_label_id"),
                batch_u.get("emitter_id"),
                batch_u.get("source_label_id"),
            )
        if cluster_labels is None:
            cluster_labels = batch_u.get("global_label_id")
        if cluster_logits is not None and cluster_labels is not None and int(cluster_logits.shape[-1]) > 0:
            self._collect_cluster(task or "clustering", cluster_logits.argmax(dim=-1), cluster_labels, batch_u)
        score = packed.get("openset_score", packed.get("openset_energy"))
        embed = packed.get("cluster_embedding", packed.get("task_pooled"))
        if score is not None:
            y_unknown = torch.zeros(int(score.reshape(-1).shape[0]), dtype=torch.bool)
            label_t = self._cpu_1d(cluster_labels)
            if label_t is not None:
                y_unknown = label_t[: y_unknown.numel()] < 0
            logits_osr = packed.get("task_logits", packed.get("cluster_logits"))
            conf = None
            if logits_osr is not None and torch.is_tensor(logits_osr):
                conf = logits_osr.detach().softmax(dim=-1).max(dim=-1).values.reshape(-1).cpu()
            self._val_openset.append(
                (score.detach().reshape(-1).cpu(), y_unknown.cpu(), conf, None)
            )
            if embed is not None and torch.is_tensor(embed) and (not self.training):
                self._val_absorb_embeds.append(embed.detach().cpu())
                self._val_absorb_scores.append(score.detach().reshape(-1).cpu())
        if kind in ("prediction", "imputation"):
            self._collect_recon(task or kind, packed, batch_u)

    def _is_finite_metric(self, value: Any) -> bool:
        if value is None:
            return False
        if torch.is_tensor(value):
            return bool(value.numel() == 1 and torch.isfinite(value).all())
        try:
            return bool(math.isfinite(float(value)))
        except (TypeError, ValueError):
            return False

    def _log_scalar(self, name: str, value: float, *, prog_bar: bool = False) -> None:
        if not self._is_finite_metric(value):
            return
        self.log(name, float(value), on_epoch=True, sync_dist=True, prog_bar=prog_bar, batch_size=1)

    def _log_classification_report(self, task: str, report: dict[str, Any]) -> None:
        self._log_scalar(f"val/acc_{task}", report["acc"], prog_bar=True)
        self._log_scalar(f"val/f1_{task}", report["f1"], prog_bar=True)
        self._log_scalar(f"val/miss_rate_{task}", report["miss_rate"])
        for dataset, row in (report.get("datasets") or {}).items():
            self._log_scalar(f"val/acc_{task}/{dataset}", row["acc"])
            self._log_scalar(f"val/f1_{task}/{dataset}", row["f1"])
            self._log_scalar(f"val/miss_rate_{task}/{dataset}", row["miss_rate"])

    def _flush_val_scores(self) -> dict[str, Any]:
        names = self._dataset_names()
        report: dict[str, Any] = {}
        for task, parts in self._val_cls.items():
            pred = torch.cat([item[0] for item in parts], dim=0)
            labels = torch.cat([item[1] for item in parts], dim=0)
            dataset_ids = torch.cat([item[2] for item in parts], dim=0)
            info = classification_epoch_scores(pred, labels, dataset_ids, dataset_names=names)
            report[task] = info
            self._log_classification_report(task, info)
        for task, parts in self._val_cluster.items():
            pred = torch.cat([item[0] for item in parts], dim=0)
            labels = torch.cat([item[1] for item in parts], dim=0)
            dataset_ids = torch.cat([item[2] for item in parts], dim=0)
            info = clustering_epoch_scores(pred, labels, dataset_ids, dataset_names=names)
            report[task] = info
            self._log_scalar(f"val/acc_{task}", info["acc"], prog_bar=True)
            self._log_scalar("val/nmi" if task == "clustering" else f"val/nmi_{task}", info["nmi"], prog_bar=True)
            # 选模兼容：单库时与 nmi 相同；多库时为各 H5 NMI 宏平均
            self._log_scalar("val/nmi_within_domain", info["mean_nmi"])
            self._log_scalar(f"val/macro_nmi_{task}", info["mean_nmi"])
            for dataset, row in (info.get("datasets") or {}).items():
                self._log_scalar(f"val/acc_{task}/{dataset}", row["acc"])
                self._log_scalar(f"val/nmi_{task}/{dataset}", row["nmi"])
        for task, bucket in self._val_recon.items():
            mse = float(bucket["mse_weighted"] / bucket["n"]) if bucket["n"] else 0.0
            mae = float(bucket["mae_weighted"] / bucket["n"]) if bucket["n"] else 0.0
            per_dataset: dict[str, dict[str, float]] = {}
            for dataset, row in (bucket.get("datasets") or {}).items():
                ds_mse = float(row["mse_weighted"] / row["n"]) if row["n"] else 0.0
                ds_mae = float(row["mae_weighted"] / row["n"]) if row["n"] else 0.0
                per_dataset[dataset] = {"mse": ds_mse, "mae": ds_mae, "n": float(row["n"])}
            info = reconstruction_epoch_scores(mse, mae, int(bucket["n"]), per_dataset)
            report[task] = info
            self._log_scalar(f"val/mse_{task}", mse, prog_bar=True)
            self._log_scalar(f"val/mae_{task}", mae, prog_bar=True)
            for dataset, row in per_dataset.items():
                self._log_scalar(f"val/mse_{task}/{dataset}", row["mse"])
                self._log_scalar(f"val/mae_{task}/{dataset}", row["mae"])
        if self._val_openset:
            scores = torch.cat([item[0] for item in self._val_openset], dim=0)
            y_unknown = torch.cat([item[1] for item in self._val_openset], dim=0)
            confs = [item[2] for item in self._val_openset if item[2] is not None]
            conf = torch.cat(confs, dim=0) if confs else None
            osr = openset_detection_metrics(y_unknown, scores, confidences=conf, correct=None)
            for key, value in osr.items():
                if value == value:
                    self._log_scalar(f"val/osr_{key}", value)
            report["openset"] = osr
        return report

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        hint = self._val_source_hint(dataloader_idx)
        total, parts, packed = self._forward_and_loss(batch, train=False, default_source=hint)
        batch_size = max(1, int(packed["n_tokens"].shape[0]))
        task = str(packed.get("_task") or "")
        if self._is_finite_metric(total):
            self.log("val/loss", total, on_epoch=True, prog_bar=True, add_dataloader_idx=True, batch_size=batch_size)
            if task:
                self.log(f"val/{task}/loss", total, on_epoch=True, batch_size=batch_size)
        for name, value in parts.items():
            if not should_log_loss_part(name, self.loss_weights):
                continue
            if self._is_finite_metric(value):
                self.log(f"val/{name}", value, on_epoch=True, add_dataloader_idx=True, batch_size=batch_size)
        if self.stage == "pretrain":
            recon = reconstruction_monitor_loss(parts)
            if self._is_finite_metric(recon):
                self.log("val/recon", recon, on_epoch=True, prog_bar=True, add_dataloader_idx=True, batch_size=batch_size)
            pred, target, mask = reconstruction_eval_pair(packed, kind="pretrain")
            wave_len = packed.get("iq_length")
            if torch.is_tensor(wave_len):
                wave_len = int(wave_len.reshape(-1)[0].item())
            elif wave_len is not None:
                wave_len = int(wave_len)
            ssim_pred = packed.get("mae_pred")
            ssim_tgt = packed.get("patch_targets")
            if ssim_pred is None:
                ssim_pred = pred
            if ssim_tgt is None:
                ssim_tgt = target
            ssim = ssim_iq(ssim_pred, ssim_tgt, mask, length=wave_len)
            mse = float(masked_patch_mse(pred, target, mask).detach())
            mae = float(masked_patch_mae(pred, target, mask).detach())
            self.log("val/recon_mse", mse, on_epoch=True, batch_size=batch_size)
            self.log("val/recon_mae", mae, on_epoch=True, batch_size=batch_size)
            self.log("val/ssim", ssim, on_epoch=True, batch_size=batch_size)
            batch_u = packed.get("_batch") or {}
            n = int(pred.shape[0])
            ds = self._resolve_dataset_ids(batch_u, n)
            names = self._dataset_names()
            if not hasattr(self, "_val_pretrain_by_ds") or self._val_pretrain_by_ds is None:
                self._val_pretrain_by_ds = {}
            for did in ds.unique():
                ident = int(did)
                if ident < 0:
                    continue
                select = ds == did
                n_ds = int(select.sum())
                if n_ds == 0:
                    continue
                mask_ds = mask[:n][select] if mask is not None else None
                name = dataset_display_name(ident, names)
                row = self._val_pretrain_by_ds.setdefault(
                    name, {"mse_weighted": 0.0, "mae_weighted": 0.0, "n": 0}
                )
                ds_mse = float(masked_patch_mse(pred[:n][select], target[:n][select], mask_ds).detach())
                ds_mae = float(masked_patch_mae(pred[:n][select], target[:n][select], mask_ds).detach())
                row["mse_weighted"] += ds_mse * n_ds
                row["mae_weighted"] += ds_mae * n_ds
                row["n"] += n_ds
            return total
        for item in packed.get("_all") or [packed]:
            self._collect_val_item(item)
        return total

    def _merge_val_report_into_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        for task, info in (self._last_val_report or {}).items():
            if not isinstance(info, dict):
                continue
            if info.get("kind") == "classification":
                metrics[f"val/acc_{task}"] = info["acc"]
                metrics[f"val/f1_{task}"] = info["f1"]
                metrics[f"val/miss_rate_{task}"] = info["miss_rate"]
                for dataset, row in (info.get("datasets") or {}).items():
                    metrics[f"val/acc_{task}/{dataset}"] = row["acc"]
                    metrics[f"val/f1_{task}/{dataset}"] = row["f1"]
                    metrics[f"val/miss_rate_{task}/{dataset}"] = row["miss_rate"]
            elif info.get("kind") == "clustering":
                metrics.setdefault(f"val/acc_{task}", info["acc"])
                metrics.setdefault("val/nmi" if task == "clustering" else f"val/nmi_{task}", info["nmi"])
                metrics.setdefault("val/nmi_within_domain", info["mean_nmi"])
                metrics.setdefault(f"val/macro_nmi_{task}", info["mean_nmi"])
                for dataset, row in (info.get("datasets") or {}).items():
                    metrics.setdefault(f"val/acc_{task}/{dataset}", row["acc"])
                    metrics.setdefault(f"val/nmi_{task}/{dataset}", row["nmi"])
            elif info.get("kind") == "reconstruction":
                metrics.setdefault(f"val/mse_{task}", info["mse"])
                metrics.setdefault(f"val/mae_{task}", info["mae"])
                for dataset, row in (info.get("datasets") or {}).items():
                    metrics.setdefault(f"val/mse_{task}/{dataset}", row["mse"])
                    metrics.setdefault(f"val/mae_{task}/{dataset}", row["mae"])
        return metrics

    def _flush_pretrain_dataset_scores(self) -> dict[str, Any] | None:
        """预训练：按 H5 stem 汇总 recon_mse / mae。"""
        buckets = getattr(self, "_val_pretrain_by_ds", None) or {}
        if not buckets:
            return None
        datasets: dict[str, dict[str, float]] = {}
        mse_w = mae_w = n_tot = 0.0
        for name, row in sorted(buckets.items()):
            n = float(row.get("n", 0) or 0)
            if n <= 0:
                continue
            mse = float(row["mse_weighted"] / n)
            mae = float(row["mae_weighted"] / n)
            datasets[name] = {"mse": mse, "mae": mae, "n": n}
            self._log_scalar(f"val/recon_mse/{name}", mse)
            self._log_scalar(f"val/recon_mae/{name}", mae)
            mse_w += float(row["mse_weighted"])
            mae_w += float(row["mae_weighted"])
            n_tot += n
        if n_tot <= 0:
            return None
        return {
            "kind": "reconstruction",
            "mse": mse_w / n_tot,
            "mae": mae_w / n_tot,
            "n": n_tot,
            "datasets": datasets,
        }

    def on_validation_epoch_end(self) -> None:
        if getattr(self.trainer, "sanity_checking", False):
            self._reset_val_buffers()
            return
        if self.stage != "pretrain":
            self._last_val_report = self._flush_val_scores()
        metrics = dict(self.trainer.callback_metrics)
        if self.stage == "pretrain":
            pretrain_report = self._flush_pretrain_dataset_scores()
            if pretrain_report is not None:
                self._last_val_report = {"pretrain": pretrain_report}
            value = aggregate_val_monitor(metrics)
            if value is not None and math.isfinite(value):
                self.log("val/monitor", value, prog_bar=True, on_epoch=True, sync_dist=True)
            self._reset_val_buffers()
            return
        if self.stage != "pretrain":
            metrics = prune_inactive_task_val_metrics(metrics, self._last_val_report)
        metrics = self._merge_val_report_into_metrics(metrics)
        if self.stage in ("stage2", "downstream"):
            geo = aggregate_multitask_geomean(metrics)
            if geo is not None:
                self.log("val/multitask_geomean", geo, prog_bar=False, on_epoch=True, sync_dist=True)
            cls_geo = aggregate_classification_geomean(metrics)
            if cls_geo is not None:
                self.log("val/classification_geomean", cls_geo, prog_bar=True, on_epoch=True, sync_dist=True)
        if self.stage == "joint":
            scores = self.train_cfg.get("specialist_scores") or {}
            geo = aggregate_specialist_geomean(
                metrics, {str(k): float(v) for k, v in scores.items()} if scores else None
            )
            if geo is not None:
                self.log("val/specialist_geomean", geo, prog_bar=True, on_epoch=True, sync_dist=True)

    def on_validation_end(self) -> None:
        if getattr(self.trainer, "sanity_checking", False):
            return
        datamodule = getattr(self.trainer, "datamodule", None)
        source_names = list(
            getattr(datamodule, "val_source_names", None)
            or getattr(datamodule, "source_names", [])
            or []
        )
        text = format_val_epoch_metrics(
            dict(self.trainer.callback_metrics),
            epoch=int(self.current_epoch),
            source_names=source_names,
            task_report=self._last_val_report,
        )
        if not text:
            return
        print(text, flush=True)
        logging.getLogger("resmamba").info("%s", text)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self.mix is not None:
            checkpoint["mix_state"] = self.mix.state_dict()
        checkpoint["train_stage"] = self.stage
        checkpoint["train_cfg_epochs"] = int(self.train_cfg.get("epochs", 1))
        handle = getattr(self.model, "peft", None)
        if handle is not None:
            checkpoint["peft_cfg"] = handle.cfg.to_dict()
            checkpoint["peft_tasks"] = list(handle.tasks)
            checkpoint["peft_meta"] = handle.to_meta()
        checkpoint["build_adapters"] = bool(getattr(self.model.cfg, "build_adapters", False))
        checkpoint["build_shared_adapter"] = bool(getattr(self.model.cfg, "build_shared_adapter", False))
        if self.ema_teacher is not None:
            checkpoint["ema_teacher"] = self.ema_teacher.state_dict()
        if self._frozen_prototypes:
            checkpoint["frozen_prototypes"] = {k: v.detach().cpu() for k, v in self._frozen_prototypes.items()}
        if self._stem_total_ema:
            checkpoint["stem_total_ema"] = dict(self._stem_total_ema)
        if self._mae_stem_ema:
            checkpoint["mae_stem_ema"] = dict(self._mae_stem_ema)
        if self._recon_mse_stem_ema:
            checkpoint["recon_mse_stem_ema"] = dict(self._recon_mse_stem_ema)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        mix_state = checkpoint.get("mix_state")
        if mix_state and self.mix is not None:
            self.mix.load_state_dict(mix_state)
        ema_state = checkpoint.get("ema_teacher")
        if ema_state:
            if self.ema_teacher is not None:
                self.ema_teacher.load_state_dict(ema_state, strict=False)
            else:
                self._ema_box["state"] = ema_state
        frozen = checkpoint.get("frozen_prototypes")
        if isinstance(frozen, dict):
            self._frozen_prototypes = {str(k): v for k, v in frozen.items()}
        stem_total = checkpoint.get("stem_total_ema")
        if isinstance(stem_total, dict):
            self._stem_total_ema = {str(k): float(v) for k, v in stem_total.items()}
        mae_stem = checkpoint.get("mae_stem_ema")
        if isinstance(mae_stem, dict):
            self._mae_stem_ema = {str(k): float(v) for k, v in mae_stem.items()}
        recon_stem = checkpoint.get("recon_mse_stem_ema")
        if isinstance(recon_stem, dict):
            self._recon_mse_stem_ema = {str(k): float(v) for k, v in recon_stem.items()}


def build_lit_module(cfg: SignalModelConfig, train_cfg: dict[str, Any], *, stage: str, mix: DynamicRatioScheduler | None) -> SignalLitModule:
    payload = {name: getattr(cfg, name) for name in SignalModelConfig.__dataclass_fields__ if name != "tokenizer"}
    payload["build_task_heads"] = stage != "pretrain"
    payload["build_task_interface"] = False
    payload["build_prototype_registry"] = stage != "pretrain"
    payload["build_adapters"] = stage in ("stage3", "joint", "continual")
    payload["build_shared_adapter"] = stage in ("joint", "continual")
    payload["tokenizer"] = dict(cfg.tokenizer.__dict__)
    model = SignalFoundationModel(SignalModelConfig.from_dict(payload))
    return SignalLitModule(model, train_cfg, stage=stage, mix=mix)
