from __future__ import annotations

import logging
import math
import time
from typing import Any

import torch
import torch.nn.functional as F

from resmamba_signal_model.models.model import SignalFoundationModel, SignalModelConfig
from resmamba_signal_model.models.peft import TOKENIZER_LAST_PREFIXES
from resmamba_signal_model.models.task_interface import SOURCE_TO_TASK
from resmamba_signal_model.training.task_catalog import resolve_task_catalog
from resmamba_signal_model.training.data_module import merge_source_batches
from resmamba_signal_model.training.checkpointing import (
    aggregate_multitask_geomean,
    aggregate_specialist_geomean,
    aggregate_val_monitor,
)
from resmamba_signal_model.training.clustering_labels import load_dataset_id_names
from resmamba_signal_model.training.freeze import iter_head_param_prefixes
from resmamba_signal_model.training.logging_utils import format_val_epoch_metrics
from resmamba_signal_model.training.losses import (
    _first_present,
    downstream_task_loss,
    negcos_temperature,
    reconstruction_monitor_loss,
    weighted_pretrain_loss,
)
from resmamba_signal_model.training.lr_schedule import (
    apply_param_group_lr_scales,
    as_torch_lr_scheduler,
    build_lr_scheduler,
    resolve_lr_schedule,
    scale_lr_for_grad_accum,
    use_cosine_lr_decay,
)
from resmamba_signal_model.training.mix import resolve_mix_strategy
from resmamba_signal_model.training.metrics import (
    classification_epoch_scores,
    clustering_epoch_scores,
    dataset_display_name,
    masked_patch_mse,
    openset_detection_metrics,
    reconstruction_epoch_scores,
    reconstruction_eval_pair,
    ssim_iq,
    ssim_iq_accumulate,
)
from resmamba_signal_model.training.mix import DynamicRatioScheduler

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


def _per_source_mae(outputs: dict[str, torch.Tensor], source_names: list[str]) -> tuple[dict[str, float], dict[str, float]]:
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
            value = float(F.smooth_l1_loss(pred[i, :length][m[:length]], target[i, :length][m[:length]]).detach())
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
            or {"mae": 1.0, "physical": 0.2, "impute": 0.2, "readout": 0.1, "domain": 0.05}
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
        self._reset_val_buffers()

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
        total = max(int(self.train_cfg.get("steps_per_epoch", 100)) * int(self.train_cfg.get("epochs", 1)), 2)
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

    def _optimizer_param_groups(self, peak_lr: float) -> list[dict[str, Any]]:
        if self.stage in ("pretrain", "downstream"):
            params = [p for p in self.model.parameters() if p.requires_grad]
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

        def take(predicate, lr: float, name: str) -> None:
            chosen = []
            for n, p in self.model.named_parameters():
                if not p.requires_grad or id(p) in used:
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
        take(lambda n, p: n.startswith(head_prefixes) or n.startswith("extra_task_heads."), heads_lr, "heads")
        take(lambda n, p: n.startswith("task_interface."), peak_lr, "uti")
        take(lambda n, p: True, peak_lr, "other")
        if not groups:
            params = [p for p in self.model.parameters() if p.requires_grad]
            if not params:
                raise RuntimeError("当前 stage 没有可训练参数，请检查 freeze 配置")
            return [{"params": params, "lr": peak_lr}]
        return groups

    def _val_source_hint(self, dataloader_idx: int) -> str | None:
        dm = getattr(self, "trainer", None)
        dm = getattr(dm, "datamodule", None) if dm is not None else None
        names = list(getattr(dm, "val_source_names", None) or getattr(dm, "source_names", None) or [])
        if 0 <= dataloader_idx < len(names):
            return str(names[dataloader_idx])
        return None

    def _schedule_temperature(self) -> float:
        total = max(int(self.train_cfg.get("steps_per_epoch", 100)) * int(self.train_cfg.get("epochs", 1)), 1)
        progress = float(getattr(self, "global_step", 0) or 0) / float(total)
        tau = negcos_temperature(
            progress,
            tau_max=float(self.train_cfg.get("negcos_tau_max", 0.5)),
            tau_min=float(self.train_cfg.get("negcos_tau_min", 0.05)),
        )
        self.model._negcos_temperature = tau
        head = getattr(self.model, "clustering_head", None)
        if head is not None:
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
        if float(self.train_cfg.get("distill_weight", 0.0) or 0.0) <= 0:
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

    def _ema_enabled(self) -> bool:
        explicit = self.train_cfg.get("ema_teacher")
        if explicit is not None:
            return bool(explicit)
        return any(
            float(self.loss_weights.get(name, 0.0) or 0.0) > 0.0
            for name in ("latent", "uti_pooled", "uti_token", "uti_query")
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
        outputs["teacher_pooled"] = teacher_out.get("uti_pooled")
        outputs["teacher_tokens"] = teacher_out.get("uti_tokens")
        outputs["teacher_query"] = teacher_out.get("uti_query")
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
        if bool(self.train_cfg.get("combine_then_pack", True)) or len(sources) != 1:
            merged = merge_source_batches(sources)
        else:
            merged = next(iter(sources.values()))
        self.model.grl.lambd = self._dann_lambda()
        self._schedule_temperature()
        outputs = self.model(merged, mode="pretrain")
        if self.ema_teacher is not None:
            outputs = self._attach_ema_teacher_outputs(outputs, merged)
        weights = dict(self.loss_weights)
        weights["domain"] = float(weights.get("domain", 0.05)) * float(self.model.grl.lambd)
        total, parts = weighted_pretrain_loss(outputs, merged, weights)
        return total, parts, {**outputs, "_batch": merged}

    def _downstream_forward(
        self, batch: Any, *, default_source: str | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        sources = _as_source_map(batch, default_name=default_source)
        total = None
        parts: dict[str, torch.Tensor] = {}
        last: dict[str, Any] | None = None
        src_losses: dict[str, float] = {}
        src_tokens: dict[str, float] = {}
        packed_all: list[dict[str, Any]] = []
        token_mass = None
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
            if self.training and self.distill_teacher is not None and float(self.train_cfg.get("distill_weight", 0.0) or 0.0) > 0:
                with torch.no_grad():
                    t_out = self.distill_teacher.model(sub, mode="task", task=task)
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
                and assign is not None
                and int(embed.shape[-1]) == int(registry.dim)
            ):
                ns = "emitter" if task == "emitter" else "modulation"
                registry.update(ns, embed.detach(), assign.detach())
            if registry is not None and self._frozen_prototypes:
                ns = "emitter" if task == "emitter" else "modulation"
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
                modulation_contrastive_weight=float(self.train_cfg.get("modulation_contrastive_weight", 0.25)),
                emitter_contrastive_weight=float(self.train_cfg.get("emitter_contrastive_weight", 0.0)),
                domain_weight=float(self.loss_weights.get("domain", 0.1)),
                recon_weight=float(self.train_cfg.get("lambda_recon", 0.1)),
                phys_weight=float(self.loss_weights.get("physical", 0.1)),
                task_kind=None if spec is None else spec.kind,
                task_catalog=self.train_cfg,
                label_field=label_field,
                supervised_clustering=supervised,
                distill_weight=float(self.train_cfg.get("distill_weight", 0.0)),
                distill_temperature=float(self.train_cfg.get("distill_temperature", 2.0)),
                distill_confidence=float(self.train_cfg.get("distill_confidence", 0.5)),
                prototype_anchor_weight=float(self.train_cfg.get("prototype_anchor_weight", 0.0)),
            )
            n_tokens = outputs["n_tokens"].sum().clamp_min(1).to(dtype=loss.dtype)
            src_losses[name] = float(loss.detach())
            src_tokens[name] = float(n_tokens.detach())
            if bool(self.train_cfg.get("token_normalized_loss", True)):
                weighted = loss * n_tokens
                total = weighted if total is None else total + weighted
            else:
                total = loss if total is None else total + loss
            token_mass = n_tokens if token_mass is None else token_mass + n_tokens
            for key, value in sub_parts.items():
                parts[f"{name}/{key}"] = value
            last = {**outputs, "_batch": sub, "_task": task, "_source": name}
            packed_all.append(last)
        assert total is not None and last is not None
        if bool(self.train_cfg.get("token_normalized_loss", True)) and token_mass is not None:
            total = total / token_mass.clamp_min(1)
        last = {**last, "_src_losses": src_losses, "_src_tokens": src_tokens, "_all": packed_all}
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
                src_losses, src_tokens = _per_source_mae(packed, source_names)
                self.mix.update(src_losses, src_tokens)
        ratios = self.mix.ratios()
        batch_size = max(1, int(packed["n_tokens"].shape[0]))
        for name, ratio in ratios.items():
            self.log(f"mix/ratio_{name}", ratio, on_step=True, batch_size=batch_size)
            self.log(f"mix/ema_loss_{name}", self.mix.ema[name], on_step=True, batch_size=batch_size)
        shares = self.mix.token_shares(int(self.train_cfg.get("token_budget", 4096)))
        for name, share in shares.items():
            self.log(f"mix/tokens_{name}", float(share), on_step=True, batch_size=batch_size)

    def training_step(self, batch: Any, batch_idx: int):
        total, parts, packed = self._forward_and_loss(batch, train=True)
        now = time.perf_counter()
        dt = max(now - self._last_step_t, 1e-6)
        self._last_step_t = now
        n_tokens = float(packed["n_tokens"].sum().detach())
        batch_size = max(1, int(packed["n_tokens"].shape[0]))
        self.log("loss/total", total, prog_bar=True, on_step=True, batch_size=batch_size)
        for name, value in parts.items():
            log_name = name if name.startswith("loss/") else f"loss/{name}"
            self.log(log_name, value, on_step=True, prog_bar=(name == "mae"), batch_size=batch_size)
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

    def _reset_val_buffers(self) -> None:
        self._val_cls: dict[str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
        self._val_cluster: dict[str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
        self._val_recon: dict[str, dict[str, Any]] = {}
        self._val_absorb_embeds = []
        self._val_absorb_scores = []
        self._val_openset = []

    def _cpu_1d(self, value: Any) -> torch.Tensor | None:
        if value is None:
            return None
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        return value.detach().reshape(-1).cpu()

    def _collect_cls(self, task: str, pred: torch.Tensor, labels: Any, dataset_id: Any) -> None:
        pred_t = self._cpu_1d(pred)
        label_t = self._cpu_1d(labels)
        if pred_t is None or label_t is None:
            return
        n = min(pred_t.numel(), label_t.numel())
        pred_t = pred_t[:n].long()
        label_t = label_t[:n].long()
        ds_t = self._cpu_1d(dataset_id)
        if ds_t is None or ds_t.numel() != n:
            ds_t = torch.full((n,), -1, dtype=torch.long)
        else:
            ds_t = ds_t[:n].long()
        self._val_cls.setdefault(task, []).append((pred_t, label_t, ds_t))

    def _collect_cluster(self, task: str, pred: torch.Tensor, labels: Any, dataset_id: Any) -> None:
        pred_t = self._cpu_1d(pred)
        label_t = self._cpu_1d(labels)
        if pred_t is None or label_t is None:
            return
        n = min(pred_t.numel(), label_t.numel())
        pred_t = pred_t[:n].long()
        label_t = label_t[:n].long()
        ds_t = self._cpu_1d(dataset_id)
        if ds_t is None or ds_t.numel() != n:
            ds_t = torch.full((n,), -1, dtype=torch.long)
        else:
            ds_t = ds_t[:n].long()
        self._val_cluster.setdefault(task, []).append((pred_t, label_t, ds_t))

    def _recon_bucket(self, task: str) -> dict[str, Any]:
        return self._val_recon.setdefault(
            task,
            {"ssim_sum": 0.0, "ssim_count": 0, "mse_weighted": 0.0, "n": 0, "datasets": {}},
        )

    def _add_recon_stats(
        self,
        bucket: dict[str, Any],
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None,
        n: int,
    ) -> None:
        total, count = ssim_iq_accumulate(pred, target, mask)
        mse = float(masked_patch_mse(pred, target, mask).detach())
        bucket["ssim_sum"] += float(total)
        bucket["ssim_count"] += int(count)
        bucket["mse_weighted"] += mse * max(n, 1)
        bucket["n"] += int(n)

    def _collect_recon(self, task: str, packed: dict[str, Any], batch_u: dict[str, Any]) -> None:
        if packed.get("recon_norm") is None and packed.get("mae_pred") is None and packed.get("pred_patches") is None:
            return
        try:
            pred, target, mask = reconstruction_eval_pair(packed)
        except (KeyError, ValueError):
            return
        if pred is None or target is None:
            return
        n = int(pred.shape[0])
        self._add_recon_stats(self._recon_bucket(task), pred, target, mask, n)
        dataset_id = batch_u.get("dataset_id")
        if dataset_id is None or not torch.is_tensor(dataset_id):
            return
        names = self._dataset_names()
        ds = dataset_id.detach().reshape(-1)
        n = min(n, int(ds.numel()))
        for did in ds[:n].unique():
            ident = int(did)
            if ident < 0:
                continue
            select = ds[:n] == did
            n_ds = int(select.sum())
            if n_ds == 0:
                continue
            mask_ds = mask[:n][select] if mask is not None else None
            name = dataset_display_name(ident, names)
            ds_bucket = self._recon_bucket(task)["datasets"].setdefault(
                name,
                {"ssim_sum": 0.0, "ssim_count": 0, "mse_weighted": 0.0, "n": 0},
            )
            self._add_recon_stats(ds_bucket, pred[:n][select], target[:n][select], mask_ds, n_ds)

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
        if kind == "classification" and logits is not None:
            labels = batch_u.get(label_field or "canonical_mod_label_id", batch_u.get("mod_label_id"))
            self._collect_cls(task or "modulation", logits.argmax(dim=-1), labels, batch_u.get("dataset_id"))
        if kind == "emitter" or (kind != "classification" and packed.get("emitter_logits") is not None):
            pred_e = packed.get("task_logits")
            if pred_e is None:
                pred_e = packed.get("emitter_logits")
            if pred_e is not None:
                labels_e = batch_u.get(label_field or "global_emitter_id", batch_u.get("emitter_id"))
                self._collect_cls(task or "emitter", pred_e.argmax(dim=-1), labels_e, batch_u.get("dataset_id"))
        cluster_logits = packed.get("cluster_logits")
        cluster_labels = batch_u.get("global_label_id")
        if cluster_logits is not None and cluster_labels is not None:
            self._collect_cluster(task or "clustering", cluster_logits.argmax(dim=-1), cluster_labels, batch_u.get("dataset_id"))
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
        self._log_scalar(f"val/acc_{task}", report["acc"])
        self._log_scalar(f"val/f1_{task}", report["f1"])
        self._log_scalar(f"val/macro_acc_{task}", report["mean_acc"])
        self._log_scalar(f"val/macro_f1_{task}", report["mean_f1"])
        for dataset, row in (report.get("datasets") or {}).items():
            self._log_scalar(f"val/acc_{task}/{dataset}", row["acc"])
            self._log_scalar(f"val/f1_{task}/{dataset}", row["f1"])

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
            self._log_scalar("val/nmi" if task == "clustering" else f"val/nmi_{task}", info["nmi"])
            self._log_scalar("val/ari" if task == "clustering" else f"val/ari_{task}", info["ari"])
            self._log_scalar("val/nmi_within_domain", info["mean_nmi"])
            self._log_scalar(f"val/macro_nmi_{task}", info["mean_nmi"])
            self._log_scalar(f"val/macro_ari_{task}", info["mean_ari"])
            for dataset, row in (info.get("datasets") or {}).items():
                self._log_scalar(f"val/nmi_{task}/{dataset}", row["nmi"])
        for task, bucket in self._val_recon.items():
            ssim = float(bucket["ssim_sum"] / bucket["ssim_count"]) if bucket["ssim_count"] else 0.0
            mse = float(bucket["mse_weighted"] / bucket["n"]) if bucket["n"] else 0.0
            per_dataset: dict[str, dict[str, float]] = {}
            for dataset, row in (bucket.get("datasets") or {}).items():
                ds_ssim = float(row["ssim_sum"] / row["ssim_count"]) if row["ssim_count"] else 0.0
                ds_mse = float(row["mse_weighted"] / row["n"]) if row["n"] else 0.0
                per_dataset[dataset] = {"ssim": ds_ssim, "mse": ds_mse, "n": float(row["n"])}
            info = reconstruction_epoch_scores(ssim, mse, int(bucket["n"]), per_dataset)
            report[task] = info
            self._log_scalar("val/impute_mse" if task in ("prediction", "imputation", "pretrain") else f"val/mse_{task}", mse)
            if task == "prediction":
                self._log_scalar("val/ssim", ssim)
                self._log_scalar("val/ssim_prediction", ssim)
            elif task == "imputation":
                self._log_scalar("val/ssim_imputation", ssim)
            else:
                self._log_scalar(f"val/ssim_{task}", ssim)
            self._log_scalar(f"val/macro_ssim_{task}", info["mean_ssim"])
            for dataset, row in per_dataset.items():
                self._log_scalar(f"val/ssim_{task}/{dataset}", row["ssim"])
                self._log_scalar(f"val/mse_{task}/{dataset}", row["mse"])
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
        if self._is_finite_metric(total):
            self.log("val/loss", total, on_epoch=True, prog_bar=True, add_dataloader_idx=True, batch_size=batch_size)
        for name, value in parts.items():
            if self._is_finite_metric(value):
                self.log(f"val/{name}", value, on_epoch=True, add_dataloader_idx=True, batch_size=batch_size)
        if self.stage == "pretrain":
            recon = reconstruction_monitor_loss(parts)
            if self._is_finite_metric(recon):
                self.log("val/recon", recon, on_epoch=True, prog_bar=True, add_dataloader_idx=True, batch_size=batch_size)
            pred, target, mask = reconstruction_eval_pair(packed)
            ssim = ssim_iq(pred, target, mask)
            mse = masked_patch_mse(pred, target, mask)
            # recon_mse：全部 target patch 上的 MSE；保留 impute_mse 别名兼容旧日志/门控
            self.log("val/recon_mse", mse, on_epoch=True, batch_size=batch_size)
            self.log("val/impute_mse", mse, on_epoch=True, batch_size=batch_size)
            self.log("val/ssim", ssim, on_epoch=True, batch_size=batch_size)
            return total
        for item in packed.get("_all") or [packed]:
            self._collect_val_item(item)
        return total

    def _merge_val_report_into_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        for task, info in (self._last_val_report or {}).items():
            if not isinstance(info, dict):
                continue
            if info.get("kind") == "classification":
                metrics.setdefault(f"val/acc_{task}", info["acc"])
                metrics.setdefault(f"val/f1_{task}", info["f1"])
                metrics.setdefault(f"val/macro_acc_{task}", info["mean_acc"])
                for dataset, row in (info.get("datasets") or {}).items():
                    metrics.setdefault(f"val/acc_{task}/{dataset}", row["acc"])
            elif info.get("kind") == "clustering":
                metrics.setdefault("val/nmi" if task == "clustering" else f"val/nmi_{task}", info["nmi"])
            elif info.get("kind") == "reconstruction":
                key = "val/ssim_prediction" if task == "prediction" else f"val/ssim_{task}"
                metrics.setdefault(key, info["ssim"])
        return metrics

    def on_validation_epoch_end(self) -> None:
        if getattr(self.trainer, "sanity_checking", False):
            self._reset_val_buffers()
            return
        if self.stage != "pretrain":
            self._last_val_report = self._flush_val_scores()
        metrics = dict(self.trainer.callback_metrics)
        if self.stage == "pretrain":
            value = aggregate_val_monitor(metrics)
            if value is not None and math.isfinite(value):
                self.log("val/monitor", value, prog_bar=True, on_epoch=True, sync_dist=True)
            return
        metrics = self._merge_val_report_into_metrics(metrics)
        if self.stage in ("stage2", "downstream"):
            geo = aggregate_multitask_geomean(metrics)
            if geo is not None:
                self.log("val/multitask_geomean", geo, prog_bar=True, on_epoch=True, sync_dist=True)
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
        source_names = list(getattr(datamodule, "source_names", []) or [])
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


def build_lit_module(cfg: SignalModelConfig, train_cfg: dict[str, Any], *, stage: str, mix: DynamicRatioScheduler | None) -> SignalLitModule:
    payload = {name: getattr(cfg, name) for name in SignalModelConfig.__dataclass_fields__ if name != "tokenizer"}
    payload["build_task_heads"] = stage != "pretrain"
    payload["build_task_interface"] = True
    payload["build_prototype_registry"] = stage != "pretrain"
    payload["build_adapters"] = stage in ("stage3", "joint", "continual")
    payload["build_shared_adapter"] = stage in ("joint", "continual")
    payload["tokenizer"] = dict(cfg.tokenizer.__dict__)
    model = SignalFoundationModel(SignalModelConfig.from_dict(payload))
    return SignalLitModule(model, train_cfg, stage=stage, mix=mix)
