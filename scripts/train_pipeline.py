#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resmamba_signal_model.thread_env import normalize_thread_env

normalize_thread_env()

import argparse
import json
import math
import time
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Subset, Sampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from resmamba_signal_model import ResMambaSignalConfig, ResMambaSignalModel
from resmamba_signal_model.data.rfdata import RFDataPoolDataset, build_rfdata_pool, pad_iq_collate, rfdata_dataloader_worker_init, variable_length_collate
from resmamba_signal_model.data.val_subset import (
    build_length_bucket_val_batches,
    build_per_dataset_class_balanced_val_indices,
    format_val_subset_plan,
)
from resmamba_signal_model.data.sampling import (
    BalancedSamplingStrategy,
    LengthBucketBalancedBatchSampler,
    LengthBucketPKBatchSampler,
    build_dataset_balanced_sampler,
    build_uniform_sampler,
    format_sampling_plan,
    resolve_balanced_sampling_strategy,
    resolve_length_bucket_weight_mode,
    resolve_pk_sampling_params,
)
from resmamba_signal_model.training.clustering_labels import load_dataset_id_names, resolve_modulation_labels
from resmamba_signal_model.training.emitter_labels import GlobalEmitterLabelMap, build_global_emitter_label_map, global_emitter_labels
from resmamba_signal_model.training.early_stopping import EarlyStopping
from resmamba_signal_model.training.lr_schedule import LinearWarmupLR, WarmupCosineLR, scale_lr_for_grad_accum
from resmamba_signal_model.training.losses import stage2_task_loss, weighted_pretrain_loss
from resmamba_signal_model.training.metrics import accuracy, macro_f1, nmi_score, ssim_iq, ssim_iq_accumulate
from resmamba_signal_model.training.selection import (
    compute_selection_score,
    resolve_selection_metric_name,
    selection_higher_is_better,
)
from resmamba_signal_model.training.param_stats import format_param_stats
from resmamba_signal_model.training.stages import configure_peft_stage2, configure_pretrain, configure_stage2_heads

DEFAULT_VAL_POOL = {
    "pretrain_train": "pretrain_val",
    "downstream_modulation_train": "downstream_modulation_val",
    "downstream_emitter_train": "downstream_emitter_val",
    "downstream_prediction_train": "downstream_prediction_val",
    "downstream_source_train": "downstream_source_val",
    "clustering_train": "clustering_val",
}

# H5 后缀用途：*_train→预训练；*_val→各阶段验证；*_test→下游头与微调训练（经 *_train pool 读取）。
# task_pools 中名为 *_train 的下游 pool 实际指向 *_test.h5，见 label_maps.json。


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_train_config(path: str | Path, *, stage: str, task: str) -> dict[str, Any]:
    defaults_path = ROOT / "configs" / "val_subset.yaml"
    defaults = load_yaml(defaults_path) if defaults_path.is_file() else {}
    cfg = {**defaults, **load_yaml(path)}
    return merge_task_config(cfg, stage=stage, task=task)


def merge_task_config(train_cfg: dict[str, Any], *, stage: str, task: str) -> dict[str, Any]:
    """将 stage2_heads.yaml 中 task_configs.<task> 覆盖到当前训练配置。"""
    if stage != "stage2":
        return train_cfg
    overrides = (train_cfg.get("task_configs") or {}).get(task)
    if not overrides:
        return train_cfg
    merged = dict(train_cfg)
    for key, value in overrides.items():
        if key == "task_configs":
            continue
        merged[key] = value
    return merged


def resolve_rfdata_root(cli_value: str | None, train_cfg: dict[str, Any]) -> Path:
    """解析 RFData 根目录：CLI > 训练 YAML > 项目 dataset/；相对路径相对项目根。"""
    for raw in (cli_value, train_cfg.get("rfdata_root"), str(ROOT / "dataset")):
        if not raw:
            continue
        root = Path(str(raw)).expanduser()
        if not root.is_absolute():
            root = ROOT / root
        return root.resolve()
    return (ROOT / "dataset").resolve()


def validate_rfdata_root(root: Path) -> None:
    label_maps = root / "label_maps.json"
    if label_maps.is_file():
        return
    raise FileNotFoundError(
        f"在 {root} 下找不到 label_maps.json。"
        "请先执行数据准备（README「数据准备」），或设置：\n"
        f"  export RFDATA_ROOT={ROOT / 'dataset'}\n"
        f"  --rfdata-root {ROOT / 'dataset'}"
    )


def build_model(
    path: str | None,
    *,
    sequence_packing: bool = False,
    model_overrides: dict[str, Any] | None = None,
) -> ResMambaSignalModel:
    raw = load_yaml(path) if path else {}
    m = raw.get("model", raw)
    fields = ResMambaSignalConfig.__dataclass_fields__
    cfg = ResMambaSignalConfig(**{k: m[k] for k in fields if k in m})
    cfg.sequence_packing = bool(sequence_packing or cfg.sequence_packing)
    if model_overrides:
        for key, value in model_overrides.items():
            if key in fields:
                setattr(cfg, key, value)
    return ResMambaSignalModel(cfg)


def resolve_peft_mode(train_cfg: dict[str, Any], *, freeze_backbone: bool) -> PeftMode:
    if not freeze_backbone:
        return "full_backbone"
    mode = str(train_cfg.get("peft_mode", "task_path"))
    if mode not in {"head_only", "task_path", "task_path_shared", "lora_task_path", "full_backbone"}:
        raise ValueError(f"Unsupported peft_mode: {mode}")
    return mode  # type: ignore[return-value]


def resolve_amp_dtype(device: torch.device, train_cfg: dict[str, Any]) -> torch.dtype | None:
    """AMP dtype：CUDA 上默认 bfloat16（避免 emitter task_path 在 fp16 下 NaN），可 YAML 覆盖。"""
    if device.type != "cuda":
        return None
    explicit = train_cfg.get("amp_dtype")
    if explicit is not None:
        name = str(explicit).strip().lower()
        if name in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if name in {"fp16", "float16", "half"}:
            return torch.float16
        raise ValueError(f"Unsupported amp_dtype: {explicit!r} (use bfloat16 or float16)")
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _autocast_kwargs(device: torch.device, use_amp: bool, amp_dtype: torch.dtype | None) -> dict[str, Any]:
    if not use_amp or device.type != "cuda" or amp_dtype is None:
        return {"device_type": "cuda", "enabled": use_amp}
    return {"device_type": "cuda", "enabled": True, "dtype": amp_dtype}


def _sanitize_metrics_for_json(metrics: dict[str, float]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key, value in metrics.items():
        out[key] = None if not math.isfinite(float(value)) else float(value)
    return out


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# H5 后缀：*_train→预训练；*_val→各阶段验证/选模（无独立测试集）；*_test→下游头与微调训练（经 *_train pool 读取）。
DEPRECATED_TEST_POOLS = frozenset({
    "pretrain_test",
    "downstream_modulation_test",
    "downstream_source_test",
    "downstream_emitter_test",
    "downstream_prediction_test",
    "clustering_test",
})


def infer_val_pool(train_pool: str, val_pool: str | None) -> str:
    if val_pool:
        if val_pool in DEPRECATED_TEST_POOLS:
            raise ValueError(
                f"验证请使用 *_val pool（如 pretrain_val），不再使用 {val_pool!r}；"
                f"*_test.h5 仅用于下游/微调训练（downstream_*_train、clustering_train）"
            )
        return val_pool
    if train_pool in DEPRECATED_TEST_POOLS:
        raise ValueError(f"{train_pool!r} 已废弃；下游训练请用 downstream_*_train / clustering_train")
    if train_pool in DEFAULT_VAL_POOL:
        return DEFAULT_VAL_POOL[train_pool]
    if train_pool.endswith("_train"):
        return train_pool.replace("_train", "_val")
    raise ValueError(f"无法推断验证集 pool，请显式指定 --val-pool（train pool={train_pool!r}）")


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    non_blocking = device.type == "cuda"
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if key == "iq" and isinstance(value, list):
            out[key] = [tensor.to(device, non_blocking=non_blocking) for tensor in value]
        elif torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=non_blocking)
        else:
            out[key] = value
    return out


def link_autodl_tensorboard(log_dir: Path) -> None:
    """AutoDL 默认监控 /root/tf-logs；将当前 run 链到该目录便于面板读取。"""
    autodl_root = Path("/root/tf-logs")
    if not Path("/root").is_dir():
        return
    autodl_root.mkdir(parents=True, exist_ok=True)
    link = autodl_root / "resmamba_current"
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.exists():
        return
    link.symlink_to(log_dir.resolve())
    print(f"AutoDL TensorBoard: {link} -> {log_dir.resolve()}")


def _filter_state_dict_for_model(model: ResMambaSignalModel, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    model_state = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in state.items():
        if key not in model_state:
            continue
        if model_state[key].shape != value.shape:
            skipped.append(key)
            continue
        filtered[key] = value
    if skipped:
        preview = ", ".join(skipped[:5])
        suffix = " ..." if len(skipped) > 5 else ""
        print(f"[checkpoint] skipped {len(skipped)} shape-mismatched keys: {preview}{suffix}")
    return filtered


def load_model_weights(model: ResMambaSignalModel, path: str | Path, device: torch.device) -> None:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state = _filter_state_dict_for_model(model, state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[checkpoint] missing keys: {len(missing)}")
    if unexpected:
        print(f"[checkpoint] unexpected keys: {len(unexpected)}")


def load_checkpoint(model: ResMambaSignalModel, path: str | Path, device: torch.device) -> None:
    """仅加载模型权重（stage2 --pretrained-checkpoint）。"""
    load_model_weights(model, path, device)


def save_checkpoint(
    path: Path,
    *,
    model: ResMambaSignalModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, float],
    scaler: torch.amp.GradScaler | None = None,
    global_step: int = 0,
    best_val: float | None = None,
    best_metrics: dict[str, float] | None = None,
    best_epoch: int | None = None,
    checkpoint_role: str | None = None,
    lr_scheduler: LinearWarmupLR | WarmupCosineLR | None = None,
    early_stopping: EarlyStopping | None = None,
    log_dir: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "args": vars(args),
        "metrics": metrics,
    }
    if best_val is not None:
        payload["best_val"] = best_val
    if best_metrics is not None:
        payload["best_metrics"] = best_metrics
    if best_epoch is not None:
        payload["best_epoch"] = best_epoch
    if checkpoint_role is not None:
        payload["checkpoint_role"] = checkpoint_role
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if lr_scheduler is not None:
        payload["lr_scheduler"] = lr_scheduler.state_dict()
    if early_stopping is not None and early_stopping.enabled:
        payload["early_stopping"] = early_stopping.state_dict()
    if log_dir is not None:
        payload["log_dir"] = str(log_dir)
    torch.save(payload, path)


def write_checkpoint_manifest(
    output_dir: Path,
    *,
    best_epoch: int,
    best_val: float,
    best_metrics: dict[str, float],
    last_epoch: int,
    last_metrics: dict[str, float],
) -> None:
    manifest = {
        "best": {
            "path": "best.pt",
            "epoch": best_epoch,
            "val_loss": best_val if math.isfinite(float(best_val)) else None,
            "metrics": _sanitize_metrics_for_json(best_metrics),
        },
        "last": {
            "path": "last.pt",
            "epoch": last_epoch,
            "val_loss": (
                last_metrics.get("loss")
                if last_metrics.get("loss") is not None and math.isfinite(float(last_metrics.get("loss")))
                else None
            ),
            "metrics": _sanitize_metrics_for_json(last_metrics),
        },
    }
    (output_dir / "checkpoints.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_training_checkpoint(
    path: str | Path,
    *,
    model: ResMambaSignalModel,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    lr_scheduler: LinearWarmupLR | WarmupCosineLR | None = None,
    early_stopping: EarlyStopping | None = None,
    load_optimizer: bool = True,
    load_lr_scheduler: bool = True,
) -> dict[str, Any]:
    """从 checkpoint 恢复完整训练状态（--resume）。"""
    ckpt_path = Path(path).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"无法续训：{ckpt_path} 不是完整的训练 checkpoint")

    load_model_weights(model, ckpt_path, device)

    if optimizer is not None and load_optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    elif optimizer is not None and load_optimizer:
        print("[resume] checkpoint 无 optimizer 状态，使用新 optimizer")
    elif optimizer is not None and not load_optimizer:
        print("[resume] 跳过恢复 optimizer 状态，使用新 optimizer")

    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    if lr_scheduler is not None and load_lr_scheduler and "lr_scheduler" in ckpt:
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    elif lr_scheduler is not None and load_lr_scheduler:
        print("[resume] checkpoint 无 lr_scheduler 状态，LR 调度从当前步重新计算")
    elif lr_scheduler is not None and not load_lr_scheduler:
        print("[resume] 跳过恢复 lr_scheduler 状态，按当前 YAML/CLI 重新构建 LR 调度")

    if early_stopping is not None and early_stopping.enabled and "early_stopping" in ckpt:
        early_stopping.load_state_dict(ckpt["early_stopping"])

    epoch = int(ckpt.get("epoch", 0))
    global_step = int(ckpt.get("global_step", 0))
    best_val = float(ckpt.get("best_val", ckpt.get("metrics", {}).get("loss", float("inf"))))
    best_metrics = ckpt.get("best_metrics") or ckpt.get("metrics") or {}
    best_epoch = int(ckpt.get("best_epoch", epoch if ckpt.get("checkpoint_role") == "best" else 0))
    saved_args = ckpt.get("args") or {}

    print(
        f"[resume] loaded {ckpt_path.name}: epoch={epoch} best_epoch={best_epoch}"
        f" global_step={global_step} best_val={best_val:.6f} next_epoch={epoch + 1}"
    )
    return {
        "epoch": epoch,
        "global_step": global_step,
        "best_val": best_val,
        "best_metrics": best_metrics,
        "best_epoch": best_epoch,
        "log_dir": ckpt.get("log_dir"),
        "output_dir": str(ckpt_path.parent),
        "saved_args": saved_args,
    }


def forward_batch(model: ResMambaSignalModel, batch: dict[str, Any], stage: str, task: str) -> dict[str, torch.Tensor]:
    if stage == "pretrain":
        return model(batch, mode="mae")
    return model(batch, mode="task", task=task)


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    stage: str,
    task: str,
    loss_weights: dict[str, float],
    emitter_offset_lookup: torch.Tensor | None = None,
    emitter_contrastive_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if stage == "pretrain":
        return weighted_pretrain_loss(outputs, batch, loss_weights)
    return stage2_task_loss(
        outputs,
        batch,
        task,
        emitter_offset_lookup=emitter_offset_lookup,
        emitter_contrastive_weight=emitter_contrastive_weight,
    )


def log_static_info(
    writer: SummaryWriter,
    *,
    args: argparse.Namespace,
    train_pool: str,
    val_pool: str,
    train_size: int,
    val_size: int,
    total_params: int,
    trainable_params: int,
) -> None:
    writer.add_text("run/stage", args.stage, 0)
    writer.add_text("run/task", args.task if args.stage == "stage2" else "pretrain", 0)
    writer.add_text("data/train_pool", train_pool, 0)
    writer.add_text("data/val_pool", val_pool, 0)
    writer.add_text("data/rfdata_root", str(args.rfdata_root), 0)
    writer.add_scalar("data/train_samples", train_size, 0)
    writer.add_scalar("data/val_samples", val_size, 0)
    writer.add_scalar("model/total_params", total_params, 0)
    writer.add_scalar("model/trainable_params", trainable_params, 0)
    writer.add_scalar("train/gradient_accumulation_steps", args.gradient_accumulation_steps, 0)
    writer.add_scalar("train/base_lr", args.base_lr, 0)
    writer.add_scalar("train/peak_lr", args.peak_lr, 0)
    writer.add_scalar("train/warmup_steps", args.warmup_steps, 0)
    writer.add_scalar("train/lr_min_ratio", getattr(args, "lr_min_ratio", 0.1), 0)
    writer.add_text("train/lr_schedule", str(getattr(args, "lr_schedule", "warmup_cosine")), 0)


def _grads_are_finite(model: torch.nn.Module) -> bool:
    for param in model.parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            return False
    return True


def _optimizer_step(
    model: ResMambaSignalModel,
    optimizer: torch.optim.Optimizer,
    *,
    use_amp: bool,
    scaler: torch.amp.GradScaler | None,
    lr_scheduler: LinearWarmupLR | WarmupCosineLR | None,
    max_grad_norm: float | None,
) -> tuple[float, bool]:
    lr_used = float(optimizer.param_groups[0]["lr"])
    if use_amp and scaler is not None:
        scaler.unscale_(optimizer)
        if not _grads_are_finite(model):
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            return lr_used, False
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        if not _grads_are_finite(model):
            optimizer.zero_grad(set_to_none=True)
            return lr_used, False
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
    if lr_scheduler is not None:
        lr_scheduler.step()
    return lr_used, True


def train_one_epoch(
    model: ResMambaSignalModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    stage: str,
    task: str,
    loss_weights: dict[str, float],
    epoch: int,
    writer: SummaryWriter,
    global_step: int,
    max_steps: int | None,
    use_amp: bool,
    amp_dtype: torch.dtype | None,
    scaler: torch.amp.GradScaler | None,
    gradient_accumulation_steps: int,
    lr_scheduler: LinearWarmupLR | WarmupCosineLR | None,
    max_grad_norm: float | None,
    empty_cache_every: int = 0,
    emitter_offset_lookup: torch.Tensor | None = None,
    emitter_contrastive_weight: float = 0.0,
) -> tuple[int, float]:
    model.train()
    running_loss = 0.0
    n_micro_batches = 0
    accum_count = 0
    skipped_nonfinite = 0
    loss_ema: float | None = None
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=f"train epoch {epoch}", leave=False, dynamic_ncols=True)
    for batch in pbar:
        if max_steps is not None and global_step >= max_steps:
            break
        batch = move_batch(batch, device)
        with torch.amp.autocast(**_autocast_kwargs(device, use_amp, amp_dtype)):
            outputs = forward_batch(model, batch, stage, task)
            loss, loss_parts = compute_loss(
                outputs,
                batch,
                stage=stage,
                task=task,
                loss_weights=loss_weights,
                emitter_offset_lookup=emitter_offset_lookup,
                emitter_contrastive_weight=emitter_contrastive_weight,
            )

        if not torch.isfinite(loss):
            skipped_nonfinite += 1
            optimizer.zero_grad(set_to_none=True)
            accum_count = 0
            pbar.set_postfix(
                loss="nan",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                amp="on" if use_amp else "off",
                accum=f"{accum_count}/{gradient_accumulation_steps}",
                skip=skipped_nonfinite,
            )
            continue

        scaled_loss = loss / gradient_accumulation_steps
        if use_amp and scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        loss_val = float(loss.detach())
        pending_loss_parts: dict[str, float] | None = None
        if accum_count + 1 >= gradient_accumulation_steps:
            pending_loss_parts = {
                name: float(part.detach())
                for name, part in loss_parts.items()
                if torch.isfinite(part)
            }
        del outputs, loss, loss_parts, scaled_loss, batch
        running_loss += loss_val
        n_micro_batches += 1
        accum_count += 1
        lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(
            loss=f"{loss_val:.4f}",
            lr=f"{lr:.2e}",
            amp="on" if use_amp else "off",
            accum=f"{accum_count}/{gradient_accumulation_steps}",
        )

        if accum_count < gradient_accumulation_steps:
            continue

        lr_used, stepped = _optimizer_step(
            model,
            optimizer,
            use_amp=use_amp,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            max_grad_norm=max_grad_norm,
        )
        if not stepped:
            skipped_nonfinite += 1
            optimizer.zero_grad(set_to_none=True)
            accum_count = 0
            pbar.set_postfix(
                loss="nan-grad",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                amp="on" if use_amp else "off",
                accum=f"{accum_count}/{gradient_accumulation_steps}",
                skip=skipped_nonfinite,
            )
            continue
        optimizer.zero_grad(set_to_none=True)
        accum_count = 0
        global_step += 1
        loss_ema = loss_val if loss_ema is None else 0.98 * loss_ema + 0.02 * loss_val
        writer.add_scalar("train/loss", loss_val, global_step)
        writer.add_scalar("train/loss_ema", loss_ema, global_step)
        writer.add_scalar("train/lr", lr_used, global_step)
        if pending_loss_parts:
            for name, part_val in pending_loss_parts.items():
                writer.add_scalar(f"train/loss_{name}", part_val, global_step)
        writer.flush()
        if empty_cache_every > 0 and global_step % empty_cache_every == 0 and device.type == "cuda":
            torch.cuda.empty_cache()

    if accum_count > 0 and (max_steps is None or global_step < max_steps):
        _optimizer_step(
            model,
            optimizer,
            use_amp=use_amp,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            max_grad_norm=max_grad_norm,
        )
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        writer.flush()

    if skipped_nonfinite:
        print(f"[warn] epoch {epoch}: skipped {skipped_nonfinite} non-finite loss batches")

    avg_loss = running_loss / max(n_micro_batches, 1)
    writer.add_scalar("epoch/train_loss", avg_loss, epoch)
    return global_step, avg_loss


def _dataset_display_name(dataset_id: int, dataset_name_map: dict[int, str] | None) -> str:
    if dataset_name_map:
        return dataset_name_map.get(int(dataset_id), f"dataset_{int(dataset_id)}")
    return f"dataset_{int(dataset_id)}"


def _log_per_dataset_classification_metrics(
    metrics: dict[str, float],
    writer: SummaryWriter,
    epoch: int,
    preds: torch.Tensor,
    labels: torch.Tensor,
    dataset_ids: torch.Tensor,
    dataset_name_map: dict[int, str] | None,
) -> None:
    for ds_id in torch.unique(dataset_ids):
        mask = dataset_ids == ds_id
        name = _dataset_display_name(int(ds_id), dataset_name_map)
        metrics[f"acc/{name}"] = accuracy(preds[mask], labels[mask])
        metrics[f"f1/{name}"] = macro_f1(preds[mask], labels[mask])
        writer.add_scalar(f"val/acc_{name}", metrics[f"acc/{name}"], epoch)
        writer.add_scalar(f"val/f1_{name}", metrics[f"f1/{name}"], epoch)


def _log_per_dataset_nmi_metrics(
    metrics: dict[str, float],
    writer: SummaryWriter,
    epoch: int,
    preds: torch.Tensor,
    labels: torch.Tensor,
    dataset_ids: torch.Tensor,
    dataset_name_map: dict[int, str] | None,
) -> None:
    for ds_id in torch.unique(dataset_ids):
        mask = dataset_ids == ds_id
        if mask.sum() < 2:
            continue
        name = _dataset_display_name(int(ds_id), dataset_name_map)
        metrics[f"nmi/{name}"] = nmi_score(preds[mask], labels[mask])
        writer.add_scalar(f"val/nmi_{name}", metrics[f"nmi/{name}"], epoch)


def build_val_dataloader(
    val_ds: RFDataPoolDataset,
    *,
    batch_size: int,
    collate_fn: Any,
    num_workers: int,
    pin_memory: bool,
    stage: str,
    task: str,
    subset_fraction: float,
    subset_seed: int,
    epoch: int,
    resample_each_epoch: bool,
    max_per_dataset: int | None = None,
    full_datasets: list[str] | set[str] | None = None,
    length_bucket_batching: bool = False,
    sequence_packing: bool = False,
) -> DataLoader:
    selected: list[int] | None = None
    if subset_fraction < 1.0:
        seed = subset_seed + (epoch if resample_each_epoch else 0)
        selected, _ = build_per_dataset_class_balanced_val_indices(
            val_ds,
            stage=stage,
            task=task,
            fraction=subset_fraction,
            seed=seed,
            max_per_dataset=max_per_dataset,
            full_datasets=full_datasets,
        )
    loader_kwargs = {
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    # 长度分桶 batch 使用 pool 全局下标，不能包 Subset（Subset 只接受 0..len-1 局部下标）
    if length_bucket_batching and not sequence_packing:
        pool_indices = selected if selected is not None else list(range(len(val_ds)))
        batches = build_length_bucket_val_batches(val_ds, pool_indices, batch_size)
        return DataLoader(val_ds, batch_sampler=_FixedIndexBatchSampler(batches), **loader_kwargs)
    dataset: RFDataPoolDataset | Subset = val_ds
    if selected is not None:
        dataset = Subset(val_ds, selected)
    return DataLoader(dataset, shuffle=False, batch_size=batch_size, **loader_kwargs)


class _FixedIndexBatchSampler(Sampler[list[int]]):
    def __init__(self, batches: list[list[int]]) -> None:
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def validate_one_epoch(
    model: ResMambaSignalModel,
    loader: DataLoader,
    device: torch.device,
    *,
    stage: str,
    task: str,
    loss_weights: dict[str, float],
    epoch: int,
    writer: SummaryWriter,
    max_batches: int | None = None,
    use_amp: bool = False,
    amp_dtype: torch.dtype | None = None,
    emitter_offset_lookup: torch.Tensor | None = None,
    emitter_label_map: GlobalEmitterLabelMap | None = None,
    emitter_contrastive_weight: float = 0.0,
    dataset_name_map: dict[int, str] | None = None,
) -> dict[str, float]:
    pbar = tqdm(loader, desc=f"val epoch {epoch}", leave=False, dynamic_ncols=True)
    total_loss = 0.0
    n_batches = 0
    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_dataset_ids: list[torch.Tensor] = []
    ssim_total = 0.0
    ssim_count = 0
    ssim_by_dataset: dict[int, list[tuple[float, int]]] = {}
    skipped_nonfinite = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = move_batch(batch, device)
            with torch.amp.autocast(**_autocast_kwargs(device, use_amp, amp_dtype)):
                outputs = forward_batch(model, batch, stage, task)
                loss, _ = compute_loss(
                    outputs,
                    batch,
                    stage=stage,
                    task=task,
                    loss_weights=loss_weights,
                    emitter_offset_lookup=emitter_offset_lookup,
                    emitter_contrastive_weight=emitter_contrastive_weight,
                )
            loss_val = float(loss.detach())
            if not math.isfinite(loss_val):
                skipped_nonfinite += 1
                pbar.set_postfix(loss="nan", skip=skipped_nonfinite)
                continue
            total_loss += loss_val
            n_batches += 1
            pbar.set_postfix(loss=f"{loss_val:.4f}")

            if stage == "stage2" and task == "modulation":
                labels = resolve_modulation_labels(
                    batch["mod_label_id"],
                    batch.get("source_label_id"),
                )
                valid = labels >= 0
                if valid.any():
                    preds = outputs["modulation_logits"].argmax(dim=-1)
                    all_preds.append(preds[valid].cpu())
                    all_labels.append(labels[valid].cpu())
                    all_dataset_ids.append(batch["dataset_id"][valid].cpu())
            elif stage == "stage2" and task == "emitter":
                raw_labels = batch["emitter_id"]
                if emitter_offset_lookup is not None:
                    labels = global_emitter_labels(batch["dataset_id"], raw_labels, emitter_offset_lookup)
                else:
                    labels = raw_labels
                valid = labels >= 0
                if valid.any():
                    preds = outputs["emitter_logits"].argmax(dim=-1)
                    all_preds.append(preds[valid].cpu())
                    all_labels.append(labels[valid].cpu())
                    all_dataset_ids.append(batch["dataset_id"][valid].cpu())
            elif stage == "stage2" and task == "clustering":
                labels = batch.get("global_label_id")
                if labels is not None:
                    valid = labels >= 0
                    if valid.any():
                        preds = outputs["cluster_logits"].argmax(dim=-1)
                        all_preds.append(preds[valid].cpu())
                        all_labels.append(labels[valid].cpu())
                        all_dataset_ids.append(batch["dataset_id"][valid].cpu())
            elif stage == "stage2" and task == "prediction":
                batch_score, batch_count = ssim_iq_accumulate(
                    outputs["mae_pred"], outputs["patch_targets"], outputs["mae_mask"]
                )
                if batch_count > 0:
                    ssim_total += batch_score
                    ssim_count += batch_count
                    if "dataset_id" in batch:
                        for ds_id in torch.unique(batch["dataset_id"]):
                            ds_mask = batch["dataset_id"] == ds_id
                            if not ds_mask.any():
                                continue
                            score, count = ssim_iq_accumulate(
                                outputs["mae_pred"][ds_mask],
                                outputs["patch_targets"][ds_mask],
                                outputs["mae_mask"][ds_mask],
                            )
                            if count > 0:
                                ssim_by_dataset.setdefault(int(ds_id), []).append((score, count))

    metrics: dict[str, float] = {"loss": total_loss / max(n_batches, 1)}
    if skipped_nonfinite:
        print(f"[warn] val epoch {epoch}: skipped {skipped_nonfinite} non-finite loss batches")
    writer.add_scalar("val/loss", metrics["loss"], epoch)

    if stage == "stage2" and task in ("modulation", "emitter") and all_preds:
        preds_cat = torch.cat(all_preds)
        labels_cat = torch.cat(all_labels)
        metrics["acc"] = accuracy(preds_cat, labels_cat)
        metrics["f1"] = macro_f1(preds_cat, labels_cat)
        writer.add_scalar("val/acc", metrics["acc"], epoch)
        writer.add_scalar("val/f1", metrics["f1"], epoch)
        if all_dataset_ids:
            dataset_ids_cat = torch.cat(all_dataset_ids)
            if task == "emitter":
                for ds_id in torch.unique(dataset_ids_cat):
                    ds_mask = dataset_ids_cat == ds_id
                    name = (
                        emitter_label_map.dataset_name(int(ds_id))
                        if emitter_label_map
                        else _dataset_display_name(int(ds_id), dataset_name_map)
                    )
                    metrics[f"acc/{name}"] = accuracy(preds_cat[ds_mask], labels_cat[ds_mask])
                    metrics[f"f1/{name}"] = macro_f1(preds_cat[ds_mask], labels_cat[ds_mask])
                    writer.add_scalar(f"val/acc_{name}", metrics[f"acc/{name}"], epoch)
                    writer.add_scalar(f"val/f1_{name}", metrics[f"f1/{name}"], epoch)
            else:
                _log_per_dataset_classification_metrics(
                    metrics, writer, epoch, preds_cat, labels_cat, dataset_ids_cat, dataset_name_map
                )
    elif stage == "stage2" and task == "clustering" and all_preds:
        preds_cat = torch.cat(all_preds)
        labels_cat = torch.cat(all_labels)
        metrics["nmi"] = nmi_score(preds_cat, labels_cat)
        writer.add_scalar("val/nmi", metrics["nmi"], epoch)
        if all_dataset_ids:
            _log_per_dataset_nmi_metrics(
                metrics, writer, epoch, preds_cat, labels_cat, torch.cat(all_dataset_ids), dataset_name_map
            )
    elif stage == "stage2" and task == "prediction" and ssim_count > 0:
        metrics["ssim"] = float(ssim_total / ssim_count)
        writer.add_scalar("val/ssim", metrics["ssim"], epoch)
        for ds_id, chunks in ssim_by_dataset.items():
            total = sum(score for score, _ in chunks)
            count = sum(count for _, count in chunks)
            if count <= 0:
                continue
            name = _dataset_display_name(ds_id, dataset_name_map)
            metrics[f"ssim/{name}"] = float(total / count)
            writer.add_scalar(f"val/ssim_{name}", metrics[f"ssim/{name}"], epoch)

    return metrics


def resolve_dataset_balanced_sampling(args: argparse.Namespace, train_cfg: dict[str, Any]) -> bool:
    if args.dataset_balanced_sampling is not None:
        return bool(args.dataset_balanced_sampling)
    return bool(train_cfg.get("dataset_balanced_sampling", False))


def resolve_balanced_sampling_strategy_arg(args: argparse.Namespace, train_cfg: dict[str, Any]) -> BalancedSamplingStrategy:
    strategy = args.balanced_sampling_strategy
    if strategy is None:
        strategy = train_cfg.get("balanced_sampling_strategy")
    enabled = resolve_dataset_balanced_sampling(args, train_cfg)
    return resolve_balanced_sampling_strategy(enabled=enabled, strategy=strategy)


def resolve_gradient_accumulation_steps(args: argparse.Namespace, train_cfg: dict[str, Any]) -> int:
    if args.gradient_accumulation_steps is not None:
        steps = int(args.gradient_accumulation_steps)
    else:
        steps = int(train_cfg.get("gradient_accumulation_steps", 4))
    if steps < 1:
        raise ValueError(f"gradient_accumulation_steps 必须 >= 1，当前为 {steps}")
    return steps


def resolve_warmup_steps(args: argparse.Namespace, train_cfg: dict[str, Any]) -> int:
    if args.warmup_steps is not None:
        return max(0, int(args.warmup_steps))
    return max(0, int(train_cfg.get("warmup_steps", 500)))


def resolve_lr_scale_with_grad_accum(args: argparse.Namespace, train_cfg: dict[str, Any]) -> bool:
    if args.lr_scale_with_grad_accum is not None:
        return bool(args.lr_scale_with_grad_accum)
    return bool(train_cfg.get("lr_scale_with_grad_accum", True))


def resolve_prefetch_factor(args: argparse.Namespace, train_cfg: dict[str, Any], num_workers: int) -> int:
    if num_workers <= 0:
        return 2
    if args.prefetch_factor is not None:
        return max(1, int(args.prefetch_factor))
    return max(1, int(train_cfg.get("prefetch_factor", 4)))


def resolve_max_grad_norm(args: argparse.Namespace, train_cfg: dict[str, Any]) -> float | None:
    if args.max_grad_norm is not None:
        value = float(args.max_grad_norm)
        return None if value <= 0 else value
    if "max_grad_norm" in train_cfg:
        value = float(train_cfg["max_grad_norm"])
        return None if value <= 0 else value
    return 1.0


def resolve_lr_min_ratio(args: argparse.Namespace, train_cfg: dict[str, Any]) -> float:
    if args.lr_min_ratio is not None:
        return float(args.lr_min_ratio)
    return float(train_cfg.get("lr_min_ratio", 0.1))


def resolve_lr_schedule(args: argparse.Namespace, train_cfg: dict[str, Any]) -> str:
    if args.lr_schedule is not None:
        return str(args.lr_schedule)
    return str(train_cfg.get("lr_schedule", "warmup_cosine"))


def use_cosine_lr_decay(lr_schedule: str) -> bool:
    return lr_schedule in {"warmup_cosine", "cosine", "warmup+cosine"}


def count_optimizer_steps(num_micro_batches: int, gradient_accumulation_steps: int, epochs: int) -> int:
    steps_per_epoch = (num_micro_batches + gradient_accumulation_steps - 1) // gradient_accumulation_steps
    return max(1, steps_per_epoch * epochs)


def peek_checkpoint_meta(path: str | Path) -> dict[str, Any]:
    ckpt_path = Path(path).expanduser().resolve()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"无法读取 checkpoint 元数据：{ckpt_path}")
    sched = ckpt.get("lr_scheduler") or {}
    saved_args = ckpt.get("args") or {}
    return {
        "epoch": int(ckpt.get("epoch", 0)),
        "global_step": int(ckpt.get("global_step", 0)),
        "completed_steps": int(sched.get("completed_steps", 0)),
        "checkpoint_peak_lr": float(sched.get("peak_lr", saved_args.get("peak_lr", 0.0) or 0.0)),
    }


def resolve_reset_lr_schedule_on_resume(args: argparse.Namespace, train_cfg: dict[str, Any]) -> bool | None:
    if getattr(args, "reset_lr_schedule", None) is not None:
        return bool(args.reset_lr_schedule)
    if "reset_lr_schedule_on_resume" in train_cfg:
        return bool(train_cfg["reset_lr_schedule_on_resume"])
    return None


def should_reset_lr_schedule_on_resume(
    *,
    reset_lr_schedule: bool | None,
    checkpoint_completed_steps: int,
    total_optimizer_steps: int,
    checkpoint_peak_lr: float,
    peak_lr: float,
) -> bool:
    if reset_lr_schedule is True:
        return True
    if reset_lr_schedule is False:
        return False
    if checkpoint_completed_steps >= total_optimizer_steps:
        return True
    if checkpoint_peak_lr > 0 and abs(checkpoint_peak_lr - peak_lr) / max(peak_lr, 1e-12) > 0.05:
        return True
    return False


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    peak_lr: float,
    warmup_steps: int,
    total_optimizer_steps: int,
    lr_min_ratio: float,
    use_cosine_decay: bool,
) -> LinearWarmupLR | WarmupCosineLR:
    if use_cosine_decay and lr_min_ratio < 1.0:
        return WarmupCosineLR(
            optimizer,
            peak_lr=peak_lr,
            warmup_steps=warmup_steps,
            total_steps=total_optimizer_steps,
            min_lr_ratio=lr_min_ratio,
        )
    return LinearWarmupLR(optimizer, peak_lr=peak_lr, warmup_steps=warmup_steps)


def log_dataset_sampling(
    writer: SummaryWriter,
    pool: RFDataPoolDataset,
    *,
    strategy: BalancedSamplingStrategy,
) -> None:
    writer.add_text("data/sampling", format_sampling_plan(pool, strategy), 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="ResMamba RF signal training pipeline (pretrain / stage2).")
    parser.add_argument("--model-config", default=str(ROOT / "configs" / "model_tiny.yaml"))
    parser.add_argument("--config", default=None, help="训练配置 YAML（pretrain.yaml / stage2_heads.yaml）")
    parser.add_argument("--rfdata-root", default=str(ROOT / "dataset"))
    parser.add_argument("--pool", default="pretrain_train")
    parser.add_argument("--val-pool", default=None)
    parser.add_argument("--stage", choices=["pretrain", "stage2"], default="pretrain")
    parser.add_argument("--task", choices=["modulation", "emitter", "prediction", "clustering"], default="modulation")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数（默认读取训练 YAML，否则为 1）")
    parser.add_argument("--max-steps", type=int, default=None, help="调试用步数上限，设置后覆盖 epochs")
    parser.add_argument("--val-max-batches", type=int, default=None, help="验证批次数上限（调试用）")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="混合精度训练（CUDA 上默认开启）")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="验证 loss 无改善的容忍 epoch 数，0 表示禁用（默认读取训练 YAML，否则为 10）",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=None,
        help="视为改善的最小 val loss 下降量（默认读取训练 YAML，否则为 0）",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="micro-batch 大小（默认读取训练 YAML，否则为 2）")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker 数（默认读取训练 YAML，否则为 0）")
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "checkpoints"))
    parser.add_argument("--log-dir", default=str(ROOT / "runs" / "tensorboard"))
    parser.add_argument("--pretrained-checkpoint", default=None)
    parser.add_argument(
        "--resume",
        default=None,
        help="从训练 checkpoint 续训（best.pt / last.pt），恢复 optimizer/LR/步数",
    )
    parser.add_argument(
        "--dataset-balanced-sampling",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="开启跨子 H5 平衡采样（默认读取训练 YAML，pretrain 通常为 true）",
    )
    parser.add_argument(
        "--balanced-sampling-strategy",
        choices=[
            "uniform",
            "dataset",
            "length_bucket",
            "length_bucket_proportional",
            "length_bucket_class",
            "length_bucket_pk",
        ],
        default=None,
        help="采样策略：uniform | dataset | length_bucket | length_bucket_proportional | length_bucket_class | length_bucket_pk",
    )
    parser.add_argument(
        "--pk-num-classes",
        type=int,
        default=None,
        help="PK 采样每 batch 类别数 P（须满足 batch_size = P × K；默认读 YAML）",
    )
    parser.add_argument(
        "--pk-samples-per-class",
        type=int,
        default=None,
        help="PK 采样每类样本数 K（须满足 batch_size = P × K；默认读 YAML）",
    )
    parser.add_argument(
        "--emitter-contrastive-weight",
        type=float,
        default=None,
        help="Stage2 emitter 监督对比 loss 权重（默认读 YAML，通常 0.25）",
    )
    parser.add_argument(
        "--reset-lr-schedule",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="续训时重置 LR 调度（重新 warmup+cosine；默认自动：步数溢出或 peak_lr 变化时重置）",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="梯度累计步数（默认读取训练 YAML，通常为 4；设为 1 等价于关闭）",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="线性 warmup 的优化器步数（默认读取训练 YAML）",
    )
    parser.add_argument(
        "--lr-scale-with-grad-accum",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否按梯度累计步数线性缩放 peak lr（默认读取训练 YAML，通常为 true）",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=None,
        help="DataLoader prefetch_factor（num_workers>0 时生效，默认读 YAML 为 4）",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=None,
        help="梯度裁剪 max_norm（默认读 YAML 为 1.0；0 表示禁用）",
    )
    parser.add_argument(
        "--lr-min-ratio",
        type=float,
        default=None,
        help="cosine 衰减最低 lr = peak_lr × ratio（默认读 YAML 为 0.1）",
    )
    parser.add_argument(
        "--lr-schedule",
        choices=["warmup_cosine", "warmup_constant"],
        default=None,
        help="学习率调度：warmup_cosine（默认）或 warmup_constant（仅线性 warmup 后保持 peak_lr）",
    )
    args = parser.parse_args()

    if args.config is None:
        args.config = str(ROOT / "configs" / ("pretrain.yaml" if args.stage == "pretrain" else "stage2_heads.yaml"))
    train_cfg = load_train_config(args.config, stage=args.stage, task=args.task)
    args.rfdata_root = str(resolve_rfdata_root(args.rfdata_root, train_cfg))
    validate_rfdata_root(Path(args.rfdata_root))
    dataset_name_map = load_dataset_id_names(args.rfdata_root)
    iq_normalize = str(train_cfg.get("iq_normalize", "none"))
    loss_weights = {k: float(v) for k, v in (train_cfg.get("loss_weights") or {}).items()}
    base_lr = float(args.lr if args.lr is not None else train_cfg.get("learning_rate", 1e-4))
    batch_size = int(args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 2))
    epochs = int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 1))
    num_workers = int(args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 0))
    early_stopping_patience = int(
        args.early_stopping_patience
        if args.early_stopping_patience is not None
        else train_cfg.get("early_stopping_patience", 10)
    )
    early_stopping_min_delta = float(
        args.early_stopping_min_delta
        if args.early_stopping_min_delta is not None
        else train_cfg.get("early_stopping_min_delta", 0.0)
    )
    args.batch_size = batch_size
    args.epochs = epochs
    args.num_workers = num_workers
    args.early_stopping_patience = early_stopping_patience
    args.early_stopping_min_delta = early_stopping_min_delta
    if args.pool in DEPRECATED_TEST_POOLS:
        raise ValueError(
            f"--pool {args.pool!r} 已废弃；下游/微调训练请用 downstream_*_train 或 clustering_train（读取 *_test.h5）"
        )
    dataset_balanced_sampling = resolve_dataset_balanced_sampling(args, train_cfg)
    balanced_sampling_strategy = resolve_balanced_sampling_strategy_arg(args, train_cfg)
    length_bucket_weight_mode = resolve_length_bucket_weight_mode(balanced_sampling_strategy, train_cfg)
    pk_num_classes: int | None = None
    pk_samples_per_class: int | None = None
    if balanced_sampling_strategy == "length_bucket_pk":
        if not (args.stage == "stage2" and args.task == "emitter"):
            raise ValueError("length_bucket_pk 采样仅适用于 stage2 emitter 任务")
        pk_num_classes, pk_samples_per_class = resolve_pk_sampling_params(
            train_cfg,
            batch_size,
            pk_num_classes=args.pk_num_classes,
            pk_samples_per_class=args.pk_samples_per_class,
        )
        args.pk_num_classes = pk_num_classes
        args.pk_samples_per_class = pk_samples_per_class
    infer_emitter_classes = args.stage == "stage2" and args.task == "emitter"
    selection_metric_name = resolve_selection_metric_name(train_cfg, stage=args.stage, task=args.task)
    selection_maximize = selection_higher_is_better(selection_metric_name)
    val_subset_fraction = float(train_cfg.get("val_subset_fraction", 0.2))
    val_subset_seed = int(train_cfg.get("val_subset_seed", 42))
    val_subset_resample_each_epoch = bool(train_cfg.get("val_subset_resample_each_epoch", True))
    raw_val_max_per_dataset = train_cfg.get("val_subset_max_per_dataset")
    val_subset_max_per_dataset = int(raw_val_max_per_dataset) if raw_val_max_per_dataset is not None else None
    val_subset_full_datasets = [str(name) for name in train_cfg.get("val_subset_full_datasets", [])]
    val_length_bucket_batching = bool(
        train_cfg.get(
            "val_length_bucket_batching",
            args.stage == "stage2" and args.task == "prediction",
        )
    )
    gradient_accumulation_steps = resolve_gradient_accumulation_steps(args, train_cfg)
    warmup_steps = resolve_warmup_steps(args, train_cfg)
    lr_scale_with_grad_accum = resolve_lr_scale_with_grad_accum(args, train_cfg)
    prefetch_factor = resolve_prefetch_factor(args, train_cfg, num_workers)
    max_grad_norm = resolve_max_grad_norm(args, train_cfg)
    lr_min_ratio = resolve_lr_min_ratio(args, train_cfg)
    lr_schedule = resolve_lr_schedule(args, train_cfg)
    sequence_packing = bool(train_cfg.get("sequence_packing", False))
    peak_lr = scale_lr_for_grad_accum(
        base_lr,
        gradient_accumulation_steps,
        enabled=lr_scale_with_grad_accum,
    )
    args.gradient_accumulation_steps = gradient_accumulation_steps
    args.warmup_steps = warmup_steps
    args.lr_scale_with_grad_accum = lr_scale_with_grad_accum
    args.base_lr = base_lr
    args.peak_lr = peak_lr
    args.prefetch_factor = prefetch_factor
    args.max_grad_norm = max_grad_norm
    args.lr_min_ratio = lr_min_ratio
    args.lr_schedule = lr_schedule
    args.sequence_packing = sequence_packing

    val_pool = infer_val_pool(args.pool, args.val_pool)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = resolve_amp_dtype(device, train_cfg) if use_amp else None
    scaler = torch.amp.GradScaler("cuda") if use_amp and amp_dtype == torch.float16 else None
    early_stopping = EarlyStopping(
        patience=0 if args.max_steps is not None else early_stopping_patience,
        min_delta=early_stopping_min_delta,
        mode="max" if selection_maximize else "min",
    )

    resume_ckpt: Path | None = None
    if args.resume:
        if args.pretrained_checkpoint:
            print("[warn] 同时指定 --resume 与 --pretrained-checkpoint，仅使用 --resume")
        resume_ckpt = Path(args.resume).expanduser()
        if not resume_ckpt.is_absolute():
            resume_ckpt = (ROOT / resume_ckpt).resolve()
        if not resume_ckpt.is_file():
            raise FileNotFoundError(f"--resume checkpoint 不存在: {resume_ckpt}")
        run_name = resume_ckpt.parent.name
        output_dir = resume_ckpt.parent
    else:
        run_name = f"{args.stage}"
        if args.stage == "stage2":
            run_name += f"_{args.task}"
        run_name += f"_{time.strftime('%Y%m%d_%H%M%S')}"
        output_dir = Path(args.output_dir) / run_name

    log_dir = Path(args.log_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    link_autodl_tensorboard(log_dir)

    emitter_label_map: GlobalEmitterLabelMap | None = None
    emitter_offset_lookup: torch.Tensor | None = None
    emitter_contrastive_weight = float(
        args.emitter_contrastive_weight
        if args.emitter_contrastive_weight is not None
        else train_cfg.get("emitter_contrastive_weight", 0.0)
    )
    model_overrides: dict[str, Any] = {}
    freeze_backbone = bool(train_cfg.get("freeze_backbone", True))
    if args.stage == "stage2" and args.task == "modulation":
        default_unfreeze_mod, default_unfreeze_emit = True, False
    elif args.stage == "stage2" and args.task == "emitter":
        default_unfreeze_mod, default_unfreeze_emit = False, True
    else:
        default_unfreeze_mod, default_unfreeze_emit = False, False
    unfreeze_modulation_backbone = bool(train_cfg.get("unfreeze_modulation_backbone", default_unfreeze_mod))
    unfreeze_emitter_backbone = bool(train_cfg.get("unfreeze_emitter_backbone", default_unfreeze_emit))
    peft_mode = resolve_peft_mode(train_cfg, freeze_backbone=freeze_backbone)
    lora_rank = int(train_cfg.get("lora_rank", 8))
    lora_alpha = float(train_cfg.get("lora_alpha", 16.0))
    lora_dropout = float(train_cfg.get("lora_dropout", 0.05))
    if args.stage == "stage2" and args.task == "emitter":
        emitter_label_map = build_global_emitter_label_map(args.rfdata_root, train_cfg=train_cfg)
        model_overrides["num_emitters"] = emitter_label_map.num_emitters
        print(
            f"emitter global classes={emitter_label_map.num_emitters} "
            f"datasets={list(emitter_label_map.dataset_names.values())} "
            f"offsets={emitter_label_map.offsets}"
        )

    model = build_model(
        args.model_config,
        sequence_packing=sequence_packing,
        model_overrides=model_overrides or None,
    ).to(device)
    if args.stage == "pretrain":
        configure_pretrain(model)
    else:
        if args.pretrained_checkpoint and not args.resume:
            load_checkpoint(model, args.pretrained_checkpoint, device)
        lora_injected = configure_peft_stage2(
            model,
            args.task,
            peft_mode=peft_mode,
            freeze_backbone=freeze_backbone,
            unfreeze_modulation_backbone=unfreeze_modulation_backbone,
            unfreeze_emitter_backbone=unfreeze_emitter_backbone,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        if lora_injected:
            preview = ", ".join(lora_injected[:8])
            suffix = " ..." if len(lora_injected) > 8 else ""
            print(f"peft_mode=lora_task_path injected_lora_layers={len(lora_injected)} [{preview}{suffix}]")

    if emitter_label_map is not None:
        emitter_offset_lookup = emitter_label_map.offset_lookup().to(device)

    total_params, trainable_params = count_params(model)
    effective_batch_size = batch_size * gradient_accumulation_steps
    print(
        f"device={device} amp={use_amp}"
        + (f" amp_dtype={amp_dtype}" if amp_dtype is not None else "")
        + f" early_stopping={early_stopping.enabled}"
        f" dataset_balanced_sampling={dataset_balanced_sampling}"
        f" balanced_sampling_strategy={balanced_sampling_strategy}"
        f" length_bucket_weight={length_bucket_weight_mode}"
        f" selection_metric={selection_metric_name}"
        f" gradient_accumulation_steps={gradient_accumulation_steps}"
        f" effective_batch_size={effective_batch_size}"
        f" base_lr={base_lr:.2e} peak_lr={peak_lr:.2e}"
        f" warmup_steps={warmup_steps} lr_schedule={lr_schedule} lr_min_ratio={lr_min_ratio}"
        f" lr_scale_with_grad_accum={lr_scale_with_grad_accum}"
        f" prefetch_factor={prefetch_factor} max_grad_norm={max_grad_norm} iq_normalize={iq_normalize}"
        f" sequence_packing={sequence_packing}"
        f" peft_mode={peft_mode}"
        f" total_params={total_params:,} trainable_params={trainable_params:,}"
    )
    if balanced_sampling_strategy == "length_bucket_pk":
        print(
            f"pk_sampling: P={pk_num_classes} K={pk_samples_per_class} batch_size={batch_size}"
            f" emitter_contrastive_weight={emitter_contrastive_weight}"
        )
    elif args.stage == "stage2" and args.task == "emitter" and emitter_contrastive_weight > 0:
        print(f"emitter_contrastive_weight={emitter_contrastive_weight}")
    if args.stage == "stage2":
        print(format_param_stats(model))

    train_ds = build_rfdata_pool(args.rfdata_root, args.pool, iq_normalize=iq_normalize)
    val_ds = build_rfdata_pool(args.rfdata_root, val_pool, iq_normalize=iq_normalize)
    print(f"train_pool={args.pool} samples={len(train_ds)}")
    print(f"val_pool={val_pool} samples={len(val_ds)}")
    if balanced_sampling_strategy != "none":
        print(
            format_sampling_plan(
                train_ds,
                balanced_sampling_strategy,
                bucket_weight_mode=length_bucket_weight_mode,
                infer_emitter_classes=infer_emitter_classes,
            )
        )

    collate_fn = variable_length_collate if sequence_packing else pad_iq_collate
    loader_kwargs: dict[str, Any] = {
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["worker_init_fn"] = rfdata_dataloader_worker_init
        loader_kwargs["persistent_workers"] = True
    if balanced_sampling_strategy == "none":
        train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size, **loader_kwargs)
    elif len(train_ds.datasets) < 2:
        print("[warn] 平衡采样已开启但 train pool 仅 1 个子数据集，回退为 shuffle")
        train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size, **loader_kwargs)
    elif balanced_sampling_strategy == "length_bucket_pk":
        batch_sampler = LengthBucketPKBatchSampler(
            train_ds,
            batch_size,
            pk_num_classes=int(pk_num_classes),
            pk_samples_per_class=int(pk_samples_per_class),
            bucket_weight_mode=length_bucket_weight_mode,
        )
        train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, **loader_kwargs)
    elif balanced_sampling_strategy.startswith("length_bucket"):
        batch_sampler = LengthBucketBalancedBatchSampler(
            train_ds,
            batch_size,
            bucket_weight_mode=length_bucket_weight_mode,
            infer_emitter_classes=infer_emitter_classes,
        )
        train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, **loader_kwargs)
    else:
        sampler = (
            build_uniform_sampler(train_ds)
            if balanced_sampling_strategy == "uniform"
            else build_dataset_balanced_sampler(train_ds)
        )
        train_loader = DataLoader(
            train_ds,
            shuffle=False,
            sampler=sampler,
            batch_size=batch_size,
            **loader_kwargs,
        )
    val_loader_kwargs = {
        "batch_size": batch_size,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    val_subset_plan: dict[str, Any] | None = None
    val_estimated_size = len(val_ds)
    if val_subset_fraction < 1.0:
        val_indices, val_subset_plan = build_per_dataset_class_balanced_val_indices(
            val_ds,
            stage=args.stage,
            task=args.task,
            fraction=val_subset_fraction,
            seed=val_subset_seed,
            max_per_dataset=val_subset_max_per_dataset,
            full_datasets=val_subset_full_datasets,
        )
        val_estimated_size = int(val_subset_plan["total"])
        if val_length_bucket_batching and not sequence_packing:
            val_subset_plan["length_bucket_batching"] = True
            val_subset_plan["val_batches"] = len(build_length_bucket_val_batches(val_ds, val_indices, batch_size))
        print(format_val_subset_plan(val_subset_plan))

    train_epochs = 1 if args.max_steps is not None else epochs
    resume_meta: dict[str, Any] | None = None
    remaining_train_epochs = train_epochs
    if resume_ckpt is not None:
        resume_meta = peek_checkpoint_meta(resume_ckpt)
        remaining_train_epochs = max(1, train_epochs - int(resume_meta["epoch"]) - 1 + 1)
    schedule_epochs = remaining_train_epochs if resume_ckpt is not None else train_epochs
    if args.max_steps is not None:
        total_optimizer_steps = max(1, (args.max_steps + gradient_accumulation_steps - 1) // gradient_accumulation_steps)
    else:
        total_optimizer_steps = count_optimizer_steps(len(train_loader), gradient_accumulation_steps, schedule_epochs)

    reset_lr_schedule_flag = resolve_reset_lr_schedule_on_resume(args, train_cfg)
    reset_lr_on_resume = False
    if resume_ckpt is not None and resume_meta is not None:
        reset_lr_on_resume = should_reset_lr_schedule_on_resume(
            reset_lr_schedule=reset_lr_schedule_flag,
            checkpoint_completed_steps=int(resume_meta["completed_steps"]),
            total_optimizer_steps=total_optimizer_steps,
            checkpoint_peak_lr=float(resume_meta["checkpoint_peak_lr"]),
            peak_lr=peak_lr,
        )

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=peak_lr)
    lr_scheduler = build_lr_scheduler(
        optimizer,
        peak_lr=peak_lr,
        warmup_steps=warmup_steps,
        total_optimizer_steps=total_optimizer_steps,
        lr_min_ratio=lr_min_ratio,
        use_cosine_decay=use_cosine_lr_decay(lr_schedule),
    )
    min_lr = peak_lr * lr_min_ratio if use_cosine_lr_decay(lr_schedule) else peak_lr
    if resume_ckpt is not None:
        print(
            f"resume_schedule: remaining_epochs={remaining_train_epochs}"
            f" total_optimizer_steps={total_optimizer_steps}"
            f" reset_lr_schedule={reset_lr_on_resume}"
        )
    print(
        f"total_optimizer_steps={total_optimizer_steps} lr_schedule={lr_schedule}"
        f" peak_lr={peak_lr:.2e} min_lr={min_lr:.2e}"
    )

    resume_state: dict[str, Any] | None = None
    if resume_ckpt is not None:
        resume_state = load_training_checkpoint(
            resume_ckpt,
            model=model,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            early_stopping=early_stopping,
            load_optimizer=not reset_lr_on_resume,
            load_lr_scheduler=not reset_lr_on_resume,
        )
        if reset_lr_on_resume and lr_scheduler is not None:
            if isinstance(lr_scheduler, WarmupCosineLR):
                lr_scheduler.reconfigure(
                    peak_lr=peak_lr,
                    warmup_steps=warmup_steps,
                    total_steps=total_optimizer_steps,
                    min_lr_ratio=lr_min_ratio,
                )
            else:
                lr_scheduler.peak_lr = peak_lr
                lr_scheduler.warmup_steps = warmup_steps
                lr_scheduler.reset_progress()
            print(
                f"[resume] LR 调度已重置：warmup_steps={warmup_steps}"
                f" start_lr={optimizer.param_groups[0]['lr']:.2e}"
            )
        saved_log_dir = resume_state.get("log_dir")
        if saved_log_dir:
            log_dir = Path(saved_log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            link_autodl_tensorboard(log_dir)
        saved_args = resume_state.get("saved_args") or {}
        if saved_args.get("model_config") and saved_args["model_config"] != args.model_config:
            print(
                f"[warn] 续训 model-config={args.model_config!r} 与 checkpoint 保存的"
                f" {saved_args['model_config']!r} 不一致"
            )

    writer = SummaryWriter(log_dir=str(log_dir))
    log_dataset_sampling(writer, train_ds, strategy=balanced_sampling_strategy)
    log_static_info(
        writer,
        args=args,
        train_pool=args.pool,
        val_pool=val_pool,
        train_size=len(train_ds),
        val_size=len(val_ds),
        total_params=total_params,
        trainable_params=trainable_params,
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "train_pool": args.pool,
                "val_pool": val_pool,
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
                "total_params": total_params,
                "trainable_params": trainable_params,
                "loss_weights": loss_weights,
                "base_lr": base_lr,
                "peak_lr": peak_lr,
                "warmup_steps": warmup_steps,
                "lr_min_ratio": lr_min_ratio,
                "lr_schedule": lr_schedule,
                "total_optimizer_steps": total_optimizer_steps,
                "lr_scale_with_grad_accum": lr_scale_with_grad_accum,
                "use_amp": use_amp,
                "amp_dtype": str(amp_dtype).replace("torch.", "") if amp_dtype is not None else None,
                "dataset_balanced_sampling": dataset_balanced_sampling,
                "balanced_sampling_strategy": balanced_sampling_strategy,
                "length_bucket_weight": length_bucket_weight_mode,
                "pk_num_classes": pk_num_classes,
                "pk_samples_per_class": pk_samples_per_class,
                "selection_metric": selection_metric_name,
                "val_subset_fraction": val_subset_fraction,
                "val_subset_seed": val_subset_seed,
                "val_subset_resample_each_epoch": val_subset_resample_each_epoch,
                "val_subset_max_per_dataset": val_subset_max_per_dataset,
                "val_subset_full_datasets": val_subset_full_datasets or None,
                "val_length_bucket_batching": val_length_bucket_batching,
                "val_subset_total": val_subset_plan["total"] if val_subset_plan else len(val_ds),
                "val_batches": val_subset_plan.get("val_batches") if val_subset_plan else None,
                "sequence_packing": sequence_packing,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "early_stopping_patience": early_stopping.patience,
                "early_stopping_min_delta": early_stopping.min_delta,
                "freeze_backbone": freeze_backbone,
                "peft_mode": peft_mode,
                "lora_rank": lora_rank if peft_mode == "lora_task_path" else None,
                "lora_alpha": lora_alpha if peft_mode == "lora_task_path" else None,
                "lora_dropout": lora_dropout if peft_mode == "lora_task_path" else None,
                "unfreeze_modulation_backbone": unfreeze_modulation_backbone,
                "unfreeze_emitter_backbone": unfreeze_emitter_backbone,
                "emitter_contrastive_weight": emitter_contrastive_weight,
                "emitter_num_classes": emitter_label_map.num_emitters if emitter_label_map else None,
                "emitter_offsets": emitter_label_map.offsets if emitter_label_map else None,
                "emitter_downstream_datasets": list(emitter_label_map.dataset_names.values()) if emitter_label_map else None,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    global_step = 0
    best_selection = float("-inf") if selection_maximize else float("inf")
    best_metrics: dict[str, float] = {}
    best_epoch = 0
    last_metrics: dict[str, float] = {}
    start_epoch = 1
    if resume_state is not None:
        global_step = int(resume_state["global_step"])
        best_selection = float(resume_state["best_val"])
        best_metrics = dict(resume_state.get("best_metrics") or {})
        best_epoch = int(resume_state.get("best_epoch", 0))
        start_epoch = int(resume_state["epoch"]) + 1
        if start_epoch > train_epochs:
            print(
                f"[resume] checkpoint 已完成 epoch={resume_state['epoch']}，"
                f"目标 epochs={train_epochs}，无需续训"
            )
            writer.close()
            return

    val_max_batches = args.val_max_batches
    if val_max_batches is None and args.max_steps is not None:
        val_max_batches = min(20, max(1, (val_estimated_size + batch_size - 1) // batch_size))

    epochs = train_epochs
    for epoch in range(start_epoch, epochs + 1):
        global_step, train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            stage=args.stage,
            task=args.task,
            loss_weights=loss_weights,
            epoch=epoch,
            writer=writer,
            global_step=global_step,
            max_steps=args.max_steps,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            scaler=scaler,
            gradient_accumulation_steps=gradient_accumulation_steps,
            lr_scheduler=lr_scheduler,
            max_grad_norm=max_grad_norm,
            empty_cache_every=100,
            emitter_offset_lookup=emitter_offset_lookup,
            emitter_contrastive_weight=emitter_contrastive_weight,
        )
        val_loader = build_val_dataloader(
            val_ds,
            stage=args.stage,
            task=args.task,
            subset_fraction=val_subset_fraction,
            subset_seed=val_subset_seed,
            epoch=epoch,
            resample_each_epoch=val_subset_resample_each_epoch,
            max_per_dataset=val_subset_max_per_dataset,
            full_datasets=val_subset_full_datasets,
            length_bucket_batching=val_length_bucket_batching,
            sequence_packing=sequence_packing,
            **val_loader_kwargs,
        )
        val_metrics = validate_one_epoch(
            model,
            val_loader,
            device,
            stage=args.stage,
            task=args.task,
            loss_weights=loss_weights,
            epoch=epoch,
            writer=writer,
            max_batches=val_max_batches,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            emitter_offset_lookup=emitter_offset_lookup,
            emitter_label_map=emitter_label_map,
            emitter_contrastive_weight=emitter_contrastive_weight,
            dataset_name_map=dataset_name_map,
        )
        val_loss_str = f"{val_metrics['loss']:.6f}" if math.isfinite(val_metrics["loss"]) else "nan"
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss_str}"
            + "".join(f" {k}={v:.4f}" for k, v in val_metrics.items() if k != "loss")
        )

        last_metrics = val_metrics
        selection_score = compute_selection_score(val_metrics, selection_metric_name)
        writer.add_scalar(f"val/{selection_metric_name}", selection_score, epoch)
        improved = (
            selection_score > best_selection
            if selection_maximize
            else selection_score < best_selection
        )
        if improved:
            best_selection = selection_score
            best_metrics = val_metrics
            best_epoch = epoch
            save_checkpoint(
                output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                args=args,
                metrics=val_metrics,
                scaler=scaler,
                global_step=global_step,
                best_val=best_selection,
                best_metrics=best_metrics,
                best_epoch=best_epoch,
                checkpoint_role="best",
                lr_scheduler=lr_scheduler,
                early_stopping=early_stopping,
                log_dir=log_dir,
            )
            print(
                f"[checkpoint] 更新 best.pt @ epoch={best_epoch}"
                f" {selection_metric_name}={best_selection:.6f}"
                f" val_loss={val_metrics['loss']:.6f}"
            )

        save_checkpoint(
            output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            args=args,
            metrics=val_metrics,
            scaler=scaler,
            global_step=global_step,
            best_val=best_selection,
            best_metrics=best_metrics,
            best_epoch=best_epoch,
            checkpoint_role="last",
            lr_scheduler=lr_scheduler,
            early_stopping=early_stopping,
            log_dir=log_dir,
        )
        write_checkpoint_manifest(
            output_dir,
            best_epoch=best_epoch,
            best_val=best_selection,
            best_metrics=best_metrics,
            last_epoch=epoch,
            last_metrics=last_metrics,
        )

        if args.max_steps is not None and global_step >= args.max_steps:
            break

        if early_stopping.step(selection_score):
            print(
                f"early stopping at epoch={epoch}: {selection_metric_name} 连续 {early_stopping.patience} 个 epoch 无改善"
                f"（best={early_stopping.best_score:.6f} @ epoch={best_epoch}）"
            )
            break

    writer.close()
    print(f"checkpoints -> {output_dir}")
    print(f"  best.pt @ epoch={best_epoch} {selection_metric_name}={best_selection:.6f}")
    if best_metrics:
        print(f"    val_loss={best_metrics.get('loss', float('nan')):.6f}")
    print(f"  last.pt @ epoch={epoch} val_loss={last_metrics.get('loss', float('nan')):.6f}")
    print(f"  manifest -> {output_dir / 'checkpoints.json'}")
    print(f"tensorboard -> {log_dir.resolve()}")
    print(f"  本地: tensorboard --logdir {Path(args.log_dir).resolve()} --port 6006 --bind_all")
    print(f"  AutoDL 面板: 监控 /root/tf-logs/resmamba_current（训练时自动 symlink）")


if __name__ == "__main__":
    main()
