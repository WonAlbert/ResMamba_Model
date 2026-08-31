from __future__ import annotations

import logging
import math
import re
import warnings
from pathlib import Path
from typing import Any

import torch

_LEAFSPEC_DEPRECATION = (
    r"`isinstance\(treespec, LeafSpec\)` is deprecated, "
    r"use `isinstance\(treespec, TreeSpec\) and treespec\.is_leaf\(\)` instead"
)
_LIGHTNING_FEW_DATALOADER_WORKERS = (
    r"The '.*' does not have many workers which may be a bottleneck\. "
    r"Consider increasing the value of the `num_workers` argument` to `num_workers=\d+` "
    r"in the `DataLoader` to improve performance\."
)


def silence_third_party_warnings() -> None:
    """Hide third-party noise we cannot fix locally (tiny/CI num_workers=0 is intentional)."""
    warnings.filterwarnings(
        "ignore",
        message=_LEAFSPEC_DEPRECATION,
        category=FutureWarning,
    )
    try:
        from lightning.fabric.utilities.warnings import PossibleUserWarning
    except ImportError:
        PossibleUserWarning = UserWarning  # type: ignore[misc, assignment]
    warnings.filterwarnings(
        "ignore",
        message=_LIGHTNING_FEW_DATALOADER_WORKERS,
        category=PossibleUserWarning,
    )


_VAL_DATALOADER_KEY = re.compile(r"^(?P<name>.+)/dataloader_idx_(?P<idx>\d+)$")


def _metric_to_float(value: Any) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        raw = float(value.detach().cpu().item())
        return raw if math.isfinite(raw) else None
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return None
    return raw if math.isfinite(raw) else None


def _fmt_metric(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1000 or (magnitude != 0.0 and magnitude < 1e-4):
        return f"{value:.4e}"
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


_TASK_PRINT_ORDER = (
    "pretrain",
    "ld_intrapulse",
    "ld_model",
    "tx_modulation",
    "ld_clustering",
    "tx_clustering",
    "prediction",
    "modulation",
    "emitter",
    "clustering",
    "imputation",
)
_STRUCTURED_METRIC_RE = re.compile(
    r"^(acc_|f1_|miss_rate_|nmi|mse_|mae_|recon_mse|recon_mae)"
)


def _is_structured_val_metric(short_name: str) -> bool:
    return bool(_STRUCTURED_METRIC_RE.match(short_name.split("/", 1)[0]))


def _format_task_report_lines(report: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    names = [task for task in _TASK_PRINT_ORDER if task in report]
    names.extend(task for task in report if task not in names)
    for task in names:
        info = report[task]
        if not isinstance(info, dict):
            continue
        kind = str(info.get("kind") or "")
        if kind == "classification":
            parts = [
                f"acc={_fmt_metric(float(info['acc']))}",
                f"f1={_fmt_metric(float(info['f1']))}",
                f"miss_rate={_fmt_metric(float(info['miss_rate']))}",
                f"n={int(info.get('n', 0))}",
            ]
            lines.append(f"{task}  " + "  ".join(parts))
            for dataset, row in sorted((info.get("datasets") or {}).items()):
                lines.append(
                    f"  {dataset}  acc={_fmt_metric(float(row['acc']))}  "
                    f"f1={_fmt_metric(float(row['f1']))}  "
                    f"miss_rate={_fmt_metric(float(row['miss_rate']))}  n={int(row.get('n', 0))}"
                )
        elif kind == "clustering":
            parts = [
                f"acc={_fmt_metric(float(info['acc']))}",
                f"nmi={_fmt_metric(float(info['nmi']))}",
                f"n={int(info.get('n', 0))}",
            ]
            lines.append(f"{task}  " + "  ".join(parts))
            for dataset, row in sorted((info.get("datasets") or {}).items()):
                lines.append(
                    f"  {dataset}  acc={_fmt_metric(float(row['acc']))}  "
                    f"nmi={_fmt_metric(float(row['nmi']))}  n={int(row.get('n', 0))}"
                )
        elif kind == "reconstruction":
            parts = [
                f"mse={_fmt_metric(float(info['mse']))}",
                f"mae={_fmt_metric(float(info['mae']))}",
                f"n={int(info.get('n', 0))}",
            ]
            lines.append(f"{task}  " + "  ".join(parts))
            for dataset, row in sorted((info.get("datasets") or {}).items()):
                lines.append(
                    f"  {dataset}  mse={_fmt_metric(float(row['mse']))}  "
                    f"mae={_fmt_metric(float(row['mae']))}  n={int(row.get('n', 0))}"
                )
    return lines


def format_val_epoch_metrics(
    metrics: dict[str, Any],
    *,
    epoch: int,
    source_names: list[str] | None = None,
    task_report: dict[str, Any] | None = None,
) -> str:
    """把 callback_metrics 与分任务/分数据集报告整理成验证结束可读摘要。"""
    names = list(source_names or [])
    overall: dict[str, float] = {}
    per_source: dict[int, dict[str, float]] = {}
    hide_structured = bool(task_report)
    active_task_names = set(task_report or {})
    catalog_tasks = set(active_task_names)
    for task_name in active_task_names:
        if str(task_name).endswith("_z"):
            catalog_tasks.add(str(task_name)[:-2])

    def _skip_stale_task_metric(short: str) -> bool:
        if not catalog_tasks or "/" not in short:
            return False
        task_prefix = short.split("/", 1)[0]
        base_task = task_prefix[:-2] if task_prefix.endswith("_z") else task_prefix
        return base_task not in catalog_tasks

    for key, raw in metrics.items():
        name = str(key)
        if not name.startswith("val/"):
            continue
        value = _metric_to_float(raw)
        if value is None:
            continue
        match = _VAL_DATALOADER_KEY.match(name)
        if match:
            idx = int(match.group("idx"))
            short = match.group("name")
            if short.startswith("val/"):
                short = short[4:]
            if hide_structured and _is_structured_val_metric(short):
                continue
            if _skip_stale_task_metric(short):
                continue
            label = names[idx] if idx < len(names) else f"dataloader_{idx}"
            prefix = f"{label}/"
            if short.startswith(prefix):
                short = short[len(prefix):]
            elif short.startswith("default/"):
                short = short[8:]
            per_source.setdefault(idx, {})[short] = value
        else:
            short = name[4:]
            if hide_structured and _is_structured_val_metric(short):
                continue
            if _skip_stale_task_metric(short):
                continue
            overall[short] = value
    task_lines = _format_task_report_lines(task_report) if task_report else []
    if not overall and not per_source and not task_lines:
        return ""
    lines = [f"========== val epoch {epoch} =========="]
    lines.extend(task_lines)
    if overall:
        lines.append("  ".join(f"{key}={_fmt_metric(val)}" for key, val in sorted(overall.items())))
    for idx in sorted(per_source):
        label = names[idx] if idx < len(names) else f"dataloader_{idx}"
        parts = "  ".join(f"{key}={_fmt_metric(val)}" for key, val in sorted(per_source[idx].items()))
        lines.append(f"[{label}] {parts}")
    lines.append("====================================")
    return "\n".join(lines)


def should_log_loss_part(name: str, loss_weights: dict[str, Any] | None) -> bool:
    """TensorBoard / 指标日志：仅记录 ``loss_weights`` 中权重 >0 的分项。"""
    key = str(name).rsplit("/", 1)[-1]
    if not key or key.startswith("_"):
        return False
    if key == "total":
        return True
    if not loss_weights or key not in loss_weights:
        return False
    return float(loss_weights.get(key, 0.0) or 0.0) > 0.0


def link_autodl_tensorboard(log_dir: Path, *, run_name: str | None = None) -> Path | None:
    """AutoDL 默认监控 /root/tf-logs；每 run 仅保留一个以 run_name 命名的软链，避免 tag 合并。"""
    autodl_root = Path("/root/tf-logs")
    if not Path("/root").is_dir():
        return None
    autodl_root.mkdir(parents=True, exist_ok=True)
    for entry in list(autodl_root.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.is_symlink():
            entry.unlink()
        elif entry.is_file() and entry.name in ("LOGDIR", "README.txt", "CURRENT_RUN"):
            entry.unlink()

    version_dir = log_dir / "version_0"
    version_dir.mkdir(parents=True, exist_ok=True)
    name = str(run_name or log_dir.parent.name).strip() or "resmamba_run"
    link = autodl_root / name
    link.symlink_to(version_dir.resolve())
    (autodl_root / "CURRENT_RUN").write_text(name + "\n", encoding="utf-8")
    (autodl_root / "LOGDIR").write_text(
        f"TensorBoard logdir (one run only):\n  /root/tf-logs/{name}\n\n"
        f"After each new train.py run, restart TensorBoard (AutoDL 6007 or scripts/tensorboard.sh).\n"
        f"Do NOT use runs/experiments as logdir.\n",
        encoding="utf-8",
    )
    print(f"AutoDL TensorBoard: /root/tf-logs/{name} -> {version_dir.resolve()}")
    return link


def setup_run_file_logger(run_dir: Path, *, name: str = "resmamba") -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
