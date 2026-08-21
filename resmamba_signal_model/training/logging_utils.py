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


def silence_third_party_warnings() -> None:
    """Hide a Lightning+PyTorch CombinedLoader deprecation we cannot fix locally."""
    warnings.filterwarnings(
        "ignore",
        message=_LEAFSPEC_DEPRECATION,
        category=FutureWarning,
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


_TASK_PRINT_ORDER = ("modulation", "emitter", "clustering", "prediction", "imputation")
_STRUCTURED_METRIC_RE = re.compile(
    r"^(acc_|f1_|macro_acc_|macro_f1_|macro_nmi_|macro_ssim_|nmi|ssim|impute_mse|recon_mse|mse_)"
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
                f"mean_acc={_fmt_metric(float(info['mean_acc']))}",
                f"mean_f1={_fmt_metric(float(info['mean_f1']))}",
                f"n={int(info.get('n', 0))}",
            ]
            lines.append(f"{task}  " + "  ".join(parts))
            for dataset, row in sorted((info.get("datasets") or {}).items()):
                lines.append(
                    f"  {dataset}  acc={_fmt_metric(float(row['acc']))}  "
                    f"f1={_fmt_metric(float(row['f1']))}  n={int(row.get('n', 0))}"
                )
        elif kind == "clustering":
            parts = [
                f"nmi={_fmt_metric(float(info['nmi']))}",
                f"mean_nmi={_fmt_metric(float(info['mean_nmi']))}",
                f"n={int(info.get('n', 0))}",
            ]
            lines.append(f"{task}  " + "  ".join(parts))
            for dataset, row in sorted((info.get("datasets") or {}).items()):
                lines.append(
                    f"  {dataset}  nmi={_fmt_metric(float(row['nmi']))}  n={int(row.get('n', 0))}"
                )
        elif kind == "reconstruction":
            parts = [
                f"mse={_fmt_metric(float(info['mse']))}",
                f"mean_mse={_fmt_metric(float(info['mean_mse']))}",
                f"ssim={_fmt_metric(float(info['ssim']))}",
                f"mean_ssim={_fmt_metric(float(info['mean_ssim']))}",
                f"n={int(info.get('n', 0))}",
            ]
            lines.append(f"{task}  " + "  ".join(parts))
            for dataset, row in sorted((info.get("datasets") or {}).items()):
                lines.append(
                    f"  {dataset}  mse={_fmt_metric(float(row['mse']))}  "
                    f"ssim={_fmt_metric(float(row['ssim']))}  n={int(row.get('n', 0))}"
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
    """TensorBoard / 指标日志：跳过配置权重 ≤0 的 loss 分项。

    未出现在 ``loss_weights`` 中的诊断项（如 ``structure_time``）仍记录。
    """
    key = str(name).rsplit("/", 1)[-1]
    if not key or key.startswith("_"):
        return False
    if key == "total":
        return True
    if not loss_weights or key not in loss_weights:
        return True
    return float(loss_weights.get(key, 0.0) or 0.0) > 0.0


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
