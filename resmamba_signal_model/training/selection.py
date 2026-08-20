from __future__ import annotations

import math
from typing import Any

TASK_SELECTION_DEFAULTS: dict[str, str] = {
    "modulation": "f1",
    "emitter": "per_dataset_macro_acc",
    "clustering": "nmi",
    "prediction": "ssim",
    "imputation": "ssim",
}
STAGE2_SELECTION_DEFAULTS = TASK_SELECTION_DEFAULTS


def resolve_selection_metric_name(train_cfg: dict[str, Any], *, stage: str, task: str) -> str:
    explicit = (
        train_cfg.get("selection_metric")
        or train_cfg.get("best_metric")
        or train_cfg.get("early_stopping_metric")
    )
    if explicit and str(explicit).strip().lower() not in ("", "auto"):
        return _normalize_metric_name(str(explicit))
    if stage in {"downstream", "stage2", "stage3", "stage4", "joint", "continual"}:
        if task in ("all", "", None):
            return "multitask_geomean" if stage in {"downstream", "stage2", "continual"} else "specialist_geomean"
        return TASK_SELECTION_DEFAULTS.get(task, "val_loss")
    return "val_loss"


def _normalize_metric_name(name: str) -> str:
    key = name.strip().lower()
    aliases = {
        "loss": "val_loss",
        "macro_acc": "acc",
        "macro_f1": "f1",
    }
    return aliases.get(key, key)


def multitask_metric_geomean(
    task_metrics: dict[str, dict[str, float]],
    *,
    eps: float = 1.0e-6,
    metric_by_task: dict[str, str] | None = None,
) -> tuple[float, dict[str, Any]]:
    """任务指标几何平均：``exp(mean log(clip(m_t, eps)))``，用于 stage2/downstream 选模。"""
    return specialist_relative_geomean(
        task_metrics,
        {name: 1.0 for name in task_metrics},
        eps=eps,
        drop_warn_threshold=1.0,
        metric_by_task=metric_by_task,
    )


def resolve_checkpoint_monitor_name(train_cfg: dict[str, Any], *, stage: str, task: str | None) -> str:
    """把 selection metric 映射成 Lightning ``val/...`` monitor 键。"""
    explicit = train_cfg.get("checkpoint_monitor")
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    metric = resolve_selection_metric_name(train_cfg, stage=stage, task=task or "all")
    if metric == "val_loss":
        return "val/monitor"
    if metric in ("multitask_geomean", "specialist_geomean"):
        return f"val/{metric}"
    task_name = str(task or "")
    if task_name and task_name not in ("all",):
        if metric == "nmi" and task_name == "clustering":
            return "val/nmi"
        if metric == "ssim" and task_name == "prediction":
            return "val/ssim"
        if metric == "ssim" and task_name == "imputation":
            return "val/ssim_imputation"
        if metric == "per_dataset_macro_acc":
            return f"val/macro_acc_{task_name}"
        if metric == "per_dataset_macro_f1":
            return f"val/macro_f1_{task_name}"
        return f"val/{metric}_{task_name}"
    return f"val/{metric}"


def specialist_relative_geomean(
    task_metrics: dict[str, dict[str, float]],
    specialist_scores: dict[str, float],
    *,
    eps: float = 1.0e-6,
    drop_warn_threshold: float = 0.05,
    metric_by_task: dict[str, str] | None = None,
) -> tuple[float, dict[str, Any]]:
    """相对 specialist 的标准化几何平均：``exp(mean log(clip(m_t / s_t, eps)))``。"""
    names = [task for task in TASK_SELECTION_DEFAULTS if task in task_metrics]
    names.extend(task for task in task_metrics if task not in names)
    if not names:
        raise ValueError("task_metrics 为空，无法计算相对 specialist 几何平均")
    details: dict[str, Any] = {"tasks": {}, "warnings": []}
    logs: list[float] = []
    for task in names:
        metric_name = (metric_by_task or TASK_SELECTION_DEFAULTS).get(task)
        if not metric_name:
            metric_name = next(iter(task_metrics[task]), "val_loss")
        m_t = compute_selection_score(task_metrics[task], metric_name)
        s_t = float(specialist_scores.get(task, 1.0))
        if not math.isfinite(s_t) or s_t <= 0:
            s_t = 1.0
        ratio = float(m_t) / s_t
        clipped = max(ratio, float(eps))
        logs.append(math.log(clipped))
        dropped = clipped < (1.0 - float(drop_warn_threshold))
        details["tasks"][task] = {
            "metric": metric_name,
            "m": float(m_t) if math.isfinite(m_t) else None,
            "s": s_t,
            "ratio": ratio if math.isfinite(ratio) else None,
        }
        if dropped:
            details["warnings"].append(task)
    score = math.exp(sum(logs) / len(logs))
    details["score"] = score
    return score, details


def compute_selection_score(val_metrics: dict[str, float], metric_name: str) -> float:
    metric_name = _normalize_metric_name(metric_name)
    if metric_name == "per_dataset_macro_acc":
        acc_keys = sorted(key for key in val_metrics if key.startswith("acc/"))
        if acc_keys:
            return float(sum(val_metrics[key] for key in acc_keys) / len(acc_keys))
        return float(val_metrics.get("acc", float("-inf")))

    if metric_name == "per_dataset_macro_f1":
        f1_keys = sorted(key for key in val_metrics if key.startswith("f1/"))
        if f1_keys:
            return float(sum(val_metrics[key] for key in f1_keys) / len(f1_keys))
        return float(val_metrics.get("f1", float("-inf")))

    if metric_name in ("acc",):
        return float(val_metrics.get("acc", float("-inf")))

    if metric_name in ("f1",):
        return float(val_metrics.get("f1", float("-inf")))

    if metric_name == "val_loss":
        return float(val_metrics.get("loss", float("inf")))

    if metric_name in val_metrics:
        return float(val_metrics[metric_name])

    raise KeyError(f"验证指标中不存在 selection metric {metric_name!r}，当前 keys={sorted(val_metrics)}")


def selection_higher_is_better(metric_name: str) -> bool:
    return _normalize_metric_name(metric_name) != "val_loss"
