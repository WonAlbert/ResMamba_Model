from __future__ import annotations

from typing import Any

STAGE2_SELECTION_DEFAULTS: dict[str, str] = {
    "modulation": "f1",
    "emitter": "per_dataset_macro_acc",
    "clustering": "nmi",
    "prediction": "ssim",
}


def resolve_selection_metric_name(train_cfg: dict[str, Any], *, stage: str, task: str) -> str:
    explicit = (
        train_cfg.get("selection_metric")
        or train_cfg.get("best_metric")
        or train_cfg.get("early_stopping_metric")
    )
    if explicit and str(explicit).strip().lower() not in ("", "auto"):
        return _normalize_metric_name(str(explicit))
    if stage == "stage2":
        return STAGE2_SELECTION_DEFAULTS.get(task, "val_loss")
    return "val_loss"


def _normalize_metric_name(name: str) -> str:
    key = name.strip().lower()
    aliases = {
        "loss": "val_loss",
        "macro_acc": "acc",
        "macro_f1": "f1",
    }
    return aliases.get(key, key)


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
        return -float(val_metrics.get("loss", float("inf")))

    if metric_name in val_metrics:
        return float(val_metrics[metric_name])

    raise KeyError(f"验证指标中不存在 selection metric {metric_name!r}，当前 keys={sorted(val_metrics)}")


def selection_higher_is_better(metric_name: str) -> bool:
    return _normalize_metric_name(metric_name) != "val_loss"
