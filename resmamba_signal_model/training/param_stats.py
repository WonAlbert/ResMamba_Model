from __future__ import annotations

from collections.abc import Iterable

import torch.nn as nn


def _module_param_counts(module: nn.Module) -> tuple[int, int]:
    total = 0
    trainable = 0
    for param in module.parameters():
        numel = param.numel()
        total += numel
        if param.requires_grad:
            trainable += numel
    return total, trainable


def count_params(model: nn.Module) -> tuple[int, int]:
    return _module_param_counts(model)


def count_trainable_by_module(model: nn.Module, top_level_names: Iterable[str] | None = None) -> list[dict[str, int | float | str]]:
    names = list(top_level_names) if top_level_names is not None else [name for name, _ in model.named_children()]
    total_all, _ = count_params(model)
    rows: list[dict[str, int | float | str]] = []
    for name in names:
        child = getattr(model, name, None)
        if child is None or not isinstance(child, nn.Module):
            continue
        total, trainable = _module_param_counts(child)
        rows.append(
            {
                "module": name,
                "total": total,
                "trainable": trainable,
                "trainable_pct": (100.0 * trainable / total_all) if total_all else 0.0,
            }
        )
    rows.sort(key=lambda row: int(row["trainable"]), reverse=True)
    return rows


def format_param_stats(model: nn.Module, *, top_level_names: Iterable[str] | None = None) -> str:
    total, trainable = count_params(model)
    lines = [f"total_params={total:,} trainable_params={trainable:,} trainable_ratio={100.0 * trainable / total:.2f}%"]
    for row in count_trainable_by_module(model, top_level_names):
        lines.append(
            f"  {row['module']}: trainable={int(row['trainable']):,} "
            f"total={int(row['total']):,} ({float(row['trainable_pct']):.2f}% of model)"
        )
    return "\n".join(lines)
