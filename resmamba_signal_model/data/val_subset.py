from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from resmamba_signal_model.data.rfdata import RFDataH5Dataset, RFDataPoolDataset
from resmamba_signal_model.data.sampling import pool_segments
from resmamba_signal_model.training.emitter_labels import h5_dataset_name

LABEL_FIELD_CANDIDATES = (
    "canonical_mod_label_id",
    "global_emitter_id",
    "emitter_id",
    "mod_label_id",
    "source_label_id",
    "global_label_id",
)


def resolve_val_label_field(stage: str, task: str) -> str:
    if stage == "pretrain":
        return "auto"
    if task == "emitter":
        return "global_emitter_id"
    if task == "modulation":
        return "canonical_mod_label_id"
    if task == "clustering":
        return "global_label_id"
    if task == "prediction":
        return "auto"
    return "auto"


def _infer_label_field_for_sub(sub: RFDataH5Dataset, preferred: str | None = None) -> str | None:
    candidates: list[str] = []
    if preferred and preferred != "auto":
        candidates.append(preferred)
    for field in LABEL_FIELD_CANDIDATES:
        if field not in candidates:
            candidates.append(field)
    with h5py.File(sub.h5_path, "r") as f:
        for field in candidates:
            if field not in f:
                continue
            sample = np.asarray(f[field][: min(len(f[field]), 4096)])
            if np.any(sample >= 0):
                return field
    return None


def _labels_from_h5(sub: RFDataH5Dataset, label_field: str) -> np.ndarray:
    with h5py.File(sub.h5_path, "r") as f:
        if label_field not in f:
            raise KeyError(f"{sub.h5_path.name} 缺少标签字段 {label_field!r}")
        return np.asarray(f[label_field][:], dtype=np.int64)


def _per_class_take_count(class_size: int, fraction: float) -> int:
    if fraction >= 1.0:
        return class_size
    if class_size <= 0:
        return 0
    n_take = int(math.ceil(class_size * fraction - 1e-9))
    return min(class_size, max(1, n_take))


def _uniform_dataset_take_count(dataset_size: int, fraction: float) -> int:
    if fraction >= 1.0:
        return dataset_size
    if dataset_size <= 0:
        return 0
    return min(dataset_size, max(1, int(math.ceil(dataset_size * fraction - 1e-9))))


def _cap_indices(indices: list[int], max_count: int | None, rng: np.random.Generator) -> list[int]:
    if max_count is None or len(indices) <= max_count:
        return indices
    chosen = rng.choice(len(indices), size=max_count, replace=False)
    return sorted(int(indices[i]) for i in chosen)


def build_length_bucket_val_batches(
    pool: RFDataPoolDataset,
    indices: list[int],
    batch_size: int,
) -> list[list[int]]:
    """将验证索引按子 H5 信号长度分桶后组 batch，避免 batch 内长短混 pad。"""
    if batch_size <= 0:
        raise ValueError(f"batch_size 必须为正，当前为 {batch_size}")
    if not indices:
        return []
    segments = pool_segments(pool)
    buckets: dict[int, list[int]] = defaultdict(list)
    for idx in indices:
        for seg in segments:
            if seg.offset <= idx < seg.offset + seg.size:
                buckets[seg.signal_length].append(idx)
                break
        else:
            raise IndexError(f"验证索引 {idx} 超出 pool {pool.pool_name!r} 范围")
    batches: list[list[int]] = []
    for length in sorted(buckets):
        bucket = buckets[length]
        for start in range(0, len(bucket), batch_size):
            batches.append(bucket[start : start + batch_size])
    return batches


def _effective_val_fraction(h5_filename: str, fraction: float, full_datasets: set[str]) -> float:
    if not full_datasets:
        return fraction
    try:
        if h5_dataset_name(h5_filename) in full_datasets:
            return 1.0
    except ValueError:
        pass
    return fraction


def build_per_dataset_class_balanced_val_indices(
    pool: RFDataPoolDataset,
    *,
    stage: str,
    task: str,
    label_field: str | None = None,
    fraction: float = 0.2,
    seed: int = 0,
    max_per_dataset: int | None = None,
    full_datasets: list[str] | set[str] | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """各子 H5 内按类别均匀抽取 ``fraction`` 比例样本；无可用标签时退化为该子集均匀随机抽取。"""
    full_set = {str(name) for name in (full_datasets or [])}
    if not 0.0 < fraction <= 1.0:
        if fraction >= 1.0 and not full_set:
            return list(range(len(pool))), {"fraction": fraction, "datasets": {}, "total": len(pool)}
        if fraction >= 1.0:
            fraction = 1.0
        else:
            raise ValueError(f"val_subset_fraction 必须在 (0, 1]，当前为 {fraction}")

    preferred = label_field or resolve_val_label_field(stage, task)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    report: dict[str, Any] = {
        "fraction": fraction,
        "seed": seed,
        "stage": stage,
        "task": task,
        "max_per_dataset": max_per_dataset,
        "full_datasets": sorted(full_set),
        "datasets": {},
    }
    offset = 0

    for sub in pool.datasets:
        n_total = len(sub)
        sub_fraction = _effective_val_fraction(sub.h5_path.name, fraction, full_set)
        use_full = sub_fraction >= 1.0
        field = _infer_label_field_for_sub(sub, preferred)
        dataset_selected: list[int] = []
        per_class: dict[str, dict[str, int]] = {}
        sampling_mode = "class_balanced"

        if use_full:
            sampling_mode = "full"
            dataset_selected = list(range(n_total))
            if field is not None:
                labels = _labels_from_h5(sub, field)
                by_class: dict[int, list[int]] = defaultdict(list)
                for local_idx, label in enumerate(labels):
                    if int(label) >= 0:
                        by_class[int(label)].append(local_idx)
                for cls in sorted(by_class):
                    per_class[str(cls)] = {"total": len(by_class[cls]), "selected": len(by_class[cls])}
        elif field is None:
            sampling_mode = "uniform"
            n_take = _uniform_dataset_take_count(n_total, sub_fraction)
            if n_take > 0:
                chosen = rng.choice(n_total, size=n_take, replace=False)
                dataset_selected.extend(int(i) for i in chosen)
        else:
            labels = _labels_from_h5(sub, field)
            by_class: dict[int, list[int]] = defaultdict(list)
            for local_idx, label in enumerate(labels):
                if int(label) >= 0:
                    by_class[int(label)].append(local_idx)
            if not by_class:
                sampling_mode = "uniform"
                n_take = _uniform_dataset_take_count(n_total, sub_fraction)
                if n_take > 0:
                    chosen = rng.choice(n_total, size=n_take, replace=False)
                    dataset_selected.extend(int(i) for i in chosen)
            else:
                for cls in sorted(by_class):
                    indices = by_class[cls]
                    n_take = _per_class_take_count(len(indices), sub_fraction)
                    chosen = rng.choice(np.asarray(indices, dtype=np.int64), size=n_take, replace=False)
                    dataset_selected.extend(int(i) for i in chosen)
                    per_class[str(cls)] = {"total": len(indices), "selected": int(n_take)}

        capped = dataset_selected if use_full else _cap_indices(dataset_selected, max_per_dataset, rng)
        selected.extend(offset + i for i in capped)
        report["datasets"][sub.h5_path.name] = {
            "total": n_total,
            "selected": len(capped),
            "selected_before_cap": len(dataset_selected),
            "classes": len(per_class),
            "label_field": field,
            "sampling_mode": sampling_mode,
            "fraction": sub_fraction,
            "per_class": per_class,
        }
        offset += n_total

    report["total"] = len(selected)
    return sorted(selected), report


def format_val_subset_plan(report: dict[str, Any]) -> str:
    lines = [
        f"val_subset_fraction={report['fraction']}",
        f"val_subset_seed={report.get('seed', 0)}",
        f"val_subset_total={report['total']}",
        f"stage={report.get('stage', '?')} task={report.get('task', '?')}",
    ]
    if report.get("max_per_dataset") is not None:
        lines.append(f"val_subset_max_per_dataset={report['max_per_dataset']}")
    if report.get("full_datasets"):
        lines.append(f"val_subset_full_datasets={report['full_datasets']}")
    if report.get("length_bucket_batching"):
        lines.append(f"val_length_bucket_batching=true batches={report.get('val_batches', '?')}")
    for name, info in sorted(report.get("datasets", {}).items()):
        ratio = info["selected"] / max(info["total"], 1)
        field = info.get("label_field") or "none"
        mode = info.get("sampling_mode", "class_balanced")
        lines.append(
            f"{name}: selected={info['selected']}/{info['total']} ({ratio:.1%}), "
            f"classes={info['classes']}, field={field}, mode={mode}, fraction={info.get('fraction', report['fraction'])}"
        )
    lines.append("各子 H5 内按类别均匀抽取固定比例（每类 ceil(fraction×类样本数)，至少 1 条）；无标签时均匀随机。")
    return "\n".join(lines)
