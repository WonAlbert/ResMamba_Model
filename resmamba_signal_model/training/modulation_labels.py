from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from resmamba_signal_model.training.emitter_labels import (
    GlobalEmitterLabelMap,
    _dataset_id_for_name,
    global_emitter_labels,
)
from resmamba_signal_model.training.pool_filters import (
    DEFAULT_DATASETS_CONFIG,
    load_downstream_modulation_datasets,
)

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
COMM_MODULATION_CLASSES_PER_DATASET = 11


@dataclass(frozen=True)
class CompactModulationLabelMap:
    """将下游调制用到的 ``canonical_mod_label_id`` 压缩到连续 ``[0, num_classes)``。"""

    old_to_new: dict[int, int]
    num_classes: int
    dataset_names: tuple[str, ...]

    def lookup(self, max_old_id: int | None = None) -> torch.Tensor:
        size = (max(self.old_to_new) if self.old_to_new else 0)
        if max_old_id is not None:
            size = max(size, int(max_old_id))
        lookup = torch.full((size + 1,), -1, dtype=torch.long)
        for old, new in self.old_to_new.items():
            lookup[int(old)] = int(new)
        return lookup

    def remap(self, labels: torch.Tensor, lookup: torch.Tensor | None = None) -> torch.Tensor:
        table = lookup if lookup is not None else self.lookup()
        return remap_modulation_labels(labels, table)


def remap_modulation_labels(labels: torch.Tensor, lookup: torch.Tensor) -> torch.Tensor:
    """用 ``lookup[old_id] -> new_id``（未知为 -1）压缩调制标签。"""
    labels = labels.long()
    valid = labels >= 0
    idx = labels.clamp_min(0)
    table = lookup
    if idx.numel() and int(idx.max()) >= table.numel():
        padded = torch.full((int(idx.max()) + 1,), -1, dtype=torch.long, device=table.device)
        padded[: table.numel()] = table.to(padded.device)
        table = padded
    mapped = table.to(device=labels.device)[idx]
    return torch.where(valid & (mapped >= 0), mapped, torch.full_like(labels, -1))


def _candidate_h5_paths(root: Path, dataset_name: str) -> list[Path]:
    paths: list[Path] = []
    for split in ("test", "val", "train"):
        for base in (root / "h5", root):
            path = base / f"{dataset_name}_{split}.h5"
            if path.is_file():
                paths.append(path)
    return paths


def _unique_canonical_ids_from_h5(paths: list[Path]) -> set[int]:
    found: set[int] = set()
    for path in paths:
        with h5py.File(path, "r") as handle:
            if "canonical_mod_label_id" not in handle:
                continue
            arr = np.asarray(handle["canonical_mod_label_id"][:], dtype=np.int64)
            found.update(int(x) for x in np.unique(arr) if int(x) >= 0)
    return found


def _unique_canonical_ids_from_label_maps(
    label_maps: dict,
    dataset_names: list[str],
) -> set[int]:
    """无 H5 时：用各数据集调制名表 ∩ 全局 ontology 推断出现过的 canonical ID。"""
    canonical = label_maps.get("canonical_modulations") or {}
    name_to_id = {str(name): int(cid) for name, cid in canonical.items()}
    # 兼容 id->name 存法
    if canonical and all(str(k).isdigit() for k in canonical.keys()):
        name_to_id = {str(v): int(k) for k, v in canonical.items()}
    mod_tables = label_maps.get("modulations") or {}
    found: set[int] = set()
    for dataset_name in dataset_names:
        table = mod_tables.get(dataset_name) or {}
        for local_name in table.keys():
            cid = name_to_id.get(str(local_name))
            if cid is None:
                # RML 等局部名已是别名；尽力匹配大小写无关
                cid = name_to_id.get(str(local_name).upper()) or name_to_id.get(str(local_name).lower())
            if cid is not None and int(cid) >= 0:
                found.add(int(cid))
    return found


def global_comm_modulation_labels(
    dataset_id: torch.Tensor,
    canonical_mod_label_id: torch.Tensor,
    canonical_lookup: torch.Tensor,
    offset_lookup: torch.Tensor,
) -> torch.Tensor:
    """``tx_modulation`` 方案 A：canonical→局部 0..10，再按 dataset_id 加 offset 得到 33 类全局标签。"""
    local = remap_modulation_labels(canonical_mod_label_id, canonical_lookup)
    return global_emitter_labels(dataset_id, local, offset_lookup)


def build_global_comm_modulation_label_map(
    rfdata_root: str | Path,
    *,
    dataset_names: list[str] | None = None,
    config_path: str | Path | None = None,
    train_cfg: dict | None = None,
    classes_per_dataset: int = COMM_MODULATION_CLASSES_PER_DATASET,
) -> tuple[GlobalEmitterLabelMap, CompactModulationLabelMap]:
    """通信调制下游：每数据集固定 ``classes_per_dataset`` 槽位（默认 11×3=33 类）。"""
    root = Path(rfdata_root)
    if dataset_names is None:
        dataset_names = load_downstream_modulation_datasets(root, config_path=config_path)
    canonical = build_compact_modulation_label_map(
        root,
        dataset_names=list(dataset_names),
        config_path=config_path,
        train_cfg=train_cfg,
    )
    maps_path = root / "label_maps.json"
    if not maps_path.is_file():
        raise FileNotFoundError(f"缺少 label_maps.json: {maps_path}")
    with maps_path.open("r", encoding="utf-8") as handle:
        label_maps = json.load(handle)
    dataset_id_by_name = {
        str(name): int(dataset_id) for dataset_id, name in label_maps.get("datasets", {}).items()
    }

    offsets: dict[int, int] = {}
    dataset_name_by_id: dict[int, str] = {}
    class_counts: dict[int, int] = {}
    next_offset = 0
    slots = int(classes_per_dataset)
    for dataset_name in dataset_names:
        dataset_id = _dataset_id_for_name(root, str(dataset_name), dataset_id_by_name)
        if dataset_id is None:
            continue
        if dataset_id in offsets:
            continue
        offsets[dataset_id] = next_offset
        dataset_name_by_id[dataset_id] = str(dataset_name)
        class_counts[dataset_id] = slots
        next_offset += slots

    label_map = GlobalEmitterLabelMap(
        offsets=offsets,
        dataset_names=dataset_name_by_id,
        num_emitters=next_offset,
        class_counts=class_counts,
    )
    return label_map, canonical


def build_comm_modulation_dataset_class_mask(
    label_map: GlobalEmitterLabelMap,
    canonical: CompactModulationLabelMap,
    rfdata_root: str | Path,
    *,
    num_datasets: int,
) -> torch.Tensor | None:
    """``[num_datasets, num_classes]``：每库只开放该库 H5 中实际出现的 canonical 类槽位。"""
    root = Path(rfdata_root)
    n_emitters = int(label_map.num_emitters)
    n_ds = int(num_datasets)
    if n_emitters <= 0 or n_ds <= 0 or not label_map.offsets:
        return None
    lookup = canonical.lookup()
    mask = torch.zeros(n_ds, n_emitters, dtype=torch.bool)
    for dataset_id, dataset_name in label_map.dataset_names.items():
        ds = int(dataset_id)
        if ds < 0 or ds >= n_ds:
            continue
        offset = int(label_map.offsets.get(ds, -1))
        if offset < 0:
            continue
        canonical_ids = _unique_canonical_ids_from_h5(_candidate_h5_paths(root, dataset_name))
        for cid in canonical_ids:
            if int(cid) < 0 or int(cid) >= int(lookup.numel()):
                continue
            local = int(lookup[int(cid)].item())
            if local < 0:
                continue
            global_id = offset + local
            if 0 <= global_id < n_emitters:
                mask[ds, global_id] = True
    if not bool(mask.any()):
        return None
    return mask


def build_compact_modulation_label_map(
    rfdata_root: str | Path,
    *,
    dataset_names: list[str] | None = None,
    config_path: str | Path | None = None,
    train_cfg: dict | None = None,
) -> CompactModulationLabelMap:
    root = Path(rfdata_root)
    if dataset_names is None:
        if train_cfg and train_cfg.get("modulation_downstream_datasets"):
            dataset_names = [str(name) for name in train_cfg["modulation_downstream_datasets"]]
        else:
            dataset_names = load_downstream_modulation_datasets(root, config_path=config_path)

    found: set[int] = set()
    for name in dataset_names:
        found |= _unique_canonical_ids_from_h5(_candidate_h5_paths(root, name))

    maps_path = root / "label_maps.json"
    if not found and maps_path.is_file():
        with maps_path.open("r", encoding="utf-8") as handle:
            label_maps = json.load(handle)
        found = _unique_canonical_ids_from_label_maps(label_maps, list(dataset_names))

    if not found:
        raise FileNotFoundError(
            f"无法为调制紧凑标签收集 canonical_mod_label_id，"
            f"请检查 {root} 下下游调制 H5 或 label_maps.json"
        )

    old_ids = sorted(found)
    old_to_new = {old: idx for idx, old in enumerate(old_ids)}
    return CompactModulationLabelMap(
        old_to_new=old_to_new,
        num_classes=len(old_ids),
        dataset_names=tuple(str(name) for name in dataset_names),
    )


def load_modulation_downstream_datasets_from_cfg(train_cfg: dict | None = None) -> list[str]:
    if train_cfg and train_cfg.get("modulation_downstream_datasets"):
        return [str(name) for name in train_cfg["modulation_downstream_datasets"]]
    path = DEFAULT_DATASETS_CONFIG
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        section = data.get("downstream_comm_modulation") or data.get("downstream_modulation") or {}
        if isinstance(section, dict) and section.get("datasets"):
            return [str(name) for name in section["datasets"]]
    return []
