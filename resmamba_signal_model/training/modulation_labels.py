from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from resmamba_signal_model.training.pool_filters import (
    DEFAULT_DATASETS_CONFIG,
    load_downstream_modulation_datasets,
)

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


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
