from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
DEFAULT_EMITTER_DOWNSTREAM_CONFIG = CONFIG_ROOT / "datasets.yaml"
LEGACY_EMITTER_DOWNSTREAM_CONFIG = CONFIG_ROOT / "emitter_downstream.yaml"


@dataclass(frozen=True)
class GlobalEmitterLabelMap:
    """将各子数据集的局部 emitter_id 映射到连续全局标签 [0, num_emitters)。"""

    offsets: dict[int, int]
    dataset_names: dict[int, str]
    num_emitters: int
    class_counts: dict[int, int] | None = None

    def offset_lookup(self, max_dataset_id: int | None = None) -> torch.Tensor:
        size = (max(max(self.offsets) if self.offsets else 0, max_dataset_id or 0)) + 1
        lookup = torch.full((size,), -1, dtype=torch.long)
        for dataset_id, offset in self.offsets.items():
            lookup[int(dataset_id)] = int(offset)
        return lookup

    def dataset_name(self, dataset_id: int) -> str:
        return self.dataset_names.get(int(dataset_id), f"dataset_{int(dataset_id)}")


def build_emitter_dataset_class_mask(
    rfdata_root: str | Path | None,
    *,
    num_emitters: int,
    num_datasets: int,
    compact: bool = False,
    train_cfg: dict | None = None,
    config_path: str | Path | None = None,
) -> torch.Tensor | None:
    """``[num_datasets, num_emitters]``：每行是该 dataset_id 允许的全局个体类。

    ``compact=True`` 时按下游紧凑标签图（仅 emitter_downstream 数据集）填 mask；
    否则使用全库 ``emitter_namespace``。
    """
    if rfdata_root is None or int(num_emitters) <= 0 or int(num_datasets) <= 0:
        return None
    if compact:
        label_map = build_global_emitter_label_map(
            rfdata_root,
            config_path=config_path,
            train_cfg=train_cfg,
        )
        return build_compact_emitter_dataset_class_mask(
            label_map,
            num_datasets=int(num_datasets),
        )
    path = Path(rfdata_root) / "label_maps.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    datasets = payload.get("datasets") or {}
    local_to_global = (payload.get("emitter_namespace") or {}).get("dataset_local_to_global") or {}
    if not datasets or not local_to_global:
        return None
    mask = torch.zeros(int(num_datasets), int(num_emitters), dtype=torch.bool)
    for raw_id, name in datasets.items():
        dataset_id = int(raw_id)
        if dataset_id < 0 or dataset_id >= mask.shape[0]:
            continue
        mapping = local_to_global.get(str(name)) or {}
        for raw_global in mapping.values():
            class_id = int(raw_global)
            if 0 <= class_id < mask.shape[1]:
                mask[dataset_id, class_id] = True
    if not bool(mask.any()):
        return None
    return mask


def build_compact_emitter_dataset_class_mask(
    label_map: GlobalEmitterLabelMap,
    *,
    num_datasets: int,
) -> torch.Tensor | None:
    """紧凑标签空间：每个下游数据集只开放 ``offset:offset+count`` 区间。"""
    n_emitters = int(label_map.num_emitters)
    n_ds = int(num_datasets)
    if n_emitters <= 0 or n_ds <= 0 or not label_map.offsets:
        return None
    mask = torch.zeros(n_ds, n_emitters, dtype=torch.bool)
    counts = dict(label_map.class_counts or {})
    if not counts:
        # 由相邻 offset 推断各类数
        ordered = sorted(label_map.offsets.items(), key=lambda item: item[1])
        for idx, (dataset_id, offset) in enumerate(ordered):
            end = ordered[idx + 1][1] if idx + 1 < len(ordered) else n_emitters
            counts[int(dataset_id)] = int(end) - int(offset)
    for dataset_id, offset in label_map.offsets.items():
        ds = int(dataset_id)
        if ds < 0 or ds >= n_ds:
            continue
        count = int(counts.get(ds, 0))
        start = int(offset)
        end = min(start + max(count, 0), n_emitters)
        if start < end:
            mask[ds, start:end] = True
    if not bool(mask.any()):
        return None
    return mask


def load_emitter_namespace_num_emitters(rfdata_root: str | Path | None) -> int | None:
    """读取 ``label_maps.json`` 里 ``emitter_namespace.num_emitters``。"""
    if rfdata_root is None:
        return None
    path = Path(rfdata_root) / "label_maps.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    namespace = payload.get("emitter_namespace") or {}
    if namespace.get("num_emitters") is not None:
        return int(namespace["num_emitters"])
    namespaced = namespace.get("namespaced_to_id")
    if isinstance(namespaced, dict) and namespaced:
        return len(namespaced)
    return None


def load_emitter_downstream_datasets(
    rfdata_root: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
    train_cfg: dict | None = None,
) -> list[str]:
    """通信/雷达辐射源个体识别下游白名单（当前任务范围外；保留兼容）。"""
    if train_cfg and train_cfg.get("emitter_downstream_datasets"):
        return [str(name) for name in train_cfg["emitter_downstream_datasets"]]

    if rfdata_root is not None:
        maps_path = Path(rfdata_root) / "label_maps.json"
        if maps_path.is_file():
            with maps_path.open("r", encoding="utf-8") as f:
                label_maps = json.load(f)
            if label_maps.get("emitter_downstream_datasets"):
                return [str(name) for name in label_maps["emitter_downstream_datasets"]]

    path = Path(config_path) if config_path is not None else DEFAULT_EMITTER_DOWNSTREAM_CONFIG
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        section = data.get("emitter_downstream")
        if isinstance(section, dict) and section.get("datasets"):
            return [str(name) for name in section["datasets"]]
        datasets = data.get("datasets")
        if datasets:
            return [str(name) for name in datasets]

    if LEGACY_EMITTER_DOWNSTREAM_CONFIG.is_file():
        with LEGACY_EMITTER_DOWNSTREAM_CONFIG.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        datasets = data.get("datasets")
        if datasets:
            return [str(name) for name in datasets]

    return []


def h5_dataset_name(filename: str) -> str:
    for suffix in ("_train.h5", "_val.h5", "_test.h5"):
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    raise ValueError(f"无法从 H5 文件名解析数据集名: {filename!r}")


def filter_emitter_downstream_pool(files: list[str], allowed_datasets: list[str] | set[str]) -> list[str]:
    allowed = set(allowed_datasets)
    return sorted(filename for filename in files if h5_dataset_name(filename) in allowed)


def build_global_emitter_label_map(
    rfdata_root: str | Path,
    *,
    dataset_names: list[str] | None = None,
    config_path: str | Path | None = None,
    train_cfg: dict | None = None,
) -> GlobalEmitterLabelMap:
    root = Path(rfdata_root)
    with (root / "label_maps.json").open("r", encoding="utf-8") as f:
        label_maps = json.load(f)

    if dataset_names is None:
        dataset_names = load_emitter_downstream_datasets(root, config_path=config_path, train_cfg=train_cfg)

    dataset_id_by_name = {name: int(dataset_id) for dataset_id, name in label_maps.get("datasets", {}).items()}
    emitter_tables = label_maps.get("emitters", {})
    dataset_names = [name for name in dataset_names if name in emitter_tables]

    offsets: dict[int, int] = {}
    dataset_name_by_id: dict[int, str] = {}
    class_counts: dict[int, int] = {}
    next_offset = 0
    for dataset_name in sorted(dataset_names, key=lambda name: dataset_id_by_name.get(name, 10**9)):
        dataset_id = dataset_id_by_name.get(dataset_name)
        if dataset_id is None:
            raise KeyError(f"emitter 数据集 {dataset_name!r} 不在 label_maps.datasets 中")
        n_classes = len(emitter_tables[dataset_name])
        offsets[dataset_id] = next_offset
        dataset_name_by_id[dataset_id] = dataset_name
        class_counts[dataset_id] = n_classes
        next_offset += n_classes

    return GlobalEmitterLabelMap(
        offsets=offsets,
        dataset_names=dataset_name_by_id,
        num_emitters=next_offset,
        class_counts=class_counts,
    )


def global_emitter_labels(
    dataset_id: torch.Tensor,
    emitter_id: torch.Tensor,
    offset_lookup: torch.Tensor,
) -> torch.Tensor:
    valid = (dataset_id >= 0) & (emitter_id >= 0)
    dataset_idx = dataset_id.long().clamp_min(0)
    # batch 在 GPU 时 lookup 常仍在 CPU（Lightning 不会自动搬非 buffer 张量）
    offset_lookup = offset_lookup.to(device=dataset_idx.device, dtype=torch.long)
    if dataset_idx.numel() and int(dataset_idx.max()) >= offset_lookup.numel():
        padded = torch.full((int(dataset_idx.max()) + 1,), -1, dtype=torch.long, device=dataset_idx.device)
        padded[: offset_lookup.numel()] = offset_lookup
        offset_lookup = padded
    offsets = offset_lookup[dataset_idx]
    global_id = offsets + emitter_id.long()
    has_offset = offsets >= 0
    return torch.where(valid & has_offset, global_id, torch.full_like(emitter_id.long(), -1))
