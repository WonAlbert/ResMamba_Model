from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
DEFAULT_EMITTER_DOWNSTREAM_CONFIG = CONFIG_ROOT / "emitter_downstream.yaml"


@dataclass(frozen=True)
class GlobalEmitterLabelMap:
    """将各子数据集的局部 emitter_id 映射到连续全局标签 [0, num_emitters)。"""

    offsets: dict[int, int]
    dataset_names: dict[int, str]
    num_emitters: int

    def offset_lookup(self, max_dataset_id: int | None = None) -> torch.Tensor:
        size = (max(max(self.offsets) if self.offsets else 0, max_dataset_id or 0)) + 1
        lookup = torch.full((size,), -1, dtype=torch.long)
        for dataset_id, offset in self.offsets.items():
            lookup[int(dataset_id)] = int(offset)
        return lookup

    def dataset_name(self, dataset_id: int) -> str:
        return self.dataset_names.get(int(dataset_id), f"dataset_{int(dataset_id)}")


def load_emitter_downstream_datasets(
    rfdata_root: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
    train_cfg: dict | None = None,
) -> list[str]:
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
        datasets = data.get("datasets")
        if datasets:
            return [str(name) for name in datasets]

    raise FileNotFoundError(
        f"未找到 emitter 下游数据集配置，请提供 emitter_downstream_datasets 或 {DEFAULT_EMITTER_DOWNSTREAM_CONFIG}"
    )


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

    offsets: dict[int, int] = {}
    dataset_name_by_id: dict[int, str] = {}
    next_offset = 0
    for dataset_name in sorted(dataset_names, key=lambda name: dataset_id_by_name.get(name, 10**9)):
        if dataset_name not in emitter_tables:
            raise KeyError(f"emitter 数据集 {dataset_name!r} 不在 label_maps.emitters 中")
        dataset_id = dataset_id_by_name.get(dataset_name)
        if dataset_id is None:
            raise KeyError(f"emitter 数据集 {dataset_name!r} 不在 label_maps.datasets 中")
        offsets[dataset_id] = next_offset
        dataset_name_by_id[dataset_id] = dataset_name
        next_offset += len(emitter_tables[dataset_name])

    return GlobalEmitterLabelMap(offsets=offsets, dataset_names=dataset_name_by_id, num_emitters=next_offset)


def global_emitter_labels(
    dataset_id: torch.Tensor,
    emitter_id: torch.Tensor,
    offset_lookup: torch.Tensor,
) -> torch.Tensor:
    valid = (dataset_id >= 0) & (emitter_id >= 0)
    dataset_idx = dataset_id.long().clamp_min(0)
    if dataset_idx.numel() and int(dataset_idx.max()) >= offset_lookup.numel():
        padded = torch.full((int(dataset_idx.max()) + 1,), -1, dtype=torch.long, device=offset_lookup.device)
        padded[: offset_lookup.numel()] = offset_lookup.to(padded.device)
        offset_lookup = padded
    offsets = offset_lookup[dataset_idx]
    global_id = offsets + emitter_id.long()
    has_offset = offsets >= 0
    return torch.where(valid & has_offset, global_id, torch.full_like(emitter_id.long(), -1))
