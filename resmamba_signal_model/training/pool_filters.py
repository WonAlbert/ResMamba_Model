from __future__ import annotations

import json
from pathlib import Path

import yaml

from resmamba_signal_model.training.emitter_labels import h5_dataset_name

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
DEFAULT_DATASETS_CONFIG = CONFIG_ROOT / "datasets.yaml"
# 兼容旧路径（已合并到 datasets.yaml）
LEGACY_EXCLUDED_CONFIG = CONFIG_ROOT / "excluded_datasets.yaml"
LEGACY_DOWNSTREAM_EXCLUDED_CONFIG = CONFIG_ROOT / "downstream_excluded_datasets.yaml"
LEGACY_DOWNSTREAM_MODULATION_EXTRA_CONFIG = CONFIG_ROOT / "downstream_modulation_extra_datasets.yaml"
LEGACY_PRETRAIN_CONFIG = CONFIG_ROOT / "pretrain_datasets.yaml"
LEGACY_DOWNSTREAM_MODULATION_CONFIG = CONFIG_ROOT / "downstream_modulation_datasets.yaml"


def _read_datasets_from_yaml(path: Path, section: str | None = None) -> list[str]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if section and isinstance(data.get(section), dict):
        datasets = data[section].get("datasets")
        if datasets:
            return [str(name) for name in datasets]
    datasets = data.get("datasets")
    if datasets:
        return [str(name) for name in datasets]
    return []


def _load_dataset_section(
    section: str,
    *,
    config_path: str | Path | None = None,
    legacy_path: Path,
) -> list[str]:
    if config_path is not None:
        path = Path(config_path)
        section_key = section if path.resolve() == DEFAULT_DATASETS_CONFIG.resolve() else None
        datasets = _read_datasets_from_yaml(path, section=section_key)
        if datasets:
            return datasets
    merged = _read_datasets_from_yaml(DEFAULT_DATASETS_CONFIG, section=section)
    if merged:
        return merged
    return _read_datasets_from_yaml(legacy_path, section=None)


def load_excluded_datasets(
    rfdata_root: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
) -> list[str]:
    if rfdata_root is not None:
        maps_path = Path(rfdata_root) / "label_maps.json"
        if maps_path.is_file():
            with maps_path.open("r", encoding="utf-8") as f:
                label_maps = json.load(f)
            if label_maps.get("excluded_datasets"):
                return [str(name) for name in label_maps["excluded_datasets"]]

    return _load_dataset_section("excluded", config_path=config_path, legacy_path=LEGACY_EXCLUDED_CONFIG)


def load_downstream_excluded_datasets(
    rfdata_root: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
) -> list[str]:
    if rfdata_root is not None:
        maps_path = Path(rfdata_root) / "label_maps.json"
        if maps_path.is_file():
            with maps_path.open("r", encoding="utf-8") as f:
                label_maps = json.load(f)
            if label_maps.get("downstream_excluded_datasets"):
                return [str(name) for name in label_maps["downstream_excluded_datasets"]]

    return _load_dataset_section(
        "downstream_excluded",
        config_path=config_path,
        legacy_path=LEGACY_DOWNSTREAM_EXCLUDED_CONFIG,
    )


def load_pretrain_datasets(
    rfdata_root: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
) -> list[str]:
    if rfdata_root is not None:
        maps_path = Path(rfdata_root) / "label_maps.json"
        if maps_path.is_file():
            with maps_path.open("r", encoding="utf-8") as f:
                label_maps = json.load(f)
            if label_maps.get("pretrain_datasets"):
                return [str(name) for name in label_maps["pretrain_datasets"]]

    return _load_dataset_section("pretrain", config_path=config_path, legacy_path=LEGACY_PRETRAIN_CONFIG)


def load_downstream_modulation_datasets(
    rfdata_root: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
) -> list[str]:
    if rfdata_root is not None:
        maps_path = Path(rfdata_root) / "label_maps.json"
        if maps_path.is_file():
            with maps_path.open("r", encoding="utf-8") as f:
                label_maps = json.load(f)
            if label_maps.get("downstream_modulation_datasets"):
                return [str(name) for name in label_maps["downstream_modulation_datasets"]]

    return _load_dataset_section(
        "downstream_modulation",
        config_path=config_path,
        legacy_path=LEGACY_DOWNSTREAM_MODULATION_CONFIG,
    )


def load_downstream_modulation_extra_datasets(
    rfdata_root: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
) -> list[str]:
    if rfdata_root is not None:
        maps_path = Path(rfdata_root) / "label_maps.json"
        if maps_path.is_file():
            with maps_path.open("r", encoding="utf-8") as f:
                label_maps = json.load(f)
            if label_maps.get("downstream_modulation_extra_datasets"):
                return [str(name) for name in label_maps["downstream_modulation_extra_datasets"]]

    return _load_dataset_section(
        "downstream_modulation_extra",
        config_path=config_path,
        legacy_path=LEGACY_DOWNSTREAM_MODULATION_EXTRA_CONFIG,
    )


def load_downstream_shared_datasets(
    rfdata_root: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
) -> list[str]:
    """预测 / 聚类：调制 + 个体识别下游数据集并集。"""
    modulation = load_downstream_modulation_datasets(rfdata_root=rfdata_root, config_path=config_path)
    emitter = _load_dataset_section(
        "emitter_downstream",
        config_path=config_path,
        legacy_path=CONFIG_ROOT / "emitter_downstream.yaml",
    )
    return sorted(set(modulation) | set(emitter))


def filter_dataset_pool(files: list[str], allowed_datasets: list[str] | set[str]) -> list[str]:
    allowed = set(allowed_datasets)
    return sorted(filename for filename in files if h5_dataset_name(filename) in allowed)


def filter_excluded_dataset_pool(files: list[str], excluded_datasets: list[str] | set[str]) -> list[str]:
    excluded = set(excluded_datasets)
    if not excluded:
        return sorted(files)
    return sorted(filename for filename in files if h5_dataset_name(filename) not in excluded)
