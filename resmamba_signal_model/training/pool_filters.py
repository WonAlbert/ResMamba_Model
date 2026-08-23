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

# 新任务 pool section → label_maps.json 键
_SECTION_LABEL_MAP_KEYS: dict[str, str] = {
    "pretrain": "pretrain_datasets",
    "downstream_radar_modulation": "downstream_radar_modulation_datasets",
    "downstream_radar_model": "downstream_radar_model_datasets",
    "downstream_comm_modulation": "downstream_comm_modulation_datasets",
    "clustering_radar": "clustering_radar_datasets",
    "clustering_comm": "clustering_comm_datasets",
    "prediction": "prediction_datasets",
    "downstream_modulation": "downstream_modulation_datasets",
    "emitter_downstream": "emitter_downstream_datasets",
}


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


def _load_from_label_maps(
    rfdata_root: str | Path,
    key: str,
) -> list[str] | None:
    maps_path = Path(rfdata_root) / "label_maps.json"
    if not maps_path.is_file():
        return None
    with maps_path.open("r", encoding="utf-8") as f:
        label_maps = json.load(f)
    values = label_maps.get(key)
    if values:
        return [str(name) for name in values]
    return None


def _load_section_datasets(
    section: str,
    *,
    rfdata_root: str | Path | None = None,
    config_path: str | Path | None = None,
    legacy_path: Path,
) -> list[str]:
    label_key = _SECTION_LABEL_MAP_KEYS.get(section, f"{section}_datasets")
    if rfdata_root is not None:
        cached = _load_from_label_maps(rfdata_root, label_key)
        if cached:
            return cached
    return _load_dataset_section(section, config_path=config_path, legacy_path=legacy_path)


def load_excluded_datasets(
    rfdata_root: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
) -> list[str]:
    if rfdata_root is not None:
        cached = _load_from_label_maps(rfdata_root, "excluded_datasets")
        if cached:
            return cached
    return _load_dataset_section("excluded", config_path=config_path, legacy_path=LEGACY_EXCLUDED_CONFIG)


def load_downstream_excluded_datasets(
    rfdata_root: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
) -> list[str]:
    if rfdata_root is not None:
        cached = _load_from_label_maps(rfdata_root, "downstream_excluded_datasets")
        if cached:
            return cached
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
    return _load_section_datasets(
        "pretrain",
        rfdata_root=rfdata_root,
        config_path=config_path,
        legacy_path=LEGACY_PRETRAIN_CONFIG,
    )


def load_downstream_radar_modulation_datasets(
    rfdata_root: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
) -> list[str]:
    return _load_section_datasets(
        "downstream_radar_modulation",
        rfdata_root=rfdata_root,
        config_path=config_path,
        legacy_path=DEFAULT_DATASETS_CONFIG,
    )


def load_downstream_radar_model_datasets(
    rfdata_root: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
) -> list[str]:
    return _load_section_datasets(
        "downstream_radar_model",
        rfdata_root=rfdata_root,
        config_path=config_path,
        legacy_path=DEFAULT_DATASETS_CONFIG,
    )


def load_downstream_comm_modulation_datasets(
    rfdata_root: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
) -> list[str]:
    datasets = _load_section_datasets(
        "downstream_comm_modulation",
        rfdata_root=rfdata_root,
        config_path=config_path,
        legacy_path=LEGACY_DOWNSTREAM_MODULATION_CONFIG,
    )
    if datasets:
        return datasets
    return _load_dataset_section(
        "downstream_modulation",
        config_path=config_path,
        legacy_path=LEGACY_DOWNSTREAM_MODULATION_CONFIG,
    )


def load_downstream_modulation_datasets(
    rfdata_root: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
) -> list[str]:
    """兼容别名：通信调制样式识别下游白名单。"""
    return load_downstream_comm_modulation_datasets(
        rfdata_root=rfdata_root,
        config_path=config_path,
    )


def load_downstream_modulation_extra_datasets(
    rfdata_root: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
) -> list[str]:
    if rfdata_root is not None:
        cached = _load_from_label_maps(rfdata_root, "downstream_modulation_extra_datasets")
        if cached:
            return cached
    return _load_dataset_section(
        "downstream_modulation_extra",
        config_path=config_path,
        legacy_path=LEGACY_DOWNSTREAM_MODULATION_EXTRA_CONFIG,
    )


def load_clustering_radar_datasets(
    rfdata_root: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
) -> list[str]:
    return _load_section_datasets(
        "clustering_radar",
        rfdata_root=rfdata_root,
        config_path=config_path,
        legacy_path=DEFAULT_DATASETS_CONFIG,
    )


def load_clustering_comm_datasets(
    rfdata_root: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
) -> list[str]:
    return _load_section_datasets(
        "clustering_comm",
        rfdata_root=rfdata_root,
        config_path=config_path,
        legacy_path=DEFAULT_DATASETS_CONFIG,
    )


def load_prediction_datasets(
    rfdata_root: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
) -> list[str]:
    datasets = _load_section_datasets(
        "prediction",
        rfdata_root=rfdata_root,
        config_path=config_path,
        legacy_path=DEFAULT_DATASETS_CONFIG,
    )
    if datasets:
        return datasets
    return load_downstream_shared_datasets(rfdata_root=rfdata_root, config_path=config_path)


def load_downstream_shared_datasets(
    rfdata_root: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
) -> list[str]:
    """雷达型号 + 通信调制下游数据集并集（预测 / 聚类回退）。"""
    radar_model = load_downstream_radar_model_datasets(rfdata_root=rfdata_root, config_path=config_path)
    comm_mod = load_downstream_comm_modulation_datasets(rfdata_root=rfdata_root, config_path=config_path)
    clustering_radar = load_clustering_radar_datasets(rfdata_root=rfdata_root, config_path=config_path)
    clustering_comm = load_clustering_comm_datasets(rfdata_root=rfdata_root, config_path=config_path)
    return sorted(set(radar_model) | set(comm_mod) | set(clustering_radar) | set(clustering_comm))


def filter_dataset_pool(files: list[str], allowed_datasets: list[str] | set[str]) -> list[str]:
    allowed = set(allowed_datasets)
    return sorted(filename for filename in files if h5_dataset_name(filename) in allowed)


def filter_excluded_dataset_pool(files: list[str], excluded_datasets: list[str] | set[str]) -> list[str]:
    excluded = set(excluded_datasets)
    if not excluded:
        return sorted(files)
    return sorted(filename for filename in files if h5_dataset_name(filename) not in excluded)
