from __future__ import annotations

import json
from pathlib import Path

import yaml

from resmamba_signal_model.training.emitter_labels import h5_dataset_name

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
DEFAULT_EXCLUDED_DATASETS_CONFIG = CONFIG_ROOT / "excluded_datasets.yaml"
DEFAULT_DOWNSTREAM_EXCLUDED_DATASETS_CONFIG = CONFIG_ROOT / "downstream_excluded_datasets.yaml"
DEFAULT_DOWNSTREAM_MODULATION_EXTRA_DATASETS_CONFIG = CONFIG_ROOT / "downstream_modulation_extra_datasets.yaml"


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

    path = Path(config_path) if config_path is not None else DEFAULT_EXCLUDED_DATASETS_CONFIG
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        datasets = data.get("datasets")
        if datasets:
            return [str(name) for name in datasets]

    return []


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

    path = Path(config_path) if config_path is not None else DEFAULT_DOWNSTREAM_EXCLUDED_DATASETS_CONFIG
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        datasets = data.get("datasets")
        if datasets:
            return [str(name) for name in datasets]

    return []


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

    path = Path(config_path) if config_path is not None else DEFAULT_DOWNSTREAM_MODULATION_EXTRA_DATASETS_CONFIG
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        datasets = data.get("datasets")
        if datasets:
            return [str(name) for name in datasets]

    return []


def filter_dataset_pool(files: list[str], allowed_datasets: list[str] | set[str]) -> list[str]:
    allowed = set(allowed_datasets)
    return sorted(filename for filename in files if h5_dataset_name(filename) in allowed)


def filter_excluded_dataset_pool(files: list[str], excluded_datasets: list[str] | set[str]) -> list[str]:
    excluded = set(excluded_datasets)
    if not excluded:
        return sorted(files)
    return sorted(filename for filename in files if h5_dataset_name(filename) not in excluded)
