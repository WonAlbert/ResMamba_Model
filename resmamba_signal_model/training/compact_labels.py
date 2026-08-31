from __future__ import annotations

from typing import Any

from resmamba_signal_model.training.emitter_labels import (
    GlobalEmitterLabelMap,
    build_compact_emitter_dataset_class_mask,
    build_global_emitter_label_map,
    build_global_radar_model_label_map,
)
from resmamba_signal_model.training.modulation_labels import (
    CompactModulationLabelMap,
    build_comm_modulation_dataset_class_mask,
    build_compact_modulation_label_map,
    build_global_comm_modulation_label_map,
)
from resmamba_signal_model.training.pool_filters import load_downstream_radar_modulation_datasets

COMPACT_STAGES = frozenset({"stage2", "stage3", "joint", "downstream"})


def compact_task_labels_enabled(train_cfg: dict[str, Any], *, stage: str) -> bool:
    """阶段二/三默认开启紧凑标签；预训练保持全库命名空间。"""
    if stage not in COMPACT_STAGES:
        return False
    if bool(train_cfg.get("synthetic", False)):
        return False
    explicit = train_cfg.get("compact_task_labels")
    if explicit is None:
        return True
    return bool(explicit)


def apply_compact_task_labels(
    train_cfg: dict[str, Any],
    model_cfg: Any,
    *,
    stage: str,
) -> tuple[GlobalEmitterLabelMap | None, CompactModulationLabelMap | None]:
    """把调制/个体头的类数压到下游实际使用集合，并写回 train_cfg / model_cfg。"""
    if not compact_task_labels_enabled(train_cfg, stage=stage):
        train_cfg["compact_task_labels"] = False
        return None, None

    root = train_cfg.get("rfdata_root")
    if not root:
        train_cfg["compact_task_labels"] = False
        return None, None

    emitter_map: GlobalEmitterLabelMap | None = None
    radar_model_map: GlobalEmitterLabelMap | None = None
    modulation_map: CompactModulationLabelMap | None = None

    try:
        emitter_map = build_global_emitter_label_map(root, train_cfg=train_cfg)
        if int(emitter_map.num_emitters) > 0:
            model_cfg.num_emitters = int(emitter_map.num_emitters)
            model_section = train_cfg.setdefault("model", {})
            model_section["num_emitters"] = int(emitter_map.num_emitters)
            train_cfg["compact_emitter"] = {
                "num_emitters": int(emitter_map.num_emitters),
                "offsets": {str(k): int(v) for k, v in emitter_map.offsets.items()},
                "class_counts": {str(k): int(v) for k, v in (emitter_map.class_counts or {}).items()},
                "datasets": {str(k): v for k, v in emitter_map.dataset_names.items()},
            }
        else:
            emitter_map = None
    except (FileNotFoundError, KeyError, OSError):
        emitter_map = None

    try:
        radar_model_map = build_global_radar_model_label_map(root, train_cfg=train_cfg)
        if int(radar_model_map.num_emitters) > 0:
            model_cfg.num_ld_model_classes = int(radar_model_map.num_emitters)
            model_section = train_cfg.setdefault("model", {})
            model_section["num_ld_model_classes"] = int(radar_model_map.num_emitters)
            train_cfg["compact_ld_model"] = {
                "num_emitters": int(radar_model_map.num_emitters),
                "offsets": {str(k): int(v) for k, v in radar_model_map.offsets.items()},
                "class_counts": {str(k): int(v) for k, v in (radar_model_map.class_counts or {}).items()},
                "datasets": {str(k): v for k, v in radar_model_map.dataset_names.items()},
            }
        else:
            radar_model_map = None
    except (FileNotFoundError, KeyError, OSError):
        radar_model_map = None

    try:
        comm_map, canonical_map = build_global_comm_modulation_label_map(root, train_cfg=train_cfg)
        model_cfg.num_mod_classes = int(comm_map.num_emitters)
        model_section = train_cfg.setdefault("model", {})
        model_section["num_mod_classes"] = int(comm_map.num_emitters)
        train_cfg["compact_tx_modulation"] = {
            "num_classes": int(comm_map.num_emitters),
            "classes_per_dataset": int(max(comm_map.class_counts.values()) if comm_map.class_counts else 11),
            "offsets": {str(k): int(v) for k, v in comm_map.offsets.items()},
            "class_counts": {str(k): int(v) for k, v in (comm_map.class_counts or {}).items()},
            "datasets": {str(k): v for k, v in comm_map.dataset_names.items()},
            "old_to_new": {str(k): int(v) for k, v in canonical_map.old_to_new.items()},
            "dataset_names": list(canonical_map.dataset_names),
        }
        modulation_map = canonical_map
    except (FileNotFoundError, KeyError, OSError):
        modulation_map = None

    try:
        radar_names = load_downstream_radar_modulation_datasets(rfdata_root=root)
        if radar_names:
            intrapulse_map = build_compact_modulation_label_map(root, dataset_names=radar_names)
            _write_compact_modulation(
                train_cfg,
                model_cfg,
                intrapulse_map,
                key="compact_intrapulse",
                attr="num_intrapulse_classes",
            )
    except (FileNotFoundError, KeyError, OSError):
        pass

    train_cfg["compact_task_labels"] = bool(
        emitter_map is not None or radar_model_map is not None or modulation_map is not None
    )
    return emitter_map, modulation_map


def _write_compact_modulation(
    train_cfg: dict[str, Any],
    model_cfg: Any,
    mapping: CompactModulationLabelMap,
    *,
    key: str,
    attr: str,
) -> None:
    setattr(model_cfg, attr, int(mapping.num_classes))
    model_section = train_cfg.setdefault("model", {})
    model_section[attr] = int(mapping.num_classes)
    train_cfg[key] = {
        "num_classes": int(mapping.num_classes),
        "old_to_new": {str(k): int(v) for k, v in mapping.old_to_new.items()},
        "datasets": list(mapping.dataset_names),
    }


def compact_emitter_class_mask(
    emitter_map: GlobalEmitterLabelMap | None,
    *,
    num_datasets: int,
):
    if emitter_map is None:
        return None
    return build_compact_emitter_dataset_class_mask(emitter_map, num_datasets=num_datasets)


def compact_ld_model_class_mask(
    radar_model_map: GlobalEmitterLabelMap | None,
    *,
    num_datasets: int,
):
    """``ld_model`` 头按数据集掩码（radar_mod15 / cjr_mix 类空间互不重叠）。"""
    if radar_model_map is None:
        return None
    return build_compact_emitter_dataset_class_mask(radar_model_map, num_datasets=num_datasets)


def compact_tx_modulation_class_mask(
    tx_map: GlobalEmitterLabelMap | None,
    canonical: CompactModulationLabelMap | None,
    rfdata_root: str | Path | None,
    *,
    num_datasets: int,
):
    """``tx_modulation`` 33 类方案：每 dataset 行只开放该库存在的调制槽位。"""
    if tx_map is None or canonical is None or rfdata_root is None:
        return None
    return build_comm_modulation_dataset_class_mask(
        tx_map,
        canonical,
        rfdata_root,
        num_datasets=int(num_datasets),
    )


def tx_modulation_map_from_train_cfg(train_cfg: dict | None) -> tuple[GlobalEmitterLabelMap | None, CompactModulationLabelMap | None]:
    payload = (train_cfg or {}).get("compact_tx_modulation")
    if not isinstance(payload, dict) or not payload.get("offsets"):
        return None, None
    offsets = {int(k): int(v) for k, v in dict(payload["offsets"]).items()}
    counts_raw = payload.get("class_counts") or {}
    counts = {int(k): int(v) for k, v in dict(counts_raw).items()} if counts_raw else None
    names_raw = payload.get("datasets") or {}
    names = {int(k): str(v) for k, v in dict(names_raw).items()}
    tx_map = GlobalEmitterLabelMap(
        offsets=offsets,
        dataset_names=names,
        num_emitters=int(payload.get("num_classes") or 0),
        class_counts=counts,
    )
    old_to_new_raw = payload.get("old_to_new") or {}
    if not old_to_new_raw:
        return tx_map, None
    old_to_new = {int(k): int(v) for k, v in dict(old_to_new_raw).items()}
    canonical = CompactModulationLabelMap(
        old_to_new=old_to_new,
        num_classes=len(old_to_new),
        dataset_names=tuple(str(x) for x in (payload.get("dataset_names") or names.values())),
    )
    return tx_map, canonical


def radar_model_map_from_train_cfg(train_cfg: dict | None) -> GlobalEmitterLabelMap | None:
    payload = (train_cfg or {}).get("compact_ld_model")
    if not isinstance(payload, dict) or not payload.get("offsets"):
        return None
    offsets = {int(k): int(v) for k, v in dict(payload["offsets"]).items()}
    counts_raw = payload.get("class_counts") or {}
    counts = {int(k): int(v) for k, v in dict(counts_raw).items()} if counts_raw else None
    names_raw = payload.get("datasets") or {}
    names = {int(k): str(v) for k, v in dict(names_raw).items()}
    return GlobalEmitterLabelMap(
        offsets=offsets,
        dataset_names=names,
        num_emitters=int(payload.get("num_emitters") or 0),
        class_counts=counts,
    )
