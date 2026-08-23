from __future__ import annotations

from typing import Any

from resmamba_signal_model.training.emitter_labels import (
    GlobalEmitterLabelMap,
    build_compact_emitter_dataset_class_mask,
    build_global_emitter_label_map,
)
from resmamba_signal_model.training.modulation_labels import (
    CompactModulationLabelMap,
    build_compact_modulation_label_map,
)

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
    modulation_map: CompactModulationLabelMap | None = None

    try:
        emitter_map = build_global_emitter_label_map(root, train_cfg=train_cfg)
        model_cfg.num_emitters = int(emitter_map.num_emitters)
        model_section = train_cfg.setdefault("model", {})
        model_section["num_emitters"] = int(emitter_map.num_emitters)
        train_cfg["compact_emitter"] = {
            "num_emitters": int(emitter_map.num_emitters),
            "offsets": {str(k): int(v) for k, v in emitter_map.offsets.items()},
            "class_counts": {str(k): int(v) for k, v in (emitter_map.class_counts or {}).items()},
            "datasets": {str(k): v for k, v in emitter_map.dataset_names.items()},
        }
    except (FileNotFoundError, KeyError, OSError):
        emitter_map = None

    try:
        modulation_map = build_compact_modulation_label_map(root, train_cfg=train_cfg)
        model_cfg.num_mod_classes = int(modulation_map.num_classes)
        model_section = train_cfg.setdefault("model", {})
        model_section["num_mod_classes"] = int(modulation_map.num_classes)
        train_cfg["compact_modulation"] = {
            "num_classes": int(modulation_map.num_classes),
            "old_to_new": {str(k): int(v) for k, v in modulation_map.old_to_new.items()},
            "datasets": list(modulation_map.dataset_names),
        }
    except (FileNotFoundError, KeyError, OSError):
        modulation_map = None

    train_cfg["compact_task_labels"] = bool(emitter_map is not None or modulation_map is not None)
    return emitter_map, modulation_map


def compact_emitter_class_mask(
    emitter_map: GlobalEmitterLabelMap | None,
    *,
    num_datasets: int,
):
    if emitter_map is None:
        return None
    return build_compact_emitter_dataset_class_mask(emitter_map, num_datasets=num_datasets)
