from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

# 与预训练 emitter namespace 一致，保证跨子数据集标签不碰撞。
GLOBAL_LABEL_NAMESPACE = 100_000


def local_cluster_label(
    mod_label_id: np.ndarray | torch.Tensor,
    emitter_id: np.ndarray | torch.Tensor,
    source_label_id: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """选取样本的主局部标签：modulation > emitter > source。"""
    if isinstance(mod_label_id, torch.Tensor):
        local = torch.full_like(mod_label_id, -1)
        local = torch.where(mod_label_id >= 0, mod_label_id, local)
        local = torch.where((local < 0) & (emitter_id >= 0), emitter_id, local)
        local = torch.where((local < 0) & (source_label_id >= 0), source_label_id, local)
        return local
    local = np.full(np.shape(mod_label_id), -1, dtype=np.int32)
    valid_mod = mod_label_id >= 0
    local[valid_mod] = mod_label_id[valid_mod]
    valid_emit = (local < 0) & (emitter_id >= 0)
    local[valid_emit] = emitter_id[valid_emit]
    valid_src = (local < 0) & (source_label_id >= 0)
    local[valid_src] = source_label_id[valid_src]
    return local


def resolve_cluster_eval_labels(
    mod_label_id: np.ndarray | torch.Tensor | None,
    emitter_id: np.ndarray | torch.Tensor | None,
    source_label_id: np.ndarray | torch.Tensor | None,
) -> np.ndarray | torch.Tensor | None:
    """聚类验证指标用局部标签（单 H5 内 mod > emitter > source），不做跨 dataset 全局命名。"""
    if mod_label_id is None and emitter_id is None and source_label_id is None:
        return None
    if mod_label_id is None:
        mod_label_id = emitter_id if emitter_id is not None else source_label_id
    if emitter_id is None:
        if isinstance(mod_label_id, torch.Tensor):
            emitter_id = torch.full_like(mod_label_id, -1)
        else:
            emitter_id = np.full(np.shape(mod_label_id), -1, dtype=np.int32)
    if source_label_id is None:
        if isinstance(mod_label_id, torch.Tensor):
            source_label_id = torch.full_like(mod_label_id, -1)
        else:
            source_label_id = np.full(np.shape(mod_label_id), -1, dtype=np.int32)
    return local_cluster_label(mod_label_id, emitter_id, source_label_id)


def resolve_modulation_labels(
    mod_label_id: torch.Tensor,
    source_label_id: torch.Tensor | None = None,
) -> torch.Tensor:
    """Stage2 调制标签：优先 mod_label_id，否则回退 source_label_id（辐射源数据集）。"""
    if source_label_id is None:
        return mod_label_id
    return torch.where(mod_label_id >= 0, mod_label_id, source_label_id)


def global_cluster_labels(
    dataset_id: np.ndarray | torch.Tensor,
    mod_label_id: np.ndarray | torch.Tensor,
    emitter_id: np.ndarray | torch.Tensor,
    source_label_id: np.ndarray | torch.Tensor,
    *,
    namespace: int = GLOBAL_LABEL_NAMESPACE,
) -> np.ndarray | torch.Tensor:
    local = local_cluster_label(mod_label_id, emitter_id, source_label_id)
    if isinstance(local, torch.Tensor):
        valid = local >= 0
        out = torch.full_like(local, -1)
        return torch.where(
            valid,
            dataset_id.long() * int(namespace) + local.long(),
            out,
        )
    valid = local >= 0
    out = np.full(local.shape, -1, dtype=np.int32)
    out[valid] = (dataset_id[valid].astype(np.int64) * int(namespace) + local[valid].astype(np.int64)).astype(np.int32)
    return out


def load_dataset_id_names(rfdata_root: str | Path | None) -> dict[int, str]:
    if not rfdata_root:
        return {}
    path = Path(rfdata_root) / "label_maps.json"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f).get("datasets", {})
    return {int(k): str(v) for k, v in raw.items()}
