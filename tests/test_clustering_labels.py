from __future__ import annotations

import numpy as np
import torch

from resmamba_signal_model.training.clustering_labels import (
    GLOBAL_LABEL_NAMESPACE,
    global_cluster_labels,
    local_cluster_label,
    resolve_modulation_labels,
)
from resmamba_signal_model.training.selection import (
    TASK_SELECTION_DEFAULTS,
    compute_selection_score,
    resolve_selection_metric_name,
    selection_higher_is_better,
)


def test_local_cluster_label_priority() -> None:
    mod = np.array([3, -1, -1], dtype=np.int32)
    emit = np.array([-1, 5, -1], dtype=np.int32)
    src = np.array([-1, -1, 2], dtype=np.int32)
    local = local_cluster_label(mod, emit, src)
    assert local.tolist() == [3, 5, 2]


def test_global_cluster_labels_namespace() -> None:
    dataset_id = np.array([6, 7], dtype=np.int32)
    emit = np.array([5, 5], dtype=np.int32)
    mod = np.full(2, -1, dtype=np.int32)
    src = np.full(2, -1, dtype=np.int32)
    global_id = global_cluster_labels(dataset_id, mod, emit, src)
    assert global_id.tolist() == [6 * GLOBAL_LABEL_NAMESPACE + 5, 7 * GLOBAL_LABEL_NAMESPACE + 5]


def test_global_cluster_labels_torch() -> None:
    dataset_id = torch.tensor([1, 2])
    mod = torch.tensor([4, -1])
    emit = torch.tensor([-1, 9])
    src = torch.tensor([-1, -1])
    global_id = global_cluster_labels(dataset_id, mod, emit, src)
    assert global_id.tolist() == [1 * GLOBAL_LABEL_NAMESPACE + 4, 2 * GLOBAL_LABEL_NAMESPACE + 9]


def test_resolve_modulation_labels_fallback_source() -> None:
    mod = torch.tensor([-1, 2, -1])
    src = torch.tensor([7, 9, -1])
    labels = resolve_modulation_labels(mod, src)
    assert labels.tolist() == [7, 2, -1]


def test_selection_defaults() -> None:
    assert resolve_selection_metric_name({}, stage="downstream", task="modulation") == "f1"
    assert resolve_selection_metric_name({}, stage="downstream", task="clustering") == "nmi"
    assert resolve_selection_metric_name({}, stage="downstream", task="prediction") == "mse"
    assert TASK_SELECTION_DEFAULTS["emitter"] == "per_dataset_macro_acc"


def test_selection_per_dataset_macro_acc() -> None:
    score = compute_selection_score({"acc/a": 0.5, "acc/b": 0.7, "loss": 1.0}, "per_dataset_macro_acc")
    assert abs(score - 0.6) < 1e-6


def test_selection_val_loss_sign() -> None:
    assert selection_higher_is_better("val_loss") is False
    assert compute_selection_score({"loss": 2.0}, "val_loss") == 2.0
