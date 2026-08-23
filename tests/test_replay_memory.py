from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resmamba_signal_model.training.replay_memory import (
    resolve_dynamic_samples_per_class,
    resolve_replay_memory_total,
    select_nearest_class_center_exemplars,
    stratified_scan_indices,
)


def test_resolve_replay_memory_total_defaults_to_single_task_baseline() -> None:
    cfg = {"replay_samples_per_class": 20, "replay_baseline_classes_per_task": 11}
    assert resolve_replay_memory_total(cfg) == 220


def test_dynamic_samples_per_class_shrink_with_more_replay_tasks() -> None:
    cfg = {"replay_samples_per_class": 20, "replay_baseline_classes_per_task": 10}
    one, task_one, total = resolve_dynamic_samples_per_class(
        cfg, num_replay_tasks=1, num_classes_in_task=10
    )
    two, task_two, _ = resolve_dynamic_samples_per_class(
        cfg, num_replay_tasks=2, num_classes_in_task=10
    )
    assert total == 200
    assert one == 20
    assert task_one == 200
    assert two == 10
    assert task_two == 100
    assert one * 10 == two * 10 * 2


def test_select_nearest_class_center_exemplars_picks_center_proximal() -> None:
    # 两类：离各自中心最近的样本应被选中
    c0 = torch.tensor([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0]])
    c1 = torch.tensor([[0.0, 1.0], [0.1, 0.9], [0.0, -1.0]])
    embeddings = torch.cat([c0, c1], dim=0)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    indices = torch.arange(6)
    picked = select_nearest_class_center_exemplars(
        embeddings,
        labels,
        indices,
        samples_per_class=1,
    )
    assert len(picked) == 2
    assert 0 in picked or 1 in picked
    assert 3 in picked or 4 in picked
    assert 2 not in picked
    assert 5 not in picked


def test_stratified_scan_indices_covers_classes() -> None:
    class_ids = [0] * 10 + [1] * 10 + [2] * 10
    scan = stratified_scan_indices(class_ids, max_samples=9, seed=0)
    labels = [class_ids[i] for i in scan]
    assert len(scan) == 9
    assert len(set(labels)) >= 2


def test_datamodule_replay_subset_limits_train_indices() -> None:
    from torch.utils.data import Subset

    from resmamba_signal_model.training.data_module import SignalDataModule

    cfg = {
        "synthetic": True,
        "seed": 0,
        "token_budget": 256,
        "steps_per_epoch": 2,
        "val_batches": 1,
        "num_workers": 0,
        "patch_size": 8,
        "tasks": ["tx_modulation", "ld_model"],
        "task_pools": {
            "tx_modulation": ["a", "b"],
            "ld_model": ["c", "d"],
        },
        "replay_strategy": "class_center",
        "min_ratio": 0.05,
    }
    dm = SignalDataModule(cfg, stage="stage2")
    dm.setup()
    dm.set_active_train_filter(["ld_model"], replay_sources=["tx_modulation"])
    dm.update_replay_memory({"tx_modulation": [1, 3, 5]})
    dataset, lengths = dm._train_dataset_and_lengths("tx_modulation", dm._train_sets["tx_modulation"], train=True)
    assert isinstance(dataset, Subset)
    assert list(lengths) == [dm._train_lengths["tx_modulation"][i] for i in (1, 3, 5)]
    full, full_lengths = dm._train_dataset_and_lengths("ld_model", dm._train_sets["ld_model"], train=True)
    assert full is dm._train_sets["ld_model"]
    assert full_lengths == dm._train_lengths["ld_model"]
