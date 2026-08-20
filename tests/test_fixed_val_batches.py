from __future__ import annotations

from collections import Counter

import torch

from resmamba_signal_model.data.sampling import (
    FixedBatchSampler,
    dataset_class_ids,
    plan_fixed_token_budget_batches,
    regroup_token_batches_by_length,
    resolve_sample_class_id,
)
from resmamba_signal_model.training.data_module import SignalDataModule


def _synthetic_cfg(**overrides) -> dict:
    cfg = {
        "synthetic": True,
        "synthetic_sources": ["src_a", "src_b"],
        "token_budget": 32,
        "steps_per_epoch": 1,
        "val_batches": 2,
        "val_seed": 0,
        "num_workers": 0,
        "patch_size": 8,
        "pin_memory": False,
    }
    cfg.update(overrides)
    return cfg


def _flatten(batches: list[list[int]]) -> list[int]:
    return [idx for batch in batches for idx in batch]


def test_regroup_token_batches_by_length_is_homogeneous() -> None:
    lengths = [128, 256, 128, 512, 256, 128]
    mixed = [[0, 1, 2], [3, 4, 5]]
    grouped = regroup_token_batches_by_length(mixed, lengths, token_budget=64, patch_size=16)
    used = _flatten(grouped)
    assert sorted(used) == sorted(_flatten(mixed))
    for batch in grouped:
        assert len({lengths[idx] for idx in batch}) == 1
    lengths = [128, 256, 64, 512, 192, 96, 320, 160]
    kwargs = dict(token_budget=20, patch_size=16, num_batches=4, seed=7)
    first = plan_fixed_token_budget_batches(lengths, **kwargs)
    second = plan_fixed_token_budget_batches(lengths, **kwargs)
    assert first == second
    assert first
    sampler = FixedBatchSampler(first)
    assert list(sampler) == first
    assert list(sampler) == first


def test_plan_fixed_does_not_wrap_past_dataset() -> None:
    lengths = [16] * 10
    plan = plan_fixed_token_budget_batches(
        lengths,
        token_budget=3,
        patch_size=16,
        num_batches=100,
        seed=0,
    )
    indices = _flatten(plan)
    assert len(plan) <= 100
    assert len(indices) == len(set(indices))
    assert set(indices) <= set(range(len(lengths)))
    assert len(indices) <= len(lengths)
    assert len(indices) == len(lengths)


def test_val_loader_two_epochs_same_indices_from_val_not_train() -> None:
    dm = SignalDataModule(_synthetic_cfg(val_batches=2, val_seed=0), stage="pretrain")
    dm.setup()
    assert dm._val_batch_plan
    for name, val_ds in dm._val_sets.items():
        n_val = len(val_ds)
        n_train = len(dm._train_sets[name])
        assert n_val < n_train
        plan = dm._val_batch_plan[name]
        indices = _flatten(plan)
        assert indices
        assert all(0 <= idx < n_val for idx in indices)
        val_lengths = [int(val_ds[i]["length"]) for i in range(n_val)]
        train_lengths = [int(dm._train_sets[name][i]["length"]) for i in range(n_train)]
        expected = plan_fixed_token_budget_batches(
            val_lengths,
            token_budget=dm.token_budget,
            patch_size=dm.patch_size,
            num_batches=dm.val_batches,
            seed=dm.val_seed,
            class_ids=dataset_class_ids(val_ds),
        )
        train_plan = plan_fixed_token_budget_batches(
            train_lengths,
            token_budget=dm.token_budget,
            patch_size=dm.patch_size,
            num_batches=dm.val_batches,
            seed=dm.val_seed,
            class_ids=dataset_class_ids(dm._train_sets[name]),
        )
        assert sorted(indices) == sorted(_flatten(expected))
        assert sorted(indices) != sorted(_flatten(train_plan))
        for batch in plan:
            batch_lengths = {val_lengths[idx] for idx in batch}
            assert len(batch_lengths) == 1

    loaders_a = dm._loaders(dm._val_sets, train=False)
    loaders_b = dm._loaders(dm._val_sets, train=False)
    for name, loader in loaders_a.items():
        assert loader.dataset is dm._val_sets[name]
        epoch1 = [list(batch) for batch in loader.batch_sampler]
        epoch2 = [list(batch) for batch in loader.batch_sampler]
        reloaded = [list(batch) for batch in loaders_b[name].batch_sampler]
        assert epoch1 == epoch2 == reloaded == dm._val_batch_plan[name]
        assert len(epoch1) <= dm.val_batches


def test_val_plan_stops_when_val_set_exhausted() -> None:
    dm = SignalDataModule(_synthetic_cfg(val_batches=1000, val_seed=0), stage="pretrain")
    dm.setup()
    for name, val_ds in dm._val_sets.items():
        plan = dm._val_batch_plan[name]
        indices = _flatten(plan)
        assert len(plan) <= 1000
        assert len(plan) < 1000
        assert len(indices) == len(set(indices))
        assert len(indices) <= len(val_ds)
        assert all(0 <= idx < len(val_ds) for idx in indices)
        loader = dm._loaders(dm._val_sets, train=False)[name]
        assert len(loader) == len(plan)
        assert len(loader) >= 1


def test_plan_stratified_balances_imbalanced_classes() -> None:
    lengths = [16] * 100
    class_ids = [0] * 80 + [1] * 20
    kwargs = dict(token_budget=10, patch_size=16, num_batches=4, seed=0, class_ids=class_ids)
    first = plan_fixed_token_budget_batches(lengths, **kwargs)
    second = plan_fixed_token_budget_batches(lengths, **kwargs)
    assert first == second
    counts = Counter(class_ids[idx] for idx in _flatten(first))
    assert counts[0] == 20
    assert counts[1] == 20
    shuffled = plan_fixed_token_budget_batches(lengths, token_budget=10, patch_size=16, num_batches=4, seed=0)
    shuffled_counts = Counter(class_ids[idx] for idx in _flatten(shuffled))
    assert shuffled_counts[0] >= 28


def test_plan_stratified_does_not_resample_exhausted_class() -> None:
    lengths = [16] * 30
    class_ids = [0] * 25 + [1] * 5
    plan = plan_fixed_token_budget_batches(
        lengths,
        token_budget=10,
        patch_size=16,
        num_batches=3,
        seed=1,
        class_ids=class_ids,
    )
    indices = _flatten(plan)
    counts = Counter(class_ids[idx] for idx in indices)
    assert counts[1] == 5
    assert counts[0] == 25
    assert len(indices) == len(set(indices))


def test_resolve_sample_class_id_field_priority() -> None:
    assert resolve_sample_class_id({"global_label_id": 12, "mod_label_id": 3}) == 12
    assert resolve_sample_class_id({"global_label_id": -1, "canonical_mod_label_id": 5, "mod_label_id": 3}) == 5
    assert resolve_sample_class_id({"global_label_id": -1, "mod_label_id": 3}) == 3
    assert resolve_sample_class_id({"global_label_id": -1, "mod_label_id": -1, "source_label_id": 4}) == 4
    assert resolve_sample_class_id(
        {"global_label_id": -1, "mod_label_id": -1, "source_label_id": -1, "global_emitter_id": 11, "emitter_id": 7}
    ) == 11
    assert resolve_sample_class_id({"global_label_id": -1, "mod_label_id": -1, "emitter_id": 7}) == 7
    assert resolve_sample_class_id(
        {"global_label_id": -1, "mod_label_id": -1, "source_label_id": -1, "emitter_id": -1, "dataset_id": 9}
    ) == 9


def test_val_loader_stratified_on_imbalanced_synthetic() -> None:
    from resmamba_signal_model.training.data_module import SyntheticIQDataset

    class _ImbalancedVal(SyntheticIQDataset):
        def __init__(self) -> None:
            super().__init__(n=100, lengths=(16,), n_datasets=1, source_id=0)
            self._len_idx = self._len_idx * 0
            self._ds = self._ds * 0
            self._mod = torch.tensor([0] * 80 + [1] * 10 + [2] * 10)

        def class_labels(self) -> list[int]:
            return [int(x) for x in self._mod.tolist()]

    dm = SignalDataModule(_synthetic_cfg(val_batches=3, token_budget=10, patch_size=16), stage="pretrain")
    dm.setup()
    dm._val_sets = {"src_a": _ImbalancedVal()}
    dm._build_val_plans()
    plan = dm._val_batch_plan["src_a"]
    class_ids = dm._val_sets["src_a"].class_labels()
    counts = Counter(class_ids[idx] for idx in _flatten(plan))
    assert counts.keys() == {0, 1, 2}
    assert max(counts.values()) - min(counts.values()) <= 1
    assert all(count == 10 for count in counts.values())
