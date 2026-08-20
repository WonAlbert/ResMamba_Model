from __future__ import annotations

import warnings

import torch
from torch.utils.data import DataLoader, Dataset

from resmamba_signal_model.training.data_module import SignalDataModule, merge_source_batches
from resmamba_signal_model.training.logging_utils import silence_third_party_warnings
from resmamba_signal_model.training.mix import DynamicRatioScheduler


def test_dynamic_ratio_sums_and_min_ratio() -> None:
    sched = DynamicRatioScheduler(["a", "b", "c"], alpha=0.5, tau=1.0, min_ratio=0.1)
    sched.update({"a": 10.0, "b": 1.0, "c": 1.0}, {"a": 10.0, "b": 10.0, "c": 10.0})
    ratios = sched.ratios()
    assert abs(sum(ratios.values()) - 1.0) < 1e-6
    assert min(ratios.values()) >= 0.1 - 1e-8
    shares = sched.token_shares(100)
    assert sum(shares.values()) == 100
    assert min(shares.values()) >= 1


def test_dynamic_ratio_ema_stays_order_one() -> None:
    sched = DynamicRatioScheduler(["a", "b"], alpha=0.5, tau=1.0, min_ratio=0.05)
    for _ in range(20):
        sched.update({"a": 0.2 * 64.0, "b": 0.3 * 128.0}, {"a": 64.0, "b": 128.0})
    assert 0.05 < sched.ema["a"] < 2.0
    assert 0.05 < sched.ema["b"] < 2.0
    ratios = sched.ratios()
    assert abs(sum(ratios.values()) - 1.0) < 1e-6
    sched = DynamicRatioScheduler(["x", "y"], min_ratio=0.05)
    shares = sched.token_shares(64)
    assert sum(shares.values()) == 64
    ratios = sched.ratios()
    assert abs(sum(ratios.values()) - 1.0) < 1e-6


def test_datamodule_is_lightning() -> None:
    from lightning.pytorch import LightningDataModule

    dm = SignalDataModule(
        {
            "synthetic": True,
            "synthetic_sources": ["src_a", "src_b"],
            "token_budget": 32,
            "steps_per_epoch": 1,
            "val_batches": 2,
            "num_workers": 0,
            "patch_size": 8,
        },
        stage="pretrain",
    )
    assert isinstance(dm, LightningDataModule)
    assert hasattr(dm, "_log_hyperparams")
    dm.setup()
    assert dm.mix is not None
    assert set(dm.source_names) == {"src_a", "src_b"}
    val_loaders = dm._loaders(dm._val_sets, train=False)
    assert all(len(loader) >= 1 for loader in val_loaders.values())


def test_val_loader_keeps_full_workers_without_persistent() -> None:
    dm = SignalDataModule(
        {
            "synthetic": True,
            "synthetic_sources": ["src_a", "src_b"],
            "token_budget": 32,
            "steps_per_epoch": 1,
            "val_batches": 1,
            "num_workers": 2,
            "prefetch_factor": 3,
            "patch_size": 8,
            "pin_memory": True,
        },
        stage="pretrain",
    )
    dm.setup()
    val_loader = dm._loaders(dm._val_sets, train=False)["src_a"]
    train_loader = dm._loaders(dm._train_sets, train=True)["src_a"]
    assert val_loader.num_workers == 2
    assert val_loader.prefetch_factor == 3
    assert val_loader.persistent_workers is True
    assert val_loader.pin_memory is False
    assert train_loader.num_workers == 2
    assert train_loader.prefetch_factor == 3
    assert train_loader.persistent_workers is True
    assert train_loader.pin_memory is False


def test_combined_loader_merge_then_pack() -> None:
    from lightning.pytorch.utilities import CombinedLoader

    class _IQ(Dataset):
        def __init__(self, n: int, length: int) -> None:
            self.n = n
            self.length = length

        def __len__(self) -> int:
            return self.n

        def __getitem__(self, idx: int) -> dict:
            return {
                "iq": torch.randn(2, self.length),
                "length": self.length,
                "dataset_id": 0,
                "mod_label_id": 1,
                "emitter_id": -1,
                "source_label_id": -1,
                "global_label_id": 1,
                "task_type_id": 0,
            }

    loaders = {
        "short": DataLoader(_IQ(4, 128), batch_size=2, collate_fn=lambda xs: {
            "iq": [x["iq"] for x in xs],
            "length": torch.tensor([x["length"] for x in xs]),
            "dataset_id": torch.zeros(len(xs), dtype=torch.long),
        }),
        "long": DataLoader(_IQ(4, 256), batch_size=1, collate_fn=lambda xs: {
            "iq": [x["iq"] for x in xs],
            "length": torch.tensor([x["length"] for x in xs]),
            "dataset_id": torch.ones(len(xs), dtype=torch.long),
        }),
    }
    combined = CombinedLoader(loaders, mode="max_size_cycle")
    iterator = iter(combined)
    batch = next(iterator)
    if isinstance(batch, (tuple, list)):
        batch = batch[0]
    merged = merge_source_batches(batch)
    assert isinstance(merged["iq"], list)
    assert len(merged["iq"]) >= 2
    assert set(merged["source_name"]) == {"short", "long"}
    assert 128 in merged["length"].tolist()
    assert 256 in merged["length"].tolist()
    assert "canonical_mod_label_id" in merged
    assert "global_emitter_id" in merged
    assert merged["canonical_mod_label_id"].numel() >= 2


def test_combined_loader_hides_leafspec_deprecation() -> None:
    from lightning.pytorch.utilities import CombinedLoader

    silence_third_party_warnings()
    loaders = {
        "a": DataLoader(torch.zeros(4, 1), batch_size=2),
        "b": DataLoader(torch.zeros(4, 1), batch_size=2),
    }
    with warnings.catch_warnings(record=True) as caught:
        CombinedLoader(loaders, mode="max_size_cycle")
    leftover = [str(item.message) for item in caught if "LeafSpec" in str(item.message)]
    assert leftover == []


def test_as_source_map_uses_stamped_name_and_hint() -> None:
    from resmamba_signal_model.training.lit_module import _as_source_map

    stamped = {"iq": torch.zeros(2, 2, 8), "source_name": ["emitter", "emitter"], "task": "emitter"}
    assert list(_as_source_map(stamped)) == ["emitter"]
    untagged = {"iq": torch.zeros(1, 2, 8)}
    assert list(_as_source_map(untagged)) == ["default"]
    assert list(_as_source_map(untagged, default_name="clustering")) == ["clustering"]
    combined = {
        "classification": {"iq": torch.zeros(1, 2, 8)},
        "emitter": {"iq": torch.zeros(1, 2, 8)},
    }
    assert set(_as_source_map(combined)) == {"classification", "emitter"}


def test_stage2_val_loader_stamps_source_and_task() -> None:
    dm = SignalDataModule(
        {
            "synthetic": True,
            "token_budget": 32,
            "steps_per_epoch": 1,
            "val_batches": 1,
            "num_workers": 0,
            "patch_size": 8,
            "pin_memory": False,
        },
        stage="stage2",
    )
    dm.setup()
    assert "emitter" in dm._val_sets
    loaders = dm._loaders(dm._val_sets, train=False)
    batch = next(iter(loaders["emitter"]))
    assert batch["source_name"][0] == "emitter"
    assert batch["task"] == "emitter"
    cls_batch = next(iter(loaders["classification"]))
    assert cls_batch["task"] == "modulation"
    assert "canonical_mod_label_id" in cls_batch
    assert "global_emitter_id" in cls_batch
    assert dm.val_source_names == list(dm._val_sets)




def test_mix_ema_does_not_collapse_after_one_spike() -> None:
    sched = DynamicRatioScheduler(["a", "b", "c"], alpha=0.9, tau=1.0, min_ratio=0.05, value_clip=2.0)
    sched.update({"a": 1.0e4, "b": 1.0, "c": 1.0}, {"a": 100.0, "b": 100.0, "c": 100.0})
    assert sched.ema["a"] <= 1.2
    shares = sched.token_shares(8192)
    assert max(shares.values()) < 7000
    assert min(shares.values()) >= int(0.05 * 8192) - 1


def test_train_dataloader_reports_epoch_length() -> None:
    dm = SignalDataModule(
        {
            "synthetic": True,
            "synthetic_sources": ["src_a", "src_b"],
            "token_budget": 32,
            "steps_per_epoch": 7,
            "val_batches": 3,
            "num_workers": 0,
            "patch_size": 8,
        },
        stage="pretrain",
    )
    dm.setup()
    train_loader = dm.train_dataloader()
    assert len(train_loader) == 7
    val_loader = dm.val_dataloader()
    assert len(val_loader) == 6
