from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from resmamba_signal_model.data.rfdata import (
    RFDataH5Dataset,
    RFDataPoolDataset,
    clear_iq_ram_cache,
    format_iq_ram_cache,
    iq_ram_cache_stats,
    normalize_iq,
    normalize_iq_numpy,
    pad_iq_collate,
    resolve_cache_iq_in_memory,
    rfdata_loader_worker_kwargs,
    variable_length_collate,
)


def _write_h5(path: Path, *, n: int = 8, length: int = 32) -> None:
    with h5py.File(path, "w") as f:
        iq = np.random.randn(n, 2, length).astype(np.float32)
        f.create_dataset("iq", data=iq)
        f.create_dataset("length", data=np.full(n, length, dtype=np.int32))
        f.create_dataset("dataset_id", data=np.zeros(n, dtype=np.int32))
        f.create_dataset("task_type_id", data=np.zeros(n, dtype=np.int32))
        f.create_dataset("mod_label_id", data=np.arange(n, dtype=np.int32))
        f.create_dataset("emitter_id", data=np.arange(n, dtype=np.int32))
        f.create_dataset("snr", data=np.linspace(-10, 10, n, dtype=np.float32))


def test_normalize_iq_numpy_matches_torch_formula() -> None:
    iq = torch.randn(2, 128)
    numpy_out = torch.from_numpy(normalize_iq_numpy(iq.numpy(), "joint_power"))
    power = iq.float().square().sum(dim=0).mean().clamp_min(1.0e-8)
    torch_out = (iq / torch.sqrt(power)).clamp(-5.0, 5.0)
    assert torch.allclose(numpy_out, torch_out, atol=1e-5)
    wrapped = normalize_iq(iq, "joint_power")
    assert torch.allclose(wrapped, torch_out, atol=1e-5)


def test_dataset_skips_extra_metadata_by_default(tmp_path: Path) -> None:
    path = tmp_path / "a.h5"
    _write_h5(path)
    sample = RFDataH5Dataset(path, iq_normalize="joint_power")[0]
    assert sample["iq"].shape == (2, 32)
    assert sample["mod_label_id"] == 0
    assert "snr" not in sample
    assert "h5_path" not in sample
    extra = RFDataH5Dataset(path, include_extra_metadata=True)[1]
    assert abs(float(extra["snr"]) - float(np.linspace(-10, 10, 8)[1])) < 1e-5
    assert extra["h5_path"].endswith("a.h5")


def test_pad_iq_collate_equal_length_stacks() -> None:
    batch = [
        {"iq": torch.randn(2, 16), "length": 16, "dataset_id": 1, "mod_label_id": i}
        for i in range(4)
    ]
    out = pad_iq_collate(batch)
    assert out["iq"].shape == (4, 2, 16)
    assert bool(out["sample_mask"].all())
    assert "snr" not in out
    assert out["mod_label_id"].tolist() == [0, 1, 2, 3]


def test_pad_iq_collate_pads_mixed_lengths() -> None:
    batch = [
        {"iq": torch.ones(2, 4), "length": 4, "dataset_id": 0},
        {"iq": torch.ones(2, 8) * 2, "length": 8, "dataset_id": 1},
    ]
    out = pad_iq_collate(batch)
    assert out["iq"].shape == (2, 2, 8)
    assert out["sample_mask"][0].sum() == 4
    assert out["sample_mask"][1].sum() == 8
    assert torch.equal(out["iq"][0, :, :4], torch.ones(2, 4))
    assert torch.equal(out["iq"][1], torch.ones(2, 8) * 2)


def test_variable_length_collate_keeps_list() -> None:
    batch = [
        {"iq": torch.ones(2, 3), "length": 3, "dataset_id": 0},
        {"iq": torch.ones(2, 5), "length": 5, "dataset_id": 1},
    ]
    out = variable_length_collate(batch)
    assert isinstance(out["iq"], list)
    assert out["iq"][1].shape[-1] == 5


def test_variable_length_collate_stacks_equal_length() -> None:
    batch = [
        {"iq": torch.ones(2, 8), "length": 8, "dataset_id": 0},
        {"iq": torch.ones(2, 8) * 2, "length": 8, "dataset_id": 1},
    ]
    out = variable_length_collate(batch)
    assert torch.is_tensor(out["iq"])
    assert out["iq"].shape == (2, 2, 8)


def test_loader_worker_kwargs_empty_when_no_workers() -> None:
    assert rfdata_loader_worker_kwargs(0) == {}
    kwargs = rfdata_loader_worker_kwargs(4, prefetch_factor=6)
    assert kwargs["prefetch_factor"] == 6
    assert kwargs["persistent_workers"] is True
    assert callable(kwargs["worker_init_fn"])


def test_resolve_cache_iq_in_memory_flags() -> None:
    assert resolve_cache_iq_in_memory(False, nbytes=1024) is False
    assert resolve_cache_iq_in_memory("off", nbytes=1024) is False
    assert resolve_cache_iq_in_memory(True, nbytes=1024) is True
    assert resolve_cache_iq_in_memory("auto", nbytes=1024) is True


def test_iq_ram_cache_shared_by_path_and_matches_h5(tmp_path: Path) -> None:
    clear_iq_ram_cache()
    path = tmp_path / "a.h5"
    _write_h5(path)
    cached_a = RFDataH5Dataset(path, cache_iq_in_memory=True, iq_normalize="joint_power")
    cached_b = RFDataH5Dataset(path, cache_iq_in_memory=True, iq_normalize="joint_power")
    uncached = RFDataH5Dataset(path, cache_iq_in_memory=False, iq_normalize="joint_power")
    assert cached_a._iq_ram is cached_b._iq_ram
    stats = iq_ram_cache_stats()
    assert stats["files"] == 1
    assert "a.h5" in format_iq_ram_cache()
    for i in range(len(cached_a)):
        assert torch.equal(cached_a[i]["iq"], uncached[i]["iq"])
    clear_iq_ram_cache()


def test_pool_getitems_matches_getitem(tmp_path: Path) -> None:
    clear_iq_ram_cache()
    path_a = tmp_path / "a.h5"
    path_b = tmp_path / "b.h5"
    _write_h5(path_a, n=4, length=16)
    _write_h5(path_b, n=4, length=16)
    pool = RFDataPoolDataset(
        [
            RFDataH5Dataset(path_a, cache_iq_in_memory=True, iq_normalize="joint_power"),
            RFDataH5Dataset(path_b, cache_iq_in_memory=True, iq_normalize="joint_power"),
        ],
        pool_name="toy",
    )
    items = pool.__getitems__([0, 4, 1, 7])
    for idx, item in zip([0, 4, 1, 7], items, strict=True):
        assert torch.equal(item["iq"], pool[idx]["iq"])
        assert item["mod_label_id"] == pool[idx]["mod_label_id"]
    loader = torch.utils.data.DataLoader(pool, batch_size=4, collate_fn=pad_iq_collate)
    batch = next(iter(loader))
    assert batch["iq"].shape == (4, 2, 16)
    clear_iq_ram_cache()


def test_h5_getitems_without_ram_cache_matches_getitem(tmp_path: Path) -> None:
    clear_iq_ram_cache()
    path = tmp_path / "a.h5"
    _write_h5(path, n=8, length=16)
    ds = RFDataH5Dataset(path, cache_iq_in_memory=False)
    assert ds._iq_ram is None
    indices = [7, 0, 0, 3, 5]
    items = ds.__getitems__(indices)
    for idx, item in zip(indices, items, strict=True):
        assert torch.equal(item["iq"], ds[idx]["iq"])
        assert item["mod_label_id"] == ds[idx]["mod_label_id"]
    clear_iq_ram_cache()
