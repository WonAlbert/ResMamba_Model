from __future__ import annotations

from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from resmamba_signal_model.data.rfdata import (
    RFDataH5Dataset,
    RFDataPoolDataset,
    variable_length_collate,
)
from resmamba_signal_model.data.sampling import (
    DEFAULT_FAMILY_QUOTAS,
    HomogeneousTokenBudgetSampler,
    build_homogeneous_sampling_plan,
    h5_dataset_stem,
    pool_segments,
    resolve_dataset_family,
)
from resmamba_signal_model.training.data_module import (
    is_pretrain_blocked_key,
    pretrain_collate_firewall,
)


def _write_h5(path: Path, *, n: int, length: int, dataset_id: int) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("iq", data=np.random.randn(n, 2, length).astype(np.float32))
        handle.create_dataset("length", data=np.full(n, length, dtype=np.int32))
        handle.create_dataset("dataset_id", data=np.full(n, dataset_id, dtype=np.int32))
        handle.create_dataset("task_type_id", data=np.zeros(n, dtype=np.int32))
        handle.create_dataset("mod_label_id", data=np.arange(n, dtype=np.int32) % 3)
        handle.create_dataset("emitter_id", data=np.arange(n, dtype=np.int32))


def _stem_of(idx: int, segments) -> str:
    for seg in segments:
        if seg.offset <= idx < seg.offset + seg.size:
            return Path(seg.h5_name).name
    raise IndexError(idx)


def test_resolve_dataset_family_and_wildcard() -> None:
    groups = {
        "tx_comm": ["rml2016_*", "xidian14"],
        "ld_radar": ["radchar", "radar_mod15"],
        "radcom": ["radcom_awgn"],
    }
    assert resolve_dataset_family("rml2016_10a", groups) == "tx_comm"
    assert resolve_dataset_family("rml2016_04c", groups) == "tx_comm"
    assert resolve_dataset_family("xidian14", groups) == "tx_comm"
    assert resolve_dataset_family("radchar", groups) == "ld_radar"
    assert resolve_dataset_family("radcom_dynamic", groups) is None
    assert h5_dataset_stem("rml2016_10b_train.h5") == "rml2016_10b"


def test_homogeneous_sampler_batch_has_unique_h5_stem(tmp_path: Path) -> None:
    paths = [
        tmp_path / "radchar_train.h5",
        tmp_path / "xidian14_train.h5",
        tmp_path / "radcom_awgn_train.h5",
    ]
    for i, path in enumerate(paths):
        _write_h5(path, n=16, length=32, dataset_id=i)
    pool = RFDataPoolDataset(
        [RFDataH5Dataset(path, use_labels=False) for path in paths],
        pool_name="pretrain_train",
    )
    sampler = HomogeneousTokenBudgetSampler(
        pool,
        token_budget=24,
        patch_size=8,
        num_batches=30,
        seed=7,
    )
    segments = pool_segments(pool)
    seen_stems: set[str] = set()
    for batch in sampler:
        stems = {_stem_of(idx, segments) for idx in batch}
        assert len(stems) == 1, stems
        seen_stems.update(stems)
        items = [pool[int(idx)] for idx in batch]
        collated = pretrain_collate_firewall(variable_length_collate(items))
        assert "h5_path" not in collated
        assert "dataset_id" not in collated
        assert not any(is_pretrain_blocked_key(key) for key in collated)
    assert seen_stems == {path.name for path in paths}


def test_homogeneous_sampler_does_not_need_file_id_in_samples(tmp_path: Path) -> None:
    path_a = tmp_path / "a_train.h5"
    path_b = tmp_path / "b_train.h5"
    _write_h5(path_a, n=8, length=16, dataset_id=1)
    _write_h5(path_b, n=8, length=16, dataset_id=2)
    pool = RFDataPoolDataset(
        [
            RFDataH5Dataset(path_a, use_labels=False, include_extra_metadata=False),
            RFDataH5Dataset(path_b, use_labels=False, include_extra_metadata=False),
        ],
        pool_name="toy",
    )
    sample = pool[0]
    assert "h5_path" not in sample
    assert "dataset_id" not in sample
    batch_idx = next(iter(HomogeneousTokenBudgetSampler(pool, token_budget=8, patch_size=8, num_batches=1, seed=0)))
    items = [pool[i] for i in batch_idx]
    collated = variable_length_collate(items)
    assert "h5_path" not in collated
    assert torch.is_tensor(collated["iq"]) or isinstance(collated["iq"], list)


def test_homogeneous_sampler_family_quota_distribution(tmp_path: Path) -> None:
    specs = [
        ("radchar_train.h5", 8, "ld_radar"),
        ("radar_mod15_train.h5", 8, "ld_radar"),
        ("xidian14_train.h5", 8, "tx_comm"),
        ("rml2016_10a_train.h5", 8, "tx_comm"),
        ("radcom_awgn_train.h5", 8, "radcom"),
        ("radcom_ota_train.h5", 8, "radcom"),
    ]
    paths = []
    for i, (name, n, _) in enumerate(specs):
        path = tmp_path / name
        _write_h5(path, n=n, length=32, dataset_id=i)
        paths.append(path)
    pool = RFDataPoolDataset(
        [RFDataH5Dataset(path, use_labels=False) for path in paths],
        pool_name="pretrain_train",
    )
    segments = pool_segments(pool)
    sampler = HomogeneousTokenBudgetSampler(
        pool,
        token_budget=16,
        patch_size=8,
        num_batches=6000,
        seed=123,
    )
    family_hits: Counter[str] = Counter()
    for batch in sampler:
        seg_i = next(i for i, seg in enumerate(segments) if seg.offset <= batch[0] < seg.offset + seg.size)
        family_hits[sampler.sampling_plan.segment_family[seg_i]] += 1
    total = sum(family_hits.values())
    for family, quota in DEFAULT_FAMILY_QUOTAS.items():
        observed = family_hits[family] / total
        assert observed == pytest.approx(quota, abs=0.05), (family, observed, quota, dict(family_hits))


def test_homogeneous_sampler_within_family_sqrt_size_weight(tmp_path: Path) -> None:
    small = tmp_path / "radchar_train.h5"
    large = tmp_path / "radar_mod15_train.h5"
    _write_h5(small, n=16, length=32, dataset_id=0)
    _write_h5(large, n=256, length=32, dataset_id=1)
    pool = RFDataPoolDataset(
        [RFDataH5Dataset(small, use_labels=False), RFDataH5Dataset(large, use_labels=False)],
        pool_name="ld_only",
    )
    segments = pool_segments(pool)
    plan = build_homogeneous_sampling_plan(segments)
    assert plan.families == ("ld_radar",)
    sampler = HomogeneousTokenBudgetSampler(
        pool,
        token_budget=16,
        patch_size=8,
        num_batches=4000,
        seed=99,
        source_groups={"ld_radar": ["radchar", "radar_mod15"]},
        family_quotas={"ld_radar": 1.0},
    )
    stem_hits: Counter[str] = Counter()
    for batch in sampler:
        stem_hits[h5_dataset_stem(_stem_of(batch[0], segments))] += 1
    small_share = stem_hits["radchar"] / sum(stem_hits.values())
    expected = np.sqrt(16) / (np.sqrt(16) + np.sqrt(256))
    assert small_share == pytest.approx(expected, abs=0.06)
