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


def _write_variable_h5(path: Path, lengths: list[int], dataset_id: int) -> None:
    max_l = max(lengths)
    n = len(lengths)
    iq = np.random.randn(n, 2, max_l).astype(np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("iq", data=iq)
        handle.create_dataset("length", data=np.asarray(lengths, dtype=np.int32))
        handle.create_dataset("dataset_id", data=np.full(n, dataset_id, dtype=np.int32))
        handle.create_dataset("task_type_id", data=np.zeros(n, dtype=np.int32))
        handle.create_dataset("mod_label_id", data=np.arange(n, dtype=np.int32) % 3)
        handle.create_dataset("emitter_id", data=np.arange(n, dtype=np.int32))


def test_homogeneous_sampler_length_bucket_within_h5(tmp_path: Path) -> None:
    path = tmp_path / "mixed_len_train.h5"
    lengths = [32, 32, 32, 64, 64, 128, 128]
    _write_variable_h5(path, lengths, dataset_id=0)
    pool = RFDataPoolDataset([RFDataH5Dataset(path, use_labels=False)], pool_name="toy")
    segments = pool_segments(pool)
    sampler = HomogeneousTokenBudgetSampler(
        pool,
        token_budget=16,
        patch_size=8,
        num_batches=40,
        seed=11,
        homogeneous_length_bucket=True,
    )
    for batch in sampler:
        assert len(batch) >= 1
        batch_lengths = {int(pool[i]["length"]) for i in batch}
        assert len(batch_lengths) == 1
        stem = _stem_of(batch[0], segments)
        assert stem == path.name

    flat = HomogeneousTokenBudgetSampler(
        pool,
        token_budget=16,
        patch_size=8,
        num_batches=40,
        seed=11,
        homogeneous_length_bucket=False,
    )
    mixed_seen = False
    for batch in flat:
        lens = {int(pool[i]["length"]) for i in batch}
        if len(lens) > 1:
            mixed_seen = True
            break
    assert mixed_seen


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


def test_homogeneous_sampler_stem_sticky_batches(tmp_path: Path) -> None:
    path_a = tmp_path / "a_train.h5"
    path_b = tmp_path / "b_train.h5"
    _write_h5(path_a, n=32, length=32, dataset_id=1)
    _write_h5(path_b, n=32, length=32, dataset_id=2)
    pool = RFDataPoolDataset(
        [RFDataH5Dataset(path, use_labels=False) for path in (path_a, path_b)],
        pool_name="toy",
    )
    segments = pool_segments(pool)
    sticky = 4
    sampler = HomogeneousTokenBudgetSampler(
        pool,
        token_budget=16,
        patch_size=8,
        num_batches=sticky * 2,
        seed=3,
        homogeneous_stem_sticky_batches=sticky,
    )
    batches = list(sampler)
    for block_start in range(0, len(batches), sticky):
        block = batches[block_start : block_start + sticky]
        stems = {_stem_of(idx, segments) for batch in block for idx in batch}
        assert len(stems) == 1


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


def test_homogeneous_sampler_within_family_equal_and_cap(tmp_path: Path) -> None:
    specs = [
        ("radchar_train.h5", 16),
        ("radar_mod15_train.h5", 256),
        ("cjr_mix_train.h5", 64),
        ("xidian14_train.h5", 512),
        ("rml2016_10a_train.h5", 32),
    ]
    paths = []
    for i, (name, n) in enumerate(specs):
        path = tmp_path / name
        _write_h5(path, n=n, length=32, dataset_id=i)
        paths.append(path)
    pool = RFDataPoolDataset(
        [RFDataH5Dataset(path, use_labels=False) for path in paths],
        pool_name="pretrain_train",
    )
    segments = pool_segments(pool)
    groups = {
        "ld_radar": ["radchar", "radar_mod15", "cjr_mix"],
        "tx_comm": ["rml2016_*", "xidian14"],
    }
    quotas = {"tx_comm": 0.55, "ld_radar": 0.45}
    plan = build_homogeneous_sampling_plan(
        segments,
        source_groups=groups,
        family_quotas=quotas,
        within_family_weight="equal",
        max_within_family_share=0.25,
    )
    assert plan.within_family_weight_mode == "equal"
    shares = plan.expected_stem_shares(segments)
    assert shares["radchar"] == pytest.approx(0.15, abs=1e-6)
    assert shares["radar_mod15"] == pytest.approx(0.15, abs=1e-6)
    assert shares["cjr_mix"] == pytest.approx(0.15, abs=1e-6)
    assert shares["xidian14"] == pytest.approx(0.275, abs=1e-6)
    assert shares["rml2016_10a"] == pytest.approx(0.275, abs=1e-6)

    # sqrt + cap(0.5)：大库从 ~57% 压到 50%，溢出补给仍低于 cap 的小库
    capped = build_homogeneous_sampling_plan(
        segments,
        source_groups={"ld_radar": ["radchar", "radar_mod15", "cjr_mix"]},
        family_quotas={"ld_radar": 1.0},
        within_family_weight="sqrt_size",
        max_within_family_share=0.5,
    )
    ld_shares = capped.expected_stem_shares(segments)
    assert ld_shares["radar_mod15"] == pytest.approx(0.5, abs=1e-6)
    assert ld_shares["radchar"] + ld_shares["cjr_mix"] == pytest.approx(0.5, abs=1e-6)
    assert ld_shares["radar_mod15"] < (np.sqrt(256) / (np.sqrt(16) + np.sqrt(256) + np.sqrt(64)))
    sampler = HomogeneousTokenBudgetSampler(
        pool,
        token_budget=16,
        patch_size=8,
        num_batches=5000,
        seed=3,
        source_groups=groups,
        family_quotas=quotas,
        within_family_weight="equal",
        max_within_family_share=0.25,
    )
    stem_hits: Counter[str] = Counter()
    for batch in sampler:
        stem_hits[h5_dataset_stem(_stem_of(batch[0], segments))] += 1
    total = sum(stem_hits.values())
    assert stem_hits["radar_mod15"] / total == pytest.approx(0.15, abs=0.05)
    assert stem_hits["xidian14"] / total == pytest.approx(0.275, abs=0.05)


def test_homogeneous_sampler_stem_sample_weights_upweight_hard(tmp_path: Path) -> None:
    specs = [
        ("rml2016_04c_train.h5", 64),
        ("xidian14_train.h5", 64),
        ("panoradio_hf_train.h5", 64),
        ("radchar_train.h5", 64),
        ("radar_mod15_train.h5", 64),
    ]
    paths = []
    for i, (name, n) in enumerate(specs):
        path = tmp_path / name
        _write_h5(path, n=n, length=32, dataset_id=i)
        paths.append(path)
    pool = RFDataPoolDataset(
        [RFDataH5Dataset(path, use_labels=False) for path in paths],
        pool_name="pretrain_train",
    )
    segments = pool_segments(pool)
    groups = {
        "ld_radar": ["radchar", "radar_mod15"],
        "tx_comm": ["rml2016_*", "xidian14", "panoradio_hf"],
    }
    quotas = {"tx_comm": 0.55, "ld_radar": 0.45}
    plan = build_homogeneous_sampling_plan(
        segments,
        source_groups=groups,
        family_quotas=quotas,
        within_family_weight="equal",
        max_within_family_share=0.5,
        stem_sample_weights={
            "rml2016_04c": 2.5,
            "panoradio_hf": 2.0,
            "xidian14": 0.25,
            "radchar": 2.0,
            "radar_mod15": 1.0,
        },
    )
    shares = plan.expected_stem_shares(segments)
    assert shares["rml2016_04c"] > shares["xidian14"]
    assert shares["panoradio_hf"] > shares["xidian14"]
    assert shares["rml2016_04c"] > shares["panoradio_hf"]
    # 有 max_share 时难易比会被裁剪，但仍应显著大于 1
    assert shares["rml2016_04c"] / shares["xidian14"] > 5.0
    assert shares["radchar"] == pytest.approx(shares["radar_mod15"], abs=1e-6)