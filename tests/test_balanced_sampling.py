from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from resmamba_signal_model.data.rfdata import RFDataH5Dataset, RFDataPoolDataset
from resmamba_signal_model.data.sampling import (
    LengthBucketBalancedBatchSampler,
    LengthBucketPKBatchSampler,
    build_dataset_balanced_sampler,
    build_segment_class_indices,
    build_uniform_sampler,
    format_sampling_plan,
    pool_segments,
    resolve_balanced_sampling_strategy,
    resolve_pk_sampling_params,
)


@dataclass
class _FakeSubDataset:
    h5_path: Path
    _len: int
    signal_length: int

    def __len__(self) -> int:
        return self._len


class _FakePool:
    def __init__(self, datasets: list[_FakeSubDataset], pool_name: str = "test") -> None:
        self.datasets = datasets
        self.pool_name = pool_name

    def __len__(self) -> int:
        return sum(len(d) for d in self.datasets)


def test_resolve_balanced_sampling_strategy() -> None:
    assert resolve_balanced_sampling_strategy(enabled=False, strategy=None) == "none"
    assert resolve_balanced_sampling_strategy(enabled=True, strategy=None) == "length_bucket"
    assert resolve_balanced_sampling_strategy(enabled=True, strategy="dataset") == "dataset"
    assert resolve_balanced_sampling_strategy(enabled=True, strategy="uniform") == "uniform"
    assert resolve_balanced_sampling_strategy(enabled=True, strategy="length_bucket") == "length_bucket"
    assert resolve_balanced_sampling_strategy(enabled=True, strategy="length_bucket_class") == "length_bucket_class"
    assert resolve_balanced_sampling_strategy(enabled=True, strategy="length_bucket_pk") == "length_bucket_pk"
    assert (
        resolve_balanced_sampling_strategy(enabled=True, strategy="length_bucket_proportional")
        == "length_bucket_proportional"
    )


def test_pool_segments_offsets() -> None:
    pool = _FakePool(
        [
            _FakeSubDataset(Path("a.h5"), 100, 128),
            _FakeSubDataset(Path("b.h5"), 50, 2048),
        ]
    )
    segments = pool_segments(pool)  # type: ignore[arg-type]
    assert len(segments) == 2
    assert segments[0].offset == 0 and segments[0].size == 100 and segments[0].signal_length == 128
    assert segments[1].offset == 100 and segments[1].size == 50 and segments[1].signal_length == 2048


def test_length_bucket_batch_sampler_same_length_in_batch() -> None:
    pool = _FakePool(
        [
            _FakeSubDataset(Path("short_a.h5"), 1000, 128),
            _FakeSubDataset(Path("short_b.h5"), 2000, 128),
            _FakeSubDataset(Path("long_a.h5"), 500, 2048),
            _FakeSubDataset(Path("long_b.h5"), 800, 2048),
        ]
    )
    offset = 0
    index_to_length: dict[int, int] = {}
    for sub in pool.datasets:
        for i in range(len(sub)):
            index_to_length[offset + i] = sub.signal_length
        offset += len(sub)

    sampler = LengthBucketBalancedBatchSampler(pool, batch_size=16, num_batches=20, seed=42)  # type: ignore[arg-type]
    assert len(sampler) == 20
    assert len(sampler.length_buckets) == 2

    for batch in sampler:
        batch_lengths = {index_to_length[idx] for idx in batch}
        assert len(batch_lengths) == 1
        assert len(batch) == 16


def test_dataset_balanced_sampler_covers_all_indices() -> None:
    pool = _FakePool([_FakeSubDataset(Path("a.h5"), 10, 128), _FakeSubDataset(Path("b.h5"), 20, 128)])
    sampler = build_dataset_balanced_sampler(pool)  # type: ignore[arg-type]
    assert sampler.num_samples == 30


def test_uniform_sampler_equal_weights() -> None:
    pool = _FakePool([_FakeSubDataset(Path("a.h5"), 10, 128), _FakeSubDataset(Path("b.h5"), 20, 256)])
    sampler = build_uniform_sampler(pool)  # type: ignore[arg-type]
    assert sampler.num_samples == 30
    assert torch.allclose(sampler.weights, torch.ones(30, dtype=torch.double))


def test_length_bucket_within_bucket_dataset_equal() -> None:
    pool = _FakePool(
        [
            _FakeSubDataset(Path("small.h5"), 100, 128),
            _FakeSubDataset(Path("large.h5"), 400, 128),
        ]
    )
    sampler = LengthBucketBalancedBatchSampler(pool, batch_size=32, num_batches=5000, seed=0)  # type: ignore[arg-type]
    counts = {"small.h5": 0, "large.h5": 0}
    for batch in sampler:
        for idx in batch:
            if idx < 100:
                counts["small.h5"] += 1
            else:
                counts["large.h5"] += 1
    total = sum(counts.values())
    ratio_small = counts["small.h5"] / total
    ratio_large = counts["large.h5"] / total
    assert 0.45 <= ratio_small <= 0.55
    assert 0.45 <= ratio_large <= 0.55


def test_format_sampling_plan_mentions_strategy() -> None:
    pool = _FakePool([_FakeSubDataset(Path("a.h5"), 10, 128)])
    text = format_sampling_plan(pool, "length_bucket")  # type: ignore[arg-type]
    assert "balanced_sampling_strategy=length_bucket" in text
    assert "桶间" in text


def test_length_bucket_class_weight_distribution() -> None:
    pool = _FakePool(
        [
            _FakeSubDataset(Path("adsb2_test.h5"), 11300, 1200),
            _FakeSubDataset(Path("wifi150_test.h5"), 23700, 256),
            _FakeSubDataset(Path("radar_emitters_test.h5"), 800, 1000),
        ]
    )
    sampler = LengthBucketBalancedBatchSampler(
        pool,
        batch_size=64,
        num_batches=5000,
        seed=42,
        bucket_weight_mode="class_count",
        infer_emitter_classes=True,
    )  # type: ignore[arg-type]
    counts = {"adsb2_test.h5": 0, "wifi150_test.h5": 0, "radar_emitters_test.h5": 0}
    bounds = [11300, 11300 + 23700]
    for batch in sampler:
        for idx in batch:
            if idx < bounds[0]:
                counts["adsb2_test.h5"] += 1
            elif idx < bounds[1]:
                counts["wifi150_test.h5"] += 1
            else:
                counts["radar_emitters_test.h5"] += 1
    total = sum(counts.values())
    shares = {k: v / total for k, v in counts.items()}
    assert 0.34 <= shares["adsb2_test.h5"] <= 0.42
    assert 0.52 <= shares["wifi150_test.h5"] <= 0.62
    assert 0.02 <= shares["radar_emitters_test.h5"] <= 0.06


def _write_emitter_h5(path: Path, num_samples: int, signal_length: int, num_classes: int) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("iq", data=np.zeros((num_samples, 2, signal_length), dtype=np.float32))
        f.create_dataset("emitter_id", data=(np.arange(num_samples) % num_classes).astype(np.int32))


def _build_real_pool(tmp_path: Path) -> RFDataPoolDataset:
    h5_dir = tmp_path / "h5"
    h5_dir.mkdir()
    paths = [
        h5_dir / "adsb2_test.h5",
        h5_dir / "wifi150_test.h5",
        h5_dir / "radar_emitters_test.h5",
    ]
    _write_emitter_h5(paths[0], 200, 1200, 20)
    _write_emitter_h5(paths[1], 300, 256, 30)
    _write_emitter_h5(paths[2], 120, 1000, 5)
    return RFDataPoolDataset(
        [RFDataH5Dataset(path) for path in paths],
        pool_name="test_emitter",
    )


def test_resolve_pk_sampling_params() -> None:
    p, k = resolve_pk_sampling_params({"pk_num_classes": 64, "pk_samples_per_class": 8}, 512)
    assert (p, k) == (64, 8)
    p, k = resolve_pk_sampling_params({}, 512)
    assert p * k == 512


def test_build_segment_class_indices_real(tmp_path: Path) -> None:
    pool = _build_real_pool(tmp_path)
    indices = build_segment_class_indices(pool)
    assert len(indices) == 3
    assert sum(len(v) for v in indices[0].values()) == 200


def _emitter_label_at(pool: RFDataPoolDataset, idx: int) -> int:
    offset = 0
    for sub in pool.datasets:
        if offset <= idx < offset + len(sub):
            local = idx - offset
            with h5py.File(sub.h5_path, "r") as f:
                return int(f["emitter_id"][local])
        offset += len(sub)
    raise IndexError(idx)


def test_length_bucket_pk_batch_sampler(tmp_path: Path) -> None:
    pool = _build_real_pool(tmp_path)
    index_to_length: dict[int, int] = {}
    offset = 0
    for sub in pool.datasets:
        for i in range(len(sub)):
            index_to_length[offset + i] = sub.signal_length
        offset += len(sub)

    sampler = LengthBucketPKBatchSampler(
        pool,
        batch_size=16,
        pk_num_classes=4,
        pk_samples_per_class=4,
        num_batches=30,
        seed=0,
    )
    for batch in sampler:
        assert len(batch) == 16
        batch_lengths = {index_to_length[idx] for idx in batch}
        assert len(batch_lengths) == 1
        by_label: dict[int, int] = {}
        for idx in batch:
            label = _emitter_label_at(pool, idx)
            by_label[label] = by_label.get(label, 0) + 1
        assert len(by_label) == 4
        assert all(count == 4 for count in by_label.values())


def test_format_sampling_plan_pk() -> None:
    pool = _FakePool([_FakeSubDataset(Path("a.h5"), 10, 128)])
    text = format_sampling_plan(pool, "length_bucket_pk")  # type: ignore[arg-type]
    assert "length_bucket_pk" in text
    assert "PK" in text
