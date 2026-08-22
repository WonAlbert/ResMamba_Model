from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from resmamba_signal_model.data.rfdata import (
    RFDataH5Dataset,
    RFDataPoolDataset,
    variable_length_collate,
)
from resmamba_signal_model.data.sampling import HomogeneousTokenBudgetSampler, pool_segments
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


def test_homogeneous_sampler_batch_has_unique_h5_stem(tmp_path: Path) -> None:
    paths = [
        tmp_path / "radar_emitters_train.h5",
        tmp_path / "xidian14_train.h5",
        tmp_path / "open_real_data_train.h5",
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
