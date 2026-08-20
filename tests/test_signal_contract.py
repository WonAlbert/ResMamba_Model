from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from resmamba_signal_model.data.contracts import (
    MISSING_METADATA,
    SignalBatch,
    SignalSpec,
    collate_signal_batch,
)
from resmamba_signal_model.data.rfdata import RFDataH5Dataset, pad_iq_collate


def test_signal_spec_supports_arbitrary_channels_and_validates_pairs() -> None:
    imu = SignalSpec(
        num_channels=6,
        modality_id="imu",
        channel_names=("ax", "ay", "az", "gx", "gy", "gz"),
        sample_rate_hz=200.0,
    )
    assert imu.num_channels == 6
    assert imu.complex_pairs == ()
    assert SignalSpec.rf().complex_pairs == ((0, 1),)
    with pytest.raises(ValueError, match="超出"):
        SignalSpec(num_channels=1, complex_pairs=((0, 1),))


def test_collate_signal_batch_pads_channels_time_and_metadata() -> None:
    samples = [
        {
            "values": torch.tensor([[1.0, 2.0, 3.0]]),
            "length": 3,
            "modality_id": "sonar",
            "sample_rate_hz": 2.0,
            "receiver_id": "hydrophone-1",
        },
        {
            "values": torch.arange(30, dtype=torch.float32).reshape(6, 5),
            "length": 4,
            "modality_id": "imu",
            "session_id": "walk-7",
        },
    ]
    batch = collate_signal_batch(samples)
    assert batch.values.shape == (2, 6, 5)
    assert batch.channel_mask.tolist() == [
        [True, False, False, False, False, False],
        [True, True, True, True, True, True],
    ]
    assert batch.sample_mask[0].tolist() == [True, True, True, False, False]
    assert batch.sample_mask[1].tolist() == [True, True, True, True, False]
    assert torch.allclose(batch.time_coordinates[0, :3], torch.tensor([0.0, 0.5, 1.0]))
    assert batch.coordinate_unit == ("seconds", "sample_index")
    assert batch.metadata["receiver_id"] == ("hydrophone-1", MISSING_METADATA)
    assert batch.metadata["session_id"] == (MISSING_METADATA, "walk-7")

    legacy = batch.to_legacy_dict()
    assert legacy["iq"] is legacy["values"]
    restored = SignalBatch.from_legacy_dict(legacy)
    assert torch.equal(restored.values, batch.values)
    assert restored.complex_pairs == batch.complex_pairs


def test_pad_iq_collate_keeps_legacy_iq_for_mixed_channel_counts() -> None:
    out = pad_iq_collate([
        {"iq": torch.ones(2, 4), "length": 4, "dataset_id": 0},
        {"iq": torch.ones(3, 6), "length": 6, "dataset_id": 1},
    ])
    assert out["iq"].shape == (2, 3, 6)
    assert out["values"] is out["iq"]
    assert out["channel_mask"].tolist() == [[True, True, False], [True, True, True]]
    assert out["dataset_id"].tolist() == [0, 1]


def test_rfdata_preserves_capture_metadata_and_marks_missing(tmp_path: Path) -> None:
    path = tmp_path / "arbitrary_channels.h5"
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as handle:
        handle.attrs["channel_axis"] = 1
        handle.create_dataset("iq", data=np.ones((2, 3, 8), dtype=np.float32))
        handle.create_dataset("length", data=np.array([8, 6], dtype=np.int32))
        handle.create_dataset("dataset_id", data=np.array([4, 4], dtype=np.int32))
        handle.create_dataset("task_type_id", data=np.zeros(2, dtype=np.int8))
        handle.create_dataset(
            "receiver_id",
            data=np.asarray(["rx-a", MISSING_METADATA], dtype=object),
            dtype=string_dtype,
        )
        handle.create_dataset(
            "capture_id",
            data=np.asarray(["capture-a", MISSING_METADATA], dtype=object),
            dtype=string_dtype,
        )
    dataset = RFDataH5Dataset(path)
    assert dataset.num_channels == 3
    assert dataset.signal_spec.num_channels == 3
    assert dataset[0]["iq"].shape == (3, 8)
    assert dataset[0]["receiver_id"] == "rx-a"
    assert dataset[0]["capture_id"] == "capture-a"
    assert dataset[1]["receiver_id"] == MISSING_METADATA
    assert dataset[1]["session_id"] == MISSING_METADATA
