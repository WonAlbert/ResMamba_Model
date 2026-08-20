from __future__ import annotations

from pathlib import Path
import json

import h5py
import numpy as np
import pytest
import torch

from resmamba_signal_model.data.contracts import SignalBatch, collate_signal_batch
from resmamba_signal_model.data.labels import build_modulation_ontology
from resmamba_signal_model.data.rfdata import RFDataH5Dataset
from resmamba_signal_model.data.splits import build_split_manifest

ROOT = Path(__file__).resolve().parents[1]
H5_DIR = ROOT / "dataset" / "h5"
SLICE_DATASET = "rml2016_04c"
SLICE_FILES = tuple(f"{SLICE_DATASET}_{split}.h5" for split in ("train", "val", "test"))


def _require_slice_files() -> Path:
    missing = [name for name in SLICE_FILES if not (H5_DIR / name).is_file()]
    if missing:
        pytest.skip(f"缺少真实 H5 切片: {missing}")
    return H5_DIR


def _iq_fingerprints(path: Path) -> set[bytes]:
    with h5py.File(path, "r") as handle:
        iq = np.asarray(handle["iq"][:], dtype=np.float32)
    return {row.tobytes() for row in iq}


def test_real_h5_loads_signal_batch_canonical_labels_and_disjoint_splits(tmp_path: Path) -> None:
    h5_dir = _require_slice_files()
    train_path = h5_dir / f"{SLICE_DATASET}_train.h5"
    dataset = RFDataH5Dataset(train_path)
    samples = [dataset[i] for i in range(min(8, len(dataset)))]
    batch = collate_signal_batch(samples)
    assert isinstance(batch, SignalBatch)
    assert batch.values.ndim == 3
    assert batch.values.shape[0] == len(samples)
    assert batch.values.shape[1] == 2
    assert batch.sample_mask.dtype == torch.bool
    assert bool(batch.sample_mask.any())
    assert batch.iq is batch.values

    canonical = [int(sample["canonical_mod_label_id"]) for sample in samples]
    local = [int(sample["mod_label_id"]) for sample in samples]
    assert all(value >= 0 for value in canonical)
    assert len(set(canonical)) >= 1

    maps_path = ROOT / "dataset" / "label_maps.json"
    maps = json.loads(maps_path.read_text(encoding="utf-8"))
    ontology = build_modulation_ontology(maps["modulations"])
    expected = ontology.map_local(SLICE_DATASET, np.asarray(local, dtype=np.int32))
    assert canonical == expected.tolist()
    qam16 = ontology.canonical_to_id["16QAM"]
    local_qam16 = int(maps["modulations"][SLICE_DATASET]["QAM16"])
    assert ontology.map_local(SLICE_DATASET, np.asarray([local_qam16]))[0] == qam16

    staged = tmp_path / "h5"
    staged.mkdir()
    for name in SLICE_FILES:
        (staged / name).symlink_to(h5_dir / name)
    payload = build_split_manifest(staged)
    hashes = [item["sha256"] for item in payload["datasets"][SLICE_DATASET]["files"]]
    assert len(hashes) == 3
    assert len(set(hashes)) == 3
    integrity = payload["datasets"][SLICE_DATASET]["group_integrity"]
    assert integrity["status"].startswith("unverifiable")
    assert integrity.get("claim_group_held_out") is not True

    fingerprints = {split: _iq_fingerprints(h5_dir / f"{SLICE_DATASET}_{split}.h5") for split in ("train", "val", "test")}
    assert not (fingerprints["train"] & fingerprints["val"])
    assert not (fingerprints["train"] & fingerprints["test"])
    assert not (fingerprints["val"] & fingerprints["test"])
