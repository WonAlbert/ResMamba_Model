from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_datasets import infer_npy_labels, quality_mask


def test_infer_npy_labels_numeric(tmp_path: Path) -> None:
    root = tmp_path / "adsb"
    root.mkdir()
    np.save(root / "Y_train.npy", np.array([0, 1, 99, 99], dtype=np.int64))
    np.save(root / "Y_val.npy", np.array([50, 50], dtype=np.int64))
    labels = infer_npy_labels(root, [("train", "train"), ("val", "val")], None)
    assert labels == {"0": 0, "1": 1, "50": 50, "99": 99}


def test_quality_mask_per_class() -> None:
    iq = np.ones((4, 2, 8), dtype=np.float32)
    iq[0] *= 1e-6
    iq[1] *= 1.0
    iq[2] *= 1e-6
    iq[3] *= 1.0
    labels = np.array([0, 0, 1, 1], dtype=np.int32)
    keep, _ = quality_mask(iq, labels=labels)
    assert keep.tolist() == [True, True, True, True]


def test_quality_mask_strict_is_subset() -> None:
    rng = np.random.default_rng(0)
    iq = rng.standard_normal((64, 2, 128)).astype(np.float32)
    labels = np.repeat(np.arange(4), 16).astype(np.int32)
    keep_loose, _ = quality_mask(iq, labels=labels, strict=False)
    keep_strict, _ = quality_mask(iq, labels=labels, strict=True)
    assert keep_strict.sum() <= keep_loose.sum()
    assert np.all(keep_loose[keep_strict])
