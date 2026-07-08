from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_datasets import Context, Writer, stamp_mod_label_ids


def test_stamp_mod_label_ids_from_source(tmp_path: Path) -> None:
    ctx = Context(tmp_path)
    ctx.maps["sources"]["demo"] = {"A": 0, "B": 1}
    path = ctx.h5 / "demo_test.h5"
    writer = Writer(path, length=8, dtype=np.float32, dataset_id=0, task_id=0, source=path)
    labels = np.array([0, 1, 0, 1], dtype=np.int32)
    iq = np.ones((4, 2, 8), dtype=np.float32)
    writer.append_clean(iq, __import__("collections").Counter(), source_label_id=labels)
    writer.close()

    stats = stamp_mod_label_ids(ctx, dataset_names=["demo"])
    assert stats == {"demo_test.h5": 4}
    assert ctx.maps["modulations"]["demo"] == {"A": 0, "B": 1}

    with h5py.File(path, "r") as f:
        assert f["mod_label_id"][:].tolist() == [0, 1, 0, 1]
