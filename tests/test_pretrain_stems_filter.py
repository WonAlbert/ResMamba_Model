from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from resmamba_signal_model.data.rfdata import build_rfdata_pool, load_task_pool


def _write_h5(path: Path, *, n: int = 4, length: int = 32) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("iq", data=np.random.randn(n, 2, length).astype(np.float32))
        f.create_dataset("length", data=np.full(n, length, dtype=np.int32))


def test_load_task_pool_include_stems(tmp_path: Path) -> None:
    h5 = tmp_path / "h5"
    h5.mkdir()
    _write_h5(h5 / "xidian14_train.h5")
    _write_h5(h5 / "rml2016_04c_train.h5")
    _write_h5(h5 / "radchar_train.h5")
    maps = {
        "task_pools": {
            "pretrain_train": [
                "xidian14_train.h5",
                "rml2016_04c_train.h5",
                "radchar_train.h5",
            ]
        }
    }
    (tmp_path / "label_maps.json").write_text(json.dumps(maps), encoding="utf-8")
    names = load_task_pool(tmp_path, "pretrain_train", include_stems=["xidian14"])
    assert names == ["xidian14_train.h5"]
    pool = build_rfdata_pool(tmp_path, "pretrain_train", use_labels=False, include_stems=["rml2016_04c"])
    assert len(pool.datasets) == 1
    assert "rml2016_04c" in pool.pool_name
    with pytest.raises(ValueError, match="无文件"):
        load_task_pool(tmp_path, "pretrain_train", include_stems=["missing_stem"])
