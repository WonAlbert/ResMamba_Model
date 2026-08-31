from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_datasets import (  # noqa: E402
    Context,
    ensure_missing_test_splits,
    finalize_task_pools,
)
from resmamba_signal_model.training.data_module import resolve_dataset_h5  # noqa: E402


def _write_h5(
    path: Path,
    *,
    n: int,
    task_id: int,
    dataset_id: int,
    field: str,
    labels: np.ndarray,
    value: float = 1.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("iq", data=np.full((n, 2, 16), value, dtype=np.float32))
        handle.create_dataset("length", data=np.full(n, 16, dtype=np.int32))
        handle.create_dataset("task_type_id", data=np.full(n, task_id, dtype=np.int32))
        handle.create_dataset("dataset_id", data=np.full(n, dataset_id, dtype=np.int32))
        handle.create_dataset(field, data=np.asarray(labels, dtype=np.int32))


def test_finalize_task_pools_maps_train_test_val(tmp_path: Path) -> None:
    h5 = tmp_path / "h5"
    h5.mkdir()
    mod_labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
    radar_labels = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)
    for split, value in (("train", 1.0), ("val", 2.0), ("test", 3.0)):
        _write_h5(
            h5 / f"rml2016_04c_{split}.h5",
            n=6,
            task_id=0,
            dataset_id=2,
            field="mod_label_id",
            labels=mod_labels,
            value=value,
        )
        _write_h5(
            h5 / f"rml2016_10a_{split}.h5",
            n=6,
            task_id=0,
            dataset_id=3,
            field="mod_label_id",
            labels=mod_labels,
            value=value,
        )
        _write_h5(
            h5 / f"radar_mod15_{split}.h5",
            n=6,
            task_id=0,
            dataset_id=10,
            field="mod_label_id",
            labels=radar_labels,
            value=value,
        )
        _write_h5(
            h5 / f"electromagnetic_0926_{split}.h5",
            n=6,
            task_id=0,
            dataset_id=0,
            field="mod_label_id",
            labels=mod_labels,
            value=value,
        )
        _write_h5(
            h5 / f"communication_emitters_{split}.h5",
            n=4,
            task_id=1,
            dataset_id=8,
            field="emitter_id",
            labels=np.arange(4, dtype=np.int32),
            value=value,
        )
    ctx = Context(tmp_path)
    finalize_task_pools(ctx)
    pools = ctx.maps["task_pools"]
    assert "rml2016_04c_train.h5" in pools["pretrain_train"]
    assert "radar_mod15_train.h5" in pools["pretrain_train"]
    assert "electromagnetic_0926_train.h5" not in pools["pretrain_train"]
    assert "communication_emitters_train.h5" not in pools["pretrain_train"]
    assert pools["downstream_comm_modulation_train"] == ["rml2016_04c_test.h5", "rml2016_10a_test.h5"]
    assert pools["downstream_comm_modulation_val"] == ["rml2016_04c_val.h5", "rml2016_10a_val.h5"]
    assert pools["downstream_radar_model_train"] == ["radar_mod15_test.h5"]
    assert pools["downstream_radar_model_val"] == ["radar_mod15_val.h5"]
    assert pools["downstream_modulation_train"] == ["rml2016_04c_test.h5", "rml2016_10a_test.h5"]
    assert pools["downstream_emitter_train"] == []
    assert "rml2016_04c_train.h5" not in pools["downstream_comm_modulation_train"]
    assert "radar_mod15_train.h5" not in pools["downstream_radar_model_train"]
    assert "rml2016_10a_test.h5" in pools["clustering_comm_train"]
    assert "rml2016_10a_val.h5" in pools["clustering_comm_val"]
    assert "radar_mod15_test.h5" in pools["clustering_radar_train"]
    assert "radar_mod15_val.h5" in pools["clustering_radar_val"]


def test_ensure_missing_test_splits_carves_only_missing(tmp_path: Path) -> None:
    h5 = tmp_path / "h5"
    h5.mkdir()
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32)
    _write_h5(
        h5 / "radar_mod15_train.h5",
        n=10,
        task_id=0,
        dataset_id=10,
        field="mod_label_id",
        labels=labels,
    )
    _write_h5(
        h5 / "radar_mod15_val.h5",
        n=4,
        task_id=0,
        dataset_id=10,
        field="mod_label_id",
        labels=np.array([0, 0, 1, 1], dtype=np.int32),
        value=9.0,
    )
    _write_h5(
        h5 / "wisig_train.h5",
        n=6,
        task_id=1,
        dataset_id=11,
        field="emitter_id",
        labels=np.arange(6, dtype=np.int32),
    )
    _write_h5(
        h5 / "wisig_val.h5",
        n=3,
        task_id=1,
        dataset_id=11,
        field="emitter_id",
        labels=np.arange(3, dtype=np.int32),
        value=2.0,
    )
    _write_h5(
        h5 / "wisig_test.h5",
        n=3,
        task_id=1,
        dataset_id=11,
        field="emitter_id",
        labels=np.array([10, 11, 12], dtype=np.int32),
        value=3.0,
    )
    val_before = (h5 / "radar_mod15_val.h5").read_bytes()
    wisig_test_before = (h5 / "wisig_test.h5").read_bytes()
    ctx = Context(tmp_path)
    stats = ensure_missing_test_splits(ctx, seed=20260822, frac=0.2)
    assert "radar_mod15" in stats
    assert "wisig" not in stats
    assert (h5 / "radar_mod15_test.h5").is_file()
    assert (h5 / "radar_mod15_val.h5").read_bytes() == val_before
    assert (h5 / "wisig_test.h5").read_bytes() == wisig_test_before
    with h5py.File(h5 / "radar_mod15_train.h5", "r") as train, h5py.File(
        h5 / "radar_mod15_test.h5", "r"
    ) as test:
        n_train = int(train["iq"].shape[0])
        n_test = int(test["iq"].shape[0])
        assert n_train + n_test == 10
        assert n_test >= 1
        assert n_train >= 1
        train_labels = np.asarray(train["mod_label_id"][:])
        test_labels = np.asarray(test["mod_label_id"][:])
        assert set(train_labels.tolist()) | set(test_labels.tolist()) == {0, 1}


def test_resolve_dataset_h5_eval_default_is_val(tmp_path: Path) -> None:
    h5_dir = tmp_path / "h5"
    h5_dir.mkdir()
    (h5_dir / "adsb2_val.h5").write_bytes(b"val")
    (h5_dir / "adsb2_test.h5").write_bytes(b"test")
    (h5_dir / "adsb2_train.h5").write_bytes(b"train")
    assert resolve_dataset_h5(tmp_path, "adsb2").name == "adsb2_val.h5"
    assert resolve_dataset_h5(tmp_path, "adsb2", "val").name == "adsb2_val.h5"
    assert resolve_dataset_h5(tmp_path, "adsb2", "test").name == "adsb2_test.h5"
