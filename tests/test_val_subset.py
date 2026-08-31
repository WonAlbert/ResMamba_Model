from resmamba_signal_model.data.rfdata import build_rfdata_pool
from resmamba_signal_model.data.val_subset import (
    _per_class_take_count,
    build_length_bucket_val_batches,
    build_per_dataset_class_balanced_val_indices,
    resolve_val_label_field,
)


def test_per_class_take_count() -> None:
    assert _per_class_take_count(1130, 0.2) == 226
    assert _per_class_take_count(80, 0.2) == 16
    assert _per_class_take_count(3, 0.2) == 1
    assert _per_class_take_count(100, 1.0) == 100


def test_resolve_val_label_field() -> None:
    assert resolve_val_label_field("pretrain", "prediction") == "auto"
    assert resolve_val_label_field("downstream", "emitter") == "global_emitter_id"
    assert resolve_val_label_field("downstream", "modulation") == "canonical_mod_label_id"
    assert resolve_val_label_field("downstream", "clustering") == "auto"
    assert resolve_val_label_field("downstream", "ld_clustering") == "auto"


def test_emitter_val_subset_per_dataset_and_class() -> None:
    import pytest

    try:
        pool = build_rfdata_pool("dataset", "downstream_emitter_val")
    except (KeyError, FileNotFoundError, RuntimeError, AssertionError):
        pytest.skip("downstream_emitter_val 未配置或为空")
    indices, report = build_per_dataset_class_balanced_val_indices(
        pool,
        stage="downstream",
        task="emitter",
        fraction=0.2,
        seed=7,
    )
    assert len(indices) == report["total"]
    assert report["stage"] == "downstream"
    assert report["task"] == "emitter"
    for name, info in report["datasets"].items():
        ratio = info["selected"] / info["total"]
        assert 0.15 <= ratio <= 0.25, f"{name} ratio={ratio:.3f}"
        for cls_info in info["per_class"].values():
            expected = _per_class_take_count(cls_info["total"], 0.2)
            assert cls_info["selected"] == expected


def test_prediction_val_subset_max_per_dataset() -> None:
    pool = build_rfdata_pool("dataset", "downstream_prediction_val")
    _, report = build_per_dataset_class_balanced_val_indices(
        pool,
        stage="downstream",
        task="prediction",
        fraction=0.2,
        seed=7,
        max_per_dataset=800,
    )
    assert report["max_per_dataset"] == 800
    for info in report["datasets"].values():
        assert info["selected"] <= 800


def test_radar_model_val_subset_per_dataset_and_class() -> None:
    import pytest

    try:
        pool = build_rfdata_pool("dataset", "downstream_radar_model_val")
    except (KeyError, FileNotFoundError, RuntimeError):
        pytest.skip("downstream_radar_model_val 未配置")
    if len(pool) == 0:
        pytest.skip("雷达型号下游 pool 为空")
    indices, report = build_per_dataset_class_balanced_val_indices(
        pool,
        stage="downstream",
        task="modulation",
        fraction=0.2,
        seed=7,
    )
    assert len(indices) == report["total"]
    assert report["task"] == "modulation"
    for name, info in report["datasets"].items():
        ratio = info["selected"] / info["total"]
        assert 0.15 <= ratio <= 0.25, f"{name} ratio={ratio:.3f}"


def test_radar_mod15_val_subset_uses_full_pool() -> None:
    pool = build_rfdata_pool("dataset", "downstream_radar_model_val")
    _, report = build_per_dataset_class_balanced_val_indices(
        pool,
        stage="downstream",
        task="modulation",
        fraction=0.2,
        seed=7,
        full_datasets=["radar_mod15"],
    )
    if "radar_mod15_val.h5" not in report["datasets"]:
        import pytest

        pytest.skip("radar_mod15 不在 downstream_radar_model_val")
    radar_info = report["datasets"]["radar_mod15_val.h5"]
    assert radar_info["selected"] == radar_info["total"]
    assert radar_info["fraction"] == 1.0
    assert radar_info["sampling_mode"] == "full"
    for name, info in report["datasets"].items():
        if name == "radar_mod15_val.h5":
            continue
        ratio = info["selected"] / info["total"]
        assert 0.15 <= ratio <= 0.25, f"{name} ratio={ratio:.3f}"


def test_length_bucket_val_batches_same_length_per_batch() -> None:
    pool = build_rfdata_pool("dataset", "downstream_prediction_val")
    indices, _ = build_per_dataset_class_balanced_val_indices(
        pool,
        stage="downstream",
        task="prediction",
        fraction=0.01,
        seed=3,
        max_per_dataset=50,
    )
    batches = build_length_bucket_val_batches(pool, indices, batch_size=16)
    assert batches
    length_by_idx: dict[int, int] = {}
    offset = 0
    for sub in pool.datasets:
        for local in range(len(sub)):
            length_by_idx[offset + local] = int(sub.signal_length)
        offset += len(sub)
    for batch in batches:
        lengths = {length_by_idx[idx] for idx in batch}
        assert len(lengths) == 1
