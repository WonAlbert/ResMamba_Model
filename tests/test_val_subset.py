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
    assert resolve_val_label_field("stage2", "emitter") == "emitter_id"
    assert resolve_val_label_field("stage2", "modulation") == "mod_label_id"
    assert resolve_val_label_field("stage2", "clustering") == "global_label_id"


def test_emitter_val_subset_per_dataset_and_class() -> None:
    pool = build_rfdata_pool("dataset", "downstream_emitter_val")
    indices, report = build_per_dataset_class_balanced_val_indices(
        pool,
        stage="stage2",
        task="emitter",
        fraction=0.2,
        seed=7,
    )
    assert len(indices) == report["total"]
    assert report["stage"] == "stage2"
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
        stage="stage2",
        task="prediction",
        fraction=0.2,
        seed=7,
        max_per_dataset=800,
    )
    assert report["max_per_dataset"] == 800
    for info in report["datasets"].values():
        assert info["selected"] <= 800


def test_open_real_data_val_subset_uses_full_pool() -> None:
    pool = build_rfdata_pool("dataset", "downstream_modulation_val")
    _, report = build_per_dataset_class_balanced_val_indices(
        pool,
        stage="stage2",
        task="modulation",
        fraction=0.2,
        seed=7,
        full_datasets=["open_real_data"],
    )
    open_info = report["datasets"]["open_real_data_val.h5"]
    assert open_info["selected"] == open_info["total"]
    assert open_info["fraction"] == 1.0
    assert open_info["sampling_mode"] == "full"
    for name, info in report["datasets"].items():
        if name == "open_real_data_val.h5":
            continue
        ratio = info["selected"] / info["total"]
        assert 0.15 <= ratio <= 0.25, f"{name} ratio={ratio:.3f}"


def test_length_bucket_val_batches_same_length_per_batch() -> None:
    pool = build_rfdata_pool("dataset", "downstream_prediction_val")
    indices, _ = build_per_dataset_class_balanced_val_indices(
        pool,
        stage="stage2",
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
