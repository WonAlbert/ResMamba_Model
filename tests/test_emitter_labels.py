from pathlib import Path

import torch

from resmamba_signal_model.training.emitter_labels import (
    build_emitter_dataset_class_mask,
    build_global_emitter_label_map,
    filter_emitter_downstream_pool,
    global_emitter_labels,
    load_emitter_downstream_datasets,
)


LEGACY_EMITTER_DATASETS = ["wisig", "adsb2"]


def test_emitter_downstream_datasets_empty_without_legacy_section() -> None:
    datasets = load_emitter_downstream_datasets(config_path=Path("configs/datasets.yaml"))
    assert datasets == []


def test_filter_emitter_downstream_pool() -> None:
    files = [
        "adsb2_train.h5",
        "communication_emitters_train.h5",
        "wifi150_train.h5",
        "radar_emitters_train.h5",
    ]
    filtered = filter_emitter_downstream_pool(files, ["adsb2", "wifi150"])
    assert filtered == ["adsb2_train.h5", "wifi150_train.h5"]


def test_global_emitter_label_map_counts() -> None:
    root = Path(__file__).resolve().parents[1] / "dataset"
    if not (root / "label_maps.json").is_file():
        return

    label_map = build_global_emitter_label_map(root, dataset_names=LEGACY_EMITTER_DATASETS)
    assert label_map.num_emitters == 250
    assert label_map.offsets[6] == 0
    assert label_map.offsets[11] == 100
    assert 7 not in label_map.offsets
    assert 9 not in label_map.offsets


def test_global_emitter_labels_avoid_collision() -> None:
    lookup = torch.tensor([-1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, 100], dtype=torch.long)
    dataset_id = torch.tensor([6, 11, 9], dtype=torch.long)
    emitter_id = torch.tensor([5, 5, 5], dtype=torch.long)
    labels = global_emitter_labels(dataset_id, emitter_id, lookup)
    assert labels.tolist() == [5, 105, -1]


def test_global_emitter_labels_cuda_index_cpu_lookup() -> None:
    """batch 在 GPU、lookup 在 CPU 时不应因设备不一致崩溃。"""
    if not torch.cuda.is_available():
        return
    lookup = torch.tensor([-1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, 100], dtype=torch.long)
    dataset_id = torch.tensor([6, 11], dtype=torch.long, device="cuda")
    emitter_id = torch.tensor([5, 5], dtype=torch.long, device="cuda")
    labels = global_emitter_labels(dataset_id, emitter_id, lookup)
    assert labels.device.type == "cuda"
    assert labels.tolist() == [5, 105]


def test_global_emitter_labels_invalid_dataset() -> None:
    lookup = torch.tensor([0, 100, -1, -1, -1, -1, -1, -1, -1, 250], dtype=torch.long)
    dataset_id = torch.tensor([8], dtype=torch.long)
    emitter_id = torch.tensor([1], dtype=torch.long)
    labels = global_emitter_labels(dataset_id, emitter_id, lookup)
    assert labels.tolist() == [-1]


def test_emitter_dataset_class_mask_wisig_adsb2() -> None:
    root = Path(__file__).resolve().parents[1] / "dataset"
    if not (root / "label_maps.json").is_file():
        return
    mask = build_emitter_dataset_class_mask(root, num_emitters=440, num_datasets=32)
    assert mask is not None
    assert mask.shape == (32, 440)
    assert int(mask[6].sum()) == 100
    assert int(mask[11].sum()) == 150
    assert not bool(torch.equal(mask[6], mask[11]))


def test_emitter_dataset_class_mask_compact() -> None:
    root = Path(__file__).resolve().parents[1] / "dataset"
    if not (root / "label_maps.json").is_file():
        return
    mask = build_emitter_dataset_class_mask(
        root,
        num_emitters=250,
        num_datasets=32,
        compact=True,
        train_cfg={"emitter_downstream_datasets": LEGACY_EMITTER_DATASETS},
    )
    assert mask is not None
    assert mask.shape == (32, 250)
    assert int(mask[6].sum()) == 100
    assert int(mask[11].sum()) == 150
    assert bool(mask[6, 0:100].all())
    assert bool(mask[11, 100:250].all())
