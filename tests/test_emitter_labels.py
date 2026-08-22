from pathlib import Path

import torch

from resmamba_signal_model.training.emitter_labels import (
    build_emitter_dataset_class_mask,
    build_global_emitter_label_map,
    filter_emitter_downstream_pool,
    global_emitter_labels,
    load_emitter_downstream_datasets,
)


def test_emitter_downstream_datasets_wisig_primary() -> None:
    datasets = load_emitter_downstream_datasets(config_path=Path("configs/datasets.yaml"))
    assert datasets[0] == "wisig"
    assert datasets == ["wisig", "adsb2"]
    assert "wifi150" not in datasets
    assert "communication_emitters" not in datasets
    assert "radar_emitters" not in datasets


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

    label_map = build_global_emitter_label_map(root)
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
