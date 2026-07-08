from pathlib import Path

import torch

from resmamba_signal_model.training.emitter_labels import (
    build_global_emitter_label_map,
    filter_emitter_downstream_pool,
    global_emitter_labels,
    load_emitter_downstream_datasets,
)


def test_emitter_downstream_datasets_exclude_communication() -> None:
    datasets = load_emitter_downstream_datasets(config_path=Path("configs/emitter_downstream.yaml"))
    assert datasets == ["adsb2", "wifi150", "radar_emitters"]
    assert "communication_emitters" not in datasets


def test_filter_emitter_downstream_pool() -> None:
    files = [
        "adsb2_test.h5",
        "communication_emitters_test.h5",
        "wifi150_test.h5",
        "radar_emitters_test.h5",
    ]
    filtered = filter_emitter_downstream_pool(files, ["adsb2", "wifi150", "radar_emitters"])
    assert filtered == ["adsb2_test.h5", "radar_emitters_test.h5", "wifi150_test.h5"]


def test_global_emitter_label_map_counts() -> None:
    root = Path(__file__).resolve().parents[1] / "dataset"
    if not (root / "label_maps.json").is_file():
        return

    label_map = build_global_emitter_label_map(root)
    assert label_map.num_emitters == 260
    assert label_map.offsets[6] == 0
    assert label_map.offsets[7] == 100
    assert label_map.offsets[9] == 250
    assert 8 not in label_map.offsets


def test_global_emitter_labels_avoid_collision() -> None:
    lookup = torch.tensor([-1, -1, -1, -1, -1, -1, 0, 100, -1, 250], dtype=torch.long)
    dataset_id = torch.tensor([6, 7, 9], dtype=torch.long)
    emitter_id = torch.tensor([5, 5, 5], dtype=torch.long)
    labels = global_emitter_labels(dataset_id, emitter_id, lookup)
    assert labels.tolist() == [5, 105, 255]


def test_global_emitter_labels_invalid_dataset() -> None:
    lookup = torch.tensor([0, 100, -1, -1, -1, -1, -1, -1, -1, 250], dtype=torch.long)
    dataset_id = torch.tensor([8], dtype=torch.long)
    emitter_id = torch.tensor([1], dtype=torch.long)
    labels = global_emitter_labels(dataset_id, emitter_id, lookup)
    assert labels.tolist() == [-1]
