from resmamba_signal_model.training.pool_filters import (
    filter_excluded_dataset_pool,
    load_clustering_comm_datasets,
    load_clustering_radar_datasets,
    load_downstream_comm_modulation_datasets,
    load_downstream_radar_modulation_datasets,
    load_downstream_radar_model_datasets,
    load_downstream_shared_datasets,
    load_excluded_datasets,
    load_pretrain_datasets,
)


def test_load_excluded_datasets() -> None:
    excluded = load_excluded_datasets()
    assert excluded == ["communication_emitters"]


def test_load_pretrain_datasets() -> None:
    datasets = load_pretrain_datasets()
    assert datasets == [
        "radar_mod15",
        "radchar",
        "cjr_mix",
        "rml2016_04c",
        "rml2016_10a",
        "rml2016_10b",
        "xidian14",
        "panoradio_hf",
    ]
    assert "radchar" in datasets
    assert "communication_emitters" not in datasets
    assert "electromagnetic_0926" not in datasets
    assert "wisig" not in datasets
    assert "adsb2" not in datasets
    assert "radar_emitters" not in datasets
    assert "rml2018_1a" not in datasets
    assert "wifi150" not in datasets


def test_load_downstream_comm_modulation_datasets() -> None:
    datasets = load_downstream_comm_modulation_datasets()
    assert datasets == ["rml2016_04c", "rml2016_10a", "rml2016_10b"]


def test_load_downstream_radar_model_datasets() -> None:
    datasets = load_downstream_radar_model_datasets()
    assert datasets == ["radar_mod15", "cjr_mix"]


def test_load_downstream_radar_modulation_datasets() -> None:
    datasets = load_downstream_radar_modulation_datasets()
    assert datasets == ["radchar"]


def test_load_clustering_pools() -> None:
    radar = load_clustering_radar_datasets()
    comm = load_clustering_comm_datasets()
    assert radar == ["radar_mod15", "cjr_mix"]
    assert comm == ["rml2016_10a", "xidian14"]


def test_load_downstream_shared_datasets() -> None:
    datasets = load_downstream_shared_datasets()
    assert datasets == [
        "cjr_mix",
        "radar_mod15",
        "rml2016_04c",
        "rml2016_10a",
        "rml2016_10b",
        "xidian14",
    ]


def test_filter_excluded_dataset_pool() -> None:
    files = [
        "adsb2_val.h5",
        "communication_emitters_val.h5",
        "rml2018_1a_val.h5",
    ]
    filtered = filter_excluded_dataset_pool(files, ["communication_emitters"])
    assert filtered == ["adsb2_val.h5", "rml2018_1a_val.h5"]
