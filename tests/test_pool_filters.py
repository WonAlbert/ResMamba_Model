from resmamba_signal_model.training.pool_filters import (
    filter_excluded_dataset_pool,
    load_downstream_modulation_datasets,
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
        "electromagnetic_0926",
        "open_real_data",
        "radar_emitters",
        "xidian14",
        "panoradio_hf",
        "radcom_awgn",
        "radcom_dynamic",
        "radcom_ota",
        "cjr_mix",
    ]


def test_load_downstream_modulation_datasets() -> None:
    datasets = load_downstream_modulation_datasets()
    assert datasets == ["rml2016_04c", "rml2016_10a", "rml2016_10b", "rml2018_1a"]


def test_load_downstream_shared_datasets() -> None:
    datasets = load_downstream_shared_datasets()
    assert datasets == [
        "adsb2",
        "rml2016_04c",
        "rml2016_10a",
        "rml2016_10b",
        "rml2018_1a",
        "wisig",
    ]


def test_filter_excluded_dataset_pool() -> None:
    files = [
        "adsb2_val.h5",
        "communication_emitters_val.h5",
        "rml2018_1a_val.h5",
    ]
    filtered = filter_excluded_dataset_pool(files, ["communication_emitters"])
    assert filtered == ["adsb2_val.h5", "rml2018_1a_val.h5"]
