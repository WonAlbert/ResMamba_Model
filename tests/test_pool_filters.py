from resmamba_signal_model.training.pool_filters import (
    filter_excluded_dataset_pool,
    load_downstream_excluded_datasets,
    load_downstream_modulation_extra_datasets,
    load_excluded_datasets,
)


def test_load_excluded_datasets() -> None:
    excluded = load_excluded_datasets(config_path="configs/excluded_datasets.yaml")
    assert excluded == ["communication_emitters"]


def test_load_downstream_excluded_datasets() -> None:
    excluded = load_downstream_excluded_datasets(config_path="configs/downstream_excluded_datasets.yaml")
    assert excluded == ["xidian14"]


def test_load_downstream_modulation_extra_datasets() -> None:
    extra = load_downstream_modulation_extra_datasets(config_path="configs/downstream_modulation_extra_datasets.yaml")
    assert extra == ["open_real_data", "electromagnetic_0926"]


def test_filter_excluded_dataset_pool() -> None:
    files = [
        "adsb2_val.h5",
        "communication_emitters_val.h5",
        "rml2018_1a_val.h5",
    ]
    filtered = filter_excluded_dataset_pool(files, ["communication_emitters"])
    assert filtered == ["adsb2_val.h5", "rml2018_1a_val.h5"]


def test_downstream_pool_excludes_xidian14_keeps_open_real() -> None:
    files = [
        "xidian14_test.h5",
        "open_real_data_test.h5",
        "electromagnetic_0926_test.h5",
        "rml2018_1a_test.h5",
    ]
    filtered = filter_excluded_dataset_pool(files, ["xidian14"])
    assert filtered == [
        "electromagnetic_0926_test.h5",
        "open_real_data_test.h5",
        "rml2018_1a_test.h5",
    ]
