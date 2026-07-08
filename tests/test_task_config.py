from scripts.train_pipeline import merge_task_config


def test_merge_task_config_modulation_sampling() -> None:
    cfg = {
        "balanced_sampling_strategy": "length_bucket",
        "unfreeze_emitter_backbone": True,
        "task_configs": {
            "modulation": {
                "balanced_sampling_strategy": "length_bucket",
                "unfreeze_emitter_backbone": False,
                "learning_rate": 5e-5,
            },
            "clustering": {"balanced_sampling_strategy": "dataset"},
        },
    }
    merged = merge_task_config(cfg, stage="stage2", task="modulation")
    assert merged["balanced_sampling_strategy"] == "length_bucket"
    assert merged["unfreeze_emitter_backbone"] is False
    assert merged["learning_rate"] == 5e-5
    assert "task_configs" in merged


def test_merge_task_config_prediction_val_subset() -> None:
    cfg = {
        "val_subset_fraction": 0.2,
        "task_configs": {
            "prediction": {
                "val_subset_fraction": 0.05,
                "val_subset_max_per_dataset": 800,
                "val_length_bucket_batching": True,
            },
        },
    }
    merged = merge_task_config(cfg, stage="stage2", task="prediction")
    assert merged["val_subset_fraction"] == 0.05
    assert merged["val_subset_max_per_dataset"] == 800
    assert merged["val_length_bucket_batching"] is True


def test_merge_task_config_pretrain_unchanged() -> None:
    cfg = {"balanced_sampling_strategy": "length_bucket", "task_configs": {"modulation": {"balanced_sampling_strategy": "dataset"}}}
    merged = merge_task_config(cfg, stage="pretrain", task="modulation")
    assert merged["balanced_sampling_strategy"] == "length_bucket"
