from pathlib import Path

import pytest

from resmamba_signal_model.config import load_yaml_config
from resmamba_signal_model.models.model import SignalModelConfig
from resmamba_signal_model.training.lr_schedule import resolve_lr_schedule


def test_load_model_yaml() -> None:
    cfg = load_yaml_config("configs/model.yaml")
    model = SignalModelConfig.from_dict(cfg["model"])
    assert model.d_model == 640
    assert model.encoder_mamba_layers == 5
    assert model.encoder_transformer_layers == 1
    assert model.decoder_mamba_layers == 1
    assert model.sequence_packing is True
    assert model.encode_visible_only is True


def test_load_tiny_and_pretrain_profile() -> None:
    tiny = load_yaml_config("configs/model_tiny.yaml")
    assert tiny["model"]["d_model"] == 64
    assert tiny["model"]["allow_fallback_mamba"] is True
    pre = load_yaml_config("configs/pretrain.yaml", profile="tiny")
    assert pre["val_seed"] == 0
    assert pre["checkpoint_monitor"] == "val/monitor"
    assert pre["synthetic"] is True
    assert pre["model_config"] == "configs/model_tiny.yaml"
    assert pre["loss_weights"]["domain"] == 0.05
    assert pre["loss_weights"]["structure_phase"] == 0.02
    assert pre["warmup_steps"] == 1
    assert pre["mix_strategy"] == "token_share"


def test_downstream_yaml() -> None:
    cfg = load_yaml_config("configs/downstream.yaml")
    assert "classification" in cfg["task_pools"]
    assert "emitter" in cfg["task_pools"]
    assert cfg["lambda_recon"] == 0.1
    assert cfg["use_dataset_bias"] is False
    assert cfg["checkpoint_monitor"] == "val/multitask_geomean"
    assert cfg["checkpoint_mode"] == "max"
    assert cfg["seed"] == 0
    assert cfg["amp_dtype"] == "bfloat16"
    assert cfg["token_normalized_loss"] is True
    assert cfg["clustering_view2"] is True


def test_continual_yaml() -> None:
    cfg = load_yaml_config("configs/continual.yaml")
    assert cfg["continual"] is True
    assert cfg["distill_weight"] == 0.5
    assert cfg["prototype_anchor_weight"] == 0.1
    assert cfg["absorb_unknown"] is True
    assert cfg["checkpoint_monitor"] == "val/multitask_geomean"
    assert cfg["token_normalized_loss"] is True
    assert "emitter" in cfg["task_pools"]
    assert len(cfg["continual_sessions"]) >= 1


def test_stage_yaml_profiles() -> None:
    s2 = load_yaml_config("configs/stage2.yaml")
    assert s2["truncate_backward"] is True
    assert s2["train_encoder"] is False
    assert "emitter" in s2["task_pools"]
    assert s2["lambda_recon"] == 0.0
    assert s2["early_stopping_patience"] == 3
    assert s2["checkpoint_monitor"] == "val/multitask_geomean"
    assert s2["checkpoint_mode"] == "max"
    assert s2["seed"] == 0
    assert s2["amp_dtype"] == "bfloat16"
    assert s2["balanced_sampling"] is False
    tiny = load_yaml_config("configs/stage2.yaml", profile="tiny")
    assert tiny["synthetic"] is True
    s3 = load_yaml_config("configs/stage3.yaml", profile="modulation")
    assert s3["task"] == "modulation"
    assert s3["loraplus_lr_ratio"] == 16
    joint = load_yaml_config("configs/joint.yaml")
    assert joint["checkpoint_monitor"] == "val/specialist_geomean"
    assert joint["peft"]["shared_lora"] is True


def test_sota_gate_experiment_yaml_inherits_pretrain() -> None:
    cfg = load_yaml_config("configs/experiments/validity_pretrain.yaml")
    assert cfg["mix_strategy"] == "token_share"
    assert cfg["sota_gate"]["window"] == "A_validity"


def test_unknown_profile() -> None:
    with pytest.raises(ValueError, match="未知 profile"):
        load_yaml_config("configs/pretrain.yaml", profile="missing")


def test_resolve_lr_default() -> None:
    class Args:
        lr_schedule = None

    assert resolve_lr_schedule(Args(), {}) == "warmup_cosine"


def test_pretrain_recipe_allows_longer_training() -> None:
    cfg = load_yaml_config("configs/pretrain.yaml")
    assert cfg["epochs"] == 30
    assert cfg["early_stopping_patience"] == 8
    assert float(cfg["early_stopping_min_delta"]) == pytest.approx(0.001)
