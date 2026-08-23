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
    assert model.num_prototypes == 32
    assert model.clustering_num_prototypes == 128


def test_load_tiny_and_pretrain_profile() -> None:
    tiny = load_yaml_config("configs/model_tiny.yaml")
    assert tiny["model"]["d_model"] == 64
    assert tiny["model"]["allow_fallback_mamba"] is True
    assert tiny["model"]["clustering_num_prototypes"] == 16
    pre = load_yaml_config("configs/pretrain.yaml", profile="tiny")
    assert pre["val_seed"] == 0
    assert pre["checkpoint_monitor"] == "val/monitor"
    assert pre["synthetic"] is True
    assert pre["model_config"] == "configs/model_tiny.yaml"
    assert pre["loss_weights"]["domain"] == 0.0
    assert pre["loss_weights"]["structure_phase"] == 0.15
    assert pre["loss_weights"]["vicreg_token"] == 0.2
    assert pre["combine_then_pack"] is False
    assert pre["homogeneous_batch"] is True
    assert pre["warmup_steps"] == 1
    assert pre["mix_strategy"] == "token_share"


def test_stage_yaml_profiles() -> None:
    s2 = load_yaml_config("configs/stage2.yaml")
    assert s2["truncate_backward"] is True
    assert s2["train_encoder"] is False
    assert "tx_modulation" in s2["task_pools"]
    assert s2["lambda_recon"] == 0.0
    assert s2["early_stopping_patience"] == 0
    assert s2["checkpoint_monitor"] == "val/multitask_geomean"
    assert s2.get("compact_task_labels") is True
    assert isinstance(s2.get("task_schedule"), list) and len(s2["task_schedule"]) >= 2
    assert int(s2.get("task_joint_epochs", 0)) == 0
    assert float(s2.get("cluster_utilization_weight", 0.0)) == pytest.approx(0.15)
    tiny = load_yaml_config("configs/stage2.yaml", profile="tiny")
    assert tiny["synthetic"] is True
    s3 = load_yaml_config("configs/stage3.yaml", profile="tx_modulation")
    assert s3["task"] == "tx_modulation"
    assert s3["loraplus_lr_ratio"] == 16
    s3_clu = load_yaml_config("configs/stage3.yaml", profile="ld_clustering")
    assert s3_clu["checkpoint_monitor"] == "val/nmi_within_domain"
    joint = load_yaml_config("configs/joint.yaml")
    assert joint["checkpoint_monitor"] == "val/specialist_geomean"
    assert joint["peft"]["shared_lora"] is False
    assert joint.get("unfreeze_tokenizer_last") is False

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
