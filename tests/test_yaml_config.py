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
    assert model.clustering_num_prototypes == 15
    assert model.ld_clustering_num_prototypes == 15
    assert model.tx_clustering_num_prototypes == 11


def test_load_tiny_and_pretrain_profile() -> None:
    tiny = load_yaml_config("configs/model_tiny.yaml")
    assert tiny["model"]["d_model"] == 64
    assert tiny["model"]["allow_fallback_mamba"] is True
    assert tiny["model"]["clustering_num_prototypes"] == 11
    assert tiny["model"]["ld_clustering_num_prototypes"] == 15
    assert tiny["model"]["tx_clustering_num_prototypes"] == 11
    pre = load_yaml_config("configs/pretrain.yaml", profile="tiny")
    assert pre["val_seed"] == 0
    assert pre["checkpoint_monitor"] == "val/monitor"
    assert pre["synthetic"] is True
    assert pre["model_config"] == "configs/model_tiny.yaml"
    assert pre["loss_weights"]["mae"] == 1.0
    assert pre["loss_weights"]["structure_phase"] == 0.10
    assert pre["loss_weights"]["vicreg"] == 0.2
    assert pre.get("stem_normalized_loss") is True
    assert pre.get("chronos_sampling", {}).get("enabled") is True
    assert pre.get("homogeneous_stem_sticky_batches") == 48
    assert "vicreg_token" not in pre["loss_weights"]
    assert "tcl" not in pre["loss_weights"]
    assert pre["loss_weights"]["readout"] == 0.1
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
    assert s2["early_stopping_patience"] == 4
    assert float(s2["early_stopping_min_delta"]) == pytest.approx(0.001)
    assert s2["checkpoint_monitor"] == "val/classification_geomean"
    assert s2.get("composite_exclude_tasks") == []
    assert s2.get("use_legacy_generation_heads") is True
    assert int(s2.get("steps_per_epoch") or 0) == 300
    assert s2.get("clustering_view2") is False
    assert s2.get("cache_iq_in_memory") is True
    assert float(s2.get("learning_rate") or 0) == pytest.approx(2.0e-3)
    schedule = s2.get("task_schedule") or []
    pred = next(item for item in schedule if item.get("task") == "prediction")
    assert pred.get("eval_only") in (None, False)
    assert int(pred.get("epochs") or 0) >= 1
    assert s2.get("compact_task_labels") is True
    assert isinstance(s2.get("task_schedule"), list) and len(s2["task_schedule"]) >= 2
    assert int(s2.get("task_joint_epochs", 0)) == 0
    assert float(s2.get("cluster_utilization_weight", 0.0)) == pytest.approx(0.0)
    assert float(s2.get("cluster_balance_mix", 0.0)) == pytest.approx(0.0)
    tiny = load_yaml_config("configs/stage2.yaml", profile="tiny")
    assert tiny["synthetic"] is True
    s3 = load_yaml_config("configs/stage3.yaml", profile="tx_modulation")
    assert s3["task"] == "tx_modulation"
    assert s3["loraplus_lr_ratio"] == 16
    s3_clu = load_yaml_config("configs/stage3.yaml", profile="ld_clustering")
    assert s3_clu["checkpoint_monitor"] == "val/nmi_ld_clustering"
    joint = load_yaml_config("configs/joint.yaml")
    assert joint["checkpoint_monitor"] == "val/specialist_geomean"
    assert joint["peft"]["shared_lora"] is False
    assert joint.get("unfreeze_tokenizer_last") is False

def test_unknown_profile() -> None:
    with pytest.raises(ValueError, match="未知 profile"):
        load_yaml_config("configs/pretrain.yaml", profile="missing")


def test_resolve_lr_default() -> None:
    class Args:
        lr_schedule = None

    assert resolve_lr_schedule(Args(), {}) == "warmup_cosine"


def test_pretrain_recipe_allows_longer_training() -> None:
    cfg = load_yaml_config("configs/pretrain.yaml")
    assert cfg["epochs"] == 80
    assert cfg["within_family_weight"] == "equal"
    assert float(cfg["max_within_family_share"]) == pytest.approx(0.40)
    assert float(cfg["family_quotas"]["tx_comm"]) == pytest.approx(0.55)
    assert float(cfg["family_quotas"]["ld_radar"]) == pytest.approx(0.45)
    assert float(cfg["stem_sample_weights"]["rml2016_04c"]) == pytest.approx(2.5)
    assert "xidian14" not in cfg["stem_sample_weights"]
    assert "xidian14" not in cfg["source_groups"]["tx_comm"]
    assert cfg["early_stopping_patience"] == 12
    assert cfg["homogeneous_length_bucket"] is True
    assert cfg["homogeneous_length_bucket_weight"] == "proportional"
    probs = cfg["model"]["mae_mask_probs"]
    assert float(probs["suffix"]) == pytest.approx(0.0)
    assert float(probs["random"]) == pytest.approx(1.0)
    assert float(probs["contiguous"]) == pytest.approx(0.0)
    assert float(cfg["model"]["mask_ratio"]) == pytest.approx(0.30)
    assert cfg["model"]["adaptive_mae_mask"] is False
    unlock = cfg["adaptive_mae_unlock"]
    assert unlock["enabled"] is True
    assert "rml2016_04c" in unlock["hard_stems"]
    assert float(cfg["learning_rate"]) == pytest.approx(5.0e-5)
