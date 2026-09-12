from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resmamba_signal_model.models.model import SignalFoundationModel, SignalModelConfig
from resmamba_signal_model.models.peft import PeftConfig, inject_hybrid_lora
from resmamba_signal_model.models.task_interface import DEFAULT_TASKS
from resmamba_signal_model.training.freeze import apply_stage_freeze


def _tiny(**kwargs) -> SignalFoundationModel:
    payload = dict(
        d_model=32,
        mamba_d_state=8,
        mamba_headdim=16,
        require_mamba_kernel=False,
        allow_fallback_mamba=True,
        attn_num_heads=4,
        patch_size=8,
        stem_channels=8,
        freq_bands=4,
        dropout=0.0,
        p_trunc=0.0,
        build_task_heads=True,
        build_task_interface=False,
        build_adapters=True,
        build_shared_adapter=True,
        task_names=DEFAULT_TASKS,
    )
    payload.update(kwargs)
    return SignalFoundationModel(SignalModelConfig(**payload))


def _trainable_names(model) -> set[str]:
    return {n for n, p in model.named_parameters() if p.requires_grad}


def test_stage2_freezes_backbone_trains_current_head() -> None:
    model = _tiny()
    apply_stage_freeze(
        model,
        "stage2",
        task="tx_modulation",
        train_cfg={"truncate_backward": True, "skip_recon": True},
    )
    names = _trainable_names(model)
    assert model.truncate_backward is True
    assert any(n.startswith("tx_modulation_head.") for n in names)
    assert any(n.startswith("z_linear_probes.tx_modulation.") for n in names)
    assert not any(n.startswith("ld_intrapulse_head.") for n in names)
    assert not any(n.startswith("task_interface.") for n in names)
    assert not any(n.startswith("encoder_pool.") for n in names)
    assert not any(n.startswith("encoder.") for n in names)
    assert not any(n.startswith("decoder.") for n in names)
    assert not any(n.startswith("tokenizer.") for n in names)
    assert not any(n.startswith("task_adapters.") for n in names)
    assert not any(n.startswith("shared_adapter.") for n in names)


def test_stage2_emits_z_probe_logits() -> None:
    model = _tiny()
    apply_stage_freeze(
        model,
        "stage2",
        task="tx_modulation",
        train_cfg={"truncate_backward": True, "skip_recon": True},
    )
    model.eval()
    out = model(
        {"iq": torch.randn(2, 2, 32), "sample_mask": torch.ones(2, 32, dtype=torch.bool)},
        mode="task",
        task="tx_modulation",
    )
    assert "z_probe_logits" in out
    assert out["z_probe_logits"].shape[0] == 2


def test_stage3_only_current_task() -> None:
    model = _tiny()
    inject_hybrid_lora(
        model,
        list(DEFAULT_TASKS),
        PeftConfig(r_attn=2, r_mamba=2, lora_alpha_attn=2, lora_alpha_mamba=2),
    )
    apply_stage_freeze(model, "stage3", task="ld_intrapulse", train_cfg={})
    names = _trainable_names(model)
    assert any("lora_A.ld_intrapulse" in n or "lora_B.ld_intrapulse" in n for n in names)
    assert not any("lora_A.tx_modulation" in n or "lora_B.tx_modulation" in n for n in names)
    assert any(n.startswith("task_adapters.ld_intrapulse.") for n in names)
    assert not any(n.startswith("task_adapters.tx_modulation.") for n in names)
    assert any(n.startswith("ld_intrapulse_head.") for n in names)
    assert not any(n.startswith("tx_modulation_head.") for n in names)
    assert not any(n.startswith("task_interface.") for n in names)


def test_stage3_prediction_trains_legacy_head_and_adapter() -> None:
    from scripts.train import load_train_bundle

    cfg = load_train_bundle("configs/stage3.yaml", profile="prediction", model_config="configs/model_tiny.yaml")
    model = _tiny(force_unified_generation=True, use_legacy_generation_heads=True)
    inject_hybrid_lora(
        model,
        ["prediction"],
        PeftConfig(r_attn=2, r_mamba=2, lora_alpha_attn=2, lora_alpha_mamba=2),
    )
    apply_stage_freeze(model, "stage3", task="prediction", train_cfg=cfg)
    names = _trainable_names(model)
    assert any(n.startswith("prediction_head.") for n in names)
    assert any(n.startswith("task_adapters.prediction.") for n in names)
    assert not any(n.startswith("decoder.") and "lora" not in n for n in names)
    model = _tiny()
    inject_hybrid_lora(
        model,
        list(DEFAULT_TASKS),
        PeftConfig(r_attn=2, r_mamba=2, lora_alpha_attn=2, lora_alpha_mamba=2, shared_lora=False),
    )
    apply_stage_freeze(model, "joint", train_cfg={})
    names = _trainable_names(model)
    assert any(n.startswith("shared_adapter.") for n in names)
    assert not any(n.startswith("tokenizer.") for n in names)
    assert not any("lora_A.shared" in n or "lora_B.shared" in n for n in names)
    assert any(n.startswith("task_adapters.ld_intrapulse.") for n in names)
    assert any(n.startswith("ld_intrapulse_head.") for n in names)
