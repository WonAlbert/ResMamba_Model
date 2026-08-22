from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resmamba_signal_model.models.model import SignalFoundationModel, SignalModelConfig
from resmamba_signal_model.models.peft import PeftConfig, inject_hybrid_lora
from resmamba_signal_model.training.freeze import apply_stage_freeze, iter_head_param_prefixes
from resmamba_signal_model.training.lit_module import SignalLitModule


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
        build_adapters=True,
        build_shared_adapter=True,
    )
    payload.update(kwargs)
    return SignalFoundationModel(SignalModelConfig(**payload))


def _trainable_names(model) -> set[str]:
    return {n for n, p in model.named_parameters() if p.requires_grad}


def test_stage2_freezes_backbone_trains_uti_heads() -> None:
    model = _tiny()
    apply_stage_freeze(model, "stage2", train_cfg={"truncate_backward": True, "skip_recon": True})
    names = _trainable_names(model)
    assert model.truncate_backward is True
    assert any(n.startswith("task_interface.") for n in names)
    assert any(n.startswith("modulation_head.") for n in names)
    assert any(n.startswith("emitter_head.") for n in names)
    assert any(n.startswith("z_linear_probes.") for n in names)
    assert any(n.startswith("encoder_pool.") for n in names)
    assert any(n.startswith("encoder_repr_norm.") for n in names)
    assert any(n.startswith("emitter_fingerprint.") for n in names)
    assert not any(n.startswith("encoder.") for n in names)
    assert not any(n.startswith("decoder.") for n in names)
    assert not any(n.startswith("tokenizer.") for n in names)
    assert not any(n.startswith("task_adapters.") for n in names)
    assert not any(n.startswith("shared_adapter.") for n in names)
    assert any(n.startswith("prediction_head.") for n in names)
    assert any(n.startswith("imputation_head.") for n in names)


def test_stage2_emits_z_probe_logits() -> None:
    model = _tiny()
    apply_stage_freeze(model, "stage2", train_cfg={"truncate_backward": True, "skip_recon": True})
    model.eval()
    out = model(
        {"iq": torch.randn(2, 2, 32), "sample_mask": torch.ones(2, 32, dtype=torch.bool)},
        mode="task",
        task="modulation",
    )
    assert "z_probe_logits" in out
    assert out["z_probe_logits"].shape[0] == 2
    assert out["z_probe_logits"].shape[1] == model.cfg.num_mod_classes


def test_stage3_only_current_task() -> None:
    model = _tiny()
    inject_hybrid_lora(model, ["modulation", "emitter"], PeftConfig(r_attn=2, r_mamba=2, lora_alpha_attn=2, lora_alpha_mamba=2))
    apply_stage_freeze(model, "stage3", task="modulation", train_cfg={})
    names = _trainable_names(model)
    assert any("lora_A.modulation" in n or "lora_B.modulation" in n for n in names)
    assert not any("lora_A.emitter" in n or "lora_B.emitter" in n for n in names)
    assert any(n.startswith("task_adapters.modulation.") for n in names)
    assert not any(n.startswith("task_adapters.emitter.") for n in names)
    assert any(n.startswith("modulation_head.") for n in names)
    assert not any(n.startswith("emitter_head.") for n in names)
    assert not any(n.startswith("emitter_fingerprint.") for n in names)
    assert not any(n.startswith("task_interface.") for n in names)


def test_stage3_emitter_trains_fingerprint_and_head() -> None:
    model = _tiny()
    inject_hybrid_lora(
        model,
        ["modulation", "emitter"],
        PeftConfig(r_attn=2, r_mamba=2, lora_alpha_attn=2, lora_alpha_mamba=2),
    )
    apply_stage_freeze(model, "stage3", task="emitter", train_cfg={})
    names = _trainable_names(model)
    assert any("lora_A.emitter" in n or "lora_B.emitter" in n for n in names)
    assert not any("lora_A.modulation" in n or "lora_B.modulation" in n for n in names)
    assert any(n.startswith("task_adapters.emitter.") for n in names)
    assert any(n.startswith("emitter_head.") for n in names)
    assert any(n.startswith("emitter_fingerprint.") for n in names)
    assert not any(n.startswith("modulation_head.") for n in names)
    assert not any(n.startswith("encoder.") and "lora_" not in n for n in names)
    model.train()
    out = model(
        {"iq": torch.randn(2, 2, 32), "sample_mask": torch.ones(2, 32, dtype=torch.bool)},
        mode="task",
        task="emitter",
    )
    assert "emitter_fingerprint" in out
    assert out["emitter_fingerprint"].shape == (2, model.cfg.d_model)
    assert out["emitter_fingerprint"].requires_grad
    assert "emitter_logits" in out
    out["emitter_logits"].sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.emitter_fingerprint.parameters())


def test_stage3_emitter_fingerprint_uses_heads_lr() -> None:
    assert "emitter_fingerprint" in iter_head_param_prefixes()
    model = _tiny()
    inject_hybrid_lora(
        model,
        ["modulation", "emitter"],
        PeftConfig(r_attn=2, r_mamba=2, lora_alpha_attn=2, lora_alpha_mamba=2),
    )
    apply_stage_freeze(model, "stage3", task="emitter", train_cfg={})
    lit = SignalLitModule(
        model,
        {"learning_rate": 2.0e-4, "heads_lr": 5.0e-5, "loraplus_lr_ratio": 16},
        stage="stage3",
    )
    groups = {g["name"]: g for g in lit._optimizer_param_groups(2.0e-4) if "name" in g}
    assert "heads" in groups
    fp_ids = {id(p) for p in model.emitter_fingerprint.parameters() if p.requires_grad}
    head_ids = {id(p) for p in groups["heads"]["params"]}
    assert fp_ids
    assert fp_ids <= head_ids
    assert groups["heads"]["lr"] == 5.0e-5


def test_joint_unfreezes_tokenizer_last_not_stem() -> None:
    model = _tiny()
    inject_hybrid_lora(
        model,
        ["modulation", "emitter"],
        PeftConfig(r_attn=2, r_mamba=2, lora_alpha_attn=2, lora_alpha_mamba=2, shared_lora=True),
    )
    apply_stage_freeze(model, "joint", train_cfg={})
    names = _trainable_names(model)
    assert any(n.startswith("tokenizer.time_fuse") for n in names)
    assert any(n.startswith("tokenizer.freq_proj") for n in names)
    assert not any(n.startswith("tokenizer.stem") for n in names)
    assert not any(n.startswith("tokenizer.time_branches") for n in names)
    assert any(n.startswith("shared_adapter.") for n in names)
    assert any(n.startswith("emitter_fingerprint.") for n in names)
    assert any("lora_A.shared" in n or "lora_B.shared" in n for n in names)
    assert any(n.startswith("encoder.") is False or "lora_" in n for n in names)
    assert not any(n.startswith("encoder.") and "lora_" not in n for n in names)
