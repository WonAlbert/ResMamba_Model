from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from resmamba_signal_model.models.model import SignalFoundationModel, SignalModelConfig
from resmamba_signal_model.models.peft import (
    MultiTaskLoRALinear,
    PeftConfig,
    classify_lora_target,
    inject_hybrid_lora,
    peft_state_dict,
    save_best_bundle,
)


def _tiny_model(**kwargs) -> SignalFoundationModel:
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
    )
    payload.update(kwargs)
    return SignalFoundationModel(SignalModelConfig(**payload))


def test_classify_whitelist() -> None:
    assert classify_lora_target("encoder.mamba_layers.0.fwd.in_proj") == "mamba"
    assert classify_lora_target("encoder.mamba_layers.0.fuse") == "mamba"
    assert classify_lora_target("encoder.transformer_layers.0.qkv") == "attn"
    assert classify_lora_target("encoder.transformer_layers.0.ffn.0") == "attn"
    assert classify_lora_target("decoder.blocks.0.ffn.w1") == "mamba"
    assert classify_lora_target("encoder.mamba_layers.0.fwd.A_log") is None
    assert classify_lora_target("tokenizer.freq_proj") is None
    assert classify_lora_target("decoder.repr_head.attn.out_proj") is None
    assert classify_lora_target("emitter_fingerprint.stat_mlp.1") is None


def test_inject_wraps_projections_and_freezes_w0() -> None:
    model = _tiny_model()
    handle = inject_hybrid_lora(model, ["modulation", "emitter"], PeftConfig(r_attn=4, r_mamba=2, lora_alpha_attn=4, lora_alpha_mamba=2))
    wrapped = [n for n, m in model.named_modules() if isinstance(m, MultiTaskLoRALinear)]
    assert handle.names
    assert any(n.endswith("in_proj") for n in wrapped)
    assert any(n.endswith("qkv") for n in wrapped)
    assert any(n.endswith("fuse") for n in wrapped)
    for module in model.modules():
        if isinstance(module, MultiTaskLoRALinear):
            assert module.weight.requires_grad is False
            assert "modulation" in module.lora_A
            assert "emitter" in module.lora_A


def test_active_task_delta_and_shared() -> None:
    linear = nn.Linear(8, 8, bias=False)
    nn.init.ones_(linear.weight)
    wrapped = MultiTaskLoRALinear(linear, ["modulation", "shared"], r=2, alpha=2.0)
    with torch.no_grad():
        wrapped.lora_B["modulation"].fill_(0.1)
        wrapped.lora_B["shared"].fill_(0.05)
    x = torch.ones(2, 8)
    wrapped.set_active_task(None)
    y0 = wrapped(x)
    wrapped.set_active_task("modulation")
    y1 = wrapped(x)
    assert not torch.allclose(y0, y1)
    peft = peft_state_dict(_tiny_model())
    assert any("task_interface" in k or "modulation_head" in k for k in peft)
    assert any(k.startswith("emitter_fingerprint.") for k in peft)


def test_save_best_bundle(tmp_path: Path) -> None:
    model = _tiny_model()
    inject_hybrid_lora(model, ["modulation"], PeftConfig(r_attn=2, r_mamba=2, lora_alpha_attn=2, lora_alpha_mamba=2))
    path = tmp_path / "best.ckpt"
    save_best_bundle(str(path), model=model, peft_cfg={"r_attn": 2})
    blob = torch.load(path, map_location="cpu", weights_only=False)
    assert blob["peft"] is True
    assert "state_dict" in blob
    assert any("lora_A" in k for k in blob["state_dict"])
