from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from resmamba_signal_model.models.emitter_fingerprint import (
    EMITTER_STAT_DIM,
    EmitterFingerprintBranch,
    raw_emitter_stats,
)
from resmamba_signal_model.models.heads import EmitterHead
from resmamba_signal_model.models.model import SignalFoundationModel, SignalModelConfig
from resmamba_signal_model.models.revin import RevIN
from resmamba_signal_model.models.task_interface import TaskFeatures


def test_raw_emitter_stats_use_unnormalized_moments() -> None:
    torch.manual_seed(0)
    iq = torch.randn(3, 2, 64)
    iq[:, 0] += 0.4
    iq[:, 1] *= 2.5
    mask = torch.ones(3, 64, dtype=torch.bool)
    stats = raw_emitter_stats(iq, mask)
    assert stats.shape == (3, EMITTER_STAT_DIM)
    # mean_I 应接近 +0.4
    assert float(stats[:, 0].mean()) > 0.2
    revin = RevIN(num_channels=2, affine=False)
    norm, rev_stats = revin.normalize(iq, mask)
    # 归一化后再抽会丢掉 DC/增益；raw 路径必须保留
    raw = raw_emitter_stats(iq, mask, revin_stats=rev_stats)
    assert not torch.allclose(raw[:, :4], raw_emitter_stats(norm, mask)[:, :4], atol=0.05)


def test_fingerprint_changes_emitter_logits() -> None:
    torch.manual_seed(1)
    head = EmitterHead(16, num_emitters=5, dropout=0.0)
    feat = TaskFeatures(
        pooled=torch.randn(2, 16),
        tokens=torch.randn(2, 4, 16),
        mask=torch.ones(2, 4, dtype=torch.bool),
    )
    base = head(feat)["emitter_logits"]
    same = head(feat, fingerprint=None)["emitter_logits"]
    assert torch.allclose(base, same)
    shifted = head(feat, fingerprint=torch.randn(2, 16))["emitter_logits"]
    assert not torch.allclose(base, shifted)


def test_old_head_and_encoder_weights_load_with_new_fingerprint() -> None:
    cfg = SignalModelConfig(
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
        build_task_interface=True,
        num_emitters=8,
        num_mod_classes=5,
    )
    src = SignalFoundationModel(cfg)
    old = {
        key: value.clone()
        for key, value in src.state_dict().items()
        if "emitter_fingerprint." not in key and ".fp_head." not in key
    }
    dst = SignalFoundationModel(cfg)
    missing, unexpected = dst.load_state_dict(old, strict=False)
    assert unexpected == []
    assert any(name.startswith("emitter_fingerprint.") for name in missing)
    assert any(".fp_head." in name for name in missing)
    for key, value in src.encoder.state_dict().items():
        assert torch.equal(dst.encoder.state_dict()[key], value)
    for key, value in src.tokenizer.state_dict().items():
        assert torch.equal(dst.tokenizer.state_dict()[key], value)
    for key in src.emitter_head.classifier.state_dict():
        assert torch.equal(
            dst.emitter_head.classifier.state_dict()[key],
            src.emitter_head.classifier.state_dict()[key],
        )


def test_fingerprint_branch_shapes() -> None:
    branch = EmitterFingerprintBranch(d_model=32, conv_channels=16, dropout=0.0)
    iq = torch.randn(4, 2, 48)
    mask = torch.ones(4, 48, dtype=torch.bool)
    mask[1, 30:] = False
    out = branch(iq, mask)
    assert out.shape == (4, 32)
    assert torch.isfinite(out).all()
    # 样本间可分
    assert float((out - out.mean(dim=0)).detach().norm(dim=-1).mean()) > 1e-4
