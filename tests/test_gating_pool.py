from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from resmamba_signal_model.models.decoder import GatingPooling, build_sequence_pool
from resmamba_signal_model.models.model import SignalFoundationModel, SignalModelConfig


def test_gating_pool_masks_pads_and_shapes() -> None:
    pool = GatingPooling(d_model=16, num_heads=4)
    x = torch.randn(3, 8, 16)
    pad = torch.zeros(3, 8, dtype=torch.bool)
    pad[:, 5:] = True
    out = pool(x, key_padding_mask=pad)
    assert out.shape == (3, 16)
    # pad 位权重应为 0：改 pad token 不改变输出
    x2 = x.clone()
    x2[:, 5:] = 1.0e3
    out2 = pool(x2, key_padding_mask=pad)
    assert torch.allclose(out, out2, atol=1e-5, rtol=1e-4)


def test_build_sequence_pool_defaults_to_gating() -> None:
    assert isinstance(build_sequence_pool("gating_pool", 32, num_heads=4), GatingPooling)
    assert build_sequence_pool("attn_pool", 32, num_heads=4).__class__.__name__ == "AttentionPooling"


def test_model_default_encoder_pool_is_gating_and_l2() -> None:
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
        build_task_heads=False,
    )
    model = SignalFoundationModel(cfg)
    assert isinstance(model.encoder_pool, GatingPooling)
    batch = {
        "iq": torch.randn(2, 2, 64),
        "sample_mask": torch.ones(2, 64, dtype=torch.bool),
        "length": torch.tensor([64, 64]),
        "dataset_id": torch.tensor([0, 1]),
    }
    out = model(batch, mode="encode")
    z = out["z_enc"]
    norms = z.float().norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4)
