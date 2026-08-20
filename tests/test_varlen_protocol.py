from __future__ import annotations

import torch

from resmamba_signal_model.models.model import SignalFoundationModel, SignalModelConfig
from resmamba_signal_model.models.varlen import apply_truncation_aug, assert_min_length, chunk_starts


def _cfg(**kwargs) -> SignalModelConfig:
    base = dict(
        d_model=32,
        encoder_mamba_layers=5,
        encoder_transformer_layers=1,
        decoder_mamba_layers=1,
        mamba_d_state=8,
        mamba_headdim=16,
        require_mamba_kernel=False,
        allow_fallback_mamba=True,
        attn_num_heads=4,
        attn_window=32,
        patch_size=8,
        stem_channels=8,
        freq_bands=4,
        dropout=0.0,
        l_min=16,
        chunk_len=64,
        p_trunc=0.0,
        sequence_packing=True,
        num_datasets=4,
    )
    base.update(kwargs)
    return SignalModelConfig(**base)


def test_mixed_length_pack_forward() -> None:
    model = SignalFoundationModel(_cfg(chunk_len=8192))
    model.eval()
    iq = [torch.randn(2, 16), torch.randn(2, 20), torch.randn(2, 128), torch.randn(2, 4096)]
    out = model(iq, mode="pretrain")
    assert out["z"].shape[0] == 4
    assert out["mae_pred"].shape[0] == 4
    assert torch.isfinite(out["z"]).all()


def test_shorter_than_patch_size_mask() -> None:
    model = SignalFoundationModel(_cfg(patch_size=32, l_min=16, stem_channels=8))
    model.eval()
    iq = torch.randn(1, 2, 20)
    mask = torch.ones(1, 20, dtype=torch.bool)
    out = model(iq, mask, mode="encode")
    assert out["patch_mask"].shape[1] == 1
    assert out["patch_mask"][0, 0]
    assert out["tokens"].shape[1] == 1


def test_truncation_aug_shapes() -> None:
    iq = torch.randn(3, 2, 80)
    mask = torch.ones(3, 80, dtype=torch.bool)
    torch.manual_seed(0)
    out_iq, out_mask = apply_truncation_aug(iq, mask, p_trunc=1.0, l_min=16)
    assert out_iq.shape == iq.shape
    assert out_mask.shape == mask.shape
    assert int(out_mask.sum(dim=1).min().item()) >= 16


def test_chunk_inference_z_stitch() -> None:
    model = SignalFoundationModel(_cfg(chunk_len=64, chunk_overlap=0.125, p_trunc=0.0))
    model.eval()
    iq = torch.randn(1, 2, 200)
    mask = torch.ones(1, 200, dtype=torch.bool)
    out = model(iq, mask, mode="encode")
    assert "chunk_z" in out
    assert out["z"].shape == (1, 32)
    assert out["recon_wave"].shape == iq.shape
    assert chunk_starts(200, 64, 8)[0] == 0


def test_l_min_raises() -> None:
    try:
        assert_min_length(8, 16)
    except ValueError as exc:
        assert "L_min" in str(exc)
    else:
        raise AssertionError("expected ValueError")
