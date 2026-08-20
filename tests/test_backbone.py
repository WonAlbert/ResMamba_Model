from __future__ import annotations

import torch

from resmamba_signal_model.data.packing import pack_valid_tokens
from resmamba_signal_model.models.backbone import HybridEncoder
from resmamba_signal_model.models.mamba_backbone import BiMamba2Block
from resmamba_signal_model.models.transformer import MemoryTransformerBlock


def _encoder(**kwargs) -> HybridEncoder:
    defaults = dict(
        d_model=32,
        encoder_mamba_layers=5,
        encoder_transformer_layers=1,
        d_state=8,
        headdim=16,
        require_mamba_kernel=False,
        allow_fallback_mamba=True,
        attn_num_heads=4,
        attn_window=16,
        dropout=0.0,
        norm_type="rmsnorm",
    )
    defaults.update(kwargs)
    return HybridEncoder(**defaults)


def test_encoder_layer_layout() -> None:
    enc = _encoder()
    assert len(enc.mamba_layers) == 5
    assert len(enc.transformer_layers) == 1
    assert all(isinstance(layer, BiMamba2Block) for layer in enc.mamba_layers)
    assert all(isinstance(layer, MemoryTransformerBlock) for layer in enc.transformer_layers)


def test_encoder_pad_and_packed_paths() -> None:
    enc = _encoder()
    x = torch.randn(2, 12, 32)
    pad = torch.zeros(2, 12, dtype=torch.bool)
    pad[1, 8:] = True
    y_pad = enc(x, key_padding_mask=pad)
    assert y_pad.shape == x.shape
    valid = ~pad
    packed, cu, seq = pack_valid_tokens(x, valid)
    y_pk = enc(packed, seq_idx=seq, cu_seqlens=cu)
    assert y_pk.shape == packed.shape
    assert torch.isfinite(y_pk).all()


def test_memory_attention_on_long_sequence() -> None:
    enc = _encoder(attn_window=8)
    x = torch.randn(1, 40, 32)
    y = enc(x)
    assert y.shape == x.shape
    assert enc.transformer_layers[0].last_used_memory is True
    short = enc(torch.randn(1, 4, 32))
    assert enc.transformer_layers[0].last_used_memory is False
    assert short.shape[1] == 4
