from __future__ import annotations

import torch
import torch.nn as nn

from resmamba_signal_model.models.decoder import DecoderBlock, SharedDecoder, UnifiedQueryDecoder
from resmamba_signal_model.models.physics import PHYS_DIM
from resmamba_signal_model.models.mamba_backbone import BiMamba2Block


def _decoder() -> SharedDecoder:
    return SharedDecoder(
        d_model=32,
        patch_size=8,
        decoder_mamba_layers=1,
        attn_num_heads=4,
        dropout=0.0,
        sequence_packing=True,
        d_state=8,
        headdim=16,
        require_mamba_kernel=False,
        allow_fallback_mamba=True,
        norm_type="rmsnorm",
    )


def test_decoder_mamba_blocks_match_config() -> None:
    dec = _decoder()
    assert len(dec.blocks) == 1
    assert isinstance(dec.blocks[0], DecoderBlock)
    assert isinstance(dec.blocks[0].mamba, BiMamba2Block)
    dec2 = SharedDecoder(
        d_model=32,
        patch_size=8,
        decoder_mamba_layers=2,
        attn_num_heads=4,
        dropout=0.0,
        sequence_packing=True,
        d_state=8,
        headdim=16,
        require_mamba_kernel=False,
        allow_fallback_mamba=True,
        norm_type="rmsnorm",
    )
    assert len(dec2.blocks) == 2
    assert all(isinstance(block, DecoderBlock) for block in dec2.blocks)


def test_visible_only_scatter_fills_mask() -> None:
    dec = _decoder()
    b, n, d = 2, 6, 32
    x_tok = torch.randn(b, n, d)
    visible = torch.tensor([[1, 1, 0, 1, 0, 1], [1, 0, 1, 1, 1, 0]], dtype=torch.bool)
    patch_mask = torch.ones(b, n, dtype=torch.bool)
    h_vis = x_tok[visible].unsqueeze(0)
    phys = torch.randn(b, n, PHYS_DIM)
    out = dec(h_vis, x_tok, patch_mask, visible, phys, packed_encoder=True)
    assert out["recon_norm"].shape == (b, n, 2, 8)
    assert out["z"].shape == (b, d)
    filled = out["patch_h"][~visible]
    assert filled.shape[0] > 0
    assert not torch.allclose(filled, dec.mask_token.squeeze(0).expand_as(filled), atol=1e-5)


def test_skip_film_backward_and_dual_path() -> None:
    dec = _decoder()
    b, n, d = 2, 5, 32
    x_tok = torch.randn(b, n, d, requires_grad=True)
    visible = torch.ones(b, n, dtype=torch.bool)
    visible[:, -2:] = False
    patch_mask = torch.ones(b, n, dtype=torch.bool)
    h_vis = x_tok.detach()[visible].unsqueeze(0)
    phys = torch.randn(b, n, PHYS_DIM, requires_grad=True)
    out = dec(h_vis, x_tok, patch_mask, visible, phys, packed_encoder=True)
    loss = out["recon_norm"].square().mean() + out["z"].square().mean()
    loss.backward()
    assert x_tok.grad is not None and x_tok.grad.abs().sum() > 0
    assert phys.grad is not None and phys.grad.abs().sum() > 0
    assert out["h_dec"].shape[0] == b


def test_decoder_pad_path() -> None:
    dec = _decoder()
    dec.sequence_packing = False
    x_tok = torch.randn(2, 4, 32)
    visible = torch.ones(2, 4, dtype=torch.bool)
    patch_mask = torch.ones(2, 4, dtype=torch.bool)
    out = dec(x_tok, x_tok, patch_mask, visible, torch.randn(2, 4, PHYS_DIM), packed_encoder=False, sequence_packing=False)
    assert out["z"].shape == (2, 32)
    assert torch.isfinite(out["recon_norm"]).all()


def test_target_token_and_physics_cannot_enter_skip_or_film() -> None:
    torch.manual_seed(3)
    dec = _decoder().eval()
    b, n, d = 2, 6, 32
    x_tok = torch.randn(b, n, d)
    physics = torch.randn(b, n, PHYS_DIM)
    visible = torch.ones(b, n, dtype=torch.bool)
    visible[:, -2:] = False
    target = ~visible
    patch_mask = torch.ones_like(visible)
    h_vis = x_tok[visible].unsqueeze(0)

    first = dec(
        h_vis,
        x_tok,
        patch_mask,
        visible,
        physics,
        target_mask=target,
        packed_encoder=True,
    )
    changed_tokens = x_tok.clone()
    changed_physics = physics.clone()
    changed_tokens[target] = torch.randn_like(changed_tokens[target]) * 1000.0
    changed_physics[target] = torch.randn_like(changed_physics[target]) * 1000.0
    second = dec(
        h_vis,
        changed_tokens,
        patch_mask,
        visible,
        changed_physics,
        target_mask=target,
        packed_encoder=True,
    )
    for key in ("patch_h", "z", "recon_norm", "global_phys_pred"):
        assert torch.allclose(first[key], second[key], atol=1.0e-6, rtol=1.0e-6), key

def test_query_decoder_gate_zero_matches_visible_only() -> None:
    torch.manual_seed(0)
    dec = UnifiedQueryDecoder(d_model=32, patch_size=8, query_dim=32, num_heads=4)
    context = torch.randn(1, 4, 32)
    visible = torch.tensor([[True, True, False, True]])
    patch_mask = torch.ones(1, 4, dtype=torch.bool)
    target = ~visible
    physics = torch.randn(1, PHYS_DIM)
    recon_a, _ = dec(context, visible, patch_mask, target, context_physics=physics)
    context_b = context.clone()
    context_b[:, 2] = torch.randn(32)
    recon_b, _ = dec(context_b, visible, patch_mask, target, context_physics=physics)
    assert torch.allclose(recon_a, recon_b, atol=1.0e-6, rtol=1.0e-6)


def test_query_decoder_reads_local_decoder_state() -> None:
    torch.manual_seed(0)
    dec = UnifiedQueryDecoder(d_model=32, patch_size=8, query_dim=32, num_heads=4)
    dec.local_query_gate.data.fill_(1.0)
    context = torch.randn(1, 4, 32)
    visible = torch.tensor([[True, True, False, True]])
    patch_mask = torch.ones(1, 4, dtype=torch.bool)
    target = ~visible
    physics = torch.randn(1, PHYS_DIM)
    recon_a, _ = dec(context, visible, patch_mask, target, context_physics=physics)
    context_b = context.clone()
    context_b[:, 2] = torch.randn(32)
    recon_b, _ = dec(context_b, visible, patch_mask, target, context_physics=physics)
    assert not torch.allclose(recon_a[:, 2], recon_b[:, 2], atol=1.0e-6, rtol=1.0e-6)



def test_query_decoder_empty_visible_backward_finite() -> None:
    torch.manual_seed(0)
    dec = UnifiedQueryDecoder(d_model=32, patch_size=8, query_dim=32, num_heads=4)
    context = torch.randn(2, 4, 32, requires_grad=True)
    visible = torch.zeros(2, 4, dtype=torch.bool)
    patch_mask = torch.ones(2, 4, dtype=torch.bool)
    patch_mask[1, 2:] = False
    target = patch_mask.clone()
    physics = torch.randn(2, PHYS_DIM)
    recon, query_h = dec(context, visible, patch_mask, target, context_physics=physics)
    loss = recon.square().mean() + query_h.square().mean()
    loss.backward()
    assert torch.isfinite(recon).all()
    assert context.grad is not None
    assert torch.isfinite(context.grad).all()
    for param in dec.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()


def test_local_recon_is_zero_initialized() -> None:
    dec = UnifiedQueryDecoder(d_model=32, patch_size=8, query_dim=32, num_heads=4)
    assert torch.count_nonzero(dec.local_recon.weight).item() == 0
    assert torch.count_nonzero(dec.local_recon.bias).item() == 0


def test_local_recon_reads_interpolated_state() -> None:
    torch.manual_seed(0)
    dec = UnifiedQueryDecoder(d_model=32, patch_size=8, query_dim=32, num_heads=4)
    nn.init.normal_(dec.local_recon.weight, std=0.05)
    context = torch.randn(1, 4, 32)
    visible = torch.tensor([[True, True, False, True]])
    patch_mask = torch.ones(1, 4, dtype=torch.bool)
    target = ~visible
    physics = torch.randn(1, PHYS_DIM)
    recon_a, _ = dec(context, visible, patch_mask, target, context_physics=physics)
    context_b = context.clone()
    context_b[:, 2] = torch.randn(32)
    recon_b, _ = dec(context_b, visible, patch_mask, target, context_physics=physics)
    assert not torch.allclose(recon_a[:, 2], recon_b[:, 2], atol=1.0e-6, rtol=1.0e-6)
