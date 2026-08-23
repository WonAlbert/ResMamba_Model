from __future__ import annotations

import torch

from resmamba_signal_model.models.moe import (
    MOE_EXPERT_NAMES,
    MoEFFN,
    MoEFusion,
    load_balancing_loss,
)


def test_moe_expert_names() -> None:
    assert MOE_EXPERT_NAMES == ("ld_intrapulse", "ld_model", "tx_modulation")


def test_moe_ffn_dense_routing() -> None:
    torch.manual_seed(0)
    moe = MoEFFN(32, num_experts=3, ffn_expand=1.0, top_k=None)
    x = torch.randn(2, 5, 32)
    y, aux = moe(x)
    assert y.shape == x.shape
    assert aux.gate_weights.shape == (2, 5, 3)
    assert torch.isfinite(aux.load_balance_loss)
    assert aux.load_balance_loss.ndim == 0


def test_moe_ffn_top_k_sparsity() -> None:
    moe = MoEFFN(16, num_experts=3, ffn_expand=1.0, top_k=2)
    x = torch.randn(1, 4, 16)
    _, aux = moe(x)
    active = (aux.gate_weights > 0).float().sum(dim=-1)
    assert (active <= 2.0 + 1e-5).all()


def test_moe_fusion_three_branches() -> None:
    fusion = MoEFusion(16, 3)
    branches = [torch.randn(2, 6, 16) for _ in range(3)]
    fused, aux = fusion(branches)
    assert fused.shape == (2, 6, 16)
    assert aux.gate_weights.shape[-1] == 3


def test_load_balancing_uniform_routing() -> None:
    probs = torch.full((10, 3), 1.0 / 3.0)
    lb = load_balancing_loss(probs, num_experts=3)
    assert torch.isfinite(lb)
    assert lb.item() >= 0.0
