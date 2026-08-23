from __future__ import annotations

import torch

from resmamba_signal_model.models.moe import (
    MOE_EXPERT_NAMES,
    MoEFFN,
    MoEFusion,
    expand_route_weights,
    load_balancing_loss,
    resolve_pretrain_stem_route_weights,
    resolve_task_route_weights,
    uniform_route_weights,
)


def test_moe_expert_names() -> None:
    assert MOE_EXPERT_NAMES == ("ld_intrapulse", "ld_model", "tx_modulation")


def test_task_route_weights() -> None:
    w = resolve_task_route_weights("ld_intrapulse", 3)
    assert torch.allclose(w, torch.tensor([1.0, 0.0, 0.0]))
    w = resolve_task_route_weights("ld_model", 3)
    assert torch.allclose(w, torch.tensor([0.0, 1.0, 0.0]))
    w = resolve_task_route_weights("tx_modulation", 3)
    assert torch.allclose(w, torch.tensor([0.0, 0.0, 1.0]))
    w = resolve_task_route_weights("ld_clustering", 3)
    assert torch.allclose(w, torch.tensor([0.5, 0.5, 0.0]))
    w = resolve_task_route_weights("prediction", 3)
    assert torch.allclose(w, torch.full((3,), 1.0 / 3.0))


def test_pretrain_stem_route_weights() -> None:
    assert torch.allclose(resolve_pretrain_stem_route_weights("radchar", 3), torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(resolve_pretrain_stem_route_weights("radar_mod15", 3), torch.tensor([0.0, 1.0, 0.0]))
    assert torch.allclose(resolve_pretrain_stem_route_weights("rml2016_04c", 3), torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(
        resolve_pretrain_stem_route_weights("radcom_awgn", 3),
        torch.full((3,), 1.0 / 3.0),
    )


def test_moe_ffn_task_routing() -> None:
    torch.manual_seed(0)
    moe = MoEFFN(32, num_experts=3, ffn_expand=1.0, top_k=None)
    x = torch.randn(2, 5, 32)
    route = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    y, aux = moe(x, route_weights=route)
    assert y.shape == x.shape
    assert aux.gate_weights.shape == (2, 5, 3)
    assert torch.allclose(aux.gate_weights[0, :, 0], torch.ones(5), atol=1e-5)
    assert torch.allclose(aux.gate_weights[1, :, 2], torch.ones(5), atol=1e-5)
    assert aux.load_balance_loss.item() == 0.0


def test_moe_ffn_top_k_sparsity() -> None:
    moe = MoEFFN(16, num_experts=3, ffn_expand=1.0, top_k=2)
    x = torch.randn(1, 4, 16)
    route = uniform_route_weights(3).unsqueeze(0)
    _, aux = moe(x, route_weights=route)
    active = (aux.gate_weights > 0).float().sum(dim=-1)
    assert (active <= 2.0 + 1e-5).all()


def test_moe_fusion_three_branches() -> None:
    fusion = MoEFusion(16, 3)
    branches = [torch.randn(2, 6, 16) for _ in range(3)]
    route = resolve_task_route_weights("ld_model", 3).unsqueeze(0).expand(2, -1)
    fused, aux = fusion(branches, route_weights=route)
    assert fused.shape == (2, 6, 16)
    assert aux.gate_weights.shape[-1] == 3
    assert torch.allclose(aux.gate_weights[:, :, 1], torch.ones(2, 6), atol=1e-5)


def test_expand_route_weights_padding() -> None:
    x = torch.randn(2, 4, 8)
    pad = torch.tensor([[False, False, True, True], [False, False, False, False]])
    w = expand_route_weights(torch.tensor([1.0, 0.0, 0.0]), x, num_experts=3, key_padding_mask=pad)
    assert w.shape == (2, 4, 3)
    assert torch.all(w[0, 2] == 0.0)


def test_load_balancing_uniform_routing() -> None:
    probs = torch.full((10, 3), 1.0 / 3.0)
    lb = load_balancing_loss(probs, num_experts=3)
    assert torch.isfinite(lb)
    assert lb.item() >= 0.0
