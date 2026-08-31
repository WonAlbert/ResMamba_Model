from __future__ import annotations

import pytest
import torch

from resmamba_signal_model.training.lit_module import SignalLitModule


class _StubModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(1.0))


def _lit_module(*, stem_normalized_loss: bool) -> SignalLitModule:
    model = _StubModel()
    train_cfg = {
        "loss_weights": {"mae": 1.0},
        "stem_normalized_loss": stem_normalized_loss,
        "stem_loss_ema_alpha": 0.9,
    }
    lit = SignalLitModule.__new__(SignalLitModule)
    torch.nn.Module.__init__(lit)
    lit.model = model  # type: ignore[assignment]
    lit.train_cfg = train_cfg
    lit.stage = "pretrain"
    lit.loss_weights = train_cfg["loss_weights"]
    lit._stem_total_ema = {}
    return lit


def test_stem_normalize_first_visit_is_unity() -> None:
    lit = _lit_module(stem_normalized_loss=True)
    total = torch.tensor(0.5, requires_grad=True)
    batch = {"moe_route_stem": "radchar"}
    norm = lit._stem_normalize_pretrain_total(total, batch)
    assert float(norm.detach()) == pytest.approx(1.0)


def test_stem_normalize_uses_previous_ema() -> None:
    lit = _lit_module(stem_normalized_loss=True)
    lit._stem_total_ema["radchar"] = 0.5
    total = torch.tensor(0.4, requires_grad=True)
    batch = {"moe_route_stem": "radchar"}
    norm = lit._stem_normalize_pretrain_total(total, batch)
    assert float(norm.detach()) == pytest.approx(0.8)


def test_stem_normalize_disabled_passthrough() -> None:
    lit = _lit_module(stem_normalized_loss=False)
    total = torch.tensor(0.5, requires_grad=True)
    batch = {"moe_route_stem": "radchar"}
    norm = lit._stem_normalize_pretrain_total(total, batch)
    assert norm is total
