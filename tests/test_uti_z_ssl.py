from __future__ import annotations

import torch

from resmamba_signal_model.models.model import SignalFoundationModel, SignalModelConfig
from resmamba_signal_model.training.losses import foundation_pretrain_losses, vicreg_token_loss, view_decorrelation_loss


def _tiny() -> SignalFoundationModel:
    return SignalFoundationModel(
        SignalModelConfig(
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
            build_task_interface=True,
            build_task_heads=False,
        )
    )


def test_vicreg_token_and_view_div_are_finite_and_have_grad() -> None:
    torch.manual_seed(0)
    model = _tiny()
    model.train()
    batch = {
        "iq": torch.randn(4, 2, 64),
        "sample_mask": torch.ones(4, 64, dtype=torch.bool),
    }
    out = model(batch, mode="pretrain", mask_mode="random")
    assert "h_enc" in out and "z_semantic" in out and "z_source" in out and "z_context" in out
    losses = foundation_pretrain_losses(
        out,
        batch,
        include={"vicreg", "vicreg_token", "view_div", "uti_pooled", "uti_token", "uti_query"},
    )
    assert float(losses["vicreg"].detach()) >= 0.0
    assert float(losses["vicreg_token"].detach()) > 0.0
    assert float(losses["view_div"].detach()) >= 0.0
    assert float(losses["uti_pooled"].detach()) == 0.0
    (losses["vicreg"] + losses["vicreg_token"] + losses["view_div"]).backward()
    assert any(
        p.grad is not None and float(p.grad.abs().sum()) > 0.0
        for p in model.task_interface.view_adapters.parameters()
    )
    assert any(
        p.grad is not None and float(p.grad.abs().sum()) > 0.0
        for p in model.encoder.parameters()
        if p.requires_grad
    )


def test_view_decorrelation_is_zero_for_orthogonal_views() -> None:
    a = torch.zeros(8, 4)
    b = torch.zeros(8, 4)
    a[:, 0] = 1.0
    b[:, 1] = 1.0
    loss = view_decorrelation_loss([a, b])
    assert float(loss) < 1.0e-6


def test_vicreg_token_uses_visible_tokens_only() -> None:
    h = torch.zeros(2, 4, 8)
    h[:, :2] = torch.randn(2, 2, 8)
    mask = torch.zeros(2, 4, dtype=torch.bool)
    mask[:, :2] = True
    loss = vicreg_token_loss(h, mask)
    assert torch.isfinite(loss)
    assert float(loss) >= 0.0
