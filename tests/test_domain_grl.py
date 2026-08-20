from __future__ import annotations

import torch

from resmamba_signal_model.models.domain import DomainDiscriminator, GradientReversal


def test_grl_flips_gradient_sign() -> None:
    x = torch.randn(4, 8, requires_grad=True)
    y = (x * 2).sum()
    y.backward()
    g_plain = x.grad.detach().clone()
    x.grad = None
    grl = GradientReversal(lambd=1.0)
    y_rev = (grl(x) * 2).sum()
    y_rev.backward()
    assert torch.allclose(x.grad, -g_plain)


def test_domain_loss_reaches_z() -> None:
    z = torch.randn(6, 16, requires_grad=True)
    disc = DomainDiscriminator(16, num_datasets=3)
    grl = GradientReversal(lambd=1.0)
    logits = disc(grl(z))
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    assert z.grad is not None
    assert z.grad.abs().sum() > 0


def test_model_domain_grl_skips_z_general() -> None:
    from resmamba_signal_model.models.model import SignalFoundationModel, SignalModelConfig

    model = SignalFoundationModel(
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
            num_datasets=4,
            build_task_interface=True,
        )
    )
    model.train()
    batch = {
        "iq": torch.randn(3, 2, 32),
        "sample_mask": torch.ones(3, 32, dtype=torch.bool),
        "dataset_id": torch.tensor([0, 1, 2]),
    }
    out = model(batch, mode="pretrain")
    out["z"].retain_grad()
    loss = torch.nn.functional.cross_entropy(out["domain_logits"], batch["dataset_id"])
    loss.backward()
    assert out["z"].grad is None or float(out["z"].grad.abs().sum()) < 1.0e-8
