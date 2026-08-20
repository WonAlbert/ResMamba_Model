from __future__ import annotations

import torch

from resmamba_signal_model.models.physics import (
    PHYS_DIM,
    MODALITY_IMU,
    MODALITY_RF,
    MODALITY_SONAR,
    patch_physics,
    physics_constraint_loss,
    physics_feature_mask,
    project_patch_energy,
    sequence_physics,
)
from resmamba_signal_model.models.revin import RevIN, clip_normalized


def test_energy_projection_aligns_power() -> None:
    torch.manual_seed(0)
    target = torch.randn(2, 5, 2, 8)
    recon = target * 0.3 + torch.randn_like(target) * 0.05
    aligned = project_patch_energy(recon, target)
    p_true = target.square().sum(dim=(-2, -1))
    p_aln = aligned.square().sum(dim=(-2, -1))
    assert torch.allclose(p_true, p_aln, rtol=1e-4, atol=1e-5)


def test_physics_loss_stays_bounded_for_degenerate_recon() -> None:
    orig = torch.randn(2, 4, 2, 8)
    recon = orig * 0.0
    recon[..., 0, :] = 1.0e-8
    mask = torch.ones(2, 4, dtype=torch.bool)
    loss = physics_constraint_loss(recon, orig, mask)
    assert torch.isfinite(loss)
    assert float(loss) < 20.0


def test_physics_loss_backward() -> None:
    recon = torch.randn(2, 4, 2, 8, requires_grad=True)
    orig = torch.randn(2, 4, 2, 8)
    mask = torch.ones(2, 4, dtype=torch.bool)
    loss = physics_constraint_loss(recon, orig, mask)
    loss.backward()
    assert recon.grad is not None
    assert recon.grad.abs().sum() > 0
    phys = patch_physics(orig)
    assert phys.shape == (2, 4, PHYS_DIM)
    assert PHYS_DIM == 12


def test_modality_mask_disables_rf_plugins_for_imu() -> None:
    patches = torch.randn(2, 3, 2, 8)
    rf = patch_physics(patches, modality_id=torch.tensor([MODALITY_RF, MODALITY_RF]))
    imu = patch_physics(patches, modality_id=torch.tensor([MODALITY_IMU, MODALITY_IMU]))
    mask = physics_feature_mask(torch.tensor([MODALITY_IMU]), n_tokens=3)
    assert mask.shape == (1, 3, PHYS_DIM)
    assert mask[0, 0, 5:8].sum() == 0
    assert rf[..., 5:8].abs().sum() > 0
    assert imu[..., 5:8].abs().sum() == 0
    sonar = patch_physics(patches[:1], modality_id=torch.tensor([MODALITY_SONAR]))
    assert sonar[..., 8:10].abs().sum() > 0
    assert sonar[..., 5:8].abs().sum() == 0


def test_zero_waveform_physics_backward_is_finite() -> None:
    iq = torch.zeros(3, 2, 32, requires_grad=True)
    mask = torch.ones(3, 32, dtype=torch.bool)
    phys = sequence_physics(iq, mask)
    loss = phys.square().mean()
    loss.backward()
    assert torch.isfinite(phys).all()
    assert iq.grad is not None
    assert torch.isfinite(iq.grad).all()


def test_physics_target_side_does_not_nan_revin_affine() -> None:
    revin = RevIN(num_channels=2, affine=True, std_min=1.0e-2, clip=8.0)
    x = torch.zeros(2, 2, 64)
    x[..., 48:] = 5.0e4
    sample_mask = torch.ones(2, 64, dtype=torch.bool)
    observed = sample_mask.clone()
    observed[:, 32:] = False
    _x_hat, stats = revin.normalize(x, sample_mask, observed_mask=observed)
    target = clip_normalized(revin.apply_stats(x, stats, sample_mask), 8.0)
    patches = target.unfold(-1, 8, 8).permute(0, 2, 1, 3).contiguous()
    loss = physics_constraint_loss(patches.detach() * 0.0, patches) + sequence_physics(target, sample_mask).square().mean()
    loss.backward()
    assert torch.isfinite(revin.gamma.grad).all()
    assert torch.isfinite(revin.beta.grad).all()


def test_envelope_sqrt_through_revin_is_finite() -> None:
    revin = RevIN(num_channels=2, affine=True, std_min=1.0e-2, clip=8.0)
    x = torch.zeros(2, 2, 64)
    x[..., 48:] = 5.0e4
    sample_mask = torch.ones(2, 64, dtype=torch.bool)
    observed = sample_mask.clone()
    observed[:, 32:] = False
    _iq, stats = revin.normalize(x, sample_mask, observed_mask=observed)
    target = clip_normalized(revin.apply_stats(x, stats, sample_mask), 8.0)
    power = target.square().sum(dim=1)
    from resmamba_signal_model.models.physics import safe_sqrt
    loss = safe_sqrt(power).mean() + physics_constraint_loss(
        target.unfold(-1, 8, 8).permute(0, 2, 1, 3).contiguous().detach() * 0.0,
        target.unfold(-1, 8, 8).permute(0, 2, 1, 3).contiguous(),
    )
    loss.backward()
    assert torch.isfinite(revin.gamma.grad).all()
    assert torch.isfinite(revin.beta.grad).all()
