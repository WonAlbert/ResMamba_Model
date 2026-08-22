from __future__ import annotations

import torch

from resmamba_signal_model.models.revin import RevIN


def test_revin_roundtrip_error() -> None:
    torch.manual_seed(0)
    revin = RevIN(num_channels=2, scale_mode="channel", affine=True)
    x = torch.randn(3, 2, 64) * 4 + 1.5
    mask = torch.ones(3, 64, dtype=torch.bool)
    x_hat, stats = revin.normalize(x, mask)
    recon = revin.denormalize(x_hat, stats)
    assert (recon - x).abs().max().item() < 1e-5


def test_revin_pad_does_not_pollute_stats() -> None:
    torch.manual_seed(1)
    revin = RevIN(num_channels=2, scale_mode="channel", affine=False)
    x = torch.zeros(2, 2, 32)
    x[0, :, :10] = 2.0
    x[1, :, :] = 99.0
    mask = torch.zeros(2, 32, dtype=torch.bool)
    mask[0, :10] = True
    mask[1, :8] = True
    x[1, :, :8] = 3.0
    _x_hat, stats = revin.normalize(x, mask)
    assert torch.allclose(stats.mean[0], torch.full((2,), 2.0), atol=1e-5)
    assert torch.allclose(stats.mean[1], torch.full((2,), 3.0), atol=1e-5)


def test_revin_observed_mask_hides_targets_but_keeps_target_normalization() -> None:
    revin = RevIN(num_channels=2, scale_mode="channel", affine=False)
    x = torch.arange(32, dtype=torch.float32).view(1, 2, 16)
    sample_mask = torch.ones(1, 16, dtype=torch.bool)
    observed = sample_mask.clone()
    observed[:, 10:] = False

    x_hat, stats = revin.normalize(x, sample_mask, observed_mask=observed)
    assert torch.count_nonzero(x_hat[..., 10:]) == 0
    assert torch.allclose(stats.mean, x[..., :10].mean(dim=-1))

    targets = revin.apply_stats(x, stats, sample_mask)
    assert torch.count_nonzero(targets[..., 10:]) > 0
    restored = revin.denormalize(targets, stats)
    assert torch.allclose(restored, x, atol=1.0e-5)


def test_revin_quiet_observed_std_floor_does_not_use_targets() -> None:
    revin = RevIN(num_channels=2, scale_mode="channel", affine=False, std_min=1.0e-2)
    x = torch.zeros(1, 2, 16)
    x[..., 10:] = 5.0
    sample_mask = torch.ones(1, 16, dtype=torch.bool)
    observed = sample_mask.clone()
    observed[:, 10:] = False
    _x_hat, stats = revin.normalize(x, sample_mask, observed_mask=observed)
    assert torch.allclose(stats.mean, torch.zeros(1, 2), atol=1.0e-6)
    assert torch.all(stats.std >= 1.0e-2 - 1.0e-8)


def test_clip_normalized_bounds_masked_pulse_without_changing_stats() -> None:
    from resmamba_signal_model.models.revin import clip_normalized

    revin = RevIN(num_channels=2, scale_mode="channel", affine=False, std_min=1.0e-2)
    x = torch.full((1, 2, 32), 1.0e-4)
    x[..., 24:] = 4.0
    sample_mask = torch.ones(1, 32, dtype=torch.bool)
    observed = sample_mask.clone()
    observed[:, 24:] = False
    x_hat, stats = revin.normalize(x, sample_mask, observed_mask=observed)
    raw_targets = revin.apply_stats(x, stats, sample_mask)
    clipped = clip_normalized(raw_targets, 8.0)
    assert raw_targets[..., 24:].abs().max().item() > 8.0
    assert clipped.abs().max().item() <= 8.0 + 1.0e-6
    assert torch.count_nonzero(x_hat[..., 24:]) == 0
    restored = revin.denormalize(raw_targets, stats)
    assert torch.allclose(restored, x, atol=1.0e-5)


def test_revin_affine_grads_finite_for_quiet_observed_huge_pulse() -> None:
    from resmamba_signal_model.models.revin import clip_normalized

    revin = RevIN(num_channels=2, scale_mode="channel", affine=True, std_min=1.0e-2, clip=8.0)
    x = torch.zeros(2, 2, 64)
    x[..., :8] = 1.0e-4
    x[..., 32:] = 6.0e4
    sample_mask = torch.ones(2, 64, dtype=torch.bool)
    observed = sample_mask.clone()
    observed[:, 16:] = False
    x_hat, stats = revin.normalize(x, sample_mask, observed_mask=observed)
    target = clip_normalized(revin.apply_stats(x, stats, sample_mask), 8.0)
    loss = target.square().mean() + x_hat.square().mean()
    loss.backward()
    assert torch.isfinite(target).all()
    assert float(target.abs().max()) <= 8.0 + 1.0e-5
    assert revin.gamma.grad is not None
    assert torch.isfinite(revin.gamma.grad).all()
    assert torch.isfinite(revin.beta.grad).all()


def test_revin_inf_zscore_does_not_nan_affine_grad() -> None:
    revin = RevIN(num_channels=2, scale_mode="channel", affine=True, std_min=1.0e-2, clip=8.0)
    x = torch.zeros(1, 2, 16)
    x[..., 8:] = float("inf")
    mask = torch.ones(1, 16, dtype=torch.bool)
    observed = mask.clone()
    observed[:, 8:] = False
    x_hat, stats = revin.normalize(x, mask, observed_mask=observed)
    target = revin.apply_stats(torch.nan_to_num(x, nan=0.0, posinf=1.0e6, neginf=-1.0e6), stats, mask)
    loss = target.square().mean() + x_hat.square().mean()
    loss.backward()
    assert torch.isfinite(revin.gamma.grad).all()
    assert torch.isfinite(revin.beta.grad).all()
