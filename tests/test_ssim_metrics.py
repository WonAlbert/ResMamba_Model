import pytest
import torch

from resmamba_signal_model.training.metrics import (
    masked_patch_mse,
    reconstruction_eval_pair,
    ssim_iq,
    ssim_iq_accumulate,
)


def test_ssim_iq_accumulate_matches_mean() -> None:
    pred = torch.randn(2, 2, 32)
    target = pred + 0.01 * torch.randn_like(pred)
    mask = torch.tensor([True, False])
    total, count = ssim_iq_accumulate(pred, target, mask)
    assert count == 2
    direct = ssim_iq(pred[:1], target[:1], mask[:1])
    assert abs(total / count - direct) < 1e-5


def test_reconstruction_eval_pair_prefers_norm_space() -> None:
    small = torch.randn(2, 3, 2, 8)
    large = small * 1.0e6
    pred, target, mask = reconstruction_eval_pair(
        {
            "recon_norm": small,
            "patch_targets_norm": torch.zeros_like(small),
            "mae_pred": large,
            "patch_targets": large,
            "mae_mask": torch.ones(2, 3, dtype=torch.bool),
        }
    )
    assert pred is small
    assert torch.equal(target, torch.zeros_like(small))
    assert mask is not None and bool(mask.all())


def test_masked_patch_mse_uses_mask() -> None:
    pred = torch.zeros(2, 2, 2, 4)
    target = torch.zeros_like(pred)
    pred[0, 0] = 2.0
    mask = torch.tensor([[True, False], [False, False]])
    mse = masked_patch_mse(pred, target, mask)
    assert float(mse) == pytest.approx(4.0)


def test_ssim_iq_stays_bounded_on_large_amplitude() -> None:
    pred = 1.0e6 * torch.randn(1, 2, 2, 32)
    target = pred + 1.0e5 * torch.randn_like(pred)
    score = ssim_iq(pred, target)
    assert -1.0 <= score <= 1.0
