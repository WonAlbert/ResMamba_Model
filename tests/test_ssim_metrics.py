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
    assert count == 1
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


def test_reconstruction_eval_pair_prediction_uses_suffix_mask_only() -> None:
    pred = torch.zeros(2, 4, 2, 4)
    target = torch.ones_like(pred)
    suffix = torch.tensor([[False, False, True, True], [False, True, True, False]])
    span = torch.ones(2, 4, dtype=torch.bool)
    target_mask = suffix | span
    out = {
        "pred_patches": pred,
        "patch_targets_norm": target,
        "suffix_mask": suffix,
        "span_mask": span,
        "target_mask": target_mask,
        "recon_mask": target_mask,
        "mae_mask": torch.ones(2, 4, dtype=torch.bool),
    }
    _, _, mask = reconstruction_eval_pair(out, kind="prediction")
    assert mask is not None
    assert torch.equal(mask, suffix)
    mse = masked_patch_mse(pred, target, mask)
    # only masked positions: 4 True cells, each channel/patch mean err = 1 -> mse=1
    assert float(mse) == pytest.approx(1.0)


def test_reconstruction_eval_pair_imputation_uses_span_mask_only() -> None:
    pred = torch.zeros(1, 4, 2, 4)
    target = torch.ones_like(pred)
    span = torch.tensor([[True, True, False, False]])
    suffix = torch.tensor([[False, False, True, True]])
    out = {
        "pred_patches": pred,
        "patch_targets_norm": target,
        "span_mask": span,
        "suffix_mask": suffix,
        "target_mask": span | suffix,
        "recon_mask": span | suffix,
    }
    _, _, mask = reconstruction_eval_pair(out, kind="imputation")
    assert mask is not None
    assert torch.equal(mask, span)


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


def test_ssim_envelope_ignores_carrier_phase() -> None:
    torch.manual_seed(0)
    target = torch.randn(2, 3, 2, 32)
    pred = torch.stack((-target[:, :, 1], target[:, :, 0]), dim=2)
    mask = torch.ones(2, 3, dtype=torch.bool)
    assert ssim_iq(pred, target, mask) > 0.99


def test_ssim_envelope_wave_uses_length_and_mask() -> None:
    torch.manual_seed(1)
    target = torch.randn(2, 4, 2, 16)
    pred = target.clone()
    pred[:, 0] = 0.0
    # 只评测远离损坏 patch 的后半段，避免 SSIM 窗跨过边界
    mask = torch.tensor([[False, False, True, True], [False, False, True, True]])
    masked_only = ssim_iq(pred, target, mask, length=64)
    all_patches = ssim_iq(pred, target, torch.ones_like(mask), length=64)
    assert masked_only > 0.99
    assert all_patches < masked_only
