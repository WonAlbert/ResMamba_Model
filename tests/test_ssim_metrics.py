import torch

from resmamba_signal_model.training.metrics import ssim_iq, ssim_iq_accumulate


def test_ssim_iq_accumulate_matches_mean() -> None:
    pred = torch.randn(2, 2, 32)
    target = pred + 0.01 * torch.randn_like(pred)
    mask = torch.tensor([True, False])
    total, count = ssim_iq_accumulate(pred, target, mask)
    assert count == 2
    direct = ssim_iq(pred[:1], target[:1], mask[:1])
    assert abs(total / count - direct) < 1e-5
