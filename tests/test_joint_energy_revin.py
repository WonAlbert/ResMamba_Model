from __future__ import annotations

import math

import torch

from resmamba_signal_model.models.model import SignalFoundationModel, SignalModelConfig
from resmamba_signal_model.models.physics import LOG_POWER_INDEX, restore_absolute_log_power, sequence_physics
from resmamba_signal_model.models.revin import AMP_AUX_DIM, RevIN


def _channel_std_ratio(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask.to(dtype=x.dtype).unsqueeze(1)
    denom = w.sum(dim=-1).clamp_min(1.0)
    mean = (x * w).sum(dim=-1) / denom
    centered = (x - mean.unsqueeze(-1)) * w
    std = (centered.square().sum(dim=-1) / denom).clamp_min(1.0e-8).sqrt()
    return std[:, 0] / std[:, 1]


def test_joint_energy_preserves_iq_std_ratio() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 2, 256)
    x[:, 0] *= 2.0
    mask = torch.ones(4, 256, dtype=torch.bool)
    in_ratio = _channel_std_ratio(x, mask)
    joint = RevIN(num_channels=2, affine=False, scale_mode="joint_energy", peak_papr_clip=0.0)
    channel = RevIN(num_channels=2, affine=False, scale_mode="channel")
    jn, _ = joint.normalize(x, mask)
    cn, _ = channel.normalize(x, mask)
    joint_ratio = _channel_std_ratio(jn, mask)
    channel_ratio = _channel_std_ratio(cn, mask)
    assert torch.allclose(joint_ratio, in_ratio, rtol=0.05, atol=0.05)
    assert float((channel_ratio - 1.0).abs().mean()) < float((joint_ratio - 1.0).abs().mean())
    assert float((channel_ratio - 1.0).abs().mean()) < 0.15


def test_joint_energy_impulse_does_not_dominate_scale() -> None:
    torch.manual_seed(1)
    x = torch.randn(2, 2, 256) * 0.4
    x_clean = x.clone()
    x[0, :, 80] = 40.0
    mask = torch.ones(2, 256, dtype=torch.bool)
    revin = RevIN(num_channels=2, affine=False, scale_mode="joint_energy", peak_papr_clip=0.0)
    _, s_imp = revin.normalize(x, mask)
    _, s_clean = revin.normalize(x_clean, mask)
    assert torch.allclose(s_imp.std[0], s_clean.std[0], rtol=0.2, atol=0.05)
    assert s_imp.amp_aux is not None
    assert float(s_imp.amp_aux.scale_gap[0]) > float(s_clean.amp_aux.scale_gap[0])


def test_joint_energy_observed_mask_excluded_from_scale() -> None:
    torch.manual_seed(2)
    x = torch.randn(1, 2, 128)
    x[..., 64:] = 80.0
    sample = torch.ones(1, 128, dtype=torch.bool)
    observed = sample.clone()
    observed[:, 64:] = False
    revin = RevIN(num_channels=2, affine=False, scale_mode="joint_energy", peak_papr_clip=0.0)
    x_hat, stats = revin.normalize(x, sample, observed_mask=observed)
    x_obs = x.clone()
    x_obs[..., 64:] = 0.0
    _, stats_obs = revin.normalize(x_obs, sample, observed_mask=observed)
    assert torch.allclose(stats.std, stats_obs.std, rtol=1.0e-4, atol=1.0e-4)
    assert torch.count_nonzero(x_hat[..., 64:]) == 0


def test_joint_energy_gain_invariance_and_amp_aux_log_scale() -> None:
    torch.manual_seed(3)
    x = torch.randn(3, 2, 128)
    mask = torch.ones(3, 128, dtype=torch.bool)
    revin = RevIN(num_channels=2, affine=False, scale_mode="joint_energy", peak_papr_clip=0.0)
    n1, s1 = revin.normalize(x, mask)
    n2, s2 = revin.normalize(x * 2.0, mask)
    assert torch.allclose(n1, n2, atol=1.0e-5, rtol=1.0e-5)
    assert s1.amp_aux is not None and s2.amp_aux is not None
    delta = s2.amp_aux.log_scale - s1.amp_aux.log_scale
    assert torch.allclose(delta, torch.full_like(delta, math.log(2.0)), atol=1.0e-4)
    vec = revin.amp_aux_vector(s1)
    assert vec is not None and vec.shape == (3, AMP_AUX_DIM)


def test_peak_papr_clip_shrinks_only_hot_timesteps() -> None:
    revin = RevIN(num_channels=2, affine=False, scale_mode="joint_energy", peak_papr_clip=16.0)
    x = torch.zeros(1, 2, 64)
    x[..., :32] = 0.5
    x[..., 40] = 8.0
    mask = torch.ones(1, 64, dtype=torch.bool)
    x_hat, _ = revin.normalize(x, mask)
    power = x_hat.square().sum(dim=1)
    assert float(power[0, 40]) <= 16.0 + 1.0e-4
    assert float(power[0, :32].mean()) < 4.0


def test_restore_absolute_log_power_adds_log_scale() -> None:
    phys = torch.zeros(2, 4, 12)
    phys[..., LOG_POWER_INDEX] = 0.25
    restored = restore_absolute_log_power(phys, torch.tensor([math.log(2.0), 0.0]))
    assert torch.allclose(restored[0, :, LOG_POWER_INDEX], torch.full((4,), 0.25 + math.log(2.0)))
    assert torch.allclose(restored[1, :, LOG_POWER_INDEX], torch.full((4,), 0.25))


def test_sequence_physics_readout_uses_absolute_log_power() -> None:
    iq = torch.randn(2, 2, 64)
    mask = torch.ones(2, 64, dtype=torch.bool)
    rel = sequence_physics(iq, mask)
    abs_p = sequence_physics(iq, mask, log_scale=torch.ones(2))
    assert torch.allclose(abs_p[..., LOG_POWER_INDEX], rel[..., LOG_POWER_INDEX] + 1.0)


def test_model_gain_invariance_of_main_iq_and_z_enc() -> None:
    torch.manual_seed(4)
    cfg = SignalModelConfig(
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
        revin_affine=False,
        revin_scale_mode="joint_energy",
        revin_peak_papr_clip=0.0,
        build_task_interface=True,
    )
    model = SignalFoundationModel(cfg).eval()
    iq = torch.randn(2, 2, 64)
    mask = torch.ones(2, 64, dtype=torch.bool)
    with torch.no_grad():
        a = model({"iq": iq, "sample_mask": mask}, mode="encode", mask_mode="none")
        b = model({"iq": iq * 2.0, "sample_mask": mask}, mode="encode", mask_mode="none")
    assert torch.allclose(a["z_enc"], b["z_enc"], atol=2.0e-4, rtol=2.0e-4)
    assert a["amp_aux"] is not None and b["amp_aux"] is not None
    delta = b["amp_aux"][:, 0] - a["amp_aux"][:, 0]
    assert torch.allclose(delta, torch.full_like(delta, math.log(2.0)), atol=1.0e-3)
    # amp_aux 不得拼进分类身份
    assert a["z_enc"].shape[-1] == cfg.d_model
