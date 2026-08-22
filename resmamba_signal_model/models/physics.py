from __future__ import annotations

import torch
import torch.nn.functional as F

# 通用结构特征 + 模态插件。缺失插件由 feature mask 置零，不进入 FiLM。
PHYS_SHARED_NAMES = ("log_power", "papr", "envelope_rate", "spectral_centroid", "spectral_spread")
PHYS_RF_NAMES = ("iq_corr", "iq_var_ratio", "cfo_proxy")
PHYS_SONAR_NAMES = ("spectral_entropy", "highband_ratio")
PHYS_IMU_NAMES = ("axis_norm", "axis_drift")
PHYS_NAMES = PHYS_SHARED_NAMES + PHYS_RF_NAMES + PHYS_SONAR_NAMES + PHYS_IMU_NAMES
PHYS_DIM = len(PHYS_NAMES)
SHARED_SLICE = slice(0, 5)
RF_SLICE = slice(5, 8)
SONAR_SLICE = slice(8, 10)
IMU_SLICE = slice(10, 12)

# 与 task_interface.MODALITY_TO_ID 对齐，避免 physics ↔ UTI 循环导入。
MODALITY_GENERIC = 0
MODALITY_RF = 1
MODALITY_SONAR = 2
MODALITY_IMU = 3


LOG_POWER_INDEX = 0


def restore_absolute_log_power(
    phys: torch.Tensor,
    log_scale: torch.Tensor | None,
) -> torch.Tensor:
    """归一化波形上的相对 log_power 加回 ``log_scale``（RSSI / 增益代理）。"""
    if log_scale is None:
        return phys
    scale = log_scale.reshape(-1, *([1] * (phys.ndim - 1))).to(device=phys.device, dtype=phys.dtype)
    out = phys.clone()
    out[..., LOG_POWER_INDEX] = out[..., LOG_POWER_INDEX] + scale.squeeze(-1)
    return out


def safe_sqrt(x: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """``sqrt(0)`` 反传是 ``Inf``，再乘 RevIN 仿射的 0 通道就是 ``0*Inf=NaN``。"""
    return x.clamp_min(eps).sqrt()


def safe_complex_abs(z: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """复数模长。不用 ``z.abs()``，避免 0 点反传不稳定。"""
    return safe_sqrt(z.real.square() + z.imag.square(), eps=eps)


def safe_angle(real: torch.Tensor, imag: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """``atan2(0,0)`` 反传是 ``NaN``。近零样本相位记 0，且切断奇异梯度。"""
    mag2 = real.square() + imag.square()
    live = mag2 > eps
    real_s = torch.where(live, real, torch.full_like(real, eps))
    imag_s = torch.where(live, imag, torch.zeros_like(imag))
    return torch.atan2(imag_s, real_s)


def _fft_logmag_and_centroid(
    i: torch.Tensor, q: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """对复数 I/Q 做双侧 ``fft``。

    返回：
    - ``logmag`` / ``mag``：完整频谱（长度 = 时域长度，不裁半）
    - ``centroid`` / ``spread``：按 ``fftfreq``（周期/样本，约 ``[-0.5, 0.5)``）加权
    - ``highband_ratio``：``|f| >= 0.25`` 的能量占比
    """
    z = torch.complex(i.float(), q.float())
    n_freq = int(z.shape[-1])
    spec = safe_complex_abs(torch.fft.fft(z, dim=-1))
    logmag = spec.log()
    freqs = torch.fft.fftfreq(n_freq, d=1.0, device=spec.device, dtype=spec.dtype)
    view = (1,) * (spec.ndim - 1) + (n_freq,)
    freq = freqs.view(view)
    mass = spec.sum(dim=-1).clamp_min(1.0e-8)
    centroid = (spec * freq).sum(dim=-1) / mass
    spread = safe_sqrt((spec * (freq - centroid.unsqueeze(-1)).square()).sum(dim=-1) / mass)
    highband_ratio = (spec * (freq.abs() >= 0.25).to(dtype=spec.dtype)).sum(dim=-1) / mass
    return logmag, centroid, spread.clamp(0.0, 1.0), spec, highband_ratio.clamp(0.0, 1.0)


def _spectral_entropy(mag: torch.Tensor) -> torch.Tensor:
    mass = mag.sum(dim=-1).clamp_min(1.0e-8)
    p = mag / mass.unsqueeze(-1)
    n_freq = max(int(mag.shape[-1]), 2)
    return -(p * p.clamp_min(1.0e-8).log()).sum(dim=-1) / mag.new_tensor(float(n_freq)).log()


def physics_feature_mask(
    modality_id: torch.Tensor | None,
    *,
    complex_pair: torch.Tensor | bool | None = None,
    n_tokens: int | None = None,
    phys_dim: int = PHYS_DIM,
) -> torch.Tensor:
    """``[B, D]`` 或 ``[B, N, D]``：共享维恒为 1，插件仅在对应模态为 1。"""
    if modality_id is None and complex_pair is None:
        raise ValueError("physics_feature_mask 需要 modality_id 或 complex_pair")
    if isinstance(complex_pair, bool) or complex_pair is None:
        if modality_id is None:
            raise ValueError("complex_pair 为标量时必须提供 modality_id 以确定 batch")
        pair_flag = torch.full(
            (int(modality_id.shape[0]),),
            True if complex_pair is None else bool(complex_pair),
            device=modality_id.device,
            dtype=torch.bool,
        )
    else:
        pair_flag = complex_pair.bool().reshape(-1)
        if modality_id is None:
            modality_id = torch.full(
                (int(pair_flag.shape[0]),),
                MODALITY_RF if bool(pair_flag.any()) else MODALITY_GENERIC,
                device=pair_flag.device,
                dtype=torch.long,
            )
        elif pair_flag.numel() == 1 and int(modality_id.shape[0]) > 1:
            pair_flag = pair_flag.expand(int(modality_id.shape[0]))
    ids = modality_id.long().reshape(-1)
    batch = int(ids.shape[0])
    if pair_flag.numel() != batch:
        pair_flag = pair_flag[:1].expand(batch)
    device = ids.device
    mask = torch.zeros(batch, phys_dim, device=device, dtype=torch.float32)
    mask[:, SHARED_SLICE] = 1.0
    rf_on = (ids == MODALITY_RF) | ((ids == MODALITY_GENERIC) & pair_flag)
    sonar_on = ids == MODALITY_SONAR
    imu_on = ids == MODALITY_IMU
    mask[rf_on, RF_SLICE] = 1.0
    mask[sonar_on, SONAR_SLICE] = 1.0
    mask[imu_on, IMU_SLICE] = 1.0
    if n_tokens is None:
        return mask
    return mask.unsqueeze(1).expand(batch, int(n_tokens), phys_dim)


def _iq_physics_from_channels(i: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """从 I/Q（或双通道代理）计算 PHYS_DIM 维特征，不含 mask。"""
    power_t = i.square() + q.square()
    power = power_t.mean(dim=-1).clamp_min(1.0e-8)
    peak = power_t.amax(dim=-1).clamp_min(1.0e-8)
    papr = (peak / power).clamp(max=64.0)
    log_power = torch.log1p(power)
    envelope = safe_sqrt(power_t)
    if envelope.shape[-1] > 1:
        envelope_rate = envelope.diff(dim=-1).abs().mean(dim=-1)
    else:
        envelope_rate = torch.zeros_like(power)

    i_mean = i.mean(dim=-1, keepdim=True)
    q_mean = q.mean(dim=-1, keepdim=True)
    ic = i - i_mean
    qc = q - q_mean
    var_i = ic.square().mean(dim=-1).clamp_min(1.0e-8)
    var_q = qc.square().mean(dim=-1).clamp_min(1.0e-8)
    corr = (ic * qc).mean(dim=-1) / (var_i.sqrt() * var_q.sqrt())
    iq_var_ratio = (var_i / var_q).clamp(1.0e-2, 64.0)

    _logmag, centroid, spread, mag, highband_ratio = _fft_logmag_and_centroid(i, q)
    phase = safe_angle(i.float(), q.float())
    dphi = torch.diff(phase, dim=-1, prepend=phase[..., :1])
    dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi))
    live = (i.square() + q.square()) > 1.0e-8
    live_f = live.to(dtype=dphi.dtype)
    cfo_proxy = (dphi * live_f).sum(dim=-1) / live_f.sum(dim=-1).clamp_min(1.0)

    entropy = _spectral_entropy(mag)

    axis_norm = (var_i + var_q).sqrt()
    if i.shape[-1] > 1:
        axis_drift = 0.5 * (i.diff(dim=-1).abs().mean(dim=-1) + q.diff(dim=-1).abs().mean(dim=-1))
    else:
        axis_drift = torch.zeros_like(power)

    stats = torch.stack(
        [
            log_power,
            papr,
            envelope_rate,
            centroid,
            spread,
            corr,
            iq_var_ratio,
            cfo_proxy,
            entropy,
            highband_ratio,
            axis_norm,
            axis_drift,
        ],
        dim=-1,
    )
    return torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)


def patch_physics(
    iq_patches: torch.Tensor,
    *,
    modality_id: torch.Tensor | None = None,
    complex_pair: torch.Tensor | bool | None = True,
    log_scale: torch.Tensor | None = None,
    return_mask: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """向量化计算每 patch 物理量。

    Args:
        iq_patches: ``[B, N, 2, P]``
        return_mask: 为真时同时返回 ``[B, N, PHYS_DIM]`` 的 modality feature mask

    Returns:
        ``[B, N, PHYS_DIM]``；默认按射频复数对打开 RF 插件、关闭声纳/IMU 插件。
    """
    if iq_patches.ndim != 4 or iq_patches.shape[2] != 2:
        raise ValueError(f"iq_patches 期望 [B,N,2,P]，当前 {tuple(iq_patches.shape)}")
    i = iq_patches[:, :, 0].float()
    q = iq_patches[:, :, 1].float()
    stats = _iq_physics_from_channels(i, q)
    batch, n_tokens, _ = stats.shape
    if modality_id is None:
        modality_id = torch.full((batch,), MODALITY_RF, device=iq_patches.device, dtype=torch.long)
    mask = physics_feature_mask(
        modality_id,
        complex_pair=True if complex_pair is None else complex_pair,
        n_tokens=n_tokens,
        phys_dim=stats.shape[-1],
    ).to(device=stats.device, dtype=stats.dtype)
    masked = stats * mask
    masked = restore_absolute_log_power(masked, log_scale)
    if return_mask:
        return masked, mask
    return masked


def sequence_physics(
    iq: torch.Tensor,
    sample_mask: torch.Tensor | None = None,
    *,
    modality_id: torch.Tensor | None = None,
    complex_pair: torch.Tensor | bool | None = True,
    log_scale: torch.Tensor | None = None,
    return_mask: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """整段波形物理量 ``[B, PHYS_DIM]``，统计只用有效采样点。"""
    if iq.ndim != 3 or iq.shape[1] != 2:
        raise ValueError(f"iq 期望 [B,2,L]，当前 {tuple(iq.shape)}")
    b, _, length = iq.shape
    if sample_mask is None:
        sample_mask = torch.ones(b, length, dtype=torch.bool, device=iq.device)
    x = iq.float().masked_fill(~sample_mask.unsqueeze(1), 0.0)
    mask = sample_mask.float()
    denom = mask.sum(dim=-1).clamp_min(1.0)
    power_t = x.square().sum(dim=1)
    power = (power_t * mask).sum(dim=-1) / denom
    peak = power_t.masked_fill(~sample_mask, 0.0).amax(dim=-1).clamp_min(1.0e-8)
    papr = (peak / power.clamp_min(1.0e-8)).clamp(max=64.0)
    log_power = torch.log1p(power.clamp_min(0.0))
    envelope = safe_sqrt(power_t)
    if length > 1:
        env_diff = envelope.diff(dim=-1).abs() * mask[:, 1:]
        envelope_rate = env_diff.sum(dim=-1) / mask[:, 1:].sum(dim=-1).clamp_min(1.0)
    else:
        envelope_rate = torch.zeros_like(power)

    mean = (x * mask.unsqueeze(1)).sum(dim=-1) / denom.unsqueeze(-1)
    centered = (x - mean.unsqueeze(-1)) * mask.unsqueeze(1)
    var = centered.square().sum(dim=-1) / denom.unsqueeze(-1)
    var_i = var[:, 0].clamp_min(1.0e-8)
    var_q = var[:, 1].clamp_min(1.0e-8)
    corr = (centered[:, 0] * centered[:, 1]).sum(dim=-1) / (denom * torch.sqrt(var_i * var_q))
    ratio = (var_i / var_q).clamp(1.0e-2, 64.0)
    _logmag, centroid, spread, mag, highband_ratio = _fft_logmag_and_centroid(x[:, 0], x[:, 1])
    phase = safe_angle(x[:, 0], x[:, 1])
    dphi = torch.diff(phase, dim=-1, prepend=phase[..., :1])
    dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi))
    live = ((x[:, 0].square() + x[:, 1].square()) > 1.0e-8).to(dtype=dphi.dtype)
    cfo_proxy = (dphi * mask * live).sum(dim=-1) / (mask * live).sum(dim=-1).clamp_min(1.0)

    entropy = _spectral_entropy(mag)
    axis_norm = (var_i + var_q).sqrt()
    if length > 1:
        drift = 0.5 * (
            (x[:, 0].diff(dim=-1).abs() * mask[:, 1:]).sum(dim=-1)
            + (x[:, 1].diff(dim=-1).abs() * mask[:, 1:]).sum(dim=-1)
        ) / mask[:, 1:].sum(dim=-1).clamp_min(1.0)
    else:
        drift = torch.zeros_like(power)

    stats = torch.stack(
        [
            log_power,
            papr,
            envelope_rate,
            centroid,
            spread,
            corr,
            ratio,
            cfo_proxy,
            entropy,
            highband_ratio,
            axis_norm,
            drift,
        ],
        dim=-1,
    )
    stats = torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)
    if modality_id is None:
        modality_id = torch.full((b,), MODALITY_RF, device=iq.device, dtype=torch.long)
    feat_mask = physics_feature_mask(
        modality_id,
        complex_pair=True if complex_pair is None else complex_pair,
        phys_dim=stats.shape[-1],
    ).to(device=stats.device, dtype=stats.dtype)
    masked = stats * feat_mask
    masked = restore_absolute_log_power(masked, log_scale)
    if return_mask:
        return masked, feat_mask
    return masked


def project_patch_energy(
    recon: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """硬约束：按 patch 能量对齐 ``recon ← recon * sqrt(P_true / P_recon)``。"""
    p_true = target.float().square().sum(dim=(-2, -1)).clamp_min(1.0e-8)
    p_recon = recon.float().square().sum(dim=(-2, -1)).clamp_min(1.0e-8)
    scale = (p_true / p_recon).sqrt().to(dtype=recon.dtype)
    aligned = recon * scale.unsqueeze(-1).unsqueeze(-1)
    if mask is None:
        return aligned
    return torch.where(mask.unsqueeze(-1).unsqueeze(-1), aligned, recon)


def stabilize_physics(stats: torch.Tensor) -> torch.Tensor:
    """把 PAPR / 方差比压到对数域，避免个别 patch 把物理损失打到几百。"""
    dim = int(stats.shape[-1])
    cols: list[torch.Tensor] = [stats[..., 0]]
    if dim > 1:
        cols.append(stats[..., 1].clamp_min(1.0).log())
    if dim > 2:
        cols.append(stats[..., 2].clamp_min(0.0))
    if dim > 3:
        cols.append(stats[..., 3].clamp(0.0, 1.0))
    if dim > 4:
        cols.append(stats[..., 4].clamp(0.0, 1.0))
    if dim > 5:
        cols.append(stats[..., 5].clamp(-1.0, 1.0))
    if dim > 6:
        cols.append(stats[..., 6].clamp_min(1.0e-4).log())
    if dim > 7:
        cols.append(stats[..., 7].clamp(-8.0, 8.0))
    if dim > 8:
        cols.extend(stats[..., 8:dim].unbind(dim=-1))
    out = torch.stack(cols, dim=-1)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def physics_constraint_loss(
    recon_patches: torch.Tensor,
    orig_patches: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """软约束：有效/被 mask 的 patch 上 SmoothL1(physics(recon), physics(orig))。"""
    pred = stabilize_physics(patch_physics(recon_patches))
    # 目标物理量是标签，不能回传到 RevIN γ/β。
    target = stabilize_physics(patch_physics(orig_patches.detach()))
    if mask is None:
        return F.smooth_l1_loss(pred, target)
    if mask.sum() == 0:
        return pred.new_tensor(0.0)
    return F.smooth_l1_loss(pred[mask], target[mask])
