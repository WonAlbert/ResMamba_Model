from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


AMP_AUX_DIM = 4
AMP_AUX_NAMES = ("log_scale", "log_peak", "papr_preclip", "scale_gap")


@dataclass
class AmplitudeAux:
    """预归一化绝对幅度旁路；不进 z_enc / 分类 / UTI semantic。"""

    log_scale: torch.Tensor
    log_peak: torch.Tensor
    papr_preclip: torch.Tensor
    scale_gap: torch.Tensor


@dataclass
class RevINStats:
    mean: torch.Tensor
    std: torch.Tensor
    amp_aux: AmplitudeAux | None = None


def clip_normalized(x: torch.Tensor, clip: float | None) -> torch.Tensor:
    """限制归一化波形幅度。不改变 stats，因此不会把目标能量泄漏进均值/方差。"""
    if clip is None or float(clip) <= 0:
        return x
    bound = float(clip)
    return x.clamp(-bound, bound)


def _masked_mean_std(
    x: torch.Tensor,
    sample_mask: torch.Tensor | None,
    eps: float,
    std_min: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """x: [B, C, L]，按样本、通道在有效长度上统计，pad 不计入。"""
    if sample_mask is None:
        mean = x.mean(dim=-1)
        var = x.var(dim=-1, unbiased=False)
    else:
        mask = sample_mask.to(dtype=x.dtype).unsqueeze(1)
        denom = mask.sum(dim=-1).clamp_min(1.0)
        mean = (x * mask).sum(dim=-1) / denom
        centered = (x - mean.unsqueeze(-1)) * mask
        var = centered.square().sum(dim=-1) / denom
    std = (var + eps).sqrt()
    if std_min > 0:
        std = std.clamp_min(float(std_min))
    return mean, std


def _finite_stats(mean: torch.Tensor, std: torch.Tensor, floor: float) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    std = torch.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(floor)
    return mean, std


def _stabilize_zscore(z: torch.Tensor, clip: float | None) -> torch.Tensor:
    """仿射前去掉 Inf/NaN，避免 ``0 * Inf`` 把 γ/β 梯度打成 NaN。"""
    z = torch.nan_to_num(z, nan=0.0, posinf=1.0e4, neginf=-1.0e4)
    if clip is None or float(clip) <= 0:
        return z
    bound = float(clip)
    return z.clamp(-bound, bound)


def _masked_joint_energy_stats(
    x: torch.Tensor,
    stats_mask: torch.Tensor | None,
    *,
    eps: float,
    std_min: float,
    winsorize_top_frac: float,
) -> tuple[torch.Tensor, torch.Tensor, AmplitudeAux]:
    """联合能量归一：去 DC → Winsorize 功率 → 标量 RMS；旁路绝对幅度。"""
    if stats_mask is None:
        mask = torch.ones(x.shape[0], x.shape[-1], dtype=x.dtype, device=x.device)
    else:
        mask = stats_mask.to(dtype=x.dtype)
    denom = mask.sum(dim=-1).clamp_min(1.0)
    mean = (x * mask.unsqueeze(1)).sum(dim=-1) / denom.unsqueeze(-1)
    centered = (x - mean.unsqueeze(-1)) * mask.unsqueeze(1)

    power_t = centered.square().sum(dim=1)
    raw_rms = torch.sqrt((power_t * mask).sum(dim=-1) / denom).clamp_min(eps)

    if float(winsorize_top_frac) > 0.0 and power_t.shape[-1] > 1:
        n_obs = mask.to(dtype=torch.long).sum(dim=-1).clamp_min(1)
        sorted_p = power_t.masked_fill(mask <= 0, float("inf")).sort(dim=-1).values
        q = 1.0 - float(winsorize_top_frac)
        idx = ((n_obs.float() - 1.0) * q).floor().long()
        idx = torch.minimum(idx.clamp_min(0), n_obs - 1)
        thresh = sorted_p.gather(1, idx.unsqueeze(1))
        thresh = torch.nan_to_num(thresh, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        capped = torch.minimum(power_t, thresh)
    else:
        capped = power_t
    winsor_rms = torch.sqrt((capped * mask).sum(dim=-1) / denom).clamp_min(eps)
    if std_min > 0:
        winsor_rms = winsor_rms.clamp_min(float(std_min))

    preclip_peak = power_t.masked_fill(mask <= 0, 0.0).amax(dim=-1).clamp_min(eps)
    preclip_power = (power_t * mask).sum(dim=-1) / denom
    papr_preclip = (preclip_peak / preclip_power.clamp_min(eps)).clamp(max=256.0)
    scale_gap = torch.log(raw_rms.clamp_min(eps)) - torch.log(winsor_rms.clamp_min(eps))

    amp_aux = AmplitudeAux(
        log_scale=torch.log(winsor_rms.clamp_min(eps)),
        log_peak=torch.log(preclip_peak),
        papr_preclip=papr_preclip,
        scale_gap=scale_gap,
    )
    std = winsor_rms.unsqueeze(-1).expand_as(mean)
    return mean, std, amp_aux


def _apply_peak_papr_clip(x: torch.Tensor, peak_papr_clip: float, sample_mask: torch.Tensor | None) -> torch.Tensor:
    """归一后若某时刻 ``I²+Q²`` 超过 ``peak_papr_clip``，只收缩该时刻。"""
    if peak_papr_clip is None or float(peak_papr_clip) <= 0:
        return x
    cap = float(peak_papr_clip)
    power_t = x.square().sum(dim=1, keepdim=True)
    shrink = (power_t / cap).clamp_min(1.0).sqrt()
    clipped = x / shrink
    if sample_mask is not None:
        clipped = clipped.masked_fill(~sample_mask.unsqueeze(1), 0.0)
    return clipped


class RevIN(nn.Module):
    """可逆实例归一化；``scale_mode=joint_energy`` 时 I/Q 共用一个能量尺度。"""

    def __init__(
        self,
        num_channels: int = 2,
        eps: float = 1e-5,
        affine: bool = True,
        std_min: float = 0.0,
        clip: float | None = None,
        *,
        scale_mode: str = "joint_energy",
        winsorize_top_frac: float = 0.01,
        peak_papr_clip: float = 16.0,
        shared_affine: bool = True,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.std_min = float(std_min)
        self.clip = None if clip is None or float(clip) <= 0 else float(clip)
        self.scale_mode = str(scale_mode).strip().lower()
        self.winsorize_top_frac = float(winsorize_top_frac)
        self.peak_papr_clip = float(peak_papr_clip)
        self.shared_affine = bool(shared_affine)
        if affine:
            if self.scale_mode == "joint_energy" and shared_affine:
                self.gamma = nn.Parameter(torch.ones(1))
                self.beta = nn.Parameter(torch.zeros(1))
            else:
                self.gamma = nn.Parameter(torch.ones(num_channels))
                self.beta = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

    def _compute_stats(
        self,
        x: torch.Tensor,
        stats_mask: torch.Tensor | None,
    ) -> RevINStats:
        if self.scale_mode == "joint_energy":
            mean, std, amp_aux = _masked_joint_energy_stats(
                x,
                stats_mask,
                eps=self.eps,
                std_min=self.std_min,
                winsorize_top_frac=self.winsorize_top_frac,
            )
            return RevINStats(mean=mean, std=std, amp_aux=amp_aux)
        mean, std = _masked_mean_std(x, stats_mask, self.eps, std_min=self.std_min)
        return RevINStats(mean=mean, std=std, amp_aux=None)

    def normalize(
        self,
        x: torch.Tensor,
        sample_mask: torch.Tensor | None = None,
        *,
        observed_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RevINStats]:
        stats_mask = sample_mask
        if observed_mask is not None:
            observed_mask = observed_mask.to(device=x.device, dtype=torch.bool)
            stats_mask = observed_mask if sample_mask is None else (sample_mask & observed_mask)
        stats = self._compute_stats(x, stats_mask)
        x_hat = self.apply_stats(x, stats, sample_mask)
        output_mask = stats_mask if observed_mask is not None else sample_mask
        if output_mask is not None:
            x_hat = x_hat.masked_fill(~output_mask.unsqueeze(1), 0.0)
        return x_hat, stats

    def apply_stats(
        self,
        x: torch.Tensor,
        stats: RevINStats,
        sample_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """使用既有 observed-only 统计量归一化，不重新读取目标值计算统计。"""
        floor = self.std_min if self.std_min > 0 else self.eps
        mean, std = _finite_stats(stats.mean, stats.std, floor)
        x_hat = (x - mean.unsqueeze(-1)) / std.unsqueeze(-1)
        x_hat = _stabilize_zscore(x_hat, self.clip)
        if self.affine:
            if self.scale_mode == "joint_energy" and self.shared_affine:
                x_hat = x_hat * self.gamma.view(1, 1, 1) + self.beta.view(1, 1, 1)
            else:
                x_hat = x_hat * self.gamma.view(1, -1, 1) + self.beta.view(1, -1, 1)
        if self.scale_mode == "joint_energy":
            x_hat = _apply_peak_papr_clip(x_hat, self.peak_papr_clip, sample_mask)
        if sample_mask is not None:
            x_hat = x_hat.masked_fill(~sample_mask.unsqueeze(1), 0.0)
        return x_hat

    def denormalize(self, x_hat: torch.Tensor, stats: RevINStats) -> torch.Tensor:
        x = x_hat
        if self.affine:
            if self.scale_mode == "joint_energy" and self.shared_affine:
                x = (x - self.beta.view(1, 1, 1)) / self.gamma.view(1, 1, 1).clamp_min(self.eps)
            else:
                x = (x - self.beta.view(1, -1, 1)) / self.gamma.view(1, -1, 1).clamp_min(self.eps)
        floor = self.std_min if self.std_min > 0 else self.eps
        mean, std = _finite_stats(stats.mean, stats.std, floor)
        return x * std.unsqueeze(-1) + mean.unsqueeze(-1)

    def amp_aux_vector(self, stats: RevINStats) -> torch.Tensor | None:
        """``[B, 4]``：log_scale, log_peak, papr_preclip, scale_gap。"""
        aux = stats.amp_aux
        if aux is None:
            return None
        return torch.stack(
            [aux.log_scale, aux.log_peak, aux.papr_preclip, aux.scale_gap],
            dim=-1,
        )

    def forward(
        self,
        x: torch.Tensor,
        sample_mask: torch.Tensor | None = None,
        *,
        stats: RevINStats | None = None,
        reverse: bool = False,
        observed_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RevINStats]:
        if reverse:
            if stats is None:
                raise ValueError("denormalize 需要传入 normalize 得到的 stats")
            return self.denormalize(x, stats), stats
        return self.normalize(x, sample_mask, observed_mask=observed_mask)
