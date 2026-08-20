from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class RevINStats:
    mean: torch.Tensor
    std: torch.Tensor


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


class RevIN(nn.Module):
    """可逆实例归一化（Kim et al., ICLR 2022），针对 I/Q 两通道。"""

    def __init__(
        self,
        num_channels: int = 2,
        eps: float = 1e-5,
        affine: bool = True,
        std_min: float = 0.0,
        clip: float | None = None,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.std_min = float(std_min)
        self.clip = None if clip is None or float(clip) <= 0 else float(clip)
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_channels))
            self.beta = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

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
        mean, std = _masked_mean_std(x, stats_mask, self.eps, std_min=self.std_min)
        stats = RevINStats(mean=mean, std=std)
        x_hat = self.apply_stats(x, stats)
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
            x_hat = x_hat * self.gamma.view(1, -1, 1) + self.beta.view(1, -1, 1)
        if sample_mask is not None:
            x_hat = x_hat.masked_fill(~sample_mask.unsqueeze(1), 0.0)
        return x_hat

    def denormalize(self, x_hat: torch.Tensor, stats: RevINStats) -> torch.Tensor:
        x = x_hat
        if self.affine:
            x = (x - self.beta.view(1, -1, 1)) / self.gamma.view(1, -1, 1).clamp_min(self.eps)
        floor = self.std_min if self.std_min > 0 else self.eps
        mean, std = _finite_stats(stats.mean, stats.std, floor)
        return x * std.unsqueeze(-1) + mean.unsqueeze(-1)

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
