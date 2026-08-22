from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from resmamba_signal_model.models.physics import PHYS_DIM, sequence_physics
from resmamba_signal_model.models.revin import RevINStats

# mean_I/Q + log std_I/Q + 原始波形物理量。RevIN 会抹掉前两项，必须从 raw IQ 回灌。
EMITTER_STAT_DIM = 4 + PHYS_DIM


def raw_emitter_stats(
    iq: torch.Tensor,
    sample_mask: torch.Tensor | None = None,
    *,
    revin_stats: RevINStats | None = None,
    modality_id: torch.Tensor | None = None,
    complex_pair: torch.Tensor | bool | None = True,
) -> torch.Tensor:
    """从 **未 RevIN** 的 I/Q 抽取个体指纹统计量 ``[B, EMITTER_STAT_DIM]``。"""
    if iq.ndim != 3 or iq.shape[1] != 2:
        raise ValueError(f"raw_emitter_stats 期望 iq [B,2,L]，当前 {tuple(iq.shape)}")
    x = iq.float()
    if sample_mask is None:
        sample_mask = torch.ones(x.shape[0], x.shape[-1], dtype=torch.bool, device=x.device)
    mask = sample_mask.to(device=x.device, dtype=torch.bool)
    weights = mask.unsqueeze(1).to(dtype=x.dtype)
    denom = weights.sum(dim=-1).clamp_min(1.0)
    if revin_stats is not None:
        mean = revin_stats.mean.float()
        std = revin_stats.std.float().clamp_min(1.0e-8)
        if mean.ndim == 1:
            mean = mean.unsqueeze(0)
        if std.ndim == 1:
            std = std.unsqueeze(0)
        if mean.shape[0] == 1 and x.shape[0] > 1:
            mean = mean.expand(x.shape[0], -1)
            std = std.expand(x.shape[0], -1)
    else:
        mean = (x * weights).sum(dim=-1) / denom
        centered = (x - mean.unsqueeze(-1)) * weights
        var = centered.square().sum(dim=-1) / denom
        std = var.clamp_min(1.0e-8).sqrt()
    log_std = std.clamp_min(1.0e-8).log()
    phys = sequence_physics(
        x,
        mask,
        modality_id=modality_id,
        complex_pair=complex_pair,
    )
    stats = torch.cat([mean, log_std, phys.float()], dim=-1)
    return torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)


class EmitterFingerprintBranch(nn.Module):
    """Raw I/Q 浅卷积 + 统计 MLP。不改 tokenizer/encoder，旧预训练权重可原样加载。"""

    def __init__(
        self,
        d_model: int,
        *,
        conv_channels: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        c = max(8, int(conv_channels))
        groups = 8 if c % 8 == 0 else 4 if c % 4 == 0 else 1
        out_groups = 8 if d_model % 8 == 0 else 4 if d_model % 4 == 0 else 1
        self.conv = nn.Sequential(
            nn.Conv1d(2, c, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(groups, c),
            nn.GELU(),
            nn.Conv1d(c, c, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(groups, c),
            nn.GELU(),
            nn.Conv1d(c, d_model, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(out_groups, d_model),
            nn.GELU(),
        )
        self.stat_mlp = nn.Sequential(
            nn.LayerNorm(EMITTER_STAT_DIM),
            nn.Linear(EMITTER_STAT_DIM, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        iq: torch.Tensor,
        sample_mask: torch.Tensor | None = None,
        *,
        revin_stats: RevINStats | None = None,
        modality_id: torch.Tensor | None = None,
        complex_pair: torch.Tensor | bool | None = True,
    ) -> torch.Tensor:
        stats = raw_emitter_stats(
            iq,
            sample_mask,
            revin_stats=revin_stats,
            modality_id=modality_id,
            complex_pair=complex_pair,
        )
        hidden = self.conv(iq.float())
        if sample_mask is None:
            pooled = hidden.mean(dim=-1)
            peak = hidden.amax(dim=-1)
        else:
            weights = sample_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(1)
            if weights.shape[-1] != hidden.shape[-1]:
                weights = F.interpolate(weights, size=hidden.shape[-1], mode="nearest")
            denom = weights.sum(dim=-1).clamp_min(1.0)
            pooled = (hidden * weights).sum(dim=-1) / denom
            peak = hidden.masked_fill(weights < 0.5, torch.finfo(hidden.dtype).min).amax(dim=-1)
            peak = torch.nan_to_num(peak, nan=0.0, posinf=0.0, neginf=0.0)
        conv_feat = 0.5 * (pooled + peak)
        fused = self.out_norm(conv_feat + self.stat_mlp(stats.to(dtype=conv_feat.dtype)))
        return F.normalize(fused.float(), dim=-1, eps=1.0e-6).to(dtype=iq.dtype)
