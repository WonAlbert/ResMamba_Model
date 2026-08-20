from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from resmamba_signal_model.models.physics import PHYS_DIM, patch_physics, safe_angle, safe_complex_abs
from resmamba_signal_model.models.varlen import n_patches, pad_time_to_patch, patchify_iq


@dataclass
class TimeFreqTokenizerConfig:
    d_model: int = 640
    patch_size: int = 16
    stem_channels: int = 64
    kernels: tuple[int, ...] = field(default_factory=lambda: (4, 8, 16, 32))
    freq_bands: int = 8
    dropout: float = 0.1
    physics_bias: bool = True
    phase_plugin: bool = False
    l_min: int = 16


def _band_pool(logmag: torch.Tensor, n_bands: int) -> torch.Tensor:
    """将 rfft log-magnitude 均分成 n_bands。``logmag``: [..., F]。"""
    n_freq = int(logmag.shape[-1])
    bands = max(1, min(int(n_bands), n_freq))
    edges = torch.linspace(0, n_freq, bands + 1, device=logmag.device)
    parts: list[torch.Tensor] = []
    for i in range(bands):
        start = int(edges[i].item())
        end = max(start + 1, int(edges[i + 1].item()))
        parts.append(logmag[..., start:end].mean(dim=-1))
    pooled = torch.stack(parts, dim=-1)
    if bands < n_bands:
        pooled = F.pad(pooled, (0, n_bands - bands))
    return pooled


class TimeFreqTokenizer(nn.Module):
    """共享 stem + 时域 depthwise 多尺度 + 与时间对齐的频域分带，不注入任务 token。"""

    def __init__(self, cfg: TimeFreqTokenizerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c = cfg.stem_channels
        self.stem = nn.Conv1d(2, c, kernel_size=7, stride=1, padding=3)
        self.time_branches = nn.ModuleList(
            [
                nn.Conv1d(c, c, kernel_size=k, stride=cfg.patch_size, padding=k // 2, groups=c)
                for k in cfg.kernels
            ]
        )
        self.time_fuse = nn.Conv1d(c * len(cfg.kernels), cfg.d_model, kernel_size=1)
        self.freq_proj = nn.Linear(cfg.freq_bands, cfg.d_model)
        self.gate = nn.Linear(cfg.d_model * 2, cfg.d_model)
        self.physics_proj = nn.Linear(PHYS_DIM, cfg.d_model) if cfg.physics_bias else None
        self.phase_proj = nn.Linear(4, cfg.d_model, bias=False) if cfg.phase_plugin else None
        self.dropout = nn.Dropout(cfg.dropout)
        self.norm = nn.LayerNorm(cfg.d_model)

    def _time_tokens(self, iq: torch.Tensor, n_tok: int) -> torch.Tensor:
        h = self.stem(iq)
        branches = []
        for conv in self.time_branches:
            y = conv(h)
            if y.shape[-1] >= n_tok:
                y = y[..., :n_tok]
            else:
                y = F.pad(y, (0, n_tok - y.shape[-1]))
            branches.append(y)
        fused = self.time_fuse(torch.cat(branches, dim=1))
        return fused.transpose(1, 2).contiguous()

    def _freq_tokens(self, iq_patches: torch.Tensor) -> torch.Tensor:
        i = iq_patches[:, :, 0].float()
        q = iq_patches[:, :, 1].float()
        z = torch.complex(i, q)
        spec = torch.fft.fft(z, dim=-1)
        logmag = safe_complex_abs(spec).log()
        logmag = logmag[..., : spec.shape[-1] // 2 + 1]
        bands = _band_pool(logmag, self.cfg.freq_bands)
        return self.freq_proj(bands.to(dtype=iq_patches.dtype))

    def _phase_tokens(self, iq_patches: torch.Tensor) -> torch.Tensor:
        """相对相位增量 + 相邻样本共轭相关，仅 RF 复数对启用。"""
        i = iq_patches[:, :, 0].float()
        q = iq_patches[:, :, 1].float()
        z = torch.complex(i, q)
        phase = safe_angle(i, q)
        dphi = torch.diff(phase, dim=-1, prepend=phase[..., :1])
        dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi))
        mean_dphi = dphi.mean(dim=-1)
        std_dphi = dphi.std(dim=-1, unbiased=False)
        if z.shape[-1] > 1:
            conj = (z[..., :-1].conj() * z[..., 1:]).mean(dim=-1)
        else:
            conj = torch.zeros_like(z[..., 0])
        stats = torch.stack([mean_dphi, std_dphi, conj.real, conj.imag], dim=-1)
        stats = torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)
        assert self.phase_proj is not None
        return self.phase_proj(stats.to(dtype=iq_patches.dtype))

    def forward(
        self,
        iq: torch.Tensor,
        sample_mask: torch.Tensor | None = None,
        *,
        modality_id: torch.Tensor | None = None,
        complex_pair: torch.Tensor | bool | None = True,
        **_unused,
    ) -> dict[str, torch.Tensor]:
        if iq.ndim != 3 or iq.shape[1] != 2:
            raise ValueError(f"tokenizer 期望 iq [B,2,L]，当前 {tuple(iq.shape)}")
        batch, _c, length = iq.shape
        if sample_mask is None:
            sample_mask = torch.ones(batch, length, dtype=torch.bool, device=iq.device)
        orig_len = length
        n_tok = n_patches(length, self.cfg.patch_size)
        iq_p, mask_p = pad_time_to_patch(iq, sample_mask, self.cfg.patch_size)
        time_tok = self._time_tokens(iq_p, n_tok)
        iq_patches, patch_mask = patchify_iq(iq, sample_mask, self.cfg.patch_size)
        n_tok = min(n_tok, iq_patches.shape[1], time_tok.shape[1])
        time_tok = time_tok[:, :n_tok]
        iq_patches = iq_patches[:, :n_tok]
        patch_mask = patch_mask[:, :n_tok]
        freq_tok = self._freq_tokens(iq_patches)
        gate = torch.sigmoid(self.gate(torch.cat([time_tok, freq_tok], dim=-1)))
        tokens = gate * time_tok + (1.0 - gate) * freq_tok
        phys, phys_mask = patch_physics(iq_patches, modality_id=modality_id, complex_pair=complex_pair, return_mask=True)
        phys = phys.detach()
        if self.physics_proj is not None:
            tokens = tokens + self.physics_proj(phys.to(dtype=tokens.dtype))
        if self.phase_proj is not None:
            tokens = tokens + self._phase_tokens(iq_patches)
        tokens = self.dropout(self.norm(tokens))
        tokens = tokens.masked_fill(~patch_mask.unsqueeze(-1), 0.0)
        return {
            "tokens": torch.nan_to_num(tokens),
            "token_mask": patch_mask,
            "patch_mask": patch_mask,
            "iq_patch_targets": iq_patches,
            "patch_physics": phys,
            "physics_mask": phys_mask,
            "orig_length": torch.full((batch,), orig_len, device=iq.device, dtype=torch.long),
            "patch_offset": 0,
        }
