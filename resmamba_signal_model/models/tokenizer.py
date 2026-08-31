from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from resmamba_signal_model.models.elastic_sampler import ElasticRoVSampler
from resmamba_signal_model.models.moe import MOE_EXPERT_NAMES, MoEAux, MoEFusion
from resmamba_signal_model.models.physics import PHYS_DIM, patch_physics, safe_angle, safe_complex_abs
from resmamba_signal_model.models.varlen import n_patches, pad_time_to_patch, patchify_iq


@dataclass
class TimeFreqTokenizerConfig:
    d_model: int = 640
    patch_size: int = 16
    stem_channels: int = 64
    kernels: tuple[int, ...] = field(default_factory=lambda: (4, 8, 16, 32))
    intrapulse_kernels: tuple[int, ...] = field(default_factory=lambda: (16, 32, 64))
    freq_bands: int = 8
    dropout: float = 0.1
    physics_bias: bool = True
    phase_plugin: bool = False
    enable_moe: bool = True
    moe_num_experts: int = 3
    moe_top_k: int | None = None
    l_min: int = 16
    # fixed_patch：均匀切分；elastic_rov：RoV 概率采样 + 弹性长度
    tokenization_mode: str = "fixed_patch"
    elastic_length_scale: float = 0.5


def _band_pool(logmag: torch.Tensor, n_bands: int) -> torch.Tensor:
    """将双侧频谱 log-magnitude 均分成 n_bands。``logmag``: [..., F]（已按频率轴排序）。"""
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


class _TimeBranchExpert(nn.Module):
    def __init__(
        self,
        stem_channels: int,
        d_model: int,
        patch_size: int,
        kernels: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    stem_channels,
                    stem_channels,
                    kernel_size=k,
                    stride=patch_size,
                    padding=k // 2,
                    groups=stem_channels,
                )
                for k in kernels
            ]
        )
        self.fuse = nn.Conv1d(stem_channels * len(kernels), d_model, kernel_size=1)

    def forward(self, stem: torch.Tensor, n_tok: int) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for conv in self.branches:
            y = conv(stem)
            if y.shape[-1] >= n_tok:
                y = y[..., :n_tok]
            else:
                y = F.pad(y, (0, n_tok - y.shape[-1]))
            parts.append(y)
        fused = self.fuse(torch.cat(parts, dim=1))
        return fused.transpose(1, 2).contiguous()


class _ElasticPatchExpert(nn.Module):
    """弹性 patch 直接嵌入（RoV 采样位置与固定 stride conv 不对齐时使用）。"""

    def __init__(self, patch_size: int, d_model: int) -> None:
        super().__init__()
        flat = 2 * int(patch_size)
        self.net = nn.Sequential(
            nn.Linear(flat, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, iq_patches: torch.Tensor) -> torch.Tensor:
        b, n, _c, p = iq_patches.shape
        x = iq_patches.reshape(b, n, -1)
        return self.net(x)


class TimeFreqTokenizer(nn.Module):
    """共享 stem + 三路具名专家（LD 脉内 / LD 型号 / TX 调制）+ task/family 硬路由融合。"""

    def __init__(self, cfg: TimeFreqTokenizerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c = cfg.stem_channels
        self.stem = nn.Conv1d(2, c, kernel_size=7, stride=1, padding=3)
        self.intrapulse_expert = _TimeBranchExpert(c, cfg.d_model, cfg.patch_size, cfg.intrapulse_kernels)
        self.model_expert = _TimeBranchExpert(c, cfg.d_model, cfg.patch_size, cfg.kernels)
        self.freq_proj = nn.Linear(cfg.freq_bands, cfg.d_model)
        self.phase_proj = nn.Linear(4, cfg.d_model, bias=False) if cfg.phase_plugin else None
        num_branches = min(int(cfg.moe_num_experts), len(MOE_EXPERT_NAMES))
        self.num_experts = max(1, num_branches)
        self.moe_fusion = (
            MoEFusion(cfg.d_model, self.num_experts, top_k=cfg.moe_top_k) if cfg.enable_moe else None
        )
        self.physics_proj = nn.Linear(PHYS_DIM, cfg.d_model) if cfg.physics_bias else None
        self.dropout = nn.Dropout(cfg.dropout)
        self.norm = nn.LayerNorm(cfg.d_model)
        self._last_moe_aux: list[MoEAux] = []
        self._elastic_mode = str(getattr(cfg, "tokenization_mode", "fixed_patch")).strip().lower() == "elastic_rov"
        if self._elastic_mode:
            self.elastic_sampler = ElasticRoVSampler(
                cfg.patch_size,
                length_scale=float(getattr(cfg, "elastic_length_scale", 0.5)),
            )
            self.intrapulse_elastic = _ElasticPatchExpert(cfg.patch_size, cfg.d_model)
            self.model_elastic = _ElasticPatchExpert(cfg.patch_size, cfg.d_model)
        else:
            self.elastic_sampler = None
            self.intrapulse_elastic = None
            self.model_elastic = None

    def pop_moe_aux(self) -> list[MoEAux]:
        aux = list(self._last_moe_aux)
        self._last_moe_aux.clear()
        return aux

    def _freq_tokens(self, iq_patches: torch.Tensor) -> torch.Tensor:
        """复数 I/Q 双侧 FFT：保留负频，fftshift 后再均分频带（TX 调制专家）。"""
        i = iq_patches[:, :, 0].float()
        q = iq_patches[:, :, 1].float()
        z = torch.complex(i, q)
        spec = torch.fft.fft(z, dim=-1)
        logmag = safe_complex_abs(spec).log()
        logmag = torch.fft.fftshift(logmag, dim=-1)
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

    def _expert_branches(
        self,
        iq_p: torch.Tensor,
        iq_patches: torch.Tensor,
        n_tok: int,
    ) -> list[torch.Tensor]:
        if self._elastic_mode:
            assert self.intrapulse_elastic is not None and self.model_elastic is not None
            intrapulse = self.intrapulse_elastic(iq_patches)
            if self.phase_proj is not None:
                intrapulse = intrapulse + self._phase_tokens(iq_patches)
            model_tok = self.model_elastic(iq_patches)
            tx_tok = self._freq_tokens(iq_patches)
            return [intrapulse, model_tok, tx_tok][: self.num_experts]
        stem = self.stem(iq_p)
        intrapulse = self.intrapulse_expert(stem, n_tok)
        if self.phase_proj is not None:
            intrapulse = intrapulse + self._phase_tokens(iq_patches)
        model_tok = self.model_expert(stem, n_tok)
        tx_tok = self._freq_tokens(iq_patches)
        branches = [intrapulse, model_tok, tx_tok][: self.num_experts]
        return branches

    def forward(
        self,
        iq: torch.Tensor,
        sample_mask: torch.Tensor | None = None,
        *,
        modality_id: torch.Tensor | None = None,
        complex_pair: torch.Tensor | bool | None = True,
        moe_route_weights: torch.Tensor | None = None,
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
        elastic_starts: torch.Tensor | None = None
        if self._elastic_mode:
            assert self.elastic_sampler is not None
            iq_patches, patch_mask, elastic_starts = self.elastic_sampler(iq, sample_mask, n_samples=n_tok)
            n_tok = int(iq_patches.shape[1])
        else:
            iq_patches, patch_mask = patchify_iq(iq, sample_mask, self.cfg.patch_size)
            n_tok = min(n_tok, iq_patches.shape[1])
            iq_patches = iq_patches[:, :n_tok]
            patch_mask = patch_mask[:, :n_tok]

        branches = self._expert_branches(iq_p, iq_patches, n_tok)
        pad_mask = ~patch_mask
        self._last_moe_aux.clear()
        if self.moe_fusion is not None:
            tokens, aux = self.moe_fusion(
                branches,
                key_padding_mask=pad_mask,
                route_weights=moe_route_weights,
            )
            self._last_moe_aux.append(aux)
        else:
            tokens = sum(branches) / float(len(branches))

        phys, phys_mask = patch_physics(
            iq_patches,
            modality_id=modality_id,
            complex_pair=complex_pair,
            return_mask=True,
        )
        phys = phys.detach()
        if self.physics_proj is not None:
            tokens = tokens + self.physics_proj(phys.to(dtype=tokens.dtype))
        tokens = self.dropout(self.norm(tokens))
        tokens = tokens.masked_fill(~patch_mask.unsqueeze(-1), 0.0)
        out: dict[str, torch.Tensor | int] = {
            "tokens": torch.nan_to_num(tokens),
            "token_mask": patch_mask,
            "patch_mask": patch_mask,
            "iq_patch_targets": iq_patches,
            "patch_physics": phys,
            "physics_mask": phys_mask,
            "orig_length": torch.full((batch,), orig_len, device=iq.device, dtype=torch.long),
            "patch_offset": 0,
        }
        if elastic_starts is not None:
            out["elastic_start_indices"] = elastic_starts
        return out
