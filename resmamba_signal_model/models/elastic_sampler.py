from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from resmamba_signal_model.models.physics import safe_angle
from resmamba_signal_model.models.varlen import n_patches, pad_time_to_patch


def compute_rov_weights(
    iq: torch.Tensor,
    sample_mask: torch.Tensor | None,
    *,
    sigma: float = 1.0e-6,
) -> torch.Tensor:
    """RF I/Q 变化率（RoV）：包络导数 + 相位跳变 + I/Q 差分。返回 ``[B, L-1]``。"""
    i = iq[:, 0].float()
    q = iq[:, 1].float()
    env = torch.sqrt(i.square() + q.square() + sigma)
    env_diff = torch.abs(torch.diff(env, dim=-1))
    phase = safe_angle(i, q)
    dphi = torch.diff(phase, dim=-1)
    dphi = torch.abs(torch.atan2(torch.sin(dphi), torch.cos(dphi)))
    iq_diff = torch.sqrt(torch.diff(i, dim=-1).square() + torch.diff(q, dim=-1).square() + sigma)
    rov = env_diff + dphi + iq_diff
    if sample_mask is not None:
        valid = sample_mask[:, 1:] & sample_mask[:, :-1]
        rov = rov * valid.to(dtype=rov.dtype)
    return rov + sigma


def rov_start_probabilities(rov: torch.Tensor, patch_size: int, max_start: int) -> torch.Tensor:
    """把 RoV 聚合为每个起点窗口的采样概率 ``[B, max_start]``。"""
    batch, rov_len = rov.shape
    p = int(patch_size)
    max_start = max(1, int(max_start))
    if rov_len < 1:
        return rov.new.ones(batch, max_start) / float(max_start)
    cs = torch.zeros(batch, rov_len + 1, device=rov.device, dtype=rov.dtype)
    cs[:, 1:] = rov.cumsum(dim=-1)
    starts = torch.arange(max_start, device=rov.device)
    end_idx = (starts + p - 1).clamp_max(rov_len)
    win = cs[:, end_idx] - cs[:, starts.clamp_max(rov_len)]
    win = win.clamp_min(1.0e-8)
    return win / win.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)


def gather_elastic_patches(
    iq: torch.Tensor,
    start_indices: torch.Tensor,
    patch_size: int,
    *,
    sample_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按起点索引提取 I/Q patch。``start_indices``: ``[B, N]`` → ``patches [B,N,2,P]``。"""
    iq_pad, mask_pad = pad_time_to_patch(iq, sample_mask, patch_size)
    batch, _, length = iq_pad.shape
    n_tok = int(start_indices.shape[1])
    p = int(patch_size)
    base = torch.arange(p, device=iq.device).view(1, 1, p)
    idx = start_indices.unsqueeze(-1) + base  # [B, N, P]
    idx = idx.clamp_max(length - 1)
    flat = iq_pad.reshape(batch, 2, length)
    idx_flat = idx.reshape(batch, n_tok * p)
    gathered = torch.gather(flat, dim=-1, index=idx_flat.unsqueeze(1).expand(-1, 2, -1))
    patches = gathered.reshape(batch, 2, n_tok, p).permute(0, 2, 1, 3).contiguous()
    if sample_mask is not None:
        m_idx = idx.reshape(batch, n_tok, p)
        patch_mask = torch.gather(mask_pad.unsqueeze(1).expand(-1, n_tok, -1), dim=-1, index=m_idx).any(dim=-1)
    else:
        patch_mask = torch.ones(batch, n_tok, dtype=torch.bool, device=iq.device)
    return patches, patch_mask


class ElasticRoVSampler(nn.Module):
    """PATK 风格 RoV 起点采样 + 可学习弹性长度 + 线性插值回固定 patch。"""

    def __init__(self, patch_size: int, *, length_scale: float = 0.5) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.length_scale = float(length_scale)
        p = self.patch_size
        self.length_weights = nn.Sequential(nn.Linear(p, 1), nn.Tanh())
        self.sign_weights = nn.Sequential(nn.Linear(p, 1), nn.ReLU(), nn.Sigmoid())

    def _elastic_interpolate(self, flat_patches: torch.Tensor) -> torch.Tensor:
        """``flat_patches [M, P]`` → 弹性截取 + 插值回 P。"""
        m, p = flat_patches.shape
        if m == 0:
            return flat_patches
        lw = 0.5 * self.length_scale * self.length_weights(flat_patches)
        elen = torch.round(p * (1.0 + lw.squeeze(-1))).to(torch.int64).clamp(min=1, max=p)
        out = flat_patches.new_zeros(m, p)
        for i in range(m):
            n = int(elen[i].item())
            seg = flat_patches[i, :n].view(1, 1, -1)
            out[i] = F.interpolate(seg, size=p, mode="linear", align_corners=True).view(-1)
        weighted = self.sign_weights(out) * out
        return weighted

    def forward(
        self,
        iq: torch.Tensor,
        sample_mask: torch.Tensor | None,
        n_samples: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 ``(patches [B,N,2,P], patch_mask [B,N], start_indices [B,N])``。"""
        batch, _, length = iq.shape
        p = self.patch_size
        iq_pad, mask_pad = pad_time_to_patch(iq, sample_mask, p)
        length = int(iq_pad.shape[-1])
        n_tok = int(n_samples) if n_samples is not None else n_patches(length, p)
        max_start = max(1, length - p + 1)
        n_draw = min(n_tok, max_start)

        rov = compute_rov_weights(iq_pad, mask_pad)
        probs = rov_start_probabilities(rov, p, max_start)

        start_list: list[torch.Tensor] = []
        for b in range(batch):
            valid_n = n_draw
            if mask_pad is not None:
                # 只允许起点 s 使 patch [s,s+p) 落在有效区
                base = torch.arange(max_start, device=iq.device)
                ends = (base + p).clamp_max(length)
                valid_starts = mask_pad[b, base] & mask_pad[b, (ends - 1).clamp_min(0)]
                if int(valid_starts.sum()) < 1:
                    valid_starts = torch.ones(max_start, dtype=torch.bool, device=iq.device)
                p_b = probs[b].clone()
                p_b = p_b * valid_starts.to(dtype=p_b.dtype)
                if float(p_b.sum()) <= 0:
                    p_b = valid_starts.to(dtype=p_b.dtype)
                p_b = p_b / p_b.sum().clamp_min(1.0e-8)
                valid_n = min(n_draw, int(valid_starts.sum().item()))
                valid_n = max(valid_n, 1)
                idx = torch.multinomial(p_b, valid_n, replacement=False)
            else:
                idx = torch.multinomial(probs[b], valid_n, replacement=False)
            idx, _ = torch.sort(idx)
            if valid_n < n_tok:
                pad_idx = idx[-1].expand(n_tok - valid_n)
                idx = torch.cat([idx, pad_idx], dim=0)
            start_list.append(idx[:n_tok])
        start_indices = torch.stack(start_list, dim=0)

        patches, patch_mask = gather_elastic_patches(iq_pad, start_indices, p, sample_mask=mask_pad)
        b, n, _c, _p = patches.shape
        flat_i = patches[:, :, 0, :].reshape(b * n, p)
        flat_q = patches[:, :, 1, :].reshape(b * n, p)
        flat_i = self._elastic_interpolate(flat_i)
        flat_q = self._elastic_interpolate(flat_q)
        patches = torch.stack(
            [flat_i.reshape(b, n, p), flat_q.reshape(b, n, p)],
            dim=2,
        )
        return patches.to(dtype=iq.dtype), patch_mask, start_indices
