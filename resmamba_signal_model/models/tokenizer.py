from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MultiScaleTokenizerConfig:
    d_model: int = 256
    patch_size: int = 8
    max_tokens: int = 1024
    num_datasets: int = 16
    num_tasks: int = 2
    kernels: tuple[int, ...] = field(default_factory=lambda: (4, 8, 16, 32))
    dropout: float = 0.1


class ResConv1DBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.GroupNorm(1, channels), nn.GELU(), nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.Dropout(dropout), nn.GroupNorm(1, channels), nn.GELU(), nn.Conv1d(channels, channels, 1),
        )
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if y.shape[-1] != x.shape[-1]:
            y = y[..., : x.shape[-1]]
        return x + y


class MultiScaleResNetTokenizer(nn.Module):
    num_special_tokens = 4
    physical_token_index = 3

    def __init__(self, cfg: MultiScaleTokenizerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        branch_dim = max(8, cfg.d_model // len(cfg.kernels))
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(2, branch_dim, kernel_size=k, stride=cfg.patch_size, padding=k // 2),
                ResConv1DBlock(branch_dim, k, dropout=cfg.dropout),
                ResConv1DBlock(branch_dim, max(3, k // 2), dropout=cfg.dropout),
            )
            for k in cfg.kernels
        ])
        self.patch_proj = nn.Sequential(nn.LayerNorm(branch_dim * len(cfg.kernels)), nn.Linear(branch_dim * len(cfg.kernels), cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.d_model))
        self.physical_proj = nn.Sequential(nn.LayerNorm(5), nn.Linear(5, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.d_model))
        self.dataset_embed = nn.Embedding(cfg.num_datasets + 1, cfg.d_model)
        self.task_embed = nn.Embedding(cfg.num_tasks, cfg.d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.pos = nn.Parameter(torch.zeros(1, cfg.max_tokens + self.num_special_tokens, cfg.d_model))
        self.dropout = nn.Dropout(cfg.dropout)
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.pos, std=0.02)

    @staticmethod
    def physical_stats(iq: torch.Tensor, sample_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, _c, length = iq.shape
        if sample_mask is None:
            sample_mask = torch.ones(b, length, dtype=torch.bool, device=iq.device)
        mask = sample_mask.unsqueeze(1).float()
        denom = mask.sum(dim=-1).clamp_min(1.0)
        power_t = iq.float().square().sum(dim=1)
        power = (power_t * sample_mask.float()).sum(dim=-1) / sample_mask.float().sum(dim=-1).clamp_min(1.0)
        peak = power_t.masked_fill(~sample_mask, 0.0).max(dim=-1).values.clamp_min(1.0e-8)
        papr = peak / power.clamp_min(1.0e-8)
        mean = (iq.float() * mask).sum(dim=-1, keepdim=True) / denom.unsqueeze(-1)
        centered = (iq.float() - mean) * mask
        var = centered.square().sum(dim=-1) / denom
        corr = (centered[:, 0] * centered[:, 1]).sum(dim=-1) / (denom[:, 0] * torch.sqrt(var[:, 0] * var[:, 1]).clamp_min(1.0e-8))
        ratio = var[:, 0] / var[:, 1].clamp_min(1.0e-8)
        stats = torch.stack([torch.log1p(power), torch.log1p(peak), papr, corr, torch.log1p(ratio)], dim=-1)
        return torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)

    def _patchify_iq(self, iq: torch.Tensor, sample_mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, length = iq.shape
        p = self.cfg.patch_size
        pad = (p - length % p) % p
        if pad:
            iq = F.pad(iq, (0, pad))
            if sample_mask is not None:
                sample_mask = F.pad(sample_mask, (0, pad), value=False)
        patches = iq.unfold(-1, p, p).transpose(1, 2).contiguous()
        patch_mask = torch.ones(b, patches.shape[1], dtype=torch.bool, device=iq.device)
        if sample_mask is not None:
            patch_mask = sample_mask.unfold(-1, p, p).any(dim=-1)
        return patches, patch_mask

    def forward(self, iq: torch.Tensor, sample_mask: torch.Tensor | None = None, dataset_id: torch.Tensor | None = None, task_type_id: torch.Tensor | None = None, **_metadata) -> dict[str, torch.Tensor]:
        b, _c, length = iq.shape
        if sample_mask is None:
            sample_mask = torch.ones(b, length, dtype=torch.bool, device=iq.device)
        branch_tokens = [branch(iq) for branch in self.branches]
        min_t = min(x.shape[-1] for x in branch_tokens)
        conv = torch.cat([x[..., :min_t] for x in branch_tokens], dim=1).transpose(1, 2).contiguous()
        patch_tokens = self.patch_proj(conv)
        iq_patches, patch_mask = self._patchify_iq(iq, sample_mask)
        limit = min(patch_tokens.shape[1], patch_mask.shape[1], self.cfg.max_tokens)
        patch_tokens = patch_tokens[:, :limit]
        patch_mask = patch_mask[:, :limit]
        iq_patches = iq_patches[:, :limit]

        if dataset_id is None:
            dataset_id = torch.zeros(b, dtype=torch.long, device=iq.device)
        dataset_id = dataset_id.long().clamp(0, self.cfg.num_datasets)
        if task_type_id is None:
            task_type_id = torch.zeros(b, dtype=torch.long, device=iq.device)
        task_type_id = task_type_id.long().clamp(0, self.cfg.num_tasks - 1)
        cls = self.cls.expand(b, -1, -1)
        dataset_token = self.dataset_embed(dataset_id).unsqueeze(1)
        task_token = self.task_embed(task_type_id).unsqueeze(1)
        physical_stats = self.physical_stats(iq, sample_mask)
        physical_token = self.physical_proj(physical_stats).unsqueeze(1)
        tokens = torch.cat([cls, dataset_token, task_token, physical_token, patch_tokens], dim=1)
        token_mask = torch.cat([torch.ones(b, self.num_special_tokens, dtype=torch.bool, device=iq.device), patch_mask], dim=1)
        tokens = self.dropout(tokens + self.pos[:, : tokens.shape[1]])
        return {
            "tokens": torch.nan_to_num(tokens),
            "token_mask": token_mask,
            "patch_mask": patch_mask,
            "iq_patch_targets": iq_patches,
            "physical_stats": physical_stats,
            "patch_offset": self.num_special_tokens,
            "physical_token_index": self.physical_token_index,
        }
