from __future__ import annotations

import torch
import torch.nn as nn

from resmamba_signal_model.data.packing import repack_tokens, unpack_packed_tokens
from resmamba_signal_model.models.mamba_backbone import BiMamba2Block


class ResLocalBranch(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 5, dropout: float = 0.0) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.conv = nn.Conv1d(d_model * 2, d_model, kernel_size=kernel_size, padding=pad)
        self.out = nn.Dropout(dropout)

    def _forward_segment(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).transpose(1, 2)
        y = self.conv(h).transpose(1, 2)
        if y.shape[1] != x.shape[1]:
            y = y[:, : x.shape[1]]
        return self.out(y)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        *,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cu_seqlens is not None:
            padded, _mask = unpack_packed_tokens(x, cu_seqlens)
            y = self._forward_segment(padded)
            return repack_tokens(y, cu_seqlens)
        y = self._forward_segment(x)
        if key_padding_mask is not None:
            y = y.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return y


class ResMambaBlock(nn.Module):
    def __init__(self, d_model: int, *, d_state: int = 64, d_conv: int = 4, expand: int = 2, headdim: int = 64, dropout: float = 0.0, local_kernel: int = 5, ngroups: int = 1, chunk_size: int = 256) -> None:
        super().__init__()
        self.local = ResLocalBranch(d_model, kernel_size=local_kernel, dropout=dropout)
        self.global_branch = BiMamba2Block(d_model, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim, dropout=dropout, ngroups=ngroups, chunk_size=chunk_size)
        self.fuse = nn.Sequential(nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model))

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        *,
        seq_idx: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        local = self.local(x, key_padding_mask=key_padding_mask, cu_seqlens=cu_seqlens)
        global_out = self.global_branch(x, key_padding_mask=key_padding_mask, seq_idx=seq_idx, cu_seqlens=cu_seqlens) - x
        return x + self.fuse(torch.cat([local, global_out], dim=-1))


class ResMambaStack(nn.Module):
    def __init__(self, num_layers: int, *, d_model: int, d_state: int = 64, d_conv: int = 4, expand: int = 2, headdim: int = 64, dropout: float = 0.0, local_kernel: int = 5, ngroups: int = 1, chunk_size: int = 256) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            ResMambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim, dropout=dropout, local_kernel=local_kernel, ngroups=ngroups, chunk_size=chunk_size)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        *,
        seq_idx: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = tokens
        for layer in self.layers:
            hidden = layer(hidden, key_padding_mask=key_padding_mask, seq_idx=seq_idx, cu_seqlens=cu_seqlens)
        return self.norm(hidden)
