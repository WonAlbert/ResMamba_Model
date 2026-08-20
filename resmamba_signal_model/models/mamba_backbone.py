from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from resmamba_signal_model.data.packing import apply_segments, flip_packed_tokens, flip_segments

logger = logging.getLogger(__name__)
_MAMBA2_AVAILABLE = False
_Mamba2Class: type[nn.Module] | None = None
try:
    from mamba_ssm import Mamba2 as _Mamba2Import
    try:
        from mamba_ssm.ops.triton.ssd_combined import causal_conv1d_fwd_function
    except ImportError:
        causal_conv1d_fwd_function = None
    if causal_conv1d_fwd_function is None:
        raise ImportError("causal-conv1d CUDA extension is not loadable")
    _Mamba2Class = _Mamba2Import
    _MAMBA2_AVAILABLE = True
except ImportError as exc:
    _MAMBA2_IMPORT_ERROR = exc
    logger.warning("mamba-ssm runtime unavailable (%s); using fallback SSM blocks.", exc)
else:
    _MAMBA2_IMPORT_ERROR = None


class _FallbackMamba2(nn.Module):
    def __init__(self, d_model: int, d_conv: int = 4, expand: int = 2, **_kwargs) -> None:
        super().__init__()
        inner = d_model * expand
        self.in_proj = nn.Linear(d_model, inner * 2)
        self.conv = nn.Conv1d(inner, inner, kernel_size=d_conv, padding=d_conv - 1, groups=inner)
        self.out_proj = nn.Linear(inner, d_model)

    def _forward_segment(self, x: torch.Tensor) -> torch.Tensor:
        u, v = self.in_proj(x).chunk(2, dim=-1)
        y = F.silu(self.conv(u.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2))
        return self.out_proj(y * v)

    def forward(
        self,
        x: torch.Tensor,
        *,
        seq_idx: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        **_kwargs,
    ) -> torch.Tensor:
        if cu_seqlens is None:
            return self._forward_segment(x)
        return apply_segments(x, cu_seqlens, self._forward_segment)


class _DeviceDispatchMamba2(nn.Module):
    """Use official CUDA Mamba2 on GPU; CPU-safe fallback otherwise."""

    def __init__(self, official: nn.Module, fallback: nn.Module) -> None:
        super().__init__()
        self.official = official
        self.fallback = fallback

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if x.is_cuda:
            return self.official(x, **kwargs)
        return self.fallback(x, **kwargs)


def _build_mamba2(
    d_model: int,
    *,
    d_state: int,
    d_conv: int,
    expand: int,
    headdim: int,
    ngroups: int = 1,
    chunk_size: int = 256,
    require_mamba_kernel: bool = False,
    allow_fallback_mamba: bool = True,
) -> nn.Module:
    fallback = _FallbackMamba2(d_model=d_model, d_conv=d_conv, expand=expand)
    if _MAMBA2_AVAILABLE and _Mamba2Class is not None:
        official = _Mamba2Class(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            chunk_size=chunk_size,
        )
        if allow_fallback_mamba:
            return _DeviceDispatchMamba2(official, fallback)
        return official
    if require_mamba_kernel or not allow_fallback_mamba:
        raise RuntimeError("Mamba2 CUDA kernel is required but unavailable.")
    return fallback


def _apply_padding_mask(x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
    return x if key_padding_mask is None else x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)


def _flip_with_mask(x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
    if key_padding_mask is None:
        return torch.flip(x, dims=[1])
    lengths = (~key_padding_mask).sum(dim=1)
    out = x.clone()
    for batch_idx, length in enumerate(lengths.tolist()):
        if length > 0:
            out[batch_idx, : int(length)] = torch.flip(x[batch_idx, : int(length)], dims=[0])
    return out


class UniMamba2Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        dropout: float = 0.0,
        ngroups: int = 1,
        chunk_size: int = 256,
        require_mamba_kernel: bool = False,
        allow_fallback_mamba: bool = True,
        norm: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.norm = norm if norm is not None else nn.LayerNorm(d_model)
        self.mamba = _build_mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            chunk_size=chunk_size,
            require_mamba_kernel=require_mamba_kernel,
            allow_fallback_mamba=allow_fallback_mamba,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        *,
        seq_idx: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if seq_idx is not None and cu_seqlens is not None:
            h = self.norm(x)
            mamba_kwargs = {"seq_idx": seq_idx}
            if x.is_cuda:
                out = self.mamba(h, **mamba_kwargs)
            else:
                out = apply_segments(h, cu_seqlens, self.mamba)
            return x + self.dropout(out)

        h = _apply_padding_mask(self.norm(x), key_padding_mask)
        return x + self.dropout(self.mamba(h))


class BiMamba2Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        dropout: float = 0.0,
        ngroups: int = 1,
        chunk_size: int = 256,
        share_bidirectional_weights: bool = False,
        require_mamba_kernel: bool = False,
        allow_fallback_mamba: bool = True,
        norm: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.norm = norm if norm is not None else nn.LayerNorm(d_model)
        kwargs = dict(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            chunk_size=chunk_size,
            require_mamba_kernel=require_mamba_kernel,
            allow_fallback_mamba=allow_fallback_mamba,
        )
        self.fwd = _build_mamba2(**kwargs)
        if share_bidirectional_weights:
            self.bwd = self.fwd
        else:
            self.bwd = _build_mamba2(**kwargs)
        self.fuse = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        *,
        seq_idx: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if seq_idx is not None and cu_seqlens is not None:
            h = self.norm(x)
            mamba_kwargs = {"seq_idx": seq_idx}
            if x.is_cuda:
                fwd = self.fwd(h, **mamba_kwargs)
                h_rev = flip_packed_tokens(h, cu_seqlens)
                bwd = flip_packed_tokens(self.bwd(h_rev, **mamba_kwargs), cu_seqlens)
            else:
                fwd = apply_segments(h, cu_seqlens, self.fwd)
                h_rev = flip_segments(h, cu_seqlens)
                bwd = flip_segments(apply_segments(h_rev, cu_seqlens, self.bwd), cu_seqlens)
            return x + self.dropout(self.fuse(torch.cat([fwd, bwd], dim=-1)))

        h = _apply_padding_mask(self.norm(x), key_padding_mask)
        fwd = self.fwd(h)
        bwd = _flip_with_mask(self.bwd(_flip_with_mask(h, key_padding_mask)), key_padding_mask)
        return x + self.dropout(self.fuse(torch.cat([fwd, bwd], dim=-1)))


def build_mamba_block(
    scan_direction: str,
    d_model: int,
    *,
    d_state: int = 128,
    d_conv: int = 4,
    expand: int = 2,
    headdim: int = 64,
    dropout: float = 0.0,
    ngroups: int = 1,
    chunk_size: int = 256,
    share_bidirectional_weights: bool = False,
    require_mamba_kernel: bool = False,
    allow_fallback_mamba: bool = True,
    norm: nn.Module | None = None,
) -> nn.Module:
    kwargs = dict(
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        headdim=headdim,
        dropout=dropout,
        ngroups=ngroups,
        chunk_size=chunk_size,
        require_mamba_kernel=require_mamba_kernel,
        allow_fallback_mamba=allow_fallback_mamba,
        norm=norm,
    )
    if scan_direction == "unidirectional":
        return UniMamba2Block(**kwargs)
    if scan_direction == "bidirectional":
        return BiMamba2Block(**kwargs, share_bidirectional_weights=share_bidirectional_weights)
    raise ValueError(f"Unsupported scan_direction: {scan_direction}")


class Mamba2Stack(nn.Module):
    def __init__(
        self,
        num_layers: int,
        *,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        dropout: float = 0.0,
        ngroups: int = 1,
        chunk_size: int = 256,
        activation_checkpointing: bool = False,
        scan_direction: str = "bidirectional",
        share_bidirectional_weights: bool = False,
        require_mamba_kernel: bool = False,
        allow_fallback_mamba: bool = True,
        norm_type: str = "layer_norm",
    ) -> None:
        super().__init__()
        self.activation_checkpointing = activation_checkpointing
        from resmamba_signal_model.models.norms import build_norm

        block_kwargs = dict(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            dropout=dropout,
            ngroups=ngroups,
            chunk_size=chunk_size,
            share_bidirectional_weights=share_bidirectional_weights,
            require_mamba_kernel=require_mamba_kernel,
            allow_fallback_mamba=allow_fallback_mamba,
        )
        self.layers = nn.ModuleList(
            build_mamba_block(scan_direction, norm=build_norm(norm_type, d_model), **block_kwargs)
            for _ in range(num_layers)
        )
        self.norm = build_norm(norm_type, d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        *,
        seq_idx: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = tokens
        use_checkpoint = self.activation_checkpointing and self.training and torch.is_grad_enabled()
        for layer in self.layers:
            if use_checkpoint:
                fn: Callable[..., torch.Tensor] = lambda x, layer=layer: layer(
                    x,
                    key_padding_mask=key_padding_mask,
                    seq_idx=seq_idx,
                    cu_seqlens=cu_seqlens,
                )
                hidden = checkpoint(fn, hidden, use_reentrant=False)
            else:
                hidden = layer(hidden, key_padding_mask=key_padding_mask, seq_idx=seq_idx, cu_seqlens=cu_seqlens)
        return self.norm(hidden)


def mamba2_available() -> bool:
    return _MAMBA2_AVAILABLE


def get_mamba_runtime_info(*, scan_direction: str = "bidirectional") -> dict[str, str | bool]:
    return {
        "mamba_operator": "mamba2",
        "mamba_kernel": "official_mamba_ssm" if _MAMBA2_AVAILABLE else "fallback",
        "fallback": not _MAMBA2_AVAILABLE,
        "scan_direction": scan_direction,
    }
