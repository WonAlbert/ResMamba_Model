from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from resmamba_signal_model.data.packing import repack_tokens, unpack_packed_tokens
from resmamba_signal_model.models.mamba_backbone import build_mamba_block
from resmamba_signal_model.models.norms import build_norm
from resmamba_signal_model.models.transformer import MemoryTransformerBlock


class HybridEncoder(nn.Module):
    """Encoder 布局 M-M-M-M-M-T：5×BiMamba2 + 1×RoPE MemoryTransformer。"""

    def __init__(
        self,
        *,
        d_model: int,
        encoder_mamba_layers: int = 5,
        encoder_transformer_layers: int = 1,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        dropout: float = 0.0,
        ngroups: int = 1,
        chunk_size: int = 256,
        scan_direction: str = "bidirectional",
        share_bidirectional_weights: bool = False,
        require_mamba_kernel: bool = True,
        allow_fallback_mamba: bool = False,
        norm_type: str = "rmsnorm",
        attn_num_heads: int = 8,
        attn_ffn_expand: int = 2,
        attn_window: int = 1024,
        drop_path: float = 0.0,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.activation_checkpointing = activation_checkpointing
        mamba_kwargs = dict(
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
        self.mamba_layers = nn.ModuleList(
            [
                build_mamba_block(scan_direction, norm=build_norm(norm_type, d_model), **mamba_kwargs)
                for _ in range(encoder_mamba_layers)
            ]
        )
        self.transformer_layers = nn.ModuleList(
            [
                MemoryTransformerBlock(
                    d_model,
                    num_heads=attn_num_heads,
                    ffn_expand=attn_ffn_expand,
                    dropout=dropout,
                    attn_window=attn_window,
                    drop_path=drop_path,
                    norm_type=norm_type,
                )
                for _ in range(encoder_transformer_layers)
            ]
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
        packed = seq_idx is not None and cu_seqlens is not None
        use_checkpoint = self.activation_checkpointing and self.training and torch.is_grad_enabled()
        for layer in self.mamba_layers:
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
        for layer in self.transformer_layers:
            if packed:
                padded, valid = unpack_packed_tokens(hidden, cu_seqlens)
                padded = layer(padded, key_padding_mask=~valid)
                hidden = repack_tokens(padded, cu_seqlens)
            else:
                hidden = layer(hidden, key_padding_mask=key_padding_mask)
        return self.norm(hidden)
