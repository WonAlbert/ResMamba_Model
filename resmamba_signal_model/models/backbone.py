from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from resmamba_signal_model.data.packing import repack_tokens, unpack_packed_tokens
from resmamba_signal_model.models.mamba_backbone import BiMamba2Block, build_mamba_block
from resmamba_signal_model.models.moe import MoEAux, MoEFFN
from resmamba_signal_model.models.norms import build_norm
from resmamba_signal_model.models.transformer import MemoryTransformerBlock


class MambaMoEFFNBlock(nn.Module):
    """BiMamba2 + 后置 MoE-FFN（与 Mamba kernel 解耦）。"""

    def __init__(self, mamba: nn.Module, moe_ffn: MoEFFN, *, norm_type: str = "rmsnorm", d_model: int) -> None:
        super().__init__()
        self.mamba = mamba
        self.moe_ffn = moe_ffn
        self.ffn_norm = build_norm(norm_type, d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        *,
        seq_idx: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, MoEAux | None]:
        x = self.mamba(x, key_padding_mask=key_padding_mask, seq_idx=seq_idx, cu_seqlens=cu_seqlens)
        y, aux = self.moe_ffn(self.ffn_norm(x), key_padding_mask=key_padding_mask)
        return x + y, aux


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
        enable_moe: bool = True,
        moe_num_experts: int = 3,
        moe_top_k: int | None = None,
        moe_ffn_expand: float = 2.0,
        moe_encoder_layers: int = 2,
    ) -> None:
        super().__init__()
        self.activation_checkpointing = activation_checkpointing
        self._last_moe_aux: list[MoEAux] = []
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
        moe_layers = max(0, min(int(moe_encoder_layers), int(encoder_mamba_layers))) if enable_moe else 0
        plain_layers = int(encoder_mamba_layers) - moe_layers
        self.mamba_layers = nn.ModuleList()
        for _ in range(plain_layers):
            self.mamba_layers.append(
                build_mamba_block(scan_direction, norm=build_norm(norm_type, d_model), **mamba_kwargs)
            )
        for _ in range(moe_layers):
            mamba = build_mamba_block(scan_direction, norm=build_norm(norm_type, d_model), **mamba_kwargs)
            moe_ffn = MoEFFN(
                d_model,
                num_experts=max(1, int(moe_num_experts)),
                ffn_expand=float(moe_ffn_expand),
                top_k=moe_top_k,
            )
            self.mamba_layers.append(MambaMoEFFNBlock(mamba, moe_ffn, norm_type=norm_type, d_model=d_model))

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
                    enable_moe=enable_moe,
                    moe_num_experts=moe_num_experts,
                    moe_top_k=moe_top_k,
                    moe_ffn_expand=moe_ffn_expand,
                )
                for _ in range(encoder_transformer_layers)
            ]
        )
        self.norm = build_norm(norm_type, d_model)

    def pop_moe_aux(self) -> list[MoEAux]:
        aux = list(self._last_moe_aux)
        self._last_moe_aux.clear()
        return aux

    def _run_mamba_layer(
        self,
        layer: nn.Module,
        hidden: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        seq_idx: torch.Tensor | None,
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        if isinstance(layer, MambaMoEFFNBlock):
            hidden, aux = layer(
                hidden,
                key_padding_mask=key_padding_mask,
                seq_idx=seq_idx,
                cu_seqlens=cu_seqlens,
            )
            if aux is not None:
                self._last_moe_aux.append(aux)
            return hidden
        return layer(hidden, key_padding_mask=key_padding_mask, seq_idx=seq_idx, cu_seqlens=cu_seqlens)

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
        self._last_moe_aux.clear()
        for layer in self.mamba_layers:
            if use_checkpoint:
                fn: Callable[..., torch.Tensor] = lambda x, layer=layer: self._run_mamba_layer(
                    layer, x, key_padding_mask, seq_idx, cu_seqlens
                )
                hidden = checkpoint(fn, hidden, use_reentrant=False)
            else:
                hidden = self._run_mamba_layer(layer, hidden, key_padding_mask, seq_idx, cu_seqlens)
        for layer in self.transformer_layers:
            if packed:
                padded, valid = unpack_packed_tokens(hidden, cu_seqlens)
                padded = layer(padded, key_padding_mask=~valid)
                if layer.last_moe_aux is not None:
                    self._last_moe_aux.append(layer.last_moe_aux)
                hidden = repack_tokens(padded, cu_seqlens)
            else:
                hidden = layer(hidden, key_padding_mask=key_padding_mask)
                if layer.last_moe_aux is not None:
                    self._last_moe_aux.append(layer.last_moe_aux)
        return self.norm(hidden)
