from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from resmamba_signal_model.models.moe import MoEAux, MoEFFN
from resmamba_signal_model.models.norms import DropPath, RMSNorm, build_norm


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dim 必须为偶数，当前 {dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=dtype)[None, None, :, :]
        sin = emb.sin().to(dtype=dtype)[None, None, :, :]
        return cos, sin


class MemoryTransformerBlock(nn.Module):
    """1 层 Pre-LN Transformer：短序列标准 self-attn；超 ``attn_window`` 则 chunk 均值记忆压缩。"""

    def __init__(
        self,
        d_model: int,
        *,
        num_heads: int = 8,
        ffn_expand: int = 2,
        dropout: float = 0.0,
        attn_window: int = 1024,
        drop_path: float = 0.0,
        norm_type: str = "rmsnorm",
        enable_moe: bool = True,
        moe_num_experts: int = 3,
        moe_top_k: int | None = None,
        moe_ffn_expand: float = 2.0,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} 不能整除 num_heads={num_heads}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attn_window = int(attn_window)
        self.norm1 = build_norm(norm_type, d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(self.head_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.norm2 = build_norm(norm_type, d_model)
        hidden = int(d_model * ffn_expand)
        if enable_moe:
            self.ffn: nn.Module = MoEFFN(
                d_model,
                num_experts=max(1, int(moe_num_experts)),
                ffn_expand=float(moe_ffn_expand),
                top_k=moe_top_k,
            )
            self._ffn_is_moe = True
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, d_model),
            )
            self._ffn_is_moe = False
        self.drop_path = DropPath(drop_path)
        self.last_used_memory = False
        self.last_moe_aux: MoEAux | None = None

    def _attend(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = self.rope(seq_len, x.device, x.dtype)
        q, k = apply_rope(q, k, cos, sin)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if key_padding_mask is not None:
            fill = torch.finfo(attn.dtype).min
            attn = attn.masked_fill(key_padding_mask[:, None, None, :], fill)
        attn = self.attn_drop(torch.softmax(attn, dim=-1))
        ctx = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(ctx)

    def _memory_tokens(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        batch, seq_len, dim = x.shape
        chunk_size = int(math.ceil(seq_len / self.attn_window))
        n_mem = int(math.ceil(seq_len / chunk_size))
        pad = n_mem * chunk_size - seq_len
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, pad), value=True)
        x = x.view(batch, n_mem, chunk_size, dim)
        if key_padding_mask is None:
            mem = x.mean(dim=2)
            mem_pad = None
        else:
            valid = (~key_padding_mask).view(batch, n_mem, chunk_size, 1).to(dtype=x.dtype)
            mem = (x * valid).sum(dim=2) / valid.sum(dim=2).clamp_min(1.0)
            mem_pad = valid.squeeze(-1).sum(dim=-1) <= 0
        return mem, mem_pad, chunk_size

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        *,
        moe_route_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = x.shape[1]
        h = self.norm1(x)
        if seq_len <= self.attn_window:
            self.last_used_memory = False
            attn_out = self._attend(h, key_padding_mask)
        else:
            self.last_used_memory = True
            mem, mem_pad, chunk_size = self._memory_tokens(h, key_padding_mask)
            mem_out = self._attend(mem, mem_pad)
            attn_out = mem_out.repeat_interleave(chunk_size, dim=1)[:, :seq_len]
        x = x + self.drop_path(attn_out)
        self.last_moe_aux = None
        ffn_in = self.norm2(x)
        if self._ffn_is_moe:
            ffn_out, aux = self.ffn(
                ffn_in,
                key_padding_mask=key_padding_mask,
                route_weights=moe_route_weights,
            )
            self.last_moe_aux = aux
        else:
            ffn_out = self.ffn(ffn_in)
        x = x + self.drop_path(ffn_out)
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return x
