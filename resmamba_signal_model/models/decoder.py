from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from resmamba_signal_model.data.packing import pack_valid_tokens, scatter_packed_tokens
from resmamba_signal_model.models.mamba_backbone import build_mamba_block
from resmamba_signal_model.models.norms import DropPath, RMSNorm, build_norm
from resmamba_signal_model.models.physics import PHYS_DIM
from resmamba_signal_model.models.revin import AMP_AUX_DIM


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = hidden_dim or int(dim * 8 / 3)
        hidden = (hidden + 7) // 8 * 8
        self.w1 = nn.Linear(dim, hidden * 2)
        self.w2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(a) * b)


class DecoderBlock(nn.Module):
    """Pre-LN：BiMamba2 混合 + SwiGLU。层数由 SharedDecoder 锁定为 1。"""

    def __init__(
        self,
        d_model: int,
        *,
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
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.mamba = build_mamba_block(
            scan_direction,
            d_model,
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
            norm=build_norm(norm_type, d_model),
        )
        self.ffn_norm = build_norm(norm_type, d_model)
        self.ffn = SwiGLU(d_model)
        self.drop_path = DropPath(drop_path)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        *,
        seq_idx: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.mamba(x, key_padding_mask=key_padding_mask, seq_idx=seq_idx, cu_seqlens=cu_seqlens)
        x = x + self.drop_path(self.ffn(self.ffn_norm(x)))
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return x


class PhysicsFiLM(nn.Module):
    def __init__(self, d_model: int, phys_dim: int = PHYS_DIM, amp_aux_dim: int = AMP_AUX_DIM) -> None:
        super().__init__()
        in_dim = int(phys_dim) + int(amp_aux_dim)
        self.net = nn.Sequential(nn.Linear(in_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model * 2))

    def forward(
        self,
        h: torch.Tensor,
        physics: torch.Tensor,
        physics_mask: torch.Tensor | None = None,
        amp_aux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feats = physics.float()
        if physics_mask is not None:
            feats = feats * physics_mask.to(device=feats.device, dtype=feats.dtype)
        if amp_aux is not None:
            aux = amp_aux.to(device=feats.device, dtype=feats.dtype)
            if aux.ndim == 2:
                aux = aux.unsqueeze(1).expand(-1, feats.shape[1], -1)
            feats = torch.cat([feats, aux], dim=-1)
        else:
            pad = torch.zeros(
                *feats.shape[:-1],
                AMP_AUX_DIM,
                device=feats.device,
                dtype=feats.dtype,
            )
            feats = torch.cat([feats, pad], dim=-1)
        gamma, beta = self.net(feats).to(dtype=h.dtype).chunk(2, dim=-1)
        return (gamma.tanh() + 1.0) * h + beta


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4) -> None:
        super().__init__()
        heads = max(1, min(num_heads, d_model))
        while d_model % heads != 0 and heads > 1:
            heads -= 1
        self.query = nn.Parameter(torch.zeros(1, 1, d_model))
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        query = self.query.expand(x.shape[0], -1, -1)
        out, _ = self.attn(query, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        return out.squeeze(1)


class ReconHead(nn.Module):
    def __init__(self, d_model: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.net = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2 * patch_size))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).view(*h.shape[:-1], 2, self.patch_size)


class ReprHead(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4) -> None:
        super().__init__()
        self.pool = AttentionPooling(d_model, num_heads=num_heads)
        self.proj = nn.Sequential(nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, d_model))

    def forward(self, dec_token: torch.Tensor, patch_h: torch.Tensor, patch_pad: torch.Tensor | None) -> torch.Tensor:
        pooled = self.pool(patch_h, key_padding_mask=patch_pad)
        z = self.proj(torch.cat([dec_token, pooled], dim=-1))
        return F.normalize(z.float(), dim=-1).to(dtype=dec_token.dtype)


class UnifiedQueryDecoder(nn.Module):
    """坐标 query + 可见 token cross-attn；局部相位走零初始化残差头。

    旧 query 只看全局物理量和可见 token，频谱容易抄到、相位抄不到，I/Q MAE
    会被钉住。mask 位的 ``patch_h`` 是 BiMamba 插值、不含目标真值，适合作为
    局部相位先验。``local_recon`` / ``local_query_gate`` 均零初始化：
    ``--init-from`` 旧权重时前向不变，续训再学会抄局部相位。
    """

    def __init__(
        self,
        d_model: int,
        patch_size: int,
        *,
        query_dim: int = 320,
        num_heads: int = 4,
        phys_dim: int = PHYS_DIM,
        condition_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dim = max(8, min(int(query_dim), int(d_model)))
        heads = max(1, min(int(num_heads), dim))
        while dim % heads != 0 and heads > 1:
            heads -= 1
        self.query_dim = dim
        self.condition_dim = max(1, int(condition_dim))
        self.context_proj = nn.Linear(d_model, dim)
        self.coord_proj = nn.Sequential(nn.Linear(2, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.physics_proj = nn.Linear(phys_dim, dim)
        self.condition_proj = nn.Linear(self.condition_dim, dim, bias=False)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.output = nn.Linear(dim, 2 * patch_size)
        self.local_recon = nn.Linear(d_model, 2 * patch_size)
        nn.init.zeros_(self.local_recon.weight)
        nn.init.zeros_(self.local_recon.bias)
        self.patch_size = int(patch_size)
        # 0 → 与旧权重前向一致；训练中再打开局部插值。
        self.local_query_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        context_tokens: torch.Tensor,
        visible: torch.Tensor,
        patch_mask: torch.Tensor,
        target_mask: torch.Tensor,
        *,
        context_physics: torch.Tensor,
        task_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, n_tokens, _ = context_tokens.shape
        if n_tokens == 0:
            empty = context_tokens.new_zeros(batch, 0, 2, self.patch_size)
            return empty, context_tokens.new_zeros(batch, 0, self.query_dim)
        positions = torch.linspace(
            0.0,
            1.0,
            n_tokens,
            device=context_tokens.device,
            dtype=context_tokens.dtype,
        ).view(1, n_tokens, 1)
        coords = torch.cat(
            [positions.expand(batch, -1, -1), target_mask.to(dtype=context_tokens.dtype).unsqueeze(-1)],
            dim=-1,
        )
        local = self.context_proj(context_tokens)
        query = self.coord_proj(coords)
        query = query + self.physics_proj(context_physics.to(dtype=context_tokens.dtype)).unsqueeze(1)
        if task_context is not None:
            cond = task_context.to(device=context_tokens.device, dtype=context_tokens.dtype)
            if cond.shape[-1] < self.condition_dim:
                cond = F.pad(cond, (0, self.condition_dim - cond.shape[-1]))
            elif cond.shape[-1] > self.condition_dim:
                cond = cond[..., : self.condition_dim]
            query = query + self.condition_proj(cond).unsqueeze(1)
        query = query + self.local_query_gate.to(dtype=query.dtype) * local

        context = local
        visible_keys = patch_mask & visible
        key_padding_mask = ~visible_keys
        # 整行无可见 key 时 softmax(-inf) 会出 NaN；让该行走一个零向量 dummy key。
        empty_rows = ~visible_keys.any(dim=-1)
        if bool(empty_rows.any()):
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[empty_rows, 0] = False
        attended, _ = self.cross_attn(
            query,
            context,
            context,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        query_h = self.attn_norm(query + attended)
        query_h = query_h + self.ffn(self.ffn_norm(query_h))
        query_h = query_h.masked_fill(~patch_mask.unsqueeze(-1), 0.0)
        recon = self.output(query_h) + self.local_recon(context_tokens)
        recon = recon.view(batch, n_tokens, 2, self.patch_size)
        recon = recon.masked_fill(~patch_mask.unsqueeze(-1).unsqueeze(-1), 0.0)
        return recon, query_h


class SharedDecoder(nn.Module):
    """共享上下文解码与坐标查询重建；旧 patch head 由兼容开关保留。"""

    def __init__(
        self,
        d_model: int,
        patch_size: int,
        *,
        decoder_mamba_layers: int = 1,
        attn_num_heads: int = 8,
        dropout: float = 0.0,
        sequence_packing: bool = True,
        query_dim: int = 320,
        condition_dim: int = 64,
        legacy_reconstruction: bool = False,
        **block_kwargs,
    ) -> None:
        super().__init__()
        if decoder_mamba_layers != 1:
            raise ValueError("SharedDecoder 锁定 decoder_mamba_layers=1，不加 Decoder Transformer")
        self.d_model = d_model
        self.sequence_packing = sequence_packing
        self.legacy_reconstruction = bool(legacy_reconstruction)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.dec_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.skip_proj = nn.Linear(d_model, d_model)
        self.skip_gate = nn.Linear(d_model * 2, d_model)
        self.pre_norm = RMSNorm(d_model)
        self.film = PhysicsFiLM(d_model)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, dropout=dropout, **block_kwargs) for _ in range(1)])
        self.query_decoder = UnifiedQueryDecoder(
            d_model,
            patch_size,
            query_dim=query_dim,
            num_heads=min(4, attn_num_heads),
            condition_dim=condition_dim,
            dropout=dropout,
        )
        # 保留旧权重和可复现实验路径；默认重建由 query_decoder 完成。
        self.recon_head = ReconHead(d_model, patch_size)
        self.repr_head = ReprHead(d_model, num_heads=min(4, attn_num_heads))
        self.readout_phys = nn.Linear(d_model, PHYS_DIM)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.dec_token, std=0.02)

    def scatter_encoder(
        self,
        h_vis: torch.Tensor,
        visible: torch.Tensor,
        patch_mask: torch.Tensor,
        *,
        packed: bool,
    ) -> torch.Tensor:
        b, n, d = patch_mask.shape[0], patch_mask.shape[1], self.d_model
        h_full = self.mask_token.expand(b, n, d).clone()
        if packed:
            if h_vis.numel() > 0:
                h_full[visible] = h_vis.reshape(-1, d)
        else:
            if h_vis.shape[:2] == (b, n):
                h_full = torch.where(visible.unsqueeze(-1), h_vis, h_full)
            else:
                h_full[visible] = h_vis.reshape(-1, d)
        masked = patch_mask & ~visible
        if masked.any():
            h_full = torch.where(masked.unsqueeze(-1), self.mask_token.expand_as(h_full), h_full)
        return h_full

    def _gated_skip(
        self,
        h_full: torch.Tensor,
        x_tok: torch.Tensor,
        visible: torch.Tensor,
    ) -> torch.Tensor:
        safe_tok = x_tok.masked_fill(~visible.unsqueeze(-1), 0.0)
        gate = torch.sigmoid(self.skip_gate(torch.cat([h_full, safe_tok], dim=-1)))
        skipped = h_full + gate * self.skip_proj(safe_tok)
        return torch.where(visible.unsqueeze(-1), skipped, h_full)

    def _run_blocks(
        self,
        h: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        seq_idx: torch.Tensor | None,
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        for block in self.blocks:
            h = block(h, key_padding_mask=key_padding_mask, seq_idx=seq_idx, cu_seqlens=cu_seqlens)
        return h

    def forward(
        self,
        h_vis: torch.Tensor,
        x_tok: torch.Tensor,
        patch_mask: torch.Tensor,
        visible: torch.Tensor,
        patch_physics: torch.Tensor,
        *,
        packed_encoder: bool,
        sequence_packing: bool | None = None,
        skip_recon: bool = False,
        target_mask: torch.Tensor | None = None,
        context_physics: torch.Tensor | None = None,
        task_context: torch.Tensor | None = None,
        legacy_reconstruction: bool | None = None,
        physics_mask: torch.Tensor | None = None,
        amp_aux: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        packing = self.sequence_packing if sequence_packing is None else sequence_packing
        if target_mask is None:
            target_mask = patch_mask & ~visible
        target_mask = target_mask & patch_mask
        h_full = self.scatter_encoder(h_vis, visible, patch_mask, packed=packed_encoder)
        h0 = self._gated_skip(h_full, x_tok, visible)
        h0 = self.pre_norm(h0)
        safe_physics = patch_physics.masked_fill(~visible.unsqueeze(-1), 0.0)
        film_h = self.film(h0, safe_physics, physics_mask=physics_mask, amp_aux=amp_aux)
        h0 = torch.where(visible.unsqueeze(-1), film_h, h0)
        h0 = h0.masked_fill(~patch_mask.unsqueeze(-1), 0.0)

        batch = x_tok.shape[0]
        dec = self.dec_token.expand(batch, -1, -1)
        tokens = torch.cat([dec, h0], dim=1)
        tok_valid = torch.cat([torch.ones(batch, 1, dtype=torch.bool, device=x_tok.device), patch_mask], dim=1)

        if packing:
            packed, cu_seqlens, seq_idx = pack_valid_tokens(tokens, tok_valid)
            dec_out = self._run_blocks(packed, None, seq_idx, cu_seqlens)
            h_dec_full = scatter_packed_tokens(dec_out, tok_valid)
        else:
            pad_mask = ~tok_valid
            h_dec_full = self._run_blocks(tokens, pad_mask, None, None)

        dec_h = h_dec_full[:, 0]
        patch_h = h_dec_full[:, 1:]
        query_h = None
        use_legacy = self.legacy_reconstruction if legacy_reconstruction is None else bool(legacy_reconstruction)
        if context_physics is None:
            denom = visible.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=patch_physics.dtype)
            context_physics = (safe_physics * visible.unsqueeze(-1)).sum(dim=1) / denom
        if skip_recon:
            recon = None
        elif use_legacy:
            recon = self.recon_head(patch_h)
        else:
            recon, query_h = self.query_decoder(
                patch_h,
                visible,
                patch_mask,
                target_mask,
                context_physics=context_physics,
                task_context=task_context,
            )
        z = self.repr_head(dec_h, patch_h, patch_pad=~patch_mask)
        global_phys_pred = self.readout_phys(z)
        return {
            "h_full": h_full,
            "h_dec": h_dec_full,
            "recon_norm": recon,
            "z": z,
            "global_phys_pred": global_phys_pred,
            "dec_h": dec_h,
            "patch_h": patch_h,
            "query_h": query_h,
        }
