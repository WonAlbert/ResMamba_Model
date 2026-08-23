from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

# 与识别任务对齐的三路专家语义（聚类/预测复用混合，不另开专家）
MOE_EXPERT_NAMES: tuple[str, ...] = ("ld_intrapulse", "ld_model", "tx_modulation")


@dataclass
class MoEAux:
    gate_weights: torch.Tensor
    load_balance_loss: torch.Tensor
    expert_names: tuple[str, ...] = field(default_factory=lambda: MOE_EXPERT_NAMES)


def load_balancing_loss(
    gate_probs: torch.Tensor,
    *,
    num_experts: int,
    key_padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Switch-style 负载均衡：N * sum_e f_e * P_e（f 来自 top-1，P 为软路由均值）。"""
    probs = gate_probs.float()
    if key_padding_mask is not None:
        valid = ~key_padding_mask
        if not bool(valid.any()):
            return probs.new_tensor(0.0)
        probs = probs[valid]
    else:
        probs = probs.reshape(-1, num_experts)
    if probs.numel() == 0:
        return probs.new_tensor(0.0)
    p = probs.mean(dim=0)
    hard = F.one_hot(probs.argmax(dim=-1), num_classes=num_experts).float()
    f = hard.mean(dim=0)
    return float(num_experts) * (f * p).sum()


class MoEGate(nn.Module):
    """内容路由 gate；禁止 dataset_id / task 名等域泄漏特征。"""

    def __init__(self, in_dim: int, num_experts: int, *, top_k: int | None = None) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError(f"moe_num_experts 必须 >= 1，当前 {num_experts}")
        self.num_experts = int(num_experts)
        self.top_k = int(top_k) if top_k is not None and int(top_k) > 0 else None
        if self.top_k is not None and self.top_k >= self.num_experts:
            self.top_k = None
        self.router = nn.Linear(in_dim, self.num_experts)

    def forward(
        self,
        gate_input: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.router(gate_input)
        if key_padding_mask is not None:
            logits = logits.masked_fill(key_padding_mask.unsqueeze(-1), float("-inf"))
        if self.top_k is not None:
            topk_vals, topk_idx = torch.topk(logits, k=self.top_k, dim=-1)
            masked = logits.new_full(logits.shape, float("-inf"))
            masked.scatter_(-1, topk_idx, topk_vals)
            logits = masked
        weights = torch.softmax(logits, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        return weights, logits


class ExpertSwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        hidden = max(8, (int(hidden_dim) + 7) // 8 * 8)
        self.w1 = nn.Linear(dim, hidden * 2)
        self.w2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(a) * b)


class MoEFFN(nn.Module):
    """稠密 softmax 混合 MoE-FFN；可选 top-k 稀疏路由。"""

    def __init__(
        self,
        d_model: int,
        *,
        num_experts: int = 3,
        ffn_expand: float = 2.0,
        top_k: int | None = None,
        gate_input_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_experts = int(num_experts)
        hidden = max(8, int(d_model * float(ffn_expand)))
        self.experts = nn.ModuleList(ExpertSwiGLU(d_model, hidden) for _ in range(self.num_experts))
        self.gate = MoEGate(gate_input_dim or d_model, self.num_experts, top_k=top_k)

    def forward(
        self,
        x: torch.Tensor,
        *,
        gate_input: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, MoEAux]:
        gate_in = x if gate_input is None else gate_input
        weights, _ = self.gate(gate_in, key_padding_mask=key_padding_mask)
        expert_outs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        mixed = (weights.unsqueeze(-1) * expert_outs).sum(dim=-2)
        lb = load_balancing_loss(weights, num_experts=self.num_experts, key_padding_mask=key_padding_mask)
        names = MOE_EXPERT_NAMES[: self.num_experts]
        if len(names) < self.num_experts:
            names = tuple(f"expert_{i}" for i in range(self.num_experts))
        return mixed, MoEAux(gate_weights=weights, load_balance_loss=lb, expert_names=names)


class MoEFusion(nn.Module):
    """多路分支输出按 token 级 gate 融合（Tokenizer 等）。"""

    def __init__(
        self,
        d_model: int,
        num_branches: int,
        *,
        top_k: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_branches = int(num_branches)
        self.gate = MoEGate(d_model * num_branches, num_branches, top_k=top_k)

    def forward(
        self,
        branches: list[torch.Tensor],
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, MoEAux]:
        if len(branches) != self.num_branches:
            raise ValueError(f"期望 {self.num_branches} 路分支，收到 {len(branches)}")
        stacked = torch.stack(branches, dim=-2)
        gate_in = torch.cat(branches, dim=-1)
        weights, _ = self.gate(gate_in, key_padding_mask=key_padding_mask)
        fused = (weights.unsqueeze(-1) * stacked).sum(dim=-2)
        lb = load_balancing_loss(weights, num_experts=self.num_branches, key_padding_mask=key_padding_mask)
        names = MOE_EXPERT_NAMES[: self.num_branches]
        if len(names) < self.num_branches:
            names = tuple(f"branch_{i}" for i in range(self.num_branches))
        return fused, MoEAux(gate_weights=weights, load_balance_loss=lb, expert_names=names)


def aggregate_moe_aux(aux_list: list[MoEAux]) -> dict[str, torch.Tensor]:
    if not aux_list:
        ref = torch.tensor(0.0)
        return {"moe_load_balance": ref, "moe_gate_weights": []}
    lb = torch.stack([a.load_balance_loss for a in aux_list]).mean()
    return {"moe_load_balance": lb, "moe_gate_weights": [a.gate_weights for a in aux_list]}
