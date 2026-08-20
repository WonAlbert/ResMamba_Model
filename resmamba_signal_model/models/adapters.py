from __future__ import annotations

import torch
import torch.nn as nn

from resmamba_signal_model.models.task_interface import TaskFeatures


class TaskAdapter(nn.Module):
    """任务专属瓶颈残差：LN-down-GELU-up，挂在 UTI 之后、头之前。"""

    def __init__(self, d_model: int, down_dim: int = 64) -> None:
        super().__init__()
        hidden = max(1, min(int(down_dim), int(d_model)))
        self.down_dim = hidden
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.alpha = nn.Parameter(torch.ones(1))

    def residual(self, h: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.net(h.float()).to(dtype=h.dtype)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.residual(h)

    def modulate(self, features: TaskFeatures) -> TaskFeatures:
        return TaskFeatures(
            pooled=self.forward(features.pooled),
            tokens=self.forward(features.tokens),
            mask=features.mask,
            query=self.forward(features.query) if features.query is not None else None,
            readout=features.readout,
            task_vec=features.task_vec,
            views=features.views,
        )


class SharedTaskAdapter(TaskAdapter):
    """联合训练新建的共享适配器，与各 TaskAdapter 残差相加。"""


def apply_task_adapters(
    features: TaskFeatures,
    *,
    task: str,
    adapters: nn.ModuleDict | None,
    shared: TaskAdapter | None,
) -> TaskFeatures:
    """h = h + α_t A_t(h) + α_s A_shared(h)。"""
    pooled = features.pooled
    tokens = features.tokens
    query = features.query
    if adapters is not None and task in adapters:
        adapter = adapters[task]
        pooled = pooled + adapter.residual(pooled)
        tokens = tokens + adapter.residual(tokens)
        if query is not None:
            query = query + adapter.residual(query)
    if shared is not None:
        pooled = pooled + shared.residual(pooled)
        tokens = tokens + shared.residual(tokens)
        if query is not None:
            query = query + shared.residual(query)
    return TaskFeatures(
        pooled=pooled,
        tokens=tokens,
        mask=features.mask,
        query=query,
        readout=features.readout,
        task_vec=features.task_vec,
        views=features.views,
    )
