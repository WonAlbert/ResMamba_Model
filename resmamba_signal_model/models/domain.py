from __future__ import annotations

import torch
import torch.nn as nn


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (lambd,) = ctx.saved_tensors
        return -lambd.to(dtype=grad_output.dtype) * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambd: float = 1.0) -> None:
        super().__init__()
        self.lambd = float(lambd)

    def forward(self, x: torch.Tensor, lambd: float | None = None) -> torch.Tensor:
        scale = x.new_tensor(self.lambd if lambd is None else float(lambd))
        return _GradientReverse.apply(x, scale)


class DomainDiscriminator(nn.Module):
    def __init__(self, d_model: int, num_datasets: int, hidden: int | None = None) -> None:
        super().__init__()
        hidden = hidden or max(32, d_model // 2)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_datasets),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.float())
