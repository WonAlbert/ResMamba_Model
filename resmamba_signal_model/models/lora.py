from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05

    def __post_init__(self) -> None:
        if self.rank < 0:
            raise ValueError(f"LoRA rank must be >= 0, got {self.rank}")
        if self.alpha <= 0:
            raise ValueError(f"LoRA alpha must be > 0, got {self.alpha}")


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, cfg: LoRAConfig) -> None:
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(linear)}")
        self.linear = linear
        self.cfg = cfg
        self.scaling = cfg.alpha / max(cfg.rank, 1)
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        if cfg.rank > 0:
            device = linear.weight.device
            dtype = linear.weight.dtype
            self.lora_a = nn.Parameter(torch.zeros(cfg.rank, linear.in_features, device=device, dtype=dtype))
            self.lora_b = nn.Parameter(torch.zeros(linear.out_features, cfg.rank, device=device, dtype=dtype))
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b)
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.linear.bias

    @property
    def in_features(self) -> int:
        return self.linear.in_features

    @property
    def out_features(self) -> int:
        return self.linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self.cfg.rank > 0 and self.lora_a is not None and self.lora_b is not None:
            lora = self.dropout(x) @ self.lora_a.transpose(0, 1) @ self.lora_b.transpose(0, 1)
            out = out + lora * self.scaling
        return out


def _replace_linear(module: nn.Module, name: str, child: nn.Linear, cfg: LoRAConfig) -> LoRALinear:
    wrapped = LoRALinear(child, cfg)
    setattr(module, name, wrapped)
    return wrapped


def inject_lora_into_module(module: nn.Module, cfg: LoRAConfig) -> list[str]:
    injected: list[str] = []
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            _replace_linear(module, name, child, cfg)
            injected.append(name)
            continue
        if isinstance(child, LoRALinear):
            continue
        injected.extend(f"{name}.{path}" for path in inject_lora_into_module(child, cfg))
    return injected


def set_lora_trainable(module: nn.Module, *, train_base: bool = False) -> None:
    for param in module.parameters():
        param.requires_grad = train_base
    for child in module.modules():
        if isinstance(child, LoRALinear):
            for param in child.linear.parameters():
                param.requires_grad = train_base
            if child.lora_a is not None:
                child.lora_a.requires_grad = True
            if child.lora_b is not None:
                child.lora_b.requires_grad = True


def count_lora_params(module: nn.Module) -> tuple[int, int]:
    total = 0
    trainable = 0
    for child in module.modules():
        if not isinstance(child, LoRALinear):
            continue
        for param in (child.lora_a, child.lora_b):
            if param is None:
                continue
            numel = param.numel()
            total += numel
            if param.requires_grad:
                trainable += numel
    return total, trainable
