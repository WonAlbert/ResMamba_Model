from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SignalSpec:
    """模型侧最小输入契约，不依赖 data 目录的 batch 实现。"""

    name: str = "generic"
    num_channels: int | None = None
    modality: str = "generic"
    complex_pairs: tuple[tuple[int, int], ...] = ()
    adapter_key: str | None = None

    @classmethod
    def from_any(cls, spec: "SignalSpec | Mapping[str, Any] | None") -> "SignalSpec":
        if spec is None:
            return cls()
        if isinstance(spec, cls):
            return spec
        payload = dict(spec)
        pairs = payload.get("complex_pairs") or ()
        payload["complex_pairs"] = tuple((int(a), int(b)) for a, b in pairs)
        return cls(**payload)


class ChannelProjectionAdapter(nn.Module):
    """可注册的轻量 1×1 通道投影；默认初始化尽量保留前两通道。"""

    def __init__(self, in_channels: int, out_channels: int = 2) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("通道数必须为正整数")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.proj = nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1, bias=False)
        with torch.no_grad():
            self.proj.weight.zero_()
            for idx in range(min(self.in_channels, self.out_channels)):
                self.proj.weight[idx, idx, 0] = 1.0
            if self.in_channels == 1 and self.out_channels > 1:
                self.proj.weight[1:, 0, 0] = 0.0

    def forward(
        self,
        values: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if values.shape[1] != self.in_channels:
            raise ValueError(
                f"适配器期望 {self.in_channels} 通道，当前 {values.shape[1]}"
            )
        if channel_mask is not None:
            mask = channel_mask.to(device=values.device, dtype=values.dtype)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            values = values * mask.unsqueeze(-1)
        return self.proj(values)


class SignalAdapterRegistry(nn.Module):
    """显式注册可学习适配器，并为未注册任意通道提供无参数保底映射。"""

    def __init__(self) -> None:
        super().__init__()
        self.adapters = nn.ModuleDict()
        self.specs: dict[str, SignalSpec] = {}

    @staticmethod
    def _key(spec: SignalSpec) -> str:
        raw = spec.adapter_key or spec.name or spec.modality or "generic"
        return str(raw).replace(".", "_").replace("/", "_")

    def register(
        self,
        spec: SignalSpec | Mapping[str, Any],
        adapter: nn.Module | None = None,
    ) -> nn.Module:
        resolved = SignalSpec.from_any(spec)
        key = self._key(resolved)
        if adapter is None:
            if resolved.num_channels is None:
                raise ValueError("自动创建通道投影需要 SignalSpec.num_channels")
            adapter = ChannelProjectionAdapter(resolved.num_channels)
        self.adapters[key] = adapter
        self.specs[key] = resolved
        return adapter

    def _deterministic(
        self,
        values: torch.Tensor,
        spec: SignalSpec,
        channel_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, channels, _ = values.shape
        if channel_mask is None:
            mask = torch.ones(batch, channels, device=values.device, dtype=values.dtype)
        else:
            mask = channel_mask.to(device=values.device, dtype=values.dtype)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            if mask.shape[0] == 1 and batch > 1:
                mask = mask.expand(batch, -1)
            if mask.shape != (batch, channels):
                raise ValueError(
                    f"channel_mask 期望 {(batch, channels)}，当前 {tuple(mask.shape)}"
                )
        values = values * mask.unsqueeze(-1)
        if spec.complex_pairs:
            first, second = spec.complex_pairs[0]
            if 0 <= first < channels and 0 <= second < channels:
                return torch.stack([values[:, first], values[:, second]], dim=1)
        if channels == 2:
            return values
        if channels == 1:
            return torch.cat([values, torch.zeros_like(values)], dim=1)

        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = values.sum(dim=1) / denom
        weights = torch.linspace(-1.0, 1.0, channels, device=values.device, dtype=values.dtype)
        weights = weights.view(1, channels) * mask
        contrast_denom = weights.abs().sum(dim=1, keepdim=True).clamp_min(1.0)
        contrast = (values * weights.unsqueeze(-1)).sum(dim=1) / contrast_denom
        return torch.stack([mean, contrast], dim=1)

    def forward(
        self,
        values: torch.Tensor,
        spec: SignalSpec | Mapping[str, Any] | None = None,
        channel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError(f"values 期望 [B,C,L]，当前 {tuple(values.shape)}")
        resolved = SignalSpec.from_any(spec)
        if resolved.num_channels is not None and int(resolved.num_channels) != values.shape[1]:
            raise ValueError(
                f"SignalSpec 声明 {resolved.num_channels} 通道，当前输入 {values.shape[1]}"
            )
        key = self._key(resolved)
        if key in self.adapters:
            adapter = self.adapters[key]
            try:
                return adapter(values, channel_mask=channel_mask)
            except TypeError:
                return adapter(values)
        return self._deterministic(values, resolved, channel_mask)
