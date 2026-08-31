from __future__ import annotations

import random
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import Sampler


def parse_chronos_sampling_cfg(train_cfg: dict[str, Any]) -> dict[str, Any]:
    """解析 ``chronos_sampling`` 配置块；未启用时返回 ``enabled=False``。"""
    raw = train_cfg.get("chronos_sampling")
    if not isinstance(raw, dict) or not bool(raw.get("enabled", False)):
        return {"enabled": False}
    mixup = raw.get("iq_mixup") if isinstance(raw.get("iq_mixup"), dict) else {}
    return {
        "enabled": True,
        "length_tier_pool": bool(raw.get("length_tier_pool", True)),
        "stem_sticky_batches": max(0, int(raw.get("stem_sticky_batches", 0) or 0)),
        "shuffle_buffer_batches": max(0, int(raw.get("shuffle_buffer_batches", 0) or 0)),
        "stem_share_cap": (
            None if raw.get("stem_share_cap") is None else float(raw.get("stem_share_cap"))
        ),
        "iq_mixup_enabled": bool(mixup.get("enabled", True)),
        "iq_mixup_k": max(2, int(mixup.get("k", 2) or 2)),
        "iq_mixup_alpha": max(1.0e-6, float(mixup.get("alpha", 0.3) or 0.3)),
    }


class ChronosShuffledBatchSampler(Sampler[list[int]]):
    """Chronos 式缓冲 shuffle：打乱 batch 消费顺序，降低连续同质段。"""

    def __init__(self, base: Sampler[list[int]], *, buffer_batches: int, seed: int | None = None) -> None:
        self.base = base
        self.buffer_batches = max(1, int(buffer_batches))
        self.seed = seed

    def __len__(self) -> int:
        return len(self.base)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed)
        buf: list[list[int]] = []
        for batch in self.base:
            buf.append(list(batch))
            if len(buf) < self.buffer_batches:
                continue
            rng.shuffle(buf)
            while buf:
                yield buf.pop()
        if buf:
            rng.shuffle(buf)
            for batch in buf:
                yield batch


def apply_iq_mixup(
    batch: dict[str, Any],
    *,
    k: int = 2,
    alpha: float = 0.3,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """TSMixup 风格：batch 内 K 路凸组合 I/Q（Chronos 思路，同长度 tier 内可跨 stem）。"""
    iq = batch.get("iq")
    if iq is None:
        iq = batch.get("values")
    if not torch.is_tensor(iq) or iq.ndim != 3:
        return batch
    b = int(iq.shape[0])
    if b < 2 or k < 2:
        return batch
    k = min(k, b)
    device = iq.device
    dtype = iq.dtype
    gen = generator if generator is not None else torch.Generator(device=device)
    if gen.device != device:
        gen = torch.Generator(device=device)
    weights = torch.from_numpy(np.random.dirichlet([float(alpha)] * k).astype(np.float32)).to(
        device=device, dtype=dtype
    )
    perm = torch.stack([torch.randperm(b, generator=gen, device=device) for _ in range(k)], dim=1)
    mixed = torch.zeros_like(iq)
    for j in range(k):
        mixed = mixed + weights[j] * iq[perm[:, j]]
    out = dict(batch)
    out["iq"] = mixed
    if "values" in out:
        out["values"] = mixed
    stems = batch.get("moe_route_stem")
    if isinstance(stems, list) and len(stems) == b:
        # 路由跟随 mixup 权重最大的源样本
        dom = int(torch.argmax(weights).item())
        out["moe_route_stem"] = [stems[int(perm[i, dom].item())] for i in range(b)]
    return out
