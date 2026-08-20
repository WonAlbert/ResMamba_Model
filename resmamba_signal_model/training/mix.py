from __future__ import annotations

import math

import torch


def resolve_mix_strategy(name: str | None) -> str:
    key = str(name or "token_share").strip().lower()
    if key in ("token_share", "dynamic", "loss_ema"):
        return "token_share"
    if key in ("equal", "uniform", "fixed"):
        return "equal"
    raise ValueError(f"未知 mix_strategy={name!r}，可选: token_share | equal")


class DynamicRatioScheduler:
    """按 token 平均损失 EMA 分配各源 token 预算份额。"""

    def __init__(
        self,
        names: list[str],
        *,
        alpha: float = 0.9,
        tau: float = 1.0,
        min_ratio: float = 0.05,
        value_clip: float = 2.0,
    ) -> None:
        if not names:
            raise ValueError("DynamicRatioScheduler 需要至少一个源名")
        if min_ratio * len(names) > 1.0 + 1e-6:
            raise ValueError(f"min_ratio={min_ratio} 与 {len(names)} 个源无法同时满足")
        self.names = list(names)
        self.alpha = float(alpha)
        self.tau = max(float(tau), 1.0e-6)
        self.min_ratio = float(min_ratio)
        self.value_clip = float(value_clip)
        self.ema: dict[str, float] = {name: 1.0 for name in self.names}

    def state_dict(self) -> dict[str, object]:
        return {
            "names": list(self.names),
            "alpha": self.alpha,
            "tau": self.tau,
            "min_ratio": self.min_ratio,
            "value_clip": self.value_clip,
            "ema": {name: float(self.ema[name]) for name in self.names},
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        ema = state.get("ema") or {}
        if not isinstance(ema, dict):
            raise TypeError("mix_state.ema 必须是 dict")
        if "value_clip" in state:
            self.value_clip = float(state["value_clip"])
        for name in self.names:
            if name in ema:
                self.ema[name] = float(ema[name])

    def update(self, losses: dict[str, float], n_tokens: dict[str, float]) -> None:
        for name in self.names:
            if name not in losses or name not in n_tokens:
                continue
            tokens = float(n_tokens[name])
            if tokens <= 0:
                continue
            value = float(losses[name]) / tokens
            if self.value_clip > 0:
                value = min(max(value, 0.0), self.value_clip)
            self.ema[name] = self.alpha * self.ema[name] + (1.0 - self.alpha) * value

    def ratios(self) -> dict[str, float]:
        xs = torch.tensor([self.ema[name] for name in self.names], dtype=torch.float64)
        logits = xs / self.tau
        r = torch.softmax(logits, dim=0)
        n = len(self.names)
        leftover = max(0.0, 1.0 - self.min_ratio * n)
        r = leftover * r + self.min_ratio
        r = r / r.sum()
        return {name: float(r[i]) for i, name in enumerate(self.names)}

    def token_shares(self, total_budget: int) -> dict[str, int]:
        ratios = self.ratios()
        raw = [max(self.min_ratio, ratios[name]) * total_budget for name in self.names]
        shares = [max(1, int(math.floor(x))) for x in raw]
        delta = int(total_budget) - sum(shares)
        i = 0
        while delta > 0 and self.names:
            shares[i % len(shares)] += 1
            delta -= 1
            i += 1
        while delta < 0:
            idx = max(range(len(shares)), key=lambda j: shares[j])
            if shares[idx] <= 1:
                break
            shares[idx] -= 1
            delta += 1
        return {name: int(share) for name, share in zip(self.names, shares)}
