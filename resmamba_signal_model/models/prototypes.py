from __future__ import annotations

from typing import Any, Iterable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

CONTENT_NAMESPACE = "modulation"
DEVICE_NAMESPACE = "emitter"
NAMESPACE_ALIASES = {
    "modulation": CONTENT_NAMESPACE,
    "content": CONTENT_NAMESPACE,
    "semantic": CONTENT_NAMESPACE,
    "emitter": DEVICE_NAMESPACE,
    "device": DEVICE_NAMESPACE,
    "source": DEVICE_NAMESPACE,
}
DEFAULT_NAMESPACES: tuple[str, ...] = (CONTENT_NAMESPACE, DEVICE_NAMESPACE)


def resolve_namespace(name: str) -> str:
    key = str(name).strip().lower()
    if key not in NAMESPACE_ALIASES:
        raise KeyError(f"未知原型命名空间 {name!r}，可选: {sorted(set(NAMESPACE_ALIASES))}")
    return NAMESPACE_ALIASES[key]


class PrototypeBank(nn.Module):
    """单一命名空间：均值、对角协方差、计数与版本。"""

    def __init__(self, num_prototypes: int, dim: int, *, name: str) -> None:
        super().__init__()
        self.name = str(name)
        self.num_prototypes = int(num_prototypes)
        self.dim = int(dim)
        self.mean = nn.Parameter(torch.randn(self.num_prototypes, self.dim) * 0.02)
        self.log_var = nn.Parameter(torch.zeros(self.num_prototypes, self.dim))
        self.register_buffer("count", torch.zeros(self.num_prototypes))
        self.register_buffer("version", torch.zeros((), dtype=torch.long))

    def normalized_mean(self) -> torch.Tensor:
        return F.normalize(self.mean.float(), dim=-1)

    def cosine_logits(self, embedding: torch.Tensor, temperature: float) -> torch.Tensor:
        z = F.normalize(embedding.float(), dim=-1)
        tau = max(float(temperature), 1.0e-6)
        return z @ self.normalized_mean().t() / tau

    def mahalanobis_sq(self, embedding: torch.Tensor) -> torch.Tensor:
        z = embedding.float()
        diff = z.unsqueeze(1) - self.mean.float().unsqueeze(0)
        var = self.log_var.float().exp().clamp_min(1.0e-4)
        return (diff.square() / var).sum(dim=-1)

    def gaussian_nll(self, embedding: torch.Tensor) -> torch.Tensor:
        var = self.log_var.float().exp().clamp_min(1.0e-4)
        maha = self.mahalanobis_sq(embedding)
        return 0.5 * (maha + var.log().sum(dim=-1).unsqueeze(0))

    @torch.no_grad()
    def ema_update(
        self,
        embedding: torch.Tensor,
        assignment: torch.Tensor,
        *,
        momentum: float = 0.99,
    ) -> None:
        if embedding.numel() == 0:
            return
        z = embedding.detach().float()
        if assignment.ndim == 1:
            k = int(self.num_prototypes)
            soft = F.one_hot(assignment.long().clamp(0, k - 1), k).to(dtype=z.dtype)
        else:
            soft = assignment.detach().float()
        mass = soft.sum(dim=0).clamp_min(1.0e-8)
        new_mean = (soft.t() @ z) / mass.unsqueeze(-1)
        centered = z.unsqueeze(1) - new_mean.unsqueeze(0)
        new_var = (soft.unsqueeze(-1) * centered.square()).sum(dim=0) / mass.unsqueeze(-1)
        decay = float(momentum)
        used = mass > 1.0e-6
        if not used.any():
            return
        self.mean.data[used] = decay * self.mean.data[used] + (1.0 - decay) * new_mean[used].to(dtype=self.mean.dtype)
        log_var = new_var.clamp_min(1.0e-4).log()
        self.log_var.data[used] = decay * self.log_var.data[used] + (1.0 - decay) * log_var[used].to(dtype=self.log_var.dtype)
        self.count[used] = self.count[used] + mass[used].to(dtype=self.count.dtype)
        self.version += 1

    @torch.no_grad()
    def absorb(
        self,
        embedding: torch.Tensor,
        *,
        min_count: float = 2.0,
        replace_frac: float = 0.1,
    ) -> int:
        """把高置信未知簇写入低计数槽位，不扩维。"""
        if embedding.ndim != 2 or embedding.shape[0] == 0:
            return 0
        z = F.normalize(embedding.detach().float(), dim=-1)
        k = int(self.num_prototypes)
        n_replace = max(1, min(k, int(round(k * float(replace_frac)))))
        empty = (self.count <= float(min_count)).nonzero(as_tuple=False).flatten()
        if empty.numel() == 0:
            empty = self.count.argsort()[:n_replace]
        else:
            empty = empty[:n_replace]
        if empty.numel() == 0:
            return 0
        centroid = F.normalize(z.mean(dim=0), dim=0)
        n_slots = int(empty.numel())
        self.mean.data[empty] = centroid.to(dtype=self.mean.dtype).unsqueeze(0).expand(n_slots, -1)
        self.log_var.data[empty] = 0.0
        self.count[empty] = float(z.shape[0])
        self.version += 1
        return n_slots


class PrototypeRegistry(nn.Module):
    """modulation/content 与 emitter/device 分命名空间，分类/聚类/开集/增量共享。"""

    def __init__(
        self,
        dim: int,
        *,
        num_prototypes: int | Mapping[str, int] = 128,
        namespaces: Iterable[str] = DEFAULT_NAMESPACES,
    ) -> None:
        super().__init__()
        names = tuple(resolve_namespace(name) for name in namespaces)
        if not names:
            names = DEFAULT_NAMESPACES
        counts: dict[str, int]
        if isinstance(num_prototypes, Mapping):
            counts = {resolve_namespace(k): int(v) for k, v in num_prototypes.items()}
        else:
            counts = {name: int(num_prototypes) for name in names}
        self.banks = nn.ModuleDict(
            {name: PrototypeBank(counts.get(name, 128), dim, name=name) for name in names}
        )
        self.dim = int(dim)

    def available_namespaces(self) -> tuple[str, ...]:
        return tuple(self.banks.keys())

    def bank(self, namespace: str) -> PrototypeBank:
        name = resolve_namespace(namespace)
        if name not in self.banks:
            raise KeyError(f"未注册命名空间 {namespace!r}，已有: {list(self.banks)}")
        return self.banks[name]  # type: ignore[return-value]

    def energy(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        tau = max(float(temperature), 1.0e-6)
        return -tau * torch.logsumexp(logits.float() / tau, dim=-1)

    def score(
        self,
        namespace: str,
        embedding: torch.Tensor,
        logits: torch.Tensor | None = None,
        *,
        temperature: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        if embedding.shape[-1] != self.dim:
            return {}
        bank = self.bank(namespace)
        if logits is None:
            logits = bank.cosine_logits(embedding, temperature)
        energy = self.energy(logits, temperature)
        maha = bank.mahalanobis_sq(embedding)
        gauss = bank.gaussian_nll(embedding)
        nearest_maha = maha.min(dim=-1).values
        nearest_gauss = gauss.min(dim=-1).values
        combined = energy + nearest_maha
        return {
            "openset_energy": energy,
            "openset_mahalanobis": nearest_maha,
            "openset_gaussian": nearest_gauss,
            "openset_score": combined,
            "openset_logits": logits,
        }

    @torch.no_grad()
    def update(
        self,
        namespace: str,
        embedding: torch.Tensor,
        assignment: torch.Tensor,
        *,
        momentum: float = 0.99,
    ) -> None:
        self.bank(namespace).ema_update(embedding, assignment, momentum=momentum)

    @torch.no_grad()
    def absorb(
        self,
        namespace: str,
        embedding: torch.Tensor,
        **kwargs: Any,
    ) -> int:
        return int(self.bank(namespace).absorb(embedding, **kwargs))

    def load_from_weight(self, namespace: str, weight: torch.Tensor) -> None:
        bank = self.bank(namespace)
        n = min(bank.num_prototypes, int(weight.shape[0]))
        d = min(bank.dim, int(weight.shape[1]))
        with torch.no_grad():
            bank.mean[:n, :d].copy_(weight[:n, :d].to(dtype=bank.mean.dtype))
