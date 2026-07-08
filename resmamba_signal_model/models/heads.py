from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x.float())


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int | None = None, dropout: float = 0.1, depth: int = 2) -> None:
        super().__init__()
        h = hidden_dim or max(in_dim * 2, 256)
        blocks = [ResidualMLPBlock(in_dim, h, dropout=dropout) for _ in range(depth)]
        self.net = nn.Sequential(*blocks, nn.LayerNorm(in_dim), nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class CosineClassifierHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int | None = None, dropout: float = 0.1, depth: int = 2, scale: float = 16.0) -> None:
        super().__init__()
        h = hidden_dim or max(in_dim * 2, 256)
        self.features = nn.Sequential(*[ResidualMLPBlock(in_dim, h, dropout=dropout) for _ in range(depth)], nn.LayerNorm(in_dim), nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout))
        self.weight = nn.Parameter(torch.empty(out_dim, h))
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = F.normalize(self.features(x.float()), dim=-1)
        weight = F.normalize(self.weight, dim=-1)
        return self.scale.clamp_min(1.0) * features @ weight.t()


class RecognitionHeads(nn.Module):
    def __init__(self, d_model: int, num_mod_classes: int = 31, num_emitters: int = 100, dropout: float = 0.1, num_datasets: int = 32, use_dataset_bias: bool = True) -> None:
        super().__init__()
        self.use_dataset_bias = use_dataset_bias
        self.shared = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 2, d_model))
        self.modulation = MLPHead(d_model, num_mod_classes, hidden_dim=d_model * 2, dropout=dropout, depth=2)
        self.emitter = CosineClassifierHead(d_model, num_emitters, hidden_dim=d_model * 4, dropout=dropout, depth=3)
        if use_dataset_bias:
            self.modulation_dataset_bias = nn.Embedding(num_datasets, num_mod_classes)
            nn.init.zeros_(self.modulation_dataset_bias.weight)

    def forward(
        self,
        h: torch.Tensor,
        dataset_id: torch.Tensor | None = None,
        *,
        heads: str = "both",
    ) -> dict[str, torch.Tensor]:
        h = h + self.shared(h)
        out: dict[str, torch.Tensor] = {}
        if heads in ("both", "modulation"):
            modulation_logits = self.modulation(h)
            if self.use_dataset_bias and dataset_id is not None:
                dataset_id = dataset_id.long().clamp(0, self.modulation_dataset_bias.num_embeddings - 1)
                modulation_logits = modulation_logits + self.modulation_dataset_bias(dataset_id)
            out["modulation_logits"] = modulation_logits
        if heads in ("both", "emitter"):
            out["emitter_logits"] = self.emitter(h)
        return out


class PrototypeClusteringHead(nn.Module):
    def __init__(self, d_model: int, proj_dim: int = 128, num_prototypes: int = 128, temperature: float = 0.1) -> None:
        super().__init__()
        self.proj = MLPHead(d_model, proj_dim, hidden_dim=d_model * 2)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, proj_dim) * 0.02)
        self.temperature = temperature

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        z = F.normalize(self.proj(h), dim=-1)
        proto = F.normalize(self.prototypes, dim=-1)
        logits = z @ proto.t() / self.temperature
        return {"cluster_embedding": z, "cluster_logits": logits, "cluster_probs": logits.softmax(dim=-1)}
