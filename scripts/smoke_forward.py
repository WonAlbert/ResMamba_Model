#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from resmamba_signal_model import ResMambaSignalConfig, ResMambaSignalModel
from resmamba_signal_model.training.losses import pretrain_smoke_losses


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def make_batch(batch_size: int = 4, length: int = 257) -> dict[str, torch.Tensor]:
    sample_mask = torch.ones(batch_size, length, dtype=torch.bool)
    lengths = torch.tensor([length, length - 9, 64, 129], dtype=torch.long)[:batch_size]
    for i, n in enumerate(lengths.tolist()):
        sample_mask[i, n:] = False
    return {
        "iq": torch.randn(batch_size, 2, length),
        "sample_mask": sample_mask,
        "length": lengths,
        "dataset_id": torch.tensor([0, 1, 2, 3], dtype=torch.long)[:batch_size],
        "task_type_id": torch.tensor([0, 1, 0, 1], dtype=torch.long)[:batch_size],
        "mod_label_id": torch.tensor([1, 1, 2, 2], dtype=torch.long)[:batch_size],
        "emitter_id": torch.tensor([5, 5, 7, 8], dtype=torch.long)[:batch_size],
    }


def main() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ResMambaSignalConfig(d_model=64, encoder_layers=1, decoder_layers=1, space_layers=1, max_tokens=64, patch_size=8, mamba_d_state=16, num_datasets=8)
    model = ResMambaSignalModel(cfg).to(device)
    batch = _to_device(make_batch(), device)
    mae = model(batch, mode="mae")
    losses = pretrain_smoke_losses(mae, batch)
    mod = model(batch, mode="task", task="modulation")
    emit = model(batch, mode="task", task="emitter")
    clu = model(batch, mode="task", task="clustering")
    print("tokens", tuple(mae["tokens"].shape), "patches", tuple(mae["patch_targets"].shape))
    print("losses", {k: round(float(v.detach()), 6) for k, v in losses.items()})
    print("modulation_logits", tuple(mod["modulation_logits"].shape))
    print("emitter_logits", tuple(emit["emitter_logits"].shape))
    print("cluster_logits", tuple(clu["cluster_logits"].shape))


if __name__ == "__main__":
    main()
