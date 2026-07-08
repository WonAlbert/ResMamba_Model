from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from resmamba_signal_model import ResMambaSignalConfig, ResMambaSignalModel
from resmamba_signal_model.data.rfdata import normalize_iq
from resmamba_signal_model.training.losses import pretrain_smoke_losses


def make_batch() -> dict[str, torch.Tensor]:
    length = 65
    sample_mask = torch.ones(4, length, dtype=torch.bool)
    sample_mask[1, 40:] = False
    sample_mask[2, 17:] = False
    return {
        "iq": torch.randn(4, 2, length),
        "sample_mask": sample_mask,
        "length": torch.tensor([65, 40, 17, 65]),
        "dataset_id": torch.tensor([0, 1, 2, 2]),
        "task_type_id": torch.tensor([0, 1, 0, 1]),
        "mod_label_id": torch.tensor([3, 3, 4, 4]),
        "emitter_id": torch.tensor([9, 9, 10, 11]),
    }


def test_joint_power_and_complex_absmax_normalization() -> None:
    iq = torch.randn(2, 256) * 4.0 + 2.0
    joint = normalize_iq(iq, "joint_power")
    ab = normalize_iq(iq, "complex_absmax")
    assert joint.shape == iq.shape
    assert ab.shape == iq.shape
    assert joint.abs().max() <= 5.0
    power = joint.square().sum(dim=0).mean()
    assert torch.isfinite(power)


def test_prediction_forward_skips_spaces() -> None:
    if not torch.cuda.is_available():
        return
    cfg = ResMambaSignalConfig(
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        space_layers=1,
        max_tokens=16,
        patch_size=8,
        mamba_d_state=8,
        mamba_headdim=16,
        num_datasets=4,
        num_mod_classes=5,
        num_emitters=16,
        num_prototypes=8,
    )
    model = ResMambaSignalModel(cfg).cuda()
    batch = make_batch()
    batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
    pred = model(batch, mode="task", task="prediction")
    assert "spaces" not in pred
    assert pred["mae_pred"].shape[2:] == (2, cfg.patch_size)


def test_tokenizer_routing_and_losses_smoke() -> None:
    if not torch.cuda.is_available():
        return
    cfg = ResMambaSignalConfig(d_model=32, encoder_layers=1, decoder_layers=1, space_layers=1, max_tokens=16, patch_size=8, mamba_d_state=8, mamba_headdim=16, num_datasets=4, num_mod_classes=5, num_emitters=16, num_prototypes=8)
    model = ResMambaSignalModel(cfg).cuda()
    batch = make_batch()
    batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
    mae = model(batch, mode="mae")
    expected_patches = 9
    assert mae["tokens"].shape[1] == expected_patches + model.tokenizer.num_special_tokens
    assert mae["patch_targets"].shape[2:] == (2, cfg.patch_size)
    assert set(mae["spaces"].keys()) == {"mod_specific", "emitter_specific", "long_context_shared", "cross_domain_shared"}
    assert mae["modulation_repr"].shape == (4, cfg.d_model)
    assert mae["emitter_repr"].shape == (4, cfg.d_model)
    assert mae["cluster_repr"].shape == (4, cfg.d_model)
    losses = pretrain_smoke_losses(mae, batch)
    assert losses
    assert all(torch.isfinite(v) for v in losses.values())

    mod = model(batch, mode="task", task="modulation")
    emit = model(batch, mode="task", task="emitter")
    clu = model(batch, mode="task", task="clustering")
    pred = model(batch, mode="task", task="prediction")
    assert mod["modulation_logits"].shape == (4, cfg.num_mod_classes)
    assert emit["emitter_logits"].shape == (4, cfg.num_emitters)
    assert clu["cluster_logits"].shape == (4, cfg.num_prototypes)
    assert pred["mae_pred"].shape[2:] == (2, cfg.patch_size)
