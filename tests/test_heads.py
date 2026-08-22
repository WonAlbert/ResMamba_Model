from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from resmamba_signal_model.models.heads import (
    CosineClassifierHead,
    EmitterHead,
    MLPHead,
    RecognitionHeads,
    remap_legacy_recognition_shared,
    ModulationHead,
    remap_task_head_checkpoints,
    ResidualMLPBlock,
)


def test_split_shared_layers_are_independent() -> None:
    torch.manual_seed(0)
    head = RecognitionHeads(d_model=8, num_mod_classes=3, num_emitters=4, dropout=0.0, use_dataset_bias=False)
    x = torch.randn(2, 8)
    both = head(x, heads="both")
    only_mod = head(x, heads="modulation")
    only_em = head(x, heads="emitter")
    assert "modulation_logits" in both and "emitter_logits" in both
    assert "emitter_logits" not in only_mod
    assert "modulation_logits" not in only_em
    assert torch.allclose(both["modulation_logits"], only_mod["modulation_logits"])
    assert torch.allclose(both["emitter_logits"], only_em["emitter_logits"])


def test_zeroing_modulation_shared_does_not_change_emitter() -> None:
    torch.manual_seed(1)
    head = RecognitionHeads(d_model=8, num_mod_classes=3, num_emitters=4, dropout=0.0, use_dataset_bias=False)
    x = torch.randn(3, 8)
    before = head(x, heads="both")
    with torch.no_grad():
        for param in head.shared_modulation.parameters():
            param.zero_()
    after = head(x, heads="both")
    assert not torch.allclose(before["modulation_logits"], after["modulation_logits"])
    assert torch.allclose(before["emitter_logits"], after["emitter_logits"])


def test_remap_legacy_shared_copies_to_both_and_drops_old() -> None:
    weight = torch.ones(4, 4)
    state = {
        "recognition_heads.shared.1.weight": weight,
        "encoder.layers.0.weight": torch.zeros(2),
        "recognition_heads.modulation.net.4.weight": torch.ones(1),
    }
    remapped = remap_legacy_recognition_shared(state)
    assert "recognition_heads.shared.1.weight" not in remapped
    assert torch.equal(remapped["recognition_heads.shared_modulation.1.weight"], weight)
    assert torch.equal(remapped["recognition_heads.shared_emitter.1.weight"], weight)
    assert "encoder.layers.0.weight" in remapped


def test_remap_does_not_overwrite_existing_split_keys() -> None:
    legacy = torch.ones(2)
    kept = torch.full((2,), 7.0)
    state = {
        "recognition_heads.shared.0.weight": legacy,
        "recognition_heads.shared_modulation.0.weight": kept,
    }
    remapped = remap_legacy_recognition_shared(state)
    assert torch.equal(remapped["recognition_heads.shared_modulation.0.weight"], kept)
    assert torch.equal(remapped["recognition_heads.shared_emitter.0.weight"], legacy)


def test_modulation_head_is_independent_of_emitter_structure() -> None:
    head = ModulationHead(d_model=8, num_mod_classes=3, dropout=0.0, use_dataset_bias=False)
    logits = head(torch.randn(2, 8))
    assert logits["modulation_logits"].shape == (2, 3)


def test_remap_task_head_checkpoints_keeps_existing_targets() -> None:
    kept = torch.full((2,), 3.0)
    state = {
        "recognition_heads.shared_modulation.0.weight": torch.ones(2),
        "modulation_head.shared.0.weight": kept,
    }
    remapped = remap_task_head_checkpoints(state)
    assert torch.equal(remapped["modulation_head.shared.0.weight"], kept)


def test_legacy_cosine_head_keeps_residual_mlp_contract() -> None:
    head = CosineClassifierHead(8, 3, hidden_dim=16, dropout=0.0, depth=2, low_rank_prototype=False)
    keys = set(head.state_dict())
    assert "features.0.net.1.weight" in keys
    assert isinstance(head.features[0], ResidualMLPBlock)
    assert not head.low_rank_prototype
    x = torch.randn(2, 8)
    logits = head(x)
    assert logits.shape == (2, 3)


def test_default_heads_use_low_rank_prototype() -> None:
    from resmamba_signal_model.config import load_yaml_config
    from resmamba_signal_model.models.model import SignalModelConfig, SignalFoundationModel

    section = dict(load_yaml_config("configs/model_tiny.yaml")["model"])
    section.update(
        {
            "build_task_heads": True,
            "build_task_interface": True,
            "require_mamba_kernel": False,
            "allow_fallback_mamba": True,
        }
    )
    model = SignalFoundationModel(SignalModelConfig.from_dict(section))
    assert model.modulation_head.classifier.low_rank_prototype
    assert model.emitter_head.classifier.low_rank_prototype
    assert model.modulation_head.classifier.weight.shape[-1] == 64
    assert model.emitter_fingerprint is not None
    assert any(n.startswith("fp_head.") for n in model.emitter_head.state_dict())


def test_low_rank_prototype_shrinks_cosine_and_mlp_and_stays_opt_in() -> None:
    torch.manual_seed(0)
    full = CosineClassifierHead(32, 16, hidden_dim=64, dropout=0.0, depth=2)
    compact = CosineClassifierHead(32, 16, dropout=0.0, low_rank_prototype=True, prototype_rank=8)
    assert sum(p.numel() for p in compact.parameters()) < sum(p.numel() for p in full.parameters())
    assert compact.weight.shape == (16, 8)
    assert compact(torch.randn(3, 32)).shape == (3, 16)

    mlp_full = MLPHead(32, 10, hidden_dim=64, dropout=0.0)
    mlp_lr = MLPHead(32, 10, dropout=0.0, low_rank_prototype=True, prototype_rank=8)
    assert sum(p.numel() for p in mlp_lr.parameters()) < sum(p.numel() for p in mlp_full.parameters())
    assert mlp_lr(torch.randn(2, 32)).shape == (2, 10)

    old = ModulationHead(d_model=8, num_mod_classes=3, dropout=0.0, low_rank_prototype=False)
    assert not old.classifier.low_rank_prototype


