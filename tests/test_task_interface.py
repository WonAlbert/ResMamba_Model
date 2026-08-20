from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from resmamba_signal_model.models.heads import (
    EmitterHead,
    ImputationHead,
    ModulationHead,
    PredictionHead,
    PrototypeClusteringHead,
    remap_legacy_recognition_shared,
    remap_recognition_heads_to_task_heads,
    remap_task_head_checkpoints,
    register_task,
)
from resmamba_signal_model.models.task_interface import (
    DEFAULT_TASKS,
    TaskFeatures,
    TaskSpec,
    UniversalTaskInterface,
    UniversalTaskInterfaceV2,
)
from resmamba_signal_model.models.model import SignalFoundationModel, SignalModelConfig


def test_uti_shapes_and_task_film() -> None:
    torch.manual_seed(0)
    uti = UniversalTaskInterface(d_model=32, rank=8, num_task_types=8, dropout=0.0)
    z = torch.randn(3, 32)
    patch_h = torch.randn(3, 5, 32)
    mask = torch.ones(3, 5, dtype=torch.bool)
    mask[0, -1] = False
    feat = uti(z, patch_h, mask, "modulation")
    assert feat.pooled.shape == (3, 32)
    assert feat.tokens.shape == (3, 5, 32)
    other = uti(z, patch_h, mask, "emitter")
    assert not torch.allclose(feat.pooled, other.pooled)


def test_heads_consume_task_features() -> None:
    feat = TaskFeatures(pooled=torch.randn(2, 16), tokens=torch.randn(2, 4, 16), mask=torch.ones(2, 4, dtype=torch.bool))
    mod = ModulationHead(16, num_mod_classes=5, dropout=0.0, use_dataset_bias=False)
    em = EmitterHead(16, num_emitters=7, dropout=0.0)
    clu = PrototypeClusteringHead(16, proj_dim=8, num_prototypes=4)
    pred = PredictionHead(16, patch_size=8, dropout=0.0)
    imp = ImputationHead(16, patch_size=8, dropout=0.0)
    assert mod(feat)["modulation_logits"].shape == (2, 5)
    assert em(feat)["emitter_logits"].shape == (2, 7)
    assert clu(feat)["cluster_logits"].shape[0] == 2
    assert pred(feat)["pred_patches"].shape[-2:] == (2, 8)
    assert "pred_patches" in imp(feat, span_mask=feat.mask)


def test_remap_to_task_heads_and_legacy() -> None:
    weight = torch.ones(4, 4)
    state = {
        "recognition_heads.shared.1.weight": weight,
        "recognition_heads.modulation.weight": torch.ones(3, 8),
    }
    remapped = remap_task_head_checkpoints(state)
    assert "recognition_heads.shared.1.weight" not in remapped
    assert "modulation_head.shared.1.weight" in remapped
    assert "emitter_head.shared.1.weight" in remapped
    assert "modulation_head.classifier.weight" in remapped
    again = remap_legacy_recognition_shared({"recognition_heads.shared.0.weight": torch.ones(2)})
    assert "recognition_heads.shared_modulation.0.weight" in again
    copied = remap_recognition_heads_to_task_heads(
        {"recognition_heads.shared_emitter.0.weight": torch.ones(2), "keep": torch.zeros(1)}
    )
    assert "emitter_head.shared.0.weight" in copied
    assert "keep" in copied


def test_register_task_on_model() -> None:
    cfg = SignalModelConfig(
        d_model=32,
        mamba_d_state=8,
        mamba_headdim=16,
        require_mamba_kernel=False,
        allow_fallback_mamba=True,
        attn_num_heads=4,
        patch_size=8,
        stem_channels=8,
        freq_bands=4,
        dropout=0.0,
        p_trunc=0.0,
        build_task_heads=True,
        build_adapters=True,
        num_task_types=8,
    )
    model = SignalFoundationModel(cfg)
    register_task("sonar", ModulationHead)
    head = model.register_task("sonar", ModulationHead, num_mod_classes=4, dropout=0.0, use_dataset_bias=False)
    assert "sonar" in model.task_interface.task_to_id
    assert "sonar" in model.task_adapters
    assert head is model.extra_task_heads["sonar"]


def test_domain_prompt_changes_condition_without_backbone_tokens() -> None:
    torch.manual_seed(0)
    uti = UniversalTaskInterfaceV2(
        d_model=32,
        rank=8,
        dropout=0.0,
        task_names=("modulation",),
        domain_prompt_size=6,
        num_datasets=4,
    )
    z = torch.randn(2, 32)
    tokens = torch.randn(2, 5, 32)
    mask = torch.ones(2, 5, dtype=torch.bool)
    base = uti(z, tokens, mask, "modulation")
    shifted = uti(z, tokens, mask, "modulation", metadata={"dataset_id": torch.tensor([1, 2])})
    assert shifted.pooled.shape == base.pooled.shape
    assert not torch.allclose(base.pooled, shifted.pooled)


def test_uti_v2_composes_semantics_instead_of_random_task_ids() -> None:
    torch.manual_seed(4)
    uti = UniversalTaskInterfaceV2(d_model=32, rank=8, dropout=0.0, task_names=())
    spec_a = TaskSpec("sonar_a", "classification", "pooled", "semantic", "sonar")
    spec_b = TaskSpec("sonar_b", "classification", "pooled", "semantic", "sonar")
    uti.add_task("sonar_a", spec_a)
    uti.add_task("sonar_b", spec_b)
    z = torch.randn(2, 32)
    tokens = torch.randn(2, 5, 32)
    mask = torch.ones(2, 5, dtype=torch.bool)

    cond_a = uti.condition_vector("sonar_a", 2, device=z.device, dtype=z.dtype)
    cond_b = uti.condition_vector("sonar_b", 2, device=z.device, dtype=z.dtype)
    assert torch.allclose(cond_a, cond_b)
    feat_a = uti(z, tokens, mask, "sonar_a")
    feat_b = uti(z, tokens, mask, "sonar_b")
    assert torch.allclose(feat_a.pooled, feat_b.pooled)
    assert torch.allclose(feat_a.tokens, feat_b.tokens)


def test_uti_v2_exposes_general_views_and_three_readouts() -> None:
    uti = UniversalTaskInterfaceV2(d_model=24, rank=6, dropout=0.0, task_names=())
    uti.add_task("pooled", TaskSpec("pooled", "classification", "pooled", "semantic", "rf"))
    uti.add_task("token", TaskSpec("token", "dense_prediction", "token", "context", "imu"))
    uti.add_task("query", TaskSpec("query", "generation", "query", "general", "sonar"))
    z = torch.randn(3, 24)
    tokens = torch.randn(3, 7, 24)
    mask = torch.ones(3, 7, dtype=torch.bool)
    views = uti.build_views(z, tokens)

    assert set(views) == {"general", "semantic", "source", "context"}
    assert torch.equal(views["general"][0], z)
    for task, readout in (("pooled", "pooled"), ("token", "token"), ("query", "query")):
        features = uti(z, tokens, mask, task, views=views)
        assert features.readout == readout
        assert features.pooled.shape == (3, 24)
        assert features.tokens.shape == (3, 7, 24)
        assert features.query is not None and features.query.shape == (3, 7, 24)


def test_uti_v2_parameter_budget_and_legacy_switch() -> None:
    v2 = UniversalTaskInterfaceV2(d_model=640, rank=64, task_names=DEFAULT_TASKS)
    assert sum(param.numel() for param in v2.parameters()) < 600_000

    legacy = UniversalTaskInterface(
        d_model=32,
        rank=8,
        num_task_types=8,
        legacy_mode=True,
    )
    z = torch.randn(2, 32)
    tokens = torch.randn(2, 4, 32)
    mask = torch.ones(2, 4, dtype=torch.bool)
    assert legacy(z, tokens, mask, "modulation").pooled.shape == (2, 32)
    old_keys = ("stem.", "task_embed.", "film.", "token_pool.", "fuse.")
    old_state = {
        key: value.clone()
        for key, value in legacy.state_dict().items()
        if key.startswith(old_keys)
    }
    restored = UniversalTaskInterface(
        d_model=32,
        rank=8,
        num_task_types=8,
        legacy_mode=True,
    )
    restored.load_state_dict(old_state, strict=False)
    for key, value in old_state.items():
        assert torch.equal(restored.state_dict()[key], value)
