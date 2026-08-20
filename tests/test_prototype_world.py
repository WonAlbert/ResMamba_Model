from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from resmamba_signal_model.models.heads import PrototypeClusteringHead
from resmamba_signal_model.models.prototypes import CONTENT_NAMESPACE, DEVICE_NAMESPACE, PrototypeRegistry
from resmamba_signal_model.training.continual import (
    absorb_unknown_embeddings,
    apply_continual_freeze,
    confidence_masked_distillation_loss,
    continual_parameter_budget,
    old_prototype_anchor_loss,
)
from resmamba_signal_model.training.losses import negcos_temperature
from resmamba_signal_model.models.model import SignalFoundationModel, SignalModelConfig
from resmamba_signal_model.models.peft import PeftConfig, inject_hybrid_lora
from resmamba_signal_model.training.freeze import apply_stage_freeze


def test_negcos_temperature_is_zero_param_and_anneals() -> None:
    assert negcos_temperature(0.0) > negcos_temperature(0.5) > negcos_temperature(1.0)
    assert math.isclose(negcos_temperature(0.0), 0.5, rel_tol=1e-6)
    assert math.isclose(negcos_temperature(1.0), 0.05, rel_tol=1e-6)


def test_prototype_namespaces_do_not_mix() -> None:
    torch.manual_seed(0)
    registry = PrototypeRegistry(8, num_prototypes=4)
    content = F.normalize(torch.randn(5, 8), dim=-1)
    device = F.normalize(torch.randn(5, 8) + 3.0, dim=-1)
    assign = torch.zeros(5, 4)
    assign[:, 0] = 1.0
    before_device = registry.bank(DEVICE_NAMESPACE).mean.detach().clone()
    registry.update(CONTENT_NAMESPACE, content, assign)
    after_device = registry.bank(DEVICE_NAMESPACE).mean
    assert torch.equal(before_device, after_device)
    scores_c = registry.score(CONTENT_NAMESPACE, content, temperature=0.1)
    scores_d = registry.score(DEVICE_NAMESPACE, device, temperature=0.1)
    assert scores_c["openset_energy"].shape == (5,)
    assert scores_d["openset_mahalanobis"].shape == (5,)
    assert not torch.equal(registry.bank(CONTENT_NAMESPACE).mean, registry.bank(DEVICE_NAMESPACE).mean)


def test_openset_energy_and_mahalanobis() -> None:
    torch.manual_seed(0)
    registry = PrototypeRegistry(4, num_prototypes=3)
    with torch.no_grad():
        registry.bank(CONTENT_NAMESPACE).mean.copy_(torch.eye(3, 4))
    known = F.normalize(torch.eye(3, 4)[:2], dim=-1)
    unknown = F.normalize(torch.ones(2, 4), dim=-1)
    known_s = registry.score(CONTENT_NAMESPACE, known, temperature=0.2)
    unk_s = registry.score(CONTENT_NAMESPACE, unknown, temperature=0.2)
    assert float(unk_s["openset_energy"].mean().detach()) > float(known_s["openset_energy"].mean().detach())
    assert float(unk_s["openset_mahalanobis"].mean().detach()) >= float(known_s["openset_mahalanobis"].mean().detach()) - 1e-5
    assert "openset_gaussian" in known_s and "openset_score" in known_s


def test_clustering_head_checkpoint_compatible_and_two_views() -> None:
    torch.manual_seed(0)
    old = PrototypeClusteringHead(16, proj_dim=8, num_prototypes=4, temperature=0.1)
    state = old.state_dict()
    new = PrototypeClusteringHead(16, proj_dim=8, num_prototypes=4, temperature=0.2)
    missing, unexpected = new.load_state_dict(state, strict=True)
    assert not missing and not unexpected
    new.train()
    out = new(torch.randn(3, 16))
    assert "cluster_embedding_view2" in out
    assert out["cluster_logits_view2"].shape == out["cluster_logits"].shape


def test_confidence_masked_distillation_ignores_low_conf() -> None:
    student = torch.zeros(4, 3, requires_grad=True)
    teacher = torch.zeros(4, 3)
    teacher[0, 0] = 8.0
    teacher[1, 1] = 0.1
    loss = confidence_masked_distillation_loss(student, teacher, temperature=2.0, confidence_threshold=0.8)
    loss.backward()
    assert torch.isfinite(loss)
    assert student.grad is not None


def test_old_prototype_anchor_and_absorb() -> None:
    registry = PrototypeRegistry(4, num_prototypes=3)
    frozen = registry.bank(CONTENT_NAMESPACE).mean.detach().clone()
    with torch.no_grad():
        registry.bank(CONTENT_NAMESPACE).mean.add_(0.5)
    loss = old_prototype_anchor_loss(
        registry.bank(CONTENT_NAMESPACE).mean,
        frozen,
        torch.tensor([4.0, 0.0, 3.0]),
    )
    assert float(loss.detach()) > 0.0
    n = absorb_unknown_embeddings(
        registry,
        F.normalize(torch.randn(6, 4), dim=-1),
        namespace=CONTENT_NAMESPACE,
        min_count=2,
    )
    assert n >= 1


def _tiny(**kwargs) -> SignalFoundationModel:
    payload = dict(
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
        build_shared_adapter=True,
        build_prototype_registry=True,
    )
    payload.update(kwargs)
    return SignalFoundationModel(SignalModelConfig(**payload))


def test_continual_freeze_uses_shared_adapter_not_per_task_lora() -> None:
    model = _tiny()
    inject_hybrid_lora(
        model,
        ["modulation", "emitter"],
        PeftConfig(r_attn=2, r_mamba=2, lora_alpha_attn=2, lora_alpha_mamba=2, shared_lora=True),
    )
    apply_stage_freeze(model, "continual")
    budget = continual_parameter_budget(model)
    assert budget["has_shared_adapter"]
    assert budget["per_task_lora"] == []
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any(n.startswith("shared_adapter.") for n in names)
    assert any(n.startswith("prototype_registry.") for n in names)
    assert not any("lora_A.modulation" in n for n in names)
