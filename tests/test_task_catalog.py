from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from resmamba_signal_model.models.heads import ModulationHead
from resmamba_signal_model.models.model import SignalFoundationModel, SignalModelConfig
from resmamba_signal_model.models.peft import PeftConfig, inject_hybrid_lora
from resmamba_signal_model.models.task_interface import UniversalTaskInterface
from resmamba_signal_model.training.freeze import apply_stage_freeze
from resmamba_signal_model.training.losses import downstream_task_loss
from resmamba_signal_model.training.task_catalog import resolve_task_catalog, builtin_spec


def test_catalog_defaults_to_five_tasks() -> None:
    catalog = resolve_task_catalog({})
    assert catalog.names == ["modulation", "emitter", "clustering", "prediction", "imputation"]
    assert catalog.source_to_task["classification"] == "modulation"
    assert builtin_spec("modulation").label_field == "canonical_mod_label_id"
    assert builtin_spec("emitter").label_field == "global_emitter_id"


def test_catalog_accepts_extra_pool_and_kind() -> None:
    catalog = resolve_task_catalog(
        {
            "task_pools": {
                "classification": ["a_train", "a_val"],
                "sonar": ["sonar_train", "sonar_val"],
            },
            "task_kinds": {"sonar": "classification"},
        }
    )
    assert "modulation" in catalog.names
    assert "sonar" in catalog.names
    assert catalog.kind("sonar") == "classification"
    assert catalog.get("sonar").source == "sonar"
    assert catalog.mask_mode("sonar") == "none"


def test_catalog_tasks_list_can_be_arbitrary() -> None:
    catalog = resolve_task_catalog(
        {
            "tasks": [
                "modulation",
                {"name": "nav_fix", "kind": "prediction", "source": "nav"},
                {"name": "sonar", "kind": "classification", "num_classes": 9},
            ]
        }
    )
    assert catalog.names == ["modulation", "nav_fix", "sonar"]
    assert catalog.kind("nav_fix") == "prediction"
    assert catalog.mask_mode("nav_fix") == "suffix"
    assert catalog.get("sonar").num_classes == 9


def test_uti_grows_when_registering_many_tasks() -> None:
    uti = UniversalTaskInterface(d_model=16, rank=4, num_task_types=2, task_names=("a", "b"))
    assert uti.task_embed.num_embeddings == 2
    for i in range(6):
        uti.add_task(f"extra_{i}")
    assert "extra_5" in uti.task_to_id
    assert uti.task_embed.num_embeddings >= 8
    feat = uti(torch.randn(2, 16), torch.randn(2, 3, 16), torch.ones(2, 3, dtype=torch.bool), "extra_5")
    assert feat.pooled.shape == (2, 16)


def test_model_builds_n_task_heads_and_stage3_isolates_extra() -> None:
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
        task_names=("modulation", "sonar", "nav_fix"),
        task_kinds={"sonar": "classification", "nav_fix": "prediction"},
        num_mod_classes=5,
    )
    model = SignalFoundationModel(cfg)
    assert model.modulation_head is not None
    assert "sonar" in model.extra_task_heads
    assert "nav_fix" in model.extra_task_heads
    assert "sonar" in model.task_adapters
    inject_hybrid_lora(model, ["modulation", "sonar", "nav_fix"], PeftConfig(r_attn=2, r_mamba=2, lora_alpha_attn=2, lora_alpha_mamba=2))
    apply_stage_freeze(model, "stage3", task="sonar", train_cfg={})
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any("lora_A.sonar" in n for n in trainable)
    assert not any("lora_A.modulation" in n for n in trainable)
    assert any(n.startswith("extra_task_heads.sonar.") for n in trainable)
    assert not any(n.startswith("modulation_head.") for n in trainable)


def test_extra_classification_loss_uses_task_logits() -> None:
    head = ModulationHead(8, num_mod_classes=3, dropout=0.0, use_dataset_bias=False, logits_key="sonar_logits")
    feat = torch.randn(4, 8)
    out = head(feat)
    assert "task_logits" in out and "sonar_logits" in out
    batch = {"mod_label_id": torch.tensor([0, 1, 2, 1]), "source_label_id": torch.tensor([-1, -1, -1, -1])}
    packed = {"z": feat, "task_logits": out["task_logits"], "sonar_logits": out["sonar_logits"]}
    loss, parts = downstream_task_loss(packed, batch, "sonar", task_kind="classification", recon_weight=0.0, phys_weight=0.0, domain_weight=0.0)
    assert torch.isfinite(loss)
    assert "task_ce" in parts
