from __future__ import annotations

import torch

from scripts.train_pipeline import build_model, count_params
from resmamba_signal_model.models.lora import LoRALinear
from resmamba_signal_model.training.stages import configure_peft_stage2, configure_stage2_heads


def _any_trainable(module) -> bool:
    return any(p.requires_grad for p in module.parameters())


def test_modulation_task_path_matches_legacy_stage2() -> None:
    legacy = build_model("configs/model_resmamba_400m.yaml")
    configure_stage2_heads(
        legacy,
        "modulation",
        freeze_backbone=True,
        unfreeze_modulation_backbone=True,
        unfreeze_emitter_backbone=False,
    )
    peft = build_model("configs/model_resmamba_400m.yaml")
    configure_peft_stage2(
        peft,
        "modulation",
        peft_mode="task_path",
        freeze_backbone=True,
        unfreeze_modulation_backbone=True,
        unfreeze_emitter_backbone=False,
    )
    assert _any_trainable(legacy.mod_fuse) == _any_trainable(peft.mod_fuse)
    assert _any_trainable(legacy.space_backbones["mod_specific"]) == _any_trainable(peft.space_backbones["mod_specific"])
    assert count_params(legacy) == count_params(peft)


def test_head_only_trains_fuse_and_heads_only() -> None:
    model = build_model("configs/model_resmamba_400m.yaml")
    configure_peft_stage2(
        model,
        "modulation",
        peft_mode="head_only",
        freeze_backbone=True,
        unfreeze_modulation_backbone=True,
        unfreeze_emitter_backbone=False,
    )
    assert _any_trainable(model.recognition_heads.modulation)
    assert _any_trainable(model.mod_fuse)
    assert not _any_trainable(model.space_backbones["mod_specific"])
    assert not _any_trainable(model.encoder)
    _, trainable = count_params(model)
    assert trainable < 25_000_000


def test_lora_task_path_freezes_space_backbone_base() -> None:
    model = build_model("configs/model_tiny.yaml")
    injected = configure_peft_stage2(
        model,
        "modulation",
        peft_mode="lora_task_path",
        freeze_backbone=True,
        unfreeze_modulation_backbone=True,
        unfreeze_emitter_backbone=False,
        lora_rank=4,
    )
    assert injected
    backbone = model.space_backbones["mod_specific"]
    assert _any_trainable(model.space_projs["mod_specific"])
    assert _any_trainable(backbone)
    base_trainable = [p.requires_grad for n, p in backbone.named_parameters() if "linear.weight" in n or "linear.bias" in n]
    lora_trainable = [p.requires_grad for n, p in backbone.named_parameters() if "lora_" in n]
    assert base_trainable and not any(base_trainable)
    assert lora_trainable and all(lora_trainable)


def test_lora_injected_after_to_device_matches_backbone_device() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    model = build_model("configs/model_tiny.yaml").to(device)
    configure_peft_stage2(
        model,
        "modulation",
        peft_mode="lora_task_path",
        freeze_backbone=True,
        unfreeze_modulation_backbone=True,
        unfreeze_emitter_backbone=False,
        lora_rank=4,
    )
    for module in model.modules():
        if not isinstance(module, LoRALinear):
            continue
        assert module.lora_a is not None and module.lora_b is not None
        assert module.lora_a.device == module.linear.weight.device
        assert module.lora_b.device == module.linear.weight.device


def test_task_path_shared_unfreezes_shared_spaces() -> None:
    model = build_model("configs/model_resmamba_400m.yaml")
    configure_peft_stage2(
        model,
        "modulation",
        peft_mode="task_path_shared",
        freeze_backbone=True,
        unfreeze_modulation_backbone=True,
        unfreeze_emitter_backbone=False,
    )
    assert _any_trainable(model.space_backbones["mod_specific"])
    assert _any_trainable(model.space_backbones["long_context_shared"])
    assert _any_trainable(model.space_backbones["cross_domain_shared"])
    assert not _any_trainable(model.encoder)
    assert not _any_trainable(model.tokenizer)
    total, trainable = count_params(model)
    assert trainable > 45_533_425


def test_emitter_task_path_baseline_param_budget() -> None:
    model = build_model("configs/model_resmamba_400m.yaml")
    configure_peft_stage2(
        model,
        "emitter",
        peft_mode="task_path",
        freeze_backbone=True,
        unfreeze_modulation_backbone=False,
        unfreeze_emitter_backbone=True,
    )
    total, trainable = count_params(model)
    assert total > 300_000_000
    assert 40_000_000 < trainable < 50_000_000
