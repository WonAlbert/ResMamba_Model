from __future__ import annotations

from typing import Literal

import torch.nn as nn

from resmamba_signal_model.models.lora import LoRAConfig, inject_lora_into_module, set_lora_trainable

PeftMode = Literal["head_only", "task_path", "task_path_shared", "lora_task_path", "full_backbone"]


def set_trainable(module: nn.Module, trainable: bool) -> None:
    for p in module.parameters():
        p.requires_grad = trainable


def unfreeze_modulation_path(model: nn.Module) -> None:
    set_trainable(model.mod_fuse, True)
    set_trainable(model.space_projs["mod_specific"], True)
    set_trainable(model.space_backbones["mod_specific"], True)


def unfreeze_emitter_path(model: nn.Module) -> None:
    set_trainable(model.emitter_fuse, True)
    set_trainable(model.space_projs["emitter_specific"], True)
    set_trainable(model.space_backbones["emitter_specific"], True)


def unfreeze_clustering_path(model: nn.Module) -> None:
    set_trainable(model.shared_fuse, True)
    unfreeze_shared_spaces(model)


def unfreeze_shared_spaces(model: nn.Module) -> None:
    for name in ("long_context_shared", "cross_domain_shared"):
        set_trainable(model.space_projs[name], True)
        set_trainable(model.space_backbones[name], True)


def _enable_recognition_heads(model: nn.Module, task: str | None) -> None:
    if task in (None, "modulation", "emitter", "recognition"):
        set_trainable(model.recognition_heads, True)


def _enable_task_heads(model: nn.Module, task: str | None) -> None:
    if task in (None, "modulation", "emitter", "recognition"):
        set_trainable(model.recognition_heads, True)
    if task in (None, "clustering"):
        set_trainable(model.clustering_head, True)
    if task in (None, "prediction"):
        set_trainable(model.decoder, True)
        set_trainable(model.iq_mae_out, True)
        set_trainable(model.future_out, True)
        set_trainable(model.physical_out, True)
        model.mask_token.requires_grad = True


def _enable_task_fuse(model: nn.Module, task: str | None) -> None:
    if task in (None, "modulation", "recognition"):
        set_trainable(model.mod_fuse, True)
    if task in (None, "emitter", "recognition"):
        set_trainable(model.emitter_fuse, True)
    if task in (None, "clustering"):
        set_trainable(model.shared_fuse, True)


def _task_space_names(task: str | None, *, unfreeze_modulation_backbone: bool, unfreeze_emitter_backbone: bool) -> list[str]:
    names: list[str] = []
    if task in (None, "modulation", "recognition") and unfreeze_modulation_backbone:
        names.append("mod_specific")
    if task in (None, "emitter", "recognition") and unfreeze_emitter_backbone:
        names.append("emitter_specific")
    if task in (None, "clustering"):
        names.extend(["long_context_shared", "cross_domain_shared"])
    return names


def _inject_lora_for_task_path(
    model: nn.Module,
    task: str | None,
    *,
    unfreeze_modulation_backbone: bool,
    unfreeze_emitter_backbone: bool,
    lora_cfg: LoRAConfig,
) -> list[str]:
    injected: list[str] = []
    space_names = _task_space_names(
        task,
        unfreeze_modulation_backbone=unfreeze_modulation_backbone,
        unfreeze_emitter_backbone=unfreeze_emitter_backbone,
    )
    for space_name in space_names:
        set_trainable(model.space_projs[space_name], True)
        backbone = model.space_backbones[space_name]
        injected.extend(f"space_backbones.{space_name}.{path}" for path in inject_lora_into_module(backbone, lora_cfg))
        set_lora_trainable(backbone, train_base=False)

    if task in (None, "modulation", "recognition") and unfreeze_modulation_backbone:
        set_trainable(model.mod_fuse, True)
    if task in (None, "emitter", "recognition") and unfreeze_emitter_backbone:
        set_trainable(model.emitter_fuse, True)
    if task in (None, "clustering"):
        set_trainable(model.shared_fuse, True)
    return injected


def configure_pretrain(model: nn.Module) -> None:
    set_trainable(model, True)
    set_trainable(model.recognition_heads, False)
    set_trainable(model.clustering_head, False)


def _configure_full_backbone(
    model: nn.Module,
    task: str | None,
    *,
    unfreeze_modulation_backbone: bool,
    unfreeze_emitter_backbone: bool,
) -> None:
    set_trainable(model, False)
    _enable_task_heads(model, task)
    set_trainable(model.tokenizer, True)
    set_trainable(model.encoder, True)
    set_trainable(model.space_projs, True)
    set_trainable(model.space_backbones, True)
    set_trainable(model.space_pool, True)
    if task == "modulation":
        set_trainable(model.mod_fuse, True)
    elif task == "emitter":
        set_trainable(model.emitter_fuse, True)
    elif task in (None, "recognition"):
        set_trainable(model.mod_fuse, True)
        set_trainable(model.emitter_fuse, True)
        set_trainable(model.shared_fuse, True)
    elif task == "clustering":
        set_trainable(model.shared_fuse, True)


def configure_peft_stage2(
    model: nn.Module,
    task: str | None = None,
    *,
    peft_mode: PeftMode = "task_path",
    freeze_backbone: bool = True,
    unfreeze_modulation_backbone: bool = True,
    unfreeze_emitter_backbone: bool = True,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
) -> list[str]:
    if peft_mode == "full_backbone" or not freeze_backbone:
        _configure_full_backbone(
            model,
            task,
            unfreeze_modulation_backbone=unfreeze_modulation_backbone,
            unfreeze_emitter_backbone=unfreeze_emitter_backbone,
        )
        return []

    set_trainable(model, False)
    _enable_task_heads(model, task)

    if peft_mode == "head_only":
        _enable_task_fuse(model, task)
        return []

    if peft_mode in ("task_path", "task_path_shared"):
        if task in (None, "modulation", "recognition") and unfreeze_modulation_backbone:
            unfreeze_modulation_path(model)
        if task in (None, "emitter", "recognition") and unfreeze_emitter_backbone:
            unfreeze_emitter_path(model)
        if task in (None, "clustering"):
            unfreeze_clustering_path(model)
        if peft_mode == "task_path_shared" and task in ("modulation", "emitter", "recognition"):
            unfreeze_shared_spaces(model)
        return []

    if peft_mode == "lora_task_path":
        _enable_task_fuse(model, task)
        lora_cfg = LoRAConfig(rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
        return _inject_lora_for_task_path(
            model,
            task,
            unfreeze_modulation_backbone=unfreeze_modulation_backbone,
            unfreeze_emitter_backbone=unfreeze_emitter_backbone,
            lora_cfg=lora_cfg,
        )

    raise ValueError(f"Unsupported peft_mode: {peft_mode}")


def configure_stage2_heads(
    model: nn.Module,
    task: str | None = None,
    *,
    freeze_backbone: bool = True,
    unfreeze_modulation_backbone: bool = True,
    unfreeze_emitter_backbone: bool = True,
) -> None:
    peft_mode: PeftMode = "full_backbone" if not freeze_backbone else "task_path"
    configure_peft_stage2(
        model,
        task,
        peft_mode=peft_mode,
        freeze_backbone=freeze_backbone,
        unfreeze_modulation_backbone=unfreeze_modulation_backbone,
        unfreeze_emitter_backbone=unfreeze_emitter_backbone,
    )
