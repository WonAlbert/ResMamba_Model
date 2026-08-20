from __future__ import annotations

from typing import Any, Iterable

import torch.nn as nn

from resmamba_signal_model.models.peft import TOKENIZER_LAST_ATTRS, MultiTaskLoRALinear
from resmamba_signal_model.models.task_interface import DEFAULT_TASKS
from resmamba_signal_model.training.task_catalog import BUILTIN_HEAD_ATTR, resolve_task_catalog

HEAD_MODULE_NAMES: dict[str, str] = dict(BUILTIN_HEAD_ATTR)


def _set_module_grad(module: nn.Module | None, enabled: bool) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = bool(enabled)


def _iter_task_heads(model: nn.Module) -> Iterable[tuple[str, nn.Module]]:
    for task, attr in HEAD_MODULE_NAMES.items():
        head = getattr(model, attr, None)
        if isinstance(head, nn.Module):
            yield task, head
    extra = getattr(model, "extra_task_heads", None)
    if isinstance(extra, nn.ModuleDict):
        for name, head in extra.items():
            yield str(name), head


def apply_stage_freeze(
    model: nn.Module,
    stage: str,
    *,
    task: str | None = None,
    train_cfg: dict[str, Any] | None = None,
) -> None:
    """按阶段冻结/解冻。stage2 同时打开截断反传开关。"""
    train_cfg = train_cfg or {}
    stage = str(stage)
    model.truncate_backward = False  # type: ignore[attr-defined]
    model.skip_recon = False  # type: ignore[attr-defined]

    if stage == "continual":
        from resmamba_signal_model.training.continual import apply_continual_freeze

        apply_continual_freeze(model)
        return

    if stage == "pretrain":
        for param in model.parameters():
            param.requires_grad = True
        return

    if stage == "downstream":
        train_encoder = bool(train_cfg.get("train_encoder", getattr(model.cfg, "train_encoder", True)))
        train_decoder = bool(train_cfg.get("train_decoder", getattr(model.cfg, "train_decoder", True)))
        train_heads = bool(train_cfg.get("train_heads", getattr(model.cfg, "train_heads", True)))
        if hasattr(model, "apply_train_flags"):
            model.apply_train_flags(train_encoder, train_decoder, train_heads)
        return

    for param in model.parameters():
        param.requires_grad = False

    if stage == "stage2":
        model.truncate_backward = bool(train_cfg.get("truncate_backward", True))  # type: ignore[attr-defined]
        model.skip_recon = bool(train_cfg.get("skip_recon", True))  # type: ignore[attr-defined]
        _set_module_grad(getattr(model, "task_interface", None), True)
        _set_module_grad(getattr(model, "prototype_registry", None), True)
        _set_module_grad(getattr(model, "z_linear_probes", None), True)
        for _name, head in _iter_task_heads(model):
            _set_module_grad(head, True)
        return

    if stage == "stage3":
        if not task:
            raise ValueError("stage3 需要指定 task")
        model.skip_recon = bool(train_cfg.get("skip_recon", True))  # type: ignore[attr-defined]
        handle = getattr(model, "peft", None)
        if handle is not None:
            handle.set_trainable(model, task)
            handle.set_active_task(model, task)
        else:
            for module in model.modules():
                if isinstance(module, MultiTaskLoRALinear):
                    module.set_trainable(task)
                    module.set_active_task(task)
        adapters = getattr(model, "task_adapters", None)
        if isinstance(adapters, nn.ModuleDict) and task in adapters:
            _set_module_grad(adapters[task], True)
        head = getattr(model, HEAD_MODULE_NAMES.get(task, ""), None)
        extra = getattr(model, "extra_task_heads", None)
        if head is None and isinstance(extra, nn.ModuleDict) and task in extra:
            head = extra[task]
        if head is None and hasattr(model, "get_task_head"):
            head = model.get_task_head(task)
        _set_module_grad(head, True)
        if bool(train_cfg.get("ssm_cotrain_dt_bias") or (train_cfg.get("peft") or {}).get("ssm_cotrain_dt_bias")):
            _unfreeze_dt_bias(model)
        return

    if stage == "joint":
        model.skip_recon = bool(train_cfg.get("skip_recon", True))  # type: ignore[attr-defined]
        _unfreeze_tokenizer_last(model)
        handle = getattr(model, "peft", None)
        if handle is not None:
            handle.set_trainable(model, None)
        else:
            for module in model.modules():
                if isinstance(module, MultiTaskLoRALinear):
                    module.set_trainable(None)
        adapters = getattr(model, "task_adapters", None)
        if isinstance(adapters, nn.ModuleDict):
            for adapter in adapters.values():
                _set_module_grad(adapter, True)
        _set_module_grad(getattr(model, "shared_adapter", None), True)
        _set_module_grad(getattr(model, "prototype_registry", None), True)
        for _name, head in _iter_task_heads(model):
            _set_module_grad(head, True)
        if bool(train_cfg.get("ssm_cotrain_dt_bias") or (train_cfg.get("peft") or {}).get("ssm_cotrain_dt_bias")):
            _unfreeze_dt_bias(model)
        return

    raise ValueError(f"未知 stage {stage!r}，可选: pretrain/downstream/stage2/stage3/joint/continual")


def _unfreeze_tokenizer_last(model: nn.Module) -> None:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return
    for name in TOKENIZER_LAST_ATTRS:
        child = getattr(tokenizer, name, None)
        if isinstance(child, nn.Module):
            _set_module_grad(child, True)
        elif isinstance(child, nn.Parameter):
            child.requires_grad = True


def _unfreeze_dt_bias(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if name.endswith("dt_bias") or ".dt_bias" in name:
            param.requires_grad = True


def iter_head_param_prefixes() -> tuple[str, ...]:
    return tuple(HEAD_MODULE_NAMES.values()) + ("extra_task_heads", "recognition_heads", "prototype_registry")


def default_stage3_tasks(train_cfg: dict[str, Any] | None = None) -> tuple[str, ...]:
    if train_cfg:
        return tuple(resolve_task_catalog(train_cfg).names)
    return DEFAULT_TASKS
