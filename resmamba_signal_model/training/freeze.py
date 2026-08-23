from __future__ import annotations

from typing import Any, Iterable

import torch.nn as nn

from resmamba_signal_model.models.peft import MultiTaskLoRALinear
from resmamba_signal_model.models.task_interface import DEFAULT_TASKS
from resmamba_signal_model.training.task_catalog import BUILTIN_HEAD_ATTR, resolve_task_catalog

HEAD_MODULE_NAMES: dict[str, str] = dict(BUILTIN_HEAD_ATTR)

FROZEN_DOWNSTREAM_PREFIXES: tuple[str, ...] = (
    "tokenizer.",
    "encoder.",
    "decoder.",
    "encoder_pool.",
    "encoder_repr_norm.",
    "revin.",
    "input_adapters.",
    "domain_disc.",
    "chunk_pool.",
)


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


def _freeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def _unfreeze_head_for_task(model: nn.Module, task: str) -> None:
    head = getattr(model, HEAD_MODULE_NAMES.get(task, ""), None)
    extra = getattr(model, "extra_task_heads", None)
    if head is None and isinstance(extra, nn.ModuleDict) and task in extra:
        head = extra[task]
    if head is None and hasattr(model, "get_task_head"):
        head = model.get_task_head(task)
    _set_module_grad(head, True)


def apply_stage_freeze(
    model: nn.Module,
    stage: str,
    *,
    task: str | None = None,
    train_cfg: dict[str, Any] | None = None,
) -> None:
    """按阶段冻结/解冻。下游路径：z_enc → TaskAdapter → 头，不训 UTI / encoder_pool。"""
    train_cfg = train_cfg or {}
    stage = str(stage)
    model.truncate_backward = False  # type: ignore[attr-defined]
    model.skip_recon = False  # type: ignore[attr-defined]

    if stage == "pretrain":
        for param in model.parameters():
            param.requires_grad = True
        return

    _freeze_all(model)

    if stage == "stage2":
        model.truncate_backward = bool(train_cfg.get("truncate_backward", True))  # type: ignore[attr-defined]
        model.skip_recon = bool(train_cfg.get("skip_recon", True))  # type: ignore[attr-defined]
        active_tasks = list(train_cfg.get("active_train_tasks") or [])
        if task:
            active_tasks = [task]
        elif not active_tasks and train_cfg.get("task"):
            active_tasks = [str(train_cfg["task"])]
        if not active_tasks:
            active_tasks = list(resolve_task_catalog(train_cfg).names)
        for t in active_tasks:
            _unfreeze_head_for_task(model, t)
        if bool(train_cfg.get("train_z_linear_probes", True)):
            probes = getattr(model, "z_linear_probes", None)
            if isinstance(probes, nn.ModuleDict):
                for t in active_tasks:
                    if t in probes:
                        _set_module_grad(probes[t], True)
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
        _unfreeze_head_for_task(model, task)
        if bool(train_cfg.get("train_z_linear_probes", False)):
            probes = getattr(model, "z_linear_probes", None)
            if isinstance(probes, nn.ModuleDict) and task in probes:
                _set_module_grad(probes[task], True)
        if bool(train_cfg.get("ssm_cotrain_dt_bias") or (train_cfg.get("peft") or {}).get("ssm_cotrain_dt_bias")):
            _unfreeze_dt_bias(model)
        return

    if stage == "joint":
        model.skip_recon = bool(train_cfg.get("skip_recon", True))  # type: ignore[attr-defined]
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

    raise ValueError(f"未知 stage {stage!r}，可选: pretrain/stage2/stage3/joint")


def _unfreeze_dt_bias(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if name.endswith("dt_bias") or ".dt_bias" in name:
            param.requires_grad = True


def iter_head_param_prefixes() -> tuple[str, ...]:
    return tuple(HEAD_MODULE_NAMES.values()) + (
        "extra_task_heads",
        "prototype_registry",
    )


def specialist_state_prefixes(task: str) -> tuple[str, ...]:
    """阶段三专家 ckpt 要带回的模块前缀。"""
    head = HEAD_MODULE_NAMES.get(task, f"{task}_head")
    return (f"{head}.", f"extra_task_heads.{task}.", f"task_adapters.{task}.")


def filter_specialist_state(state: dict[str, Any], task: str) -> dict[str, Any]:
    prefixes = specialist_state_prefixes(task)
    filtered: dict[str, Any] = {}
    for key, value in state.items():
        if f"lora_A.{task}" in key or f"lora_B.{task}" in key:
            filtered[key] = value
        elif any(key.startswith(prefix) for prefix in prefixes):
            filtered[key] = value
    return filtered


def default_stage3_tasks(train_cfg: dict[str, Any] | None = None) -> tuple[str, ...]:
    if train_cfg:
        return tuple(resolve_task_catalog(train_cfg).names)
    return DEFAULT_TASKS
