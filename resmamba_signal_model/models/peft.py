from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from resmamba_signal_model.models.task_interface import DEFAULT_TASKS

TOKENIZER_LAST_ATTRS: tuple[str, ...] = ("time_fuse", "freq_proj", "gate", "physics_proj", "norm")
TOKENIZER_LAST_PREFIXES: tuple[str, ...] = tuple(f"tokenizer.{name}" for name in TOKENIZER_LAST_ATTRS)

PEFT_STATE_MARKERS: tuple[str, ...] = (
    "lora_A.",
    "lora_B.",
    "lora_scale.",
    "task_interface.",
    "task_adapters.",
    "shared_adapter.",
    "modulation_head.",
    "emitter_head.",
    "clustering_head.",
    "prediction_head.",
    "imputation_head.",
    "recognition_heads.",
    "extra_task_heads.",
    "emitter_fingerprint.",
)


@dataclass
class PeftConfig:
    r_attn: int = 16
    r_mamba: int = 8
    lora_alpha_attn: float = 16.0
    lora_alpha_mamba: float = 8.0
    loraplus_lr_ratio: float = 16.0
    ssm_cotrain_dt_bias: bool = False
    shared_lora: bool = False
    dropout: float = 0.0
    tasks: tuple[str, ...] = DEFAULT_TASKS

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PeftConfig":
        if not data:
            return cls()
        names = {item.name for item in fields(cls)}
        payload = {k: v for k, v in data.items() if k in names}
        if "tasks" in payload and not isinstance(payload["tasks"], tuple):
            payload["tasks"] = tuple(payload["tasks"])
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tasks"] = list(self.tasks)
        return payload


def peft_config_from_train_cfg(train_cfg: dict[str, Any] | None) -> PeftConfig:
    train_cfg = train_cfg or {}
    blob = dict(train_cfg.get("peft") or {})
    for key in (
        "r_attn",
        "r_mamba",
        "lora_alpha_attn",
        "lora_alpha_mamba",
        "loraplus_lr_ratio",
        "ssm_cotrain_dt_bias",
        "shared_lora",
    ):
        if key in train_cfg and key not in blob:
            blob[key] = train_cfg[key]
    return PeftConfig.from_dict(blob)


def classify_lora_target(name: str) -> str | None:
    """显式白名单：Mamba 投影 / fuse / Transformer qkv-ffn / Decoder SwiGLU。"""
    in_encoder_mamba = "encoder.mamba_layers" in name
    in_encoder_tr = "encoder.transformer_layers" in name
    in_decoder_block = "decoder.blocks" in name
    if not (in_encoder_mamba or in_encoder_tr or in_decoder_block):
        return None
    if any(token in name for token in ("A_log", "conv1d", "dt_proj", "dt_bias")):
        return None
    if in_encoder_tr:
        if name.endswith(".qkv") or name.endswith(".out_proj") or name.endswith(".ffn.0") or name.endswith(".ffn.3"):
            return "attn"
        return None
    if name.endswith(".in_proj") or name.endswith(".out_proj") or name.endswith(".fuse"):
        return "mamba"
    if in_decoder_block and (name.endswith(".w1") or name.endswith(".w2")):
        return "mamba"
    return None


class MultiTaskLoRALinear(nn.Module):
    """冻结 W0，按任务分库 ΔW；联合阶段可再加 shared 路。"""

    def __init__(
        self,
        linear: nn.Linear,
        tasks: Iterable[str],
        *,
        r: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank 必须为正")
        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(r)
        self.active_task: str | None = None
        self.weight = nn.Parameter(linear.weight.detach().clone(), requires_grad=False)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)
        task_names = [str(t) for t in tasks]
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        for task in task_names:
            a = nn.Parameter(torch.empty(self.in_features, self.r, dtype=self.weight.dtype, device=self.weight.device))
            b = nn.Parameter(torch.zeros(self.r, self.out_features, dtype=self.weight.dtype, device=self.weight.device))
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            self.lora_A[task] = a
            self.lora_B[task] = b
        self.lora_dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def set_active_task(self, task: str | None) -> None:
        self.active_task = task

    def set_trainable(self, task: str | None) -> None:
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        for name in self.lora_A:
            train = True if task is None else name == task
            self.lora_A[name].requires_grad = train
            self.lora_B[name].requires_grad = train

    def _delta(self, x: torch.Tensor, task: str) -> torch.Tensor:
        a = self.lora_A[task].to(device=x.device, dtype=x.dtype)
        b = self.lora_B[task].to(device=x.device, dtype=x.dtype)
        h = self.lora_dropout(x) @ a
        return (h @ b) * self.scaling

    def ensure_task(self, task: str) -> None:
        name = str(task)
        if name in self.lora_A:
            return
        a = nn.Parameter(torch.empty(self.in_features, self.r, dtype=self.weight.dtype, device=self.weight.device))
        b = nn.Parameter(torch.zeros(self.r, self.out_features, dtype=self.weight.dtype, device=self.weight.device))
        nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        self.lora_A[name] = a
        self.lora_B[name] = b

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, r={self.r}, tasks={list(self.lora_A.keys())}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight.to(dtype=x.dtype), None if self.bias is None else self.bias.to(dtype=x.dtype))
        task = self.active_task
        if task and task in self.lora_A:
            y = y + self._delta(x, task)
        if task and "shared" in self.lora_A and task != "shared":
            y = y + self._delta(x, "shared")
        return y


def _assign_submodule(root: nn.Module, name: str, module: nn.Module) -> None:
    parent: nn.Module = root
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


@dataclass
class PeftHandle:
    names: list[str]
    tasks: list[str]
    cfg: PeftConfig

    def set_active_task(self, model: nn.Module, task: str | None) -> None:
        for module in model.modules():
            if isinstance(module, MultiTaskLoRALinear):
                module.set_active_task(task)

    def set_trainable(self, model: nn.Module, task: str | None) -> None:
        for module in model.modules():
            if isinstance(module, MultiTaskLoRALinear):
                module.set_trainable(task)

    def to_meta(self) -> dict[str, Any]:
        return {"names": list(self.names), "tasks": list(self.tasks), "cfg": self.cfg.to_dict()}


def inject_hybrid_lora(
    model: nn.Module,
    tasks: Iterable[str],
    cfg: PeftConfig | dict[str, Any] | None = None,
) -> PeftHandle:
    cfg = cfg if isinstance(cfg, PeftConfig) else PeftConfig.from_dict(cfg)
    task_list = [str(t) for t in tasks]
    if cfg.shared_lora and "shared" not in task_list:
        task_list.append("shared")
    wrapped: list[str] = []
    for name, module in list(model.named_modules()):
        if isinstance(module, MultiTaskLoRALinear):
            continue
        if not isinstance(module, nn.Linear):
            continue
        kind = classify_lora_target(name)
        if kind is None:
            continue
        if kind == "attn":
            r, alpha = int(cfg.r_attn), float(cfg.lora_alpha_attn)
        else:
            r, alpha = int(cfg.r_mamba), float(cfg.lora_alpha_mamba)
        _assign_submodule(
            model,
            name,
            MultiTaskLoRALinear(module, task_list, r=r, alpha=alpha, dropout=cfg.dropout),
        )
        wrapped.append(name)
    handle = PeftHandle(names=wrapped, tasks=task_list, cfg=cfg)
    model.peft = handle  # type: ignore[attr-defined]
    return handle


def peft_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """只持久化 LoRA / 适配器 / 头 / UTI / 指纹支路 / tokenizer 尾部。"""
    out: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if any(marker in name or name.startswith(marker.rstrip(".")) for marker in PEFT_STATE_MARKERS):
            out[name] = tensor
            continue
        if any(name == prefix or name.startswith(prefix + ".") for prefix in TOKENIZER_LAST_PREFIXES):
            out[name] = tensor
    return out


def save_best_bundle(
    path: str,
    *,
    model: nn.Module,
    base_state: dict[str, Any] | None = None,
    peft_cfg: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """把 base + 全部 PEFT 合成一份可推理 ckpt。"""
    merged: dict[str, Any] = {}
    if base_state:
        merged.update(base_state)
    merged.update(model.state_dict())
    payload: dict[str, Any] = {
        "state_dict": {f"model.{k}": v.detach().cpu() if torch.is_tensor(v) else v for k, v in merged.items()},
        "peft_cfg": peft_cfg,
        "peft": True,
    }
    handle = getattr(model, "peft", None)
    if handle is not None:
        payload["peft_meta"] = handle.to_meta()
        payload["peft_tasks"] = list(handle.tasks)
        if payload["peft_cfg"] is None:
            payload["peft_cfg"] = handle.cfg.to_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def state_has_lora(state: dict[str, Any]) -> bool:
    return any("lora_A." in key or "lora_B." in key for key in state)


def infer_lora_tasks(state: dict[str, Any]) -> list[str]:
    tasks: list[str] = []
    for key in state:
        if ".lora_A." not in key:
            continue
        task = key.rsplit(".lora_A.", 1)[-1]
        if task and task not in tasks:
            tasks.append(task)
    return tasks
