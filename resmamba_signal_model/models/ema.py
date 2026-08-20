from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class EMATeacher(nn.Module):
    """预训练专用 EMA teacher：部署时不构建，不增加推理参数。"""

    def __init__(self, student: nn.Module, *, momentum: float = 0.996) -> None:
        super().__init__()
        self.momentum = float(momentum)
        cfg = getattr(student, "cfg", None)
        if cfg is None:
            raise TypeError("EMATeacher 需要带 cfg 的 SignalFoundationModel")
        from resmamba_signal_model.models.model import SignalFoundationModel, SignalModelConfig

        payload = {name: getattr(cfg, name) for name in SignalModelConfig.__dataclass_fields__}
        payload["build_ema_teacher"] = False
        if not isinstance(payload.get("tokenizer"), dict):
            tok = payload.get("tokenizer")
            payload["tokenizer"] = dict(tok.__dict__) if tok is not None else {}
        self.model = SignalFoundationModel(SignalModelConfig.from_dict(payload))
        missing = self.model.load_state_dict(student.state_dict(), strict=False)
        del missing
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, student: nn.Module) -> None:
        decay = float(self.momentum)
        student_params = dict(student.named_parameters())
        for name, ema_param in self.model.named_parameters():
            src = student_params.get(name)
            if src is None or ema_param.shape != src.shape:
                continue
            ema_param.data.lerp_(src.data, 1.0 - decay)
        student_buffers = dict(student.named_buffers())
        for name, ema_buf in self.model.named_buffers():
            src = student_buffers.get(name)
            if src is None or ema_buf.shape != src.shape:
                continue
            ema_buf.data.copy_(src.data)

    @torch.no_grad()
    def forward_unmasked(self, batch: dict[str, Any]) -> dict[str, Any]:
        self.model.eval()
        out = self.model(batch, mode="pretrain", mask_mode="none", training=False)
        detached: dict[str, Any] = {}
        for key, value in out.items():
            detached[key] = value.detach() if torch.is_tensor(value) else value
        return detached
