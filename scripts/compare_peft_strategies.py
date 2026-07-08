#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_pipeline import build_model
from resmamba_signal_model.training.param_stats import format_param_stats
from resmamba_signal_model.training.stages import PeftMode, configure_peft_stage2


def _configure_for_task(model, task: str, peft_mode: PeftMode) -> None:
    if task == "modulation":
        unfreeze_mod, unfreeze_emit = True, False
    elif task == "emitter":
        unfreeze_mod, unfreeze_emit = False, True
    else:
        unfreeze_mod, unfreeze_emit = False, False
    configure_peft_stage2(
        model,
        task,
        peft_mode=peft_mode,
        freeze_backbone=peft_mode != "full_backbone",
        unfreeze_modulation_backbone=unfreeze_mod,
        unfreeze_emitter_backbone=unfreeze_emit,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PEFT trainable parameter budgets for ResMamba stage2.")
    parser.add_argument("--model-config", default=str(ROOT / "configs" / "model_resmamba_400m.yaml"))
    parser.add_argument("--task", choices=["modulation", "emitter", "clustering", "prediction"], default="modulation")
    args = parser.parse_args()

    modes: list[PeftMode] = ["head_only", "task_path", "task_path_shared", "lora_task_path", "full_backbone"]
    print(f"model_config={args.model_config} task={args.task}")
    for mode in modes:
        model = build_model(args.model_config)
        _configure_for_task(model, args.task, mode)
        print(f"\n=== peft_mode={mode} ===")
        print(format_param_stats(model))


if __name__ == "__main__":
    main()
