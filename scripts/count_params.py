#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_pipeline import build_model, merge_task_config
from resmamba_signal_model.training.param_stats import format_param_stats
from resmamba_signal_model.training.stages import configure_peft_stage2


def load_train_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Count ResMamba parameters with optional stage2 PEFT config.")
    parser.add_argument("--model-config", default=str(ROOT / "configs" / "model_tiny.yaml"))
    parser.add_argument("--train-config", default=None, help="Optional stage2 YAML to apply peft_mode / freeze flags.")
    parser.add_argument("--task", default="modulation")
    args = parser.parse_args()

    model = build_model(args.model_config)
    if args.train_config:
        train_cfg = merge_task_config(load_train_yaml(Path(args.train_config)), stage="stage2", task=args.task)
        freeze_backbone = bool(train_cfg.get("freeze_backbone", True))
        peft_mode = "full_backbone" if not freeze_backbone else str(train_cfg.get("peft_mode", "task_path"))
        default_unfreeze_mod = args.task == "modulation"
        default_unfreeze_emit = args.task == "emitter"
        configure_peft_stage2(
            model,
            args.task,
            peft_mode=peft_mode,  # type: ignore[arg-type]
            freeze_backbone=freeze_backbone,
            unfreeze_modulation_backbone=bool(train_cfg.get("unfreeze_modulation_backbone", default_unfreeze_mod)),
            unfreeze_emitter_backbone=bool(train_cfg.get("unfreeze_emitter_backbone", default_unfreeze_emit)),
            lora_rank=int(train_cfg.get("lora_rank", 8)),
            lora_alpha=float(train_cfg.get("lora_alpha", 16.0)),
            lora_dropout=float(train_cfg.get("lora_dropout", 0.05)),
        )
        print(f"train_config={args.train_config} peft_mode={peft_mode}")
    print(format_param_stats(model))


if __name__ == "__main__":
    main()
