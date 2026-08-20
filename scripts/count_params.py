#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resmamba_signal_model.config import load_yaml_config
from resmamba_signal_model.models.heads import (
    EmitterHead,
    ModulationHead,
    PrototypeClusteringHead,
)
from resmamba_signal_model.models.model import SignalFoundationModel, SignalModelConfig
from resmamba_signal_model.training.param_stats import count_params, format_param_stats


def _head_param_total(model: SignalFoundationModel) -> int:
    total = 0
    for head in model.iter_task_heads():
        total += sum(p.numel() for p in head.parameters())
    uti = getattr(model, "task_interface", None)
    if uti is not None:
        total += sum(p.numel() for p in uti.parameters())
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Count SignalFoundationModel parameters")
    parser.add_argument("--model-config", default=str(ROOT / "configs" / "model_tiny.yaml"))
    parser.add_argument("--with-heads", action="store_true", help="构建 UTI + 五头并报告头参")
    args = parser.parse_args()
    raw = load_yaml_config(args.model_config)
    section = dict(raw.get("model", raw))
    if args.with_heads:
        section["build_task_heads"] = True
        section["build_task_interface"] = True
    cfg = SignalModelConfig.from_dict(section)
    model = SignalFoundationModel(cfg)
    print(format_param_stats(model))
    if args.with_heads:
        head_total = _head_param_total(model)
        total, trainable = count_params(model)
        print(f"head_and_uti_params={head_total:,}")
        mod = getattr(model, "modulation_head", None)
        em = getattr(model, "emitter_head", None)
        clu = getattr(model, "clustering_head", None)
        if mod is not None:
            print(f"  modulation_head: {sum(p.numel() for p in mod.parameters()):,}")
        if em is not None:
            print(f"  emitter_head: {sum(p.numel() for p in em.parameters()):,}")
        if clu is not None:
            print(f"  clustering_head: {sum(p.numel() for p in clu.parameters()):,}")
        print(f"budget_check total={total:,} heads+uti={head_total:,}")


if __name__ == "__main__":
    main()
