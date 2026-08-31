#!/usr/bin/env python3
"""汇总 stage2 composite best.ckpt 与各任务 best 片段指标（分类门控优先）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLASSIFICATION_TASKS = ("ld_intrapulse", "ld_model", "tx_modulation")
CLUSTERING_TASKS = ("ld_clustering", "tx_clustering")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else {}


def _load_ckpt_meta(path: Path) -> dict[str, Any]:
    import torch

    if not path.is_file():
        return {}
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        return {}
    out: dict[str, Any] = {}
    if "composite_task_best" in ckpt:
        out["composite_task_best"] = ckpt["composite_task_best"]
    if "composite_geomean" in ckpt:
        out["composite_geomean"] = ckpt["composite_geomean"]
    return out


def summarize_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    state = _load_json(run_dir / "train_state.json")
    ckpt_path = run_dir / "ckpts" / "best.ckpt"
    ckpt_meta = _load_ckpt_meta(ckpt_path)

    task_best = ckpt_meta.get("composite_task_best") or state.get("task_best") or {}
    composite_geomean = ckpt_meta.get("composite_geomean") or state.get("best_model_score")

    classification: dict[str, Any] = {}
    clustering: dict[str, Any] = {}
    prediction: dict[str, Any] = {}

    for task, info in task_best.items():
        if not isinstance(info, dict):
            continue
        row = {
            "monitor": info.get("monitor"),
            "score": info.get("score"),
            "epoch": info.get("epoch"),
            "mode": info.get("mode"),
        }
        if task in CLASSIFICATION_TASKS:
            classification[task] = row
        elif task in CLUSTERING_TASKS:
            clustering[task] = row
        elif task == "prediction":
            prediction[task] = row

    return {
        "run_dir": str(run_dir),
        "best_ckpt": str(ckpt_path) if ckpt_path.is_file() else None,
        "monitor": state.get("monitor"),
        "best_model_score": state.get("best_model_score"),
        "composite_geomean": composite_geomean,
        "classification": classification,
        "clustering": clustering,
        "prediction": prediction,
        "task_best": task_best,
    }


def format_report(summary: dict[str, Any]) -> str:
    lines = [
        f"run: {summary.get('run_dir')}",
        f"best_ckpt: {summary.get('best_ckpt')}",
        f"composite_geomean: {summary.get('composite_geomean')}",
        "",
        "classification (stage2 门控主指标):",
    ]
    for task, row in sorted((summary.get("classification") or {}).items()):
        lines.append(
            f"  {task}: {row.get('monitor')}={row.get('score')} epoch={row.get('epoch')}"
        )
    lines.append("")
    lines.append("clustering:")
    for task, row in sorted((summary.get("clustering") or {}).items()):
        lines.append(
            f"  {task}: {row.get('monitor')}={row.get('score')} epoch={row.get('epoch')}"
        )
    lines.append("")
    pred = summary.get("prediction") or {}
    if pred:
        lines.append("prediction (零样本 baseline，正式看 stage3):")
        for task, row in sorted(pred.items()):
            lines.append(
                f"  {task}: {row.get('monitor')}={row.get('score')} epoch={row.get('epoch')}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 stage2 composite 与各任务 best 指标")
    parser.add_argument(
        "run_dir",
        type=Path,
        help="runs/experiments/stage2_* 目录",
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args()

    summary = summarize_run(args.run_dir)
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(format_report(summary))


if __name__ == "__main__":
    main()
