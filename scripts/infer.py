#!/usr/bin/env python3
"""下游任务推理：同一套 SignalFoundationModel.forward，不切换结构。"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resmamba_signal_model.config.yaml_config import load_yaml_config  # noqa: E402
from resmamba_signal_model.models.heads import remap_task_head_checkpoints  # noqa: E402
from resmamba_signal_model.models.model import SignalFoundationModel, SignalModelConfig  # noqa: E402
from resmamba_signal_model.models.peft import (  # noqa: E402
    PeftConfig,
    infer_lora_tasks,
    inject_hybrid_lora,
    peft_config_from_train_cfg,
    state_has_lora,
)
from resmamba_signal_model.models.task_interface import DEFAULT_TASKS  # noqa: E402
from resmamba_signal_model.training.checkpointing import extract_model_state_dict  # noqa: E402
from resmamba_signal_model.training.data_module import (  # noqa: E402
    build_infer_pool,
    make_fixed_eval_loader,
    resolve_dataset_h5,
)
from resmamba_signal_model.training.metrics import (  # noqa: E402
    accuracy,
    macro_f1,
    masked_patch_mse,
    nmi_score,
    openset_detection_metrics,
    reconstruction_eval_pair,
    ssim_iq,
)
from resmamba_signal_model.training.losses import _first_present  # noqa: E402
from resmamba_signal_model.training.task_catalog import builtin_spec  # noqa: E402

TASKS = ("modulation", "emitter", "prediction", "clustering", "imputation", "encode")
TASK_DATASETS: dict[str, list[str]] = {
    "modulation": ["rml2016_04c", "rml2016_10a", "rml2016_10b", "rml2018_1a"],
    "emitter": ["wisig", "adsb2"],
    "prediction": ["adsb2", "rml2016_04c", "rml2016_10a", "rml2016_10b", "rml2018_1a", "wifi150"],
    "imputation": ["adsb2", "rml2016_04c", "rml2016_10a", "rml2016_10b", "rml2018_1a", "wifi150"],
    "clustering": ["adsb2", "rml2016_04c", "rml2016_10a", "rml2016_10b", "rml2018_1a", "wifi150"],
    "encode": [],
}


def task_label_tensor(task: str, batch: dict[str, Any]) -> torch.Tensor | None:
    """与训练/验证相同的标签列：调制用 canonical，个体用 global_emitter_id。"""
    try:
        field = builtin_spec(task).label_field
    except KeyError:
        field = None
    if task == "emitter":
        value = batch.get(field or "global_emitter_id", batch.get("emitter_id"))
    elif task == "clustering":
        value = batch.get(field or "global_label_id")
    elif task in ("prediction", "imputation", "encode"):
        return None
    else:
        value = batch.get(
            field or "canonical_mod_label_id",
            batch.get("mod_label_id", batch.get("source_label_id")),
        )
    if value is None:
        return None
    return value.cpu() if torch.is_tensor(value) else torch.as_tensor(value)


def load_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"无法解析 checkpoint: {path}")
    return ckpt, remap_task_head_checkpoints(extract_model_state_dict(ckpt))


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    _ckpt, state = load_checkpoint(path)
    return state


def build_model(
    model_config: Path,
    *,
    build_task_heads: bool,
    overrides: dict[str, Any] | None = None,
    build_adapters: bool = False,
    build_shared_adapter: bool = False,
) -> SignalFoundationModel:
    raw = load_yaml_config(model_config)
    payload = dict(raw.get("model", raw))
    payload["build_task_heads"] = build_task_heads
    payload["build_adapters"] = build_adapters
    payload["build_shared_adapter"] = build_shared_adapter
    if overrides:
        payload.update(overrides)
    return SignalFoundationModel(SignalModelConfig.from_dict(payload))


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=device.type == "cuda")
        elif isinstance(value, list) and value and torch.is_tensor(value[0]):
            out[key] = [item.to(device, non_blocking=device.type == "cuda") for item in value]
        else:
            out[key] = value
    return out


def find_h5(rfdata_root: Path, dataset: str, split: str = "test") -> Path:
    return resolve_dataset_h5(rfdata_root, dataset, split)


def build_loader(
    rfdata_root: Path,
    datasets: list[str],
    *,
    split: str,
    token_budget: int,
    patch_size: int,
    val_batches: int,
    val_seed: int,
    num_workers: int,
    task: str | None = None,
) -> DataLoader:
    pool = build_infer_pool(rfdata_root, datasets, split=split)
    return make_fixed_eval_loader(
        pool,
        token_budget=token_budget,
        patch_size=patch_size,
        num_batches=val_batches,
        seed=val_seed,
        num_workers=num_workers,
        pin_memory=False,
        source_name=task,
        task=task,
    )


@torch.no_grad()
def run_task(model: SignalFoundationModel, loader: DataLoader, device: torch.device, task: str) -> dict[str, Any]:
    model.eval()
    preds: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    zs: list[torch.Tensor] = []
    ssim_vals: list[float] = []
    mse_vals: list[float] = []
    osr_scores: list[torch.Tensor] = []
    osr_unknown: list[torch.Tensor] = []
    osr_conf: list[torch.Tensor] = []
    mode = "encode" if task == "encode" else "task"
    for batch in tqdm(loader, desc=f"infer:{task}"):
        batch = move_batch(batch, device)
        out = model(batch, mode=mode, task=None if task == "encode" else task)
        zs.append(out["z"].detach().cpu())
        logits = _first_present(out, "task_logits", f"{task}_logits", "modulation_logits")
        if task == "encode":
            pass
        elif task == "emitter" and ("emitter_logits" in out or logits is not None):
            pred = out.get("emitter_logits", logits)
            preds.append(pred.argmax(dim=-1).cpu())
            label = task_label_tensor(task, batch)
            if label is not None:
                labels.append(label)
        elif task == "clustering" and "cluster_logits" in out:
            preds.append(out["cluster_logits"].argmax(dim=-1).cpu())
            label = task_label_tensor(task, batch)
            if label is not None:
                labels.append(label)
        elif logits is not None and task not in ("prediction", "imputation"):
            preds.append(logits.argmax(dim=-1).cpu())
            label = task_label_tensor(task, batch)
            if label is not None:
                labels.append(label)
        elif task in ("prediction", "imputation") or "pred_patches" in out:
            pred, target, mask = reconstruction_eval_pair(out)
            ssim_vals.append(float(ssim_iq(pred, target, mask)))
            mse_vals.append(float(masked_patch_mse(pred, target, mask).item()))
        score = out.get("openset_score", out.get("openset_energy"))
        if score is not None:
            osr_scores.append(score.detach().reshape(-1).cpu())
            label = task_label_tensor(task, batch)
            if label is not None:
                osr_unknown.append((label.reshape(-1) < 0).cpu())
            logits = _first_present(out, "task_logits", "cluster_logits")
            if logits is not None:
                osr_conf.append(logits.detach().softmax(dim=-1).max(dim=-1).values.reshape(-1).cpu())
    metrics: dict[str, Any] = {"num_batches": int(len(zs)), "z_dim": int(zs[0].shape[-1]) if zs else 0}
    if preds and labels:
        p = torch.cat(preds)
        y = torch.cat(labels)
        metrics["acc"] = float(accuracy(p, y))
        metrics["f1"] = float(macro_f1(p, y))
        if task == "clustering":
            metrics["nmi"] = float(nmi_score(p, y))
        metrics["num_samples"] = int(y.numel())
    if ssim_vals:
        metrics["ssim"] = float(sum(ssim_vals) / len(ssim_vals))
        metrics["impute_mse"] = float(sum(mse_vals) / len(mse_vals))
    if osr_scores and osr_unknown:
        scores = torch.cat(osr_scores)
        y_unknown = torch.cat(osr_unknown)
        conf = torch.cat(osr_conf) if osr_conf else None
        if int(y_unknown.any()) and int((~y_unknown.bool()).any()):
            osr = openset_detection_metrics(y_unknown, scores, confidences=conf, correct=None)
            metrics.update({f"osr_{key}": value for key, value in osr.items()})
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SignalFoundationModel 下游推理")
    p.add_argument("--task", required=True, help="任务名或 all；可为目录中任意下游任务")
    p.add_argument("--checkpoint", required=True, help="Lightning best.ckpt 或 state_dict")
    p.add_argument("--model-config", default="configs/model.yaml")
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--rfdata-root", default="dataset")
    p.add_argument("--config", default=None, help="训练 YAML，用于对齐 token_budget/val_seed/val_batches")
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--token-budget", type=int, default=None)
    p.add_argument("--val-seed", type=int, default=None)
    p.add_argument("--val-batches", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--batch-size", type=int, default=None, help="已弃用：infer 复用 val 的 token-budget 协议")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = ROOT
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rfdata_root = Path(args.rfdata_root)
    if not rfdata_root.is_absolute():
        rfdata_root = root / rfdata_root
    model_config = Path(args.model_config)
    if not model_config.is_absolute():
        model_config = root / model_config
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else root / "outputs" / f"infer_{args.task}_{stamp}"
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint) if Path(args.checkpoint).is_absolute() else root / args.checkpoint
    blob, state = load_checkpoint(ckpt_path)
    lora_tasks = blob.get("peft_tasks") or infer_lora_tasks(state) or list(DEFAULT_TASKS)
    catalog_names = [t for t in lora_tasks if t != "shared"]
    for key in state:
        if key.startswith("extra_task_heads."):
            name = key.split(".", 2)[1]
            if name not in catalog_names:
                catalog_names.append(name)
    tasks = list(catalog_names) if args.task == "all" else [args.task]
    build_adapters = bool(blob.get("build_adapters")) or any(k.startswith("task_adapters.") for k in state)
    build_shared = bool(blob.get("build_shared_adapter")) or any(k.startswith("shared_adapter.") for k in state)
    model = build_model(
        model_config,
        build_task_heads=any(t != "encode" for t in tasks),
        build_adapters=build_adapters,
        build_shared_adapter=build_shared,
        overrides={"task_names": tuple(catalog_names or DEFAULT_TASKS)},
    )
    if state_has_lora(state):
        peft_blob = blob.get("peft_cfg") or {}
        peft_cfg = PeftConfig.from_dict(peft_blob) if peft_blob else peft_config_from_train_cfg(peft_blob)
        peft_cfg.shared_lora = peft_cfg.shared_lora or ("shared" in lora_tasks)
        inject_hybrid_lora(model, lora_tasks, peft_cfg)
        print(f"[infer] injected Hybrid-LoRA+ tasks={lora_tasks}", flush=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model = model.to(device)
    print(f"[infer] missing={len(missing)} unexpected={len(unexpected)} device={device}")

    train_cfg: dict[str, Any] = {}
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.is_absolute():
            cfg_path = root / cfg_path
        train_cfg = load_yaml_config(cfg_path)
    model_raw = load_yaml_config(model_config)
    model_section = model_raw.get("model", model_raw)
    token_budget = int(args.token_budget or train_cfg.get("token_budget") or 4096)
    val_seed = int(
        args.val_seed
        if args.val_seed is not None
        else train_cfg.get("val_seed", train_cfg.get("seed", 0))
    )
    val_batches = int(args.val_batches or train_cfg.get("val_batches") or 50)
    patch_size = int(
        train_cfg.get("patch_size")
        or (train_cfg.get("model") or {}).get("patch_size")
        or model_section.get("patch_size")
        or 16
    )
    print(
        f"[infer] split={args.split} token_budget={token_budget} val_seed={val_seed} "
        f"val_batches={val_batches} patch_size={patch_size}",
        flush=True,
    )

    report: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "token_budget": token_budget,
        "val_seed": val_seed,
        "val_batches": val_batches,
        "tasks": {},
    }
    for task in tasks:
        datasets = args.datasets or TASK_DATASETS.get(task) or []
        if not datasets:
            print(f"[infer] skip {task}: 未指定 datasets")
            continue
        loader = build_loader(
            rfdata_root,
            datasets,
            split=args.split,
            token_budget=token_budget,
            patch_size=patch_size,
            val_batches=val_batches,
            val_seed=val_seed,
            num_workers=args.num_workers,
            task=None if task == "encode" else task,
        )
        metrics = run_task(model, loader, device, task)
        report["tasks"][task] = {"datasets": datasets, "split": args.split, **metrics}
        print(f"[infer] {task}: {metrics}")
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[infer] wrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
