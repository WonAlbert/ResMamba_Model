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
from resmamba_signal_model.training.emitter_labels import (  # noqa: E402
    build_global_emitter_label_map,
    build_global_radar_model_label_map,
    global_emitter_labels,
)
from resmamba_signal_model.training.modulation_labels import (  # noqa: E402
    build_global_comm_modulation_label_map,
    global_comm_modulation_labels,
    remap_modulation_labels,
)
from resmamba_signal_model.training.pool_filters import load_pretrain_datasets  # noqa: E402
from resmamba_signal_model.training.compact_labels import (  # noqa: E402
    compact_emitter_class_mask,
    compact_ld_model_class_mask,
    compact_tx_modulation_class_mask,
    radar_model_map_from_train_cfg,
    tx_modulation_map_from_train_cfg,
)

TASKS = ("modulation", "emitter", "prediction", "clustering", "imputation", "encode")
TASK_DATASETS: dict[str, list[str]] = {
    "modulation": ["rml2016_04c", "rml2016_10a", "rml2016_10b"],  # 暂不用 rml2018_1a
    "emitter": ["wisig", "adsb2"],
    "prediction": ["adsb2", "rml2016_04c", "rml2016_10a", "rml2016_10b", "wifi150"],
    "imputation": ["adsb2", "rml2016_04c", "rml2016_10a", "rml2016_10b", "wifi150"],
    "clustering": ["adsb2", "rml2016_04c", "rml2016_10a", "rml2016_10b", "wifi150"],
    "encode": [],
}


def _tensor_class_dim(state: dict[str, torch.Tensor], *suffixes: str) -> int | None:
    for key, value in state.items():
        if not torch.is_tensor(value) or value.ndim < 1:
            continue
        for suffix in suffixes:
            if key.endswith(suffix):
                return int(value.shape[0])
    return None


def resolve_compact_overrides(
    state: dict[str, torch.Tensor],
    train_cfg: dict[str, Any],
    rfdata_root: Path,
) -> dict[str, Any]:
    """从 ckpt 权重形状 / train_cfg 恢复紧凑类数，供建头与标签重映射。"""
    overrides: dict[str, Any] = {}
    model_section = dict(train_cfg.get("model") or {})
    n_mod = _tensor_class_dim(
        state,
        "tx_modulation_head.classifier.weight",
        "modulation_head.classifier.weight",
        "z_linear_probes.tx_modulation.weight",
        "z_linear_probes.modulation.weight",
    )
    n_emit = _tensor_class_dim(
        state,
        "emitter_head.classifier.weight",
        "z_linear_probes.emitter.weight",
    )
    if n_mod is None and model_section.get("num_mod_classes") is not None:
        n_mod = int(model_section["num_mod_classes"])
    if n_emit is None and model_section.get("num_emitters") is not None:
        n_emit = int(model_section["num_emitters"])
    if train_cfg.get("compact_tx_modulation", {}).get("num_classes") is not None:
        n_mod = int(train_cfg["compact_tx_modulation"]["num_classes"])
    elif train_cfg.get("compact_modulation", {}).get("num_classes") is not None:
        n_mod = int(train_cfg["compact_modulation"]["num_classes"])
    if train_cfg.get("compact_emitter", {}).get("num_emitters") is not None:
        n_emit = int(train_cfg["compact_emitter"]["num_emitters"])
    if n_mod is not None:
        overrides["num_mod_classes"] = int(n_mod)
    if n_emit is not None:
        overrides["num_emitters"] = int(n_emit)
    # 默认：类数小于全库 ontology / namespace 时启用紧凑重映射
    use_compact = bool(train_cfg.get("compact_task_labels"))
    if not use_compact and rfdata_root.is_dir():
        try:
            compact_e = build_global_emitter_label_map(rfdata_root, train_cfg=train_cfg)
            if n_emit is not None and int(n_emit) == int(compact_e.num_emitters):
                use_compact = True
        except (FileNotFoundError, KeyError, OSError):
            pass
        try:
            comm_map, _canonical = build_global_comm_modulation_label_map(rfdata_root, train_cfg=train_cfg)
            if n_mod is not None and int(n_mod) == int(comm_map.num_emitters):
                use_compact = True
        except (FileNotFoundError, KeyError, OSError):
            pass
    overrides["_use_compact"] = use_compact
    return overrides


def task_label_tensor(
    task: str,
    batch: dict[str, Any],
    *,
    emitter_offset_lookup: torch.Tensor | None = None,
    ld_model_offset_lookup: torch.Tensor | None = None,
    modulation_compact_lookup: torch.Tensor | None = None,
    modulation_offset_lookup: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """与训练/验证相同的标签列；紧凑模式下做连续 ID 重映射。"""
    try:
        field = builtin_spec(task).label_field
    except KeyError:
        field = None
    if task == "emitter":
        if emitter_offset_lookup is not None and "emitter_id" in batch and "dataset_id" in batch:
            value = global_emitter_labels(batch["dataset_id"], batch["emitter_id"], emitter_offset_lookup)
        else:
            value = batch.get(field or "global_emitter_id", batch.get("emitter_id"))
    elif task == "ld_model":
        if ld_model_offset_lookup is not None and "mod_label_id" in batch and "dataset_id" in batch:
            value = global_emitter_labels(batch["dataset_id"], batch["mod_label_id"], ld_model_offset_lookup)
        else:
            value = batch.get(field or "mod_label_id", batch.get("mod_label_id"))
    if task == "clustering" or task in ("ld_clustering", "tx_clustering"):
        from resmamba_signal_model.training.clustering_labels import resolve_cluster_eval_labels

        value = resolve_cluster_eval_labels(
            batch.get("mod_label_id"),
            batch.get("emitter_id"),
            batch.get("source_label_id"),
        )
        if value is None:
            value = batch.get(field or "global_label_id")
    elif task in ("prediction", "imputation", "encode"):
        return None
    elif task in ("tx_modulation", "modulation", "ld_intrapulse"):
        value = batch.get(
            field or "canonical_mod_label_id",
            batch.get("mod_label_id", batch.get("source_label_id")),
        )
        if (
            task in ("tx_modulation", "modulation")
            and modulation_offset_lookup is not None
            and modulation_compact_lookup is not None
            and value is not None
            and "dataset_id" in batch
        ):
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            value = global_comm_modulation_labels(
                batch["dataset_id"],
                value,
                modulation_compact_lookup,
                modulation_offset_lookup,
            )
        elif value is not None and modulation_compact_lookup is not None:
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            value = remap_modulation_labels(value, modulation_compact_lookup)
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


def find_h5(rfdata_root: Path, dataset: str, split: str = "val") -> Path:
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
def run_task(
    model: SignalFoundationModel,
    loader: DataLoader,
    device: torch.device,
    task: str,
    *,
    emitter_offset_lookup: torch.Tensor | None = None,
    ld_model_offset_lookup: torch.Tensor | None = None,
    modulation_compact_lookup: torch.Tensor | None = None,
    modulation_offset_lookup: torch.Tensor | None = None,
) -> dict[str, Any]:
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
    label_kw = {
        "emitter_offset_lookup": emitter_offset_lookup,
        "ld_model_offset_lookup": ld_model_offset_lookup,
        "modulation_compact_lookup": modulation_compact_lookup,
        "modulation_offset_lookup": modulation_offset_lookup,
    }
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
            label = task_label_tensor(task, batch, **label_kw)
            if label is not None:
                labels.append(label)
        elif task == "clustering" and "cluster_logits" in out:
            preds.append(out["cluster_logits"].argmax(dim=-1).cpu())
            label = task_label_tensor(task, batch, **label_kw)
            if label is not None:
                labels.append(label)
        elif logits is not None and task not in ("prediction", "imputation"):
            preds.append(logits.argmax(dim=-1).cpu())
            label = task_label_tensor(task, batch, **label_kw)
            if label is not None:
                labels.append(label)
        elif task in ("prediction", "imputation") or "pred_patches" in out:
            pred, target, mask = reconstruction_eval_pair(out)
            ssim_pred = out.get("mae_pred") if out.get("mae_pred") is not None else pred
            ssim_tgt = out.get("patch_targets") if out.get("patch_targets") is not None else target
            wave_len = out.get("iq_length")
            if torch.is_tensor(wave_len):
                wave_len = int(wave_len.reshape(-1)[0].item())
            elif wave_len is not None:
                wave_len = int(wave_len)
            ssim_vals.append(float(ssim_iq(ssim_pred, ssim_tgt, mask, length=wave_len)))
            mse_vals.append(float(masked_patch_mse(pred, target, mask).item()))
        score = out.get("openset_score", out.get("openset_energy"))
        if score is not None:
            osr_scores.append(score.detach().reshape(-1).cpu())
            label = task_label_tensor(task, batch, **label_kw)
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


def _macro_recon_metrics(per_dataset: dict[str, dict[str, Any]]) -> dict[str, float]:
    rows = [
        row
        for row in per_dataset.values()
        if row.get("impute_mse") is not None or row.get("ssim") is not None
    ]
    if not rows:
        return {}
    mse_vals = [float(row["impute_mse"]) for row in rows if row.get("impute_mse") is not None]
    ssim_vals = [float(row["ssim"]) for row in rows if row.get("ssim") is not None]
    out: dict[str, float] = {}
    if mse_vals:
        out["macro_mse"] = float(sum(mse_vals) / len(mse_vals))
    if ssim_vals:
        out["macro_ssim"] = float(sum(ssim_vals) / len(ssim_vals))
    out["num_datasets"] = float(len(rows))
    return out


def _print_prediction_table(per_dataset: dict[str, dict[str, Any]], macro: dict[str, float]) -> None:
    print("dataset          mse        ssim     batches", flush=True)
    for name in sorted(per_dataset):
        row = per_dataset[name]
        mse = row.get("impute_mse")
        ssim = row.get("ssim")
        n = row.get("num_batches", 0)
        if mse is None and ssim is None:
            err = row.get("error", "skip")
            print(f"{name:16s}  —          —        {err}", flush=True)
            continue
        print(
            f"{name:16s}  {float(mse):.6f}  {float(ssim):.6f}  {int(n)}",
            flush=True,
        )
    if macro:
        print(
            f"macro            {macro.get('macro_mse', 0):.6f}  {macro.get('macro_ssim', 0):.6f}  "
            f"datasets={int(macro.get('num_datasets', 0))}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SignalFoundationModel 下游推理")
    p.add_argument("--task", required=True, help="任务名或 all；可为目录中任意下游任务")
    p.add_argument("--checkpoint", required=True, help="Lightning best.ckpt 或 state_dict")
    p.add_argument("--model-config", default="configs/model.yaml")
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--rfdata-root", default="dataset")
    p.add_argument("--config", default=None, help="训练 YAML，用于对齐 token_budget/val_seed/val_batches")
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--token-budget", type=int, default=None)
    p.add_argument("--val-seed", type=int, default=None)
    p.add_argument("--val-batches", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--batch-size", type=int, default=None, help="已弃用：infer 复用 val 的 token-budget 协议")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--per-dataset",
        action="store_true",
        help="prediction/imputation：按 datasets 逐项评测并汇总 macro 指标",
    )
    p.add_argument(
        "--pretrain-sources",
        action="store_true",
        help="使用 configs/datasets.yaml pretrain 白名单作为 --datasets",
    )
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
    train_cfg: dict[str, Any] = {}
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.is_absolute():
            cfg_path = root / cfg_path
        train_cfg = load_yaml_config(cfg_path)
    else:
        # 同 run 目录的 config.yaml（训练时写出 compact_*）
        run_cfg = ckpt_path.parents[1] / "config.yaml"
        if run_cfg.is_file():
            train_cfg = load_yaml_config(run_cfg)
    compact_overrides = resolve_compact_overrides(state, train_cfg, rfdata_root)
    use_compact = bool(compact_overrides.pop("_use_compact", False))
    emitter_offset_lookup = None
    ld_model_offset_lookup = None
    modulation_compact_lookup = None
    modulation_offset_lookup = None
    compact_emitter_map = None
    ld_model_map = None
    tx_map = None
    tx_canonical = None
    if use_compact:
        try:
            compact_emitter_map = build_global_emitter_label_map(rfdata_root, train_cfg=train_cfg)
            if int(compact_emitter_map.num_emitters) > 0:
                emitter_offset_lookup = compact_emitter_map.offset_lookup()
                compact_overrides.setdefault("num_emitters", int(compact_emitter_map.num_emitters))
            else:
                compact_emitter_map = None
        except (FileNotFoundError, KeyError, OSError):
            compact_emitter_map = None
        ld_model_map = radar_model_map_from_train_cfg(train_cfg)
        if ld_model_map is None:
            try:
                ld_model_map = build_global_radar_model_label_map(rfdata_root, train_cfg=train_cfg)
            except (FileNotFoundError, KeyError, OSError):
                ld_model_map = None
        if ld_model_map is not None and int(ld_model_map.num_emitters) > 0:
            ld_model_offset_lookup = ld_model_map.offset_lookup()
            compact_overrides.setdefault("num_ld_model_classes", int(ld_model_map.num_emitters))
        tx_map, tx_canonical = tx_modulation_map_from_train_cfg(train_cfg)
        if tx_map is None:
            try:
                tx_map, tx_canonical = build_global_comm_modulation_label_map(rfdata_root, train_cfg=train_cfg)
            except (FileNotFoundError, KeyError, OSError):
                tx_map, tx_canonical = None, None
        if tx_map is not None and int(tx_map.num_emitters) > 0:
            modulation_offset_lookup = tx_map.offset_lookup()
            compact_overrides.setdefault("num_mod_classes", int(tx_map.num_emitters))
            if tx_canonical is not None:
                modulation_compact_lookup = tx_canonical.lookup()

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
    model_overrides = {"task_names": tuple(catalog_names or DEFAULT_TASKS), **compact_overrides}
    model = build_model(
        model_config,
        build_task_heads=any(t != "encode" for t in tasks),
        build_adapters=build_adapters,
        build_shared_adapter=build_shared,
        overrides=model_overrides,
    )
    if state_has_lora(state):
        peft_blob = blob.get("peft_cfg") or {}
        peft_cfg = PeftConfig.from_dict(peft_blob) if peft_blob else peft_config_from_train_cfg(peft_blob)
        peft_cfg.shared_lora = peft_cfg.shared_lora or ("shared" in lora_tasks)
        inject_hybrid_lora(model, lora_tasks, peft_cfg)
        print(f"[infer] injected Hybrid-LoRA+ tasks={lora_tasks}", flush=True)
    if ld_model_map is not None:
        mask = compact_ld_model_class_mask(ld_model_map, num_datasets=int(model.cfg.num_datasets))
        model.load_emitter_dataset_class_mask(rfdata_root, mask=mask)
    elif tx_map is not None and tx_canonical is not None:
        tx_mask = compact_tx_modulation_class_mask(
            tx_map,
            tx_canonical,
            rfdata_root,
            num_datasets=int(model.cfg.num_datasets),
        )
        model.load_tx_modulation_dataset_class_mask(tx_mask)
    elif compact_emitter_map is not None:
        mask = compact_emitter_class_mask(
            compact_emitter_map,
            num_datasets=int(model.cfg.num_datasets),
        )
        model.load_emitter_dataset_class_mask(rfdata_root, mask=mask)
    else:
        model.load_emitter_dataset_class_mask(rfdata_root)
    missing, unexpected = model.load_weights(state, strict=False)
    model = model.to(device)
    print(f"[infer] missing={len(missing)} unexpected={len(unexpected)} device={device}")
    if use_compact:
        print(
            f"[infer] compact_task_labels mod={model.cfg.num_mod_classes} emitter={model.cfg.num_emitters}",
            flush=True,
        )

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
        "compact_task_labels": use_compact,
        "tasks": {},
    }
    for task in tasks:
        if args.pretrain_sources:
            datasets = load_pretrain_datasets(rfdata_root)
        else:
            datasets = args.datasets or TASK_DATASETS.get(task) or []
        if not datasets:
            print(f"[infer] skip {task}: 未指定 datasets")
            continue
        infer_task = None if task == "encode" else task
        if args.per_dataset and task in ("prediction", "imputation"):
            per_dataset: dict[str, dict[str, Any]] = {}
            for ds in datasets:
                try:
                    h5_path = resolve_dataset_h5(rfdata_root, ds, args.split)
                except FileNotFoundError:
                    per_dataset[ds] = {"error": f"missing {ds}_{args.split}.h5"}
                    print(f"[infer] {task}/{ds}: missing H5 for split={args.split}", flush=True)
                    continue
                loader = build_loader(
                    rfdata_root,
                    [ds],
                    split=args.split,
                    token_budget=token_budget,
                    patch_size=patch_size,
                    val_batches=val_batches,
                    val_seed=val_seed,
                    num_workers=args.num_workers,
                    task=infer_task,
                )
                per_dataset[ds] = run_task(
                    model,
                    loader,
                    device,
                    task,
                    emitter_offset_lookup=emitter_offset_lookup,
                    ld_model_offset_lookup=ld_model_offset_lookup,
                    modulation_compact_lookup=modulation_compact_lookup,
                    modulation_offset_lookup=modulation_offset_lookup,
                )
                print(f"[infer] {task}/{ds}: {per_dataset[ds]}", flush=True)
            macro = _macro_recon_metrics(per_dataset)
            _print_prediction_table(per_dataset, macro)
            report["tasks"][task] = {
                "datasets": datasets,
                "split": args.split,
                "per_dataset": per_dataset,
                **macro,
            }
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
            task=infer_task,
        )
        metrics = run_task(
            model,
            loader,
            device,
            task,
            emitter_offset_lookup=emitter_offset_lookup,
            ld_model_offset_lookup=ld_model_offset_lookup,
            modulation_compact_lookup=modulation_compact_lookup,
            modulation_offset_lookup=modulation_offset_lookup,
        )
        report["tasks"][task] = {"datasets": datasets, "split": args.split, **metrics}
        print(f"[infer] {task}: {metrics}")
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[infer] wrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
