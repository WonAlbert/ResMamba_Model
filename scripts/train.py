#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resmamba_signal_model.config.yaml_config import deep_merge, load_yaml_config
from resmamba_signal_model.models.heads import remap_task_head_checkpoints
from resmamba_signal_model.models.model import SignalFoundationModel, SignalModelConfig
from resmamba_signal_model.models.peft import (
    inject_hybrid_lora,
    peft_config_from_train_cfg,
    save_best_bundle,
)
from resmamba_signal_model.models.task_interface import DEFAULT_TASKS, TASK_TO_SOURCE
from resmamba_signal_model.training.task_catalog import (
    TaskSpec,
    apply_catalog_to_model_cfg,
    apply_catalog_to_train_cfg,
    resolve_task_catalog,
)
from resmamba_signal_model.training.checkpointing import (
    TrainStateCallback,
    extract_model_state_dict,
    load_init_from_checkpoint,
    make_model_checkpoint,
    parse_trainer_devices,
    resolve_lightning_precision,
    resolve_run_and_ckpt,
    resolve_run_seed,
    resolve_train_monitor,
)
from resmamba_signal_model.data.rfdata import format_iq_ram_cache
from resmamba_signal_model.training.data_module import SignalDataModule
from resmamba_signal_model.training.early_stopping import make_early_stopping_callback
from resmamba_signal_model.training.emitter_labels import load_emitter_namespace_num_emitters
from resmamba_signal_model.training.freeze import HEAD_MODULE_NAMES, apply_stage_freeze
from resmamba_signal_model.training.lit_module import SignalLitModule
from resmamba_signal_model.training.logging_utils import link_autodl_tensorboard, setup_run_file_logger, silence_third_party_warnings
from resmamba_signal_model.training.param_stats import format_param_stats


STAGE_DEFAULT_CONFIG = {
    "pretrain": "configs/pretrain.yaml",
    "downstream": "configs/downstream.yaml",
    "stage2": "configs/stage2.yaml",
    "stage3": "configs/stage3.yaml",
    "joint": "configs/joint.yaml",
    "continual": "configs/continual.yaml",
}


def apply_named_yaml_profile(config_path: str, train_cfg: dict[str, Any], name: str | None) -> dict[str, Any]:
    if not name:
        return train_cfg
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    profiles = raw.get("profiles") or {}
    if name not in profiles or not isinstance(profiles[name], dict):
        return train_cfg
    return deep_merge(train_cfg, profiles[name])


def parse_adapter_dirs(values: list[str] | None, task_names: list[str] | None = None) -> dict[str, Path]:
    if not values:
        return {}
    known = list(task_names or DEFAULT_TASKS)
    tokens: list[str] = []
    for value in values:
        tokens.extend(part.strip() for part in str(value).split(",") if part.strip())
    mapping: dict[str, Path] = {}
    for token in tokens:
        if "=" in token:
            task, path_s = token.split("=", 1)
            mapping[task.strip()] = Path(path_s.strip())
            continue
        path = Path(token)
        hay = str(path)
        inferred = None
        for task in known:
            if f"stage3_{task}" in hay or path.name == task or path.name.startswith(f"{task}_"):
                inferred = task
                break
        if inferred:
            mapping[inferred] = path
            continue
        if path.is_dir():
            for child in sorted(path.iterdir()):
                for task in known:
                    if f"stage3_{task}" in child.name:
                        mapping[task] = child
    return mapping


def resolve_specialist_ckpt(path: Path, root: Path) -> Path:
    if not path.is_absolute():
        path = (root / path).resolve()
    else:
        path = path.resolve()
    if path.is_file():
        return path
    for candidate in (path / "ckpts" / "best.ckpt", path / "best.ckpt", path / "ckpts" / "last.ckpt"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"找不到 specialist ckpt: {path}")


def load_specialist_weights(model: SignalFoundationModel, ckpt_path: Path, task: str) -> None:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = remap_task_head_checkpoints(extract_model_state_dict(blob))
    prefixes = [HEAD_MODULE_NAMES.get(task, f"{task}_head") + ".", f"extra_task_heads.{task}."]
    filtered: dict[str, Any] = {}
    for key, value in state.items():
        if f"lora_A.{task}" in key or f"lora_B.{task}" in key:
            filtered[key] = value
        elif key.startswith(f"task_adapters.{task}."):
            filtered[key] = value
        elif any(key.startswith(prefix) for prefix in prefixes):
            filtered[key] = value
    if filtered:
        model.load_state_dict(filtered, strict=False)


def load_specialist_scores(adapter_dirs: dict[str, Path], root: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    for task, path in adapter_dirs.items():
        resolved = path if path.is_absolute() else root / path
        if resolved.is_file():
            state_path = resolved.parent.parent / "train_state.json" if resolved.parent.name == "ckpts" else resolved.parent / "train_state.json"
        else:
            state_path = resolved / "train_state.json"
        if not state_path.is_file():
            continue
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        value = payload.get("monitor_value")
        if value is not None:
            scores[task] = float(value)
    return scores


_MODEL_OVERLAY_KEYS = (
    "p_trunc",
    "l_min",
    "chunk_len",
    "sequence_packing",
    "encode_visible_only",
    "train_encoder",
    "train_decoder",
    "train_heads",
    "use_dataset_bias",
    "uti_rank",
    "adapter_down_dim",
    "build_adapters",
    "build_shared_adapter",
    "build_task_interface",
    "build_prototype_registry",
    "task_names",
    "task_kinds",
    "num_task_types",
    "require_mamba_kernel",
    "allow_fallback_mamba",
    "encoder_mamba_layers",
    "use_specialist_views",
    "low_rank_prototype",
    "prototype_rank",
    "domain_prompt_size",
    "share_bidirectional_weights",
    "phase_plugin",
    "legacy_decoder_reconstruction",
)


def load_train_bundle(config: str, *, profile: str | None, model_config: str | None) -> dict[str, Any]:
    train_cfg = load_yaml_config(config, profile=profile)
    model_path = model_config or train_cfg.get("model_config")
    if model_path:
        model_raw = load_yaml_config(model_path)
        model_section = model_raw.get("model", model_raw)
        train_cfg["model"] = deep_merge(dict(model_section), dict(train_cfg.get("model") or {}))
    model = train_cfg.setdefault("model", {})
    for key in _MODEL_OVERLAY_KEYS:
        if key in train_cfg and key not in model:
            model[key] = train_cfg[key]
    return train_cfg


class TTYTQDMProgressBar:
    """优先把进度条打到 /dev/tty，避免 stdout 被 tee 吞掉。

    中途 resume 会跳过 ``on_train_epoch_start``，默认进度条拿不到总数，显示 ``n/?`` 且 ETA=00:00。
    """

    def __new__(cls, *args: Any, **kwargs: Any):
        from lightning.pytorch.callbacks import TQDMProgressBar
        from lightning.pytorch.callbacks.progress.tqdm_progress import convert_inf

        steps_per_epoch = kwargs.pop("steps_per_epoch", None)

        class _Bar(TQDMProgressBar):
            def __init__(self, refresh_rate: int = 1) -> None:
                super().__init__(refresh_rate=refresh_rate)
                self._steps_per_epoch = int(steps_per_epoch) if steps_per_epoch else None
                self._tty = None
                try:
                    self._tty = open("/dev/tty", "w", encoding="utf-8")
                except OSError:
                    self._tty = None

            def init_train_tqdm(self):
                bar = super().init_train_tqdm()
                if self._tty is not None:
                    bar.file = self._tty
                return bar

            def _sync_train_bar(self, trainer: Any) -> None:
                bar = getattr(self, "train_progress_bar", None)
                if bar is None:
                    return
                total = convert_inf(self.total_train_batches)
                if total is None and self._steps_per_epoch:
                    total = self._steps_per_epoch
                if total is not None:
                    bar.total = int(total)
                bar.set_description(f"Epoch {trainer.current_epoch}")
                bar.refresh()

            def on_train_start(self, trainer: Any, pl_module: Any) -> None:
                super().on_train_start(trainer, pl_module)
                self._sync_train_bar(trainer)

            def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
                super().on_train_epoch_start(trainer, pl_module)
                self._sync_train_bar(trainer)

            def teardown(self, *args: Any, **kwargs: Any) -> None:
                super().teardown(*args, **kwargs)
                if self._tty is not None:
                    self._tty.close()
                    self._tty = None

        return _Bar(*args, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Signal foundation model Lightning trainer")
    parser.add_argument(
        "--stage",
        choices=("pretrain", "downstream", "stage2", "stage3", "joint", "continual"),
        default="pretrain",
    )
    parser.add_argument("--config", default=None, help="训练 YAML，默认按 --stage 选择")
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument(
        "--task",
        default=None,
        help="stage3 必填：目录中的任意任务名（默认含 modulation/emitter/clustering/prediction/imputation）",
    )
    parser.add_argument("--task-kind", default=None, help="新任务的 kind：classification|emitter|clustering|prediction|imputation")
    parser.add_argument("--tasks", default=None, help="覆盖任务目录，逗号分隔，如 modulation,emitter,sonar")
    parser.add_argument("--init-from", default=None, help="只加载权重，不恢复优化器")
    parser.add_argument(
        "--adapter-dir",
        action="append",
        default=None,
        help="joint 时传入 specialist 目录，可重复或逗号分隔；可用 task=path",
    )
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=None, help="覆盖 yaml seed，并传给 Lightning seed_everything")
    parser.add_argument("--devices", default=None, help="Lightning devices，如 auto / 1 / 0,1")
    parser.add_argument("--strategy", default=None, help="Lightning strategy，如 auto / ddp")
    parser.add_argument("--amp-dtype", dest="amp_dtype", default=None, help="bfloat16 / float16，映射到 Lightning precision")
    parser.add_argument("--precision", default=None, help="直接指定 Lightning precision")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="从 checkpoint 续训（含优化器/调度器/epoch）。省略路径时使用 --run-name 下 ckpts/best.ckpt（兼容 last.ckpt）",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="只跑验证：加载 --resume/--init-from（或 --run-name 下 best.ckpt），不训练",
    )
    return parser.parse_args()


def main() -> None:
    silence_third_party_warnings()
    args = parse_args()
    if args.stage == "stage3" and not args.task:
        raise SystemExit("stage3 需要 --task <任务名>")
    default_cfg = ROOT / STAGE_DEFAULT_CONFIG.get(args.stage, "configs/pretrain.yaml")
    config_path = args.config or str(default_cfg)
    train_cfg = load_train_bundle(config_path, profile=args.profile, model_config=args.model_config)
    if args.tasks:
        train_cfg["tasks"] = [part.strip() for part in str(args.tasks).split(",") if part.strip()]
    if args.task_kind and args.task:
        kinds = dict(train_cfg.get("task_kinds") or {})
        kinds[args.task] = args.task_kind
        train_cfg["task_kinds"] = kinds
    catalog = resolve_task_catalog(train_cfg)
    if args.stage == "stage3" and args.task:
        train_cfg = apply_named_yaml_profile(config_path, train_cfg, args.task)
        train_cfg["task"] = args.task
        if args.task not in catalog.by_name:
            catalog = catalog.with_task(
                TaskSpec(args.task, args.task_kind or "classification", args.task)
            )
        spec = catalog.require(args.task)
        source = spec.source or TASK_TO_SOURCE.get(args.task, args.task)
        pools = train_cfg.get("task_pools") or {}
        if source in pools:
            train_cfg["task_pools"] = {source: pools[source]}
        train_cfg["synthetic_sources"] = [source]
    train_cfg = apply_catalog_to_train_cfg(train_cfg, catalog)
    if args.synthetic:
        train_cfg["synthetic"] = True
    if args.max_epochs is not None:
        train_cfg["epochs"] = args.max_epochs
    if args.limit_train_batches is not None:
        train_cfg["limit_train_batches"] = args.limit_train_batches
    if args.seed is not None:
        train_cfg["seed"] = int(args.seed)
    if args.devices is not None:
        train_cfg["devices"] = args.devices
    if args.strategy is not None:
        train_cfg["strategy"] = args.strategy
    if args.amp_dtype is not None:
        train_cfg["amp_dtype"] = args.amp_dtype
    if args.precision is not None:
        train_cfg["precision"] = args.precision
    seed = resolve_run_seed(train_cfg)
    train_cfg["seed"] = seed
    train_cfg.setdefault("val_seed", seed)

    model_cfg = SignalModelConfig.from_dict(train_cfg.get("model") or {})
    model_cfg.build_task_heads = args.stage != "pretrain"
    model_cfg.build_task_interface = True
    model_cfg.build_prototype_registry = args.stage != "pretrain"
    model_cfg.build_adapters = args.stage in ("stage3", "joint", "continual")
    model_cfg.build_shared_adapter = args.stage in ("joint", "continual")
    apply_catalog_to_model_cfg(model_cfg, catalog)
    train_cfg.setdefault("patch_size", model_cfg.patch_size)
    if not bool(train_cfg.get("synthetic", False)):
        n_emitters = load_emitter_namespace_num_emitters(train_cfg.get("rfdata_root"))
        if n_emitters:
            model_cfg.num_emitters = int(n_emitters)
            model_section = train_cfg.setdefault("model", {})
            model_section["num_emitters"] = int(n_emitters)

    adapter_dirs = parse_adapter_dirs(args.adapter_dir, catalog.names)
    if args.stage == "joint" and adapter_dirs and not train_cfg.get("specialist_scores"):
        scores = load_specialist_scores(adapter_dirs, ROOT)
        if scores:
            train_cfg["specialist_scores"] = scores

    from resmamba_signal_model.training.continual import (
        ContinualSessionCallback,
        resolve_continual_sessions,
        total_continual_epochs,
    )

    sessions = resolve_continual_sessions(train_cfg, stage=args.stage)
    if args.stage == "continual" or sessions:
        train_cfg["continual"] = True if args.stage == "continual" else bool(train_cfg.get("continual", False))
        if args.stage == "continual":
            train_cfg["continual"] = True
            train_cfg.setdefault("distill_weight", 0.5)
            train_cfg.setdefault("prototype_anchor_weight", 0.1)
            train_cfg.setdefault("absorb_unknown", True)
            train_cfg.setdefault("shared_lora", True)
        if train_cfg.get("continual_sessions"):
            train_cfg["epochs"] = total_continual_epochs(sessions, default_epochs=int(train_cfg.get("epochs", 1)))

    run_name = args.run_name
    if run_name is None and args.stage == "stage3" and args.task:
        from datetime import datetime

        run_name = f"stage3_{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir, resume_ckpt = resolve_run_and_ckpt(
        root=ROOT,
        stage=args.stage,
        run_name=run_name,
        resume=args.resume,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(train_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    logger = setup_run_file_logger(run_dir)
    logger.info("stage=%s config=%s task=%s", args.stage, config_path, args.task)
    if resume_ckpt is not None:
        logger.info("resume ckpt=%s", resume_ckpt)
        print(f"resume from {resume_ckpt}", flush=True)

    import lightning as L
    L.seed_everything(seed, workers=True)

    data = SignalDataModule(train_cfg, stage=args.stage)
    data.setup()
    logger.info("sources=%s", data.source_names)
    cache_msg = format_iq_ram_cache()
    logger.info("%s", cache_msg)
    print(cache_msg, flush=True)

    model = SignalFoundationModel(model_cfg)
    if args.init_from:
        init_path = Path(args.init_from)
        if not init_path.is_absolute():
            init_path = ROOT / init_path
        missing, unexpected = load_init_from_checkpoint(model, init_path, strict=False)
        logger.info("init-from %s missing=%s unexpected=%s", init_path, len(missing), len(unexpected))
        print(f"init-from {init_path} missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    peft_cfg = None
    if args.stage in ("stage3", "joint", "continual"):
        peft_cfg = peft_config_from_train_cfg(train_cfg)
        if args.stage in ("joint", "continual"):
            peft_cfg.shared_lora = bool(train_cfg.get("shared_lora", args.stage == "continual" or peft_cfg.shared_lora))
        tasks = [args.task] if args.stage == "stage3" and args.task else list(catalog.names)
        inject_hybrid_lora(model, tasks, peft_cfg)
        logger.info("injected Hybrid-LoRA+ tasks=%s modules=%s", tasks, len(getattr(model.peft, "names", [])))

    if args.stage == "joint" and adapter_dirs:
        for task, path in adapter_dirs.items():
            ckpt = resolve_specialist_ckpt(path, ROOT)
            load_specialist_weights(model, ckpt, task)
            logger.info("loaded specialist task=%s ckpt=%s", task, ckpt)

    apply_stage_freeze(model, args.stage, task=args.task, train_cfg=train_cfg)
    logger.info("\n%s", format_param_stats(model))
    print(format_param_stats(model), flush=True)

    from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
    import torch as _torch

    if _torch.cuda.is_available():
        _torch.set_float32_matmul_precision("high")

    lit = SignalLitModule(model, train_cfg, stage=args.stage, mix=data.mix)
    tb_dir = run_dir / "tb"
    link_autodl_tensorboard(tb_dir)
    precision = resolve_lightning_precision(train_cfg)
    monitor, ckpt_mode = resolve_train_monitor(
        train_cfg,
        stage=args.stage,
        task=args.task or train_cfg.get("task"),
    )
    logger.info(
        "seed=%s precision=%s monitor=%s mode=%s mix_strategy=%s combine_then_pack=%s lr_schedule=%s devices=%s strategy=%s",
        seed,
        precision,
        monitor,
        ckpt_mode,
        train_cfg.get("mix_strategy"),
        train_cfg.get("combine_then_pack"),
        train_cfg.get("lr_schedule"),
        train_cfg.get("devices"),
        train_cfg.get("strategy"),
    )
    ckpt_dir = run_dir / "ckpts"
    ckpt_cb = make_model_checkpoint(ckpt_dir, monitor=monitor, mode=ckpt_mode)
    state_cb = TrainStateCallback(run_dir / "train_state.json", monitor=monitor, stage=args.stage)
    early_cb = make_early_stopping_callback(
        monitor=monitor,
        mode=ckpt_mode,
        patience=int(train_cfg.get("early_stopping_patience") or 0),
        min_delta=float(train_cfg.get("early_stopping_min_delta") or 0.0),
    )
    callbacks: list[Any] = [
        TTYTQDMProgressBar(refresh_rate=1, steps_per_epoch=int(train_cfg.get("steps_per_epoch") or 0) or None),
        ckpt_cb,
        state_cb,
    ]
    if early_cb is not None:
        callbacks.append(early_cb)
        logger.info(
            "early stopping monitor=%s mode=%s patience=%s",
            monitor,
            ckpt_mode,
            int(train_cfg.get("early_stopping_patience") or 0),
        )
    if sessions and (args.stage == "continual" or bool(train_cfg.get("continual")) or bool(train_cfg.get("absorb_unknown"))):
        callbacks.append(ContinualSessionCallback(sessions, train_cfg))
        logger.info("continual sessions=%s distill_weight=%s prototype_anchor_weight=%s", len(sessions), train_cfg.get("distill_weight"), train_cfg.get("prototype_anchor_weight"))

    trainer_kwargs: dict[str, Any] = dict(
        default_root_dir=str(run_dir),
        max_epochs=int(train_cfg.get("epochs", 1)),
        accumulate_grad_batches=int(train_cfg.get("gradient_accumulation_steps", 1)),
        gradient_clip_val=float(train_cfg.get("max_grad_norm", 1.0)),
        precision=precision,
        reload_dataloaders_every_n_epochs=1,
        check_val_every_n_epoch=int(train_cfg.get("check_val_every_n_epoch", 1)),
        log_every_n_steps=1,
        enable_progress_bar=True,
        callbacks=callbacks,
        logger=[
            TensorBoardLogger(save_dir=str(run_dir), name="tb", version=0),
            CSVLogger(save_dir=str(run_dir), name="csv", version=0),
        ],
    )
    limit_train = train_cfg.get("limit_train_batches")
    if limit_train is None:
        limit_train = args.limit_train_batches
    if limit_train is None:
        limit_train = train_cfg.get("steps_per_epoch")
    limit_val = train_cfg.get("limit_val_batches") or args.limit_val_batches
    if limit_train:
        trainer_kwargs["limit_train_batches"] = int(limit_train)
    if limit_val:
        trainer_kwargs["limit_val_batches"] = int(limit_val)
    devices = parse_trainer_devices(train_cfg.get("devices"))
    if devices is not None:
        trainer_kwargs["devices"] = devices
    if train_cfg.get("strategy") not in (None, ""):
        trainer_kwargs["strategy"] = train_cfg.get("strategy")
    if train_cfg.get("accelerator") not in (None, ""):
        trainer_kwargs["accelerator"] = train_cfg.get("accelerator")
    trainer = L.Trainer(**trainer_kwargs)
    if args.validate_only:
        ckpt_path = resume_ckpt
        if ckpt_path is None and args.init_from:
            init_path = Path(args.init_from)
            if not init_path.is_absolute():
                init_path = ROOT / init_path
            ckpt_path = init_path
        if ckpt_path is None:
            raise SystemExit("--validate-only 需要 --resume 或 --init-from（或 --run-name + --resume）指向 ckpt")
        logger.info("validate-only ckpt=%s", ckpt_path)
        print(f"validate-only from {ckpt_path}", flush=True)
        trainer.validate(lit, datamodule=data, ckpt_path=str(ckpt_path))
        logger.info("validation finished")
        return
    trainer.fit(lit, datamodule=data, ckpt_path=str(resume_ckpt) if resume_ckpt is not None else None)
    logger.info("training finished best=%s", ckpt_cb.best_model_path)
    if args.stage == "joint":
        bundle_path = Path(ckpt_cb.best_model_path) if ckpt_cb.best_model_path else ckpt_dir / "best.ckpt"
        if ckpt_cb.best_model_path:
            blob = _torch.load(ckpt_cb.best_model_path, map_location="cpu", weights_only=False)
            lit.model.load_state_dict(extract_model_state_dict(blob), strict=False)
        save_best_bundle(
            str(bundle_path),
            model=lit.model,
            peft_cfg=peft_cfg.to_dict() if peft_cfg is not None else None,
            extra={"train_stage": "joint"},
        )
        logger.info("wrote joint best bundle %s", bundle_path)


if __name__ == "__main__":
    main()
