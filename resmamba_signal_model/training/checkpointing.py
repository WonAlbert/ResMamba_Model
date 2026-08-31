from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from resmamba_signal_model.models.heads import remap_task_head_checkpoints
from resmamba_signal_model.training.freeze import (
    filter_stage2_task_state,
    merge_stage2_task_states,
)
from resmamba_signal_model.training.selection import (
    resolve_checkpoint_monitor_name,
    specialist_relative_geomean,
)
from resmamba_signal_model.training.task_catalog import KIND_MONITOR, resolve_task_catalog

try:
    from lightning.fabric.utilities.rank_zero import rank_zero_info
    from lightning.pytorch.callbacks import Callback, ModelCheckpoint
except ImportError:  # pragma: no cover
    Callback = object  # type: ignore[misc, assignment]
    ModelCheckpoint = None  # type: ignore[misc, assignment]
    rank_zero_info = None  # type: ignore[misc, assignment]


DEFAULT_MONITOR = "val/monitor"


def resolve_lightning_precision(train_cfg: dict[str, Any], *, cuda_available: bool | None = None) -> str:
    """把 ``precision`` / ``amp`` / ``amp_dtype`` 映射成 Lightning precision 字符串。"""
    explicit = train_cfg.get("precision")
    if explicit not in (None, ""):
        return str(explicit)
    amp = bool(train_cfg.get("amp", True))
    if not amp:
        return "32-true"
    if cuda_available is None:
        cuda_available = bool(torch.cuda.is_available())
    raw = str(train_cfg.get("amp_dtype") or "bfloat16").strip().lower()
    if raw in ("bf16", "bfloat16", "bf16-mixed"):
        mapped = "bf16-mixed"
    elif raw in ("fp16", "float16", "16", "16-mixed", "16-true"):
        mapped = "16-mixed" if raw != "16-true" else "16-true"
    elif raw in ("fp32", "float32", "32", "32-true"):
        mapped = "32-true"
    else:
        mapped = raw
    if mapped != "32-true" and not cuda_available:
        return "32-true"
    return mapped


def resolve_run_seed(train_cfg: dict[str, Any], cli_seed: int | None = None) -> int:
    if cli_seed is not None:
        return int(cli_seed)
    if train_cfg.get("seed") is not None:
        return int(train_cfg["seed"])
    return 0


def parse_trainer_devices(value: Any) -> Any:
    if value is None or value == "":
        return None
    if isinstance(value, (int, list, tuple)):
        return list(value) if isinstance(value, tuple) else value
    text = str(value).strip()
    if not text:
        return None
    if text.lower() == "auto":
        return "auto"
    if "," in text:
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    return int(text)


def resolve_train_monitor(
    train_cfg: dict[str, Any],
    *,
    stage: str,
    task: str | None = None,
) -> tuple[str, str]:
    monitor = resolve_checkpoint_monitor_name(train_cfg, stage=stage, task=task)
    return monitor, checkpoint_mode_for_monitor(monitor, train_cfg.get("checkpoint_mode"))


def checkpoint_mode_for_monitor(monitor: str, explicit: str | None = None) -> str:
    if explicit:
        mode = str(explicit).strip().lower()
        if mode not in ("min", "max"):
            raise ValueError(f"checkpoint_mode 必须是 min 或 max，当前为 {explicit!r}")
        return mode
    name = monitor.rsplit("/", 1)[-1].lower()
    if name in ("loss", "mae", "mse", "val_loss", "monitor") or "loss" in name:
        return "min"
    return "max"


def _as_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1 or not torch.isfinite(value).all():
            return None
        raw = float(value.detach().cpu().item())
    else:
        try:
            raw = float(value)
        except (TypeError, ValueError):
            return None
    if not math.isfinite(raw):
        return None
    return raw


def _collect_val_metric_means(callback_metrics: dict[str, Any], *prefixes: str) -> list[float]:
    values: list[float] = []
    for key, value in callback_metrics.items():
        name = str(key)
        if name == "val/monitor":
            continue
        for prefix in prefixes:
            if name == prefix or name.startswith(prefix + "/"):
                parsed = _as_finite_float(value)
                if parsed is not None:
                    values.append(parsed)
                break
    return values


def aggregate_val_monitor(callback_metrics: dict[str, Any]) -> float | None:
    """聚合各源重建监控标量，供 best ckpt 使用。

    优先 ``val/recon``（不含 domain / structure_phase / UTI）；若缺失再回退 ``val/loss``。
    """
    values = _collect_val_metric_means(callback_metrics, "val/recon")
    if not values:
        values = _collect_val_metric_means(callback_metrics, "val/loss")
    if not values:
        return None
    return float(sum(values) / len(values))


_DATALOADER_IDX_RE = re.compile(r"/dataloader_idx_\d+$")


def _metric_from_callbacks(callback_metrics: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        parsed = _as_finite_float(callback_metrics.get(key))
        if parsed is not None:
            return parsed
    for stored, value in callback_metrics.items():
        name = _DATALOADER_IDX_RE.sub("", str(stored))
        for key in keys:
            if name == key or name.startswith(key + "/"):
                parsed = _as_finite_float(value)
                if parsed is not None:
                    return parsed
    return None


def resolve_task_checkpoint_monitor(
    train_cfg: dict[str, Any],
    *,
    stage: str,
    task: str,
) -> tuple[str, str, tuple[str, ...]]:
    """段内选模：返回 ``(primary_monitor, mode, monitor_fallbacks)``。"""
    profiles = train_cfg.get("task_profiles") or {}
    profile = profiles.get(task) if isinstance(profiles, dict) else None
    if isinstance(profile, dict) and profile.get("checkpoint_monitor"):
        monitor = str(profile["checkpoint_monitor"]).strip()
        mode = checkpoint_mode_for_monitor(monitor, train_cfg.get("checkpoint_mode"))
        return monitor, mode, (monitor,)

    catalog = resolve_task_catalog(train_cfg)
    spec = catalog.get(task)
    metric = (spec.monitor if spec is not None else None) or KIND_MONITOR.get(
        spec.kind if spec is not None else "classification",
        "f1",
    )
    if metric == "f1":
        primary = f"val/f1_{task}"
        fallbacks = (primary, f"val/acc_{task}", f"val/macro_f1_{task}")
    elif metric == "nmi_within_domain":
        primary = f"val/macro_nmi_{task}"
        fallbacks = (primary, f"val/nmi_{task}", "val/nmi_within_domain")
    elif metric == "mse":
        primary = "val/mse_prediction" if task == "prediction" else f"val/mse_{task}"
        fallbacks = (primary, "val/mse", f"val/macro_mse_{task}")
        return primary, "min", fallbacks
    else:
        primary = f"val/{metric}_{task}"
        fallbacks = (primary,)
    mode = checkpoint_mode_for_monitor(primary, train_cfg.get("checkpoint_mode"))
    return primary, mode, fallbacks


def metric_is_improved(new: float, best: float | None, *, mode: str) -> bool:
    if best is None:
        return True
    if mode == "min":
        return new < best
    return new > best


def read_task_monitor_value(callback_metrics: dict[str, Any], fallbacks: tuple[str, ...]) -> float | None:
    for key in fallbacks:
        value = _metric_from_callbacks(callback_metrics, key)
        if value is not None:
            return value
    return None


_VAL_TASK_METRIC_RE = re.compile(
    r"^val/(?P<metric>acc|f1|miss_rate|nmi|mse|mae|macro_acc|macro_f1|macro_mse|ssim)_(?P<task>[^/]+)(?:/(?P<dataset>.+))?$"
)


def _task_metric_bases(task_report: dict[str, Any] | None) -> set[str]:
    bases: set[str] = set()
    for task in task_report or {}:
        name = str(task)
        bases.add(name[:-2] if name.endswith("_z") else name)
    return bases


def prune_inactive_task_val_metrics(
    callback_metrics: dict[str, Any],
    task_report: dict[str, Any] | None,
) -> dict[str, Any]:
    """``task_schedule`` 单任务验证时去掉其它任务残留在 callback_metrics 里的 val 指标。"""
    active = _task_metric_bases(task_report)
    if not active:
        return dict(callback_metrics)
    pruned: dict[str, Any] = {}
    for key, value in callback_metrics.items():
        name = _DATALOADER_IDX_RE.sub("", str(key))
        if not name.startswith("val/"):
            pruned[key] = value
            continue
        short = name[4:]
        if "/" in short:
            prefix = short.split("/", 1)[0]
            base = prefix[:-2] if prefix.endswith("_z") else prefix
            if base not in active:
                continue
            pruned[key] = value
            continue
        match = _VAL_TASK_METRIC_RE.match(name)
        if match:
            task = str(match.group("task"))
            base = task[:-2] if task.endswith("_z") else task
            if base not in active:
                continue
        pruned[key] = value
    return pruned


def extract_task_selection_metrics(callback_metrics: dict[str, Any]) -> dict[str, dict[str, float]]:
    """从 Lightning callback_metrics 抽出 specialist_geomean 所需的任务指标。"""
    out: dict[str, dict[str, float]] = {}

    def add(task: str, key: str, raw: Any) -> None:
        value = _as_finite_float(raw)
        if value is None:
            return
        out.setdefault(task, {})[key] = value

    for stored, raw in callback_metrics.items():
        name = _DATALOADER_IDX_RE.sub("", str(stored))
        if name in ("val/nmi", "val/nmi_within_domain", "val/macro_nmi_clustering"):
            add(
                "clustering",
                "nmi" if name == "val/nmi" else "nmi_within_domain",
                raw,
            )
            continue
        if name in ("val/mse_prediction", "val/mse"):
            add("prediction", "mse", raw)
            continue
        if name in ("val/mae_prediction", "val/mae"):
            add("prediction", "mae", raw)
            continue
        if name in ("val/mse_imputation", "val/impute_mse"):
            add("imputation", "mse", raw)
            continue
        if name in ("val/ssim", "val/ssim_prediction"):
            add("prediction", "ssim", raw)
            continue
        if name == "val/ssim_imputation":
            add("imputation", "ssim", raw)
            continue
        match = _VAL_TASK_METRIC_RE.match(name)
        if not match:
            continue
        metric = match.group("metric")
        task = match.group("task")
        dataset = match.group("dataset")
        # z_general 线性探针（val/acc_modulation_z）仅作诊断，不进 multitask geomean。
        if str(task).endswith("_z"):
            continue
        if metric == "macro_acc":
            add(task, "per_dataset_macro_acc", raw)
        elif metric == "macro_f1":
            add(task, "per_dataset_macro_f1", raw)
        elif metric == "macro_mse":
            add(task, "mse", raw)
        elif metric == "macro_ssim":
            add(task, "ssim", raw)
        elif dataset:
            add(task, f"{metric}/{dataset}", raw)
        else:
            add(task, metric, raw)

    for metrics in out.values():
        acc_keys = [key for key in metrics if key.startswith("acc/")]
        if acc_keys and "per_dataset_macro_acc" not in metrics:
            metrics["per_dataset_macro_acc"] = float(sum(metrics[key] for key in acc_keys) / len(acc_keys))
        f1_keys = [key for key in metrics if key.startswith("f1/")]
        if f1_keys and "per_dataset_macro_f1" not in metrics:
            metrics["per_dataset_macro_f1"] = float(sum(metrics[key] for key in f1_keys) / len(f1_keys))
    return out


def aggregate_specialist_geomean(
    callback_metrics: dict[str, Any],
    specialist_scores: dict[str, float] | None = None,
) -> float | None:
    task_metrics = extract_task_selection_metrics(callback_metrics)
    if not task_metrics:
        return None
    scores = specialist_scores or {name: 1.0 for name in task_metrics}
    value, _details = specialist_relative_geomean(task_metrics, scores)
    return float(value)


def aggregate_multitask_geomean(callback_metrics: dict[str, Any]) -> float | None:
    """stage2/downstream：任务指标几何平均，不使用 ``val/loss`` 均值。"""
    return aggregate_specialist_geomean(callback_metrics, None)


CLASSIFICATION_TASK_NAMES: tuple[str, ...] = ("ld_intrapulse", "ld_model", "tx_modulation")


def aggregate_classification_geomean(
    callback_metrics: dict[str, Any],
    *,
    task_names: tuple[str, ...] | None = None,
) -> float | None:
    """stage2 分类门控：仅调制/个体三类，不含聚类与 prediction。"""
    names = task_names or CLASSIFICATION_TASK_NAMES
    task_metrics = extract_task_selection_metrics(callback_metrics)
    filtered = {name: metrics for name, metrics in task_metrics.items() if name in names}
    if not filtered:
        return None
    try:
        score, _details = specialist_relative_geomean(
            filtered,
            {name: 1.0 for name in filtered},
        )
    except ValueError:
        return None
    return float(score)


def resolve_composite_exclude_tasks(train_cfg: dict[str, Any]) -> frozenset[str]:
    raw = train_cfg.get("composite_exclude_tasks") or []
    return frozenset(str(item) for item in raw)


def strip_state_prefix(state: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in state.items():
        name = str(key)
        if name.startswith("model."):
            name = name[6:]
        if name.startswith("module."):
            name = name[7:]
        out[name] = value
    return out


def extract_model_state_dict(ckpt: Any) -> dict[str, Any]:
    if not isinstance(ckpt, dict):
        raise TypeError("无法解析 checkpoint")
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state = ckpt["state_dict"]
    elif "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]
    else:
        state = ckpt
    return strip_state_prefix(state)


def load_init_from_checkpoint(model: Any, path: str | Path, *, strict: bool = False) -> Any:
    """只加载权重，不恢复优化器。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = remap_task_head_checkpoints(extract_model_state_dict(ckpt))
    loader = getattr(model, "load_weights", None)
    if callable(loader):
        return loader(state, strict=strict)
    return model.load_state_dict(state, strict=strict)


def first_existing_ckpt(directory: Path) -> Path | None:
    """目录内优先 last.ckpt（旧 run），否则 best.ckpt。"""
    for parts in (("ckpts", "last.ckpt"), ("last.ckpt",), ("ckpts", "best.ckpt"), ("best.ckpt",)):
        candidate = directory.joinpath(*parts)
        if candidate.is_file():
            return candidate
    return None


def resolve_run_and_ckpt(
    *,
    root: Path,
    stage: str,
    run_name: str | None,
    resume: str | None,
) -> tuple[Path, Path | None]:
    """解析 run 目录与续训 ckpt。``resume='auto'`` 时用 ``<run>/ckpts/best.ckpt``（兼容 last.ckpt）。"""
    experiments = root / "runs" / "experiments"
    if resume is None:
        name = run_name or f"{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return experiments / name, None

    ckpt: Path | None = None
    if resume not in ("auto", ""):
        ckpt = Path(resume)
        if not ckpt.is_absolute():
            ckpt = root / ckpt
        ckpt = ckpt.resolve()
        if ckpt.is_dir():
            found = first_existing_ckpt(ckpt)
            ckpt = found if found is not None else ckpt / "ckpts" / "best.ckpt"
        if not ckpt.is_file():
            raise FileNotFoundError(f"续训权重不存在: {ckpt}")
        if run_name:
            run_dir = experiments / run_name
        elif ckpt.parent.name == "ckpts":
            run_dir = ckpt.parent.parent
        else:
            run_dir = ckpt.parent
        return run_dir, ckpt

    if not run_name:
        raise ValueError("使用 --resume 且未给出 ckpt 路径时，必须同时指定 --run-name")
    run_dir = experiments / run_name
    ckpt = first_existing_ckpt(run_dir)
    if ckpt is None:
        raise FileNotFoundError(
            f"续训找不到 {run_dir / 'ckpts' / 'best.ckpt'}。"
            "请确认该 run 已写出 best.ckpt，或显式传入 --resume /path/to/best.ckpt"
        )
    return run_dir, ckpt


if ModelCheckpoint is not None:

    class ExceptionSafeModelCheckpoint(ModelCheckpoint):
        """ModelCheckpoint，但异常保存转到独立 ``on_exception.ckpt``，不覆盖 ``best.ckpt``。

        Lightning 自带的 ``save_on_exception`` 会把当前状态写到 ``filename``（本仓库为
        "best"），从而用当前（非最优）权重覆盖真正的最优权重。这里把异常转储重定向到
        ``on_exception.ckpt``：崩溃/中断时保留一个可 resume 的现场，同时绝不破坏 best。
        """

        def on_exception(self, trainer: Any, pl_module: Any, exception: BaseException) -> None:
            if not self._should_save_on_exception(trainer):
                return
            monitor_candidates = self._monitor_candidates(trainer)
            filepath = self.format_checkpoint_name(monitor_candidates, filename="on_exception")
            self._save_checkpoint(trainer, filepath)
            self._save_last_checkpoint(trainer, monitor_candidates)
            if rank_zero_info is not None:
                rank_zero_info(
                    f"An {type(exception).__name__} was raised with message: "
                    f"{str(exception)}, saved exception checkpoint to {filepath}"
                )


else:
    ExceptionSafeModelCheckpoint = None  # type: ignore[assignment, misc]


def make_model_checkpoint(
    dirpath: Path,
    *,
    monitor: str,
    mode: str,
    save_last: bool = False,
    save_best: bool = True,
) -> Any:
    if ModelCheckpoint is None:
        raise ImportError("需要 lightning 才能创建 ModelCheckpoint")
    dirpath.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "dirpath": str(dirpath),
        "save_on_exception": True,
        "save_weights_only": False,
        "auto_insert_metric_name": False,
        "enable_version_counter": False,
        "save_on_train_epoch_end": False,
        "every_n_epochs": 1,
        "verbose": True,
        "save_last": bool(save_last),
    }
    if save_best:
        kwargs.update(
            filename="best",
            monitor=monitor,
            mode=mode,
            save_top_k=1,
        )
    else:
        kwargs.update(
            filename="best",
            monitor=None,
            save_top_k=0,
        )
    return ExceptionSafeModelCheckpoint(**kwargs)


def _metric_float(value: Any) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class TaskCompositeCheckpointCallback(Callback):
    """``task_schedule`` stage2：段内按任务 monitor 保存 best 头，合并写入 ``best.ckpt``。"""

    def __init__(
        self,
        dirpath: Path,
        train_cfg: dict[str, Any],
        *,
        stage: str = "stage2",
    ) -> None:
        super().__init__()
        self.dirpath = Path(dirpath)
        self.train_cfg = train_cfg
        self.stage = str(stage)
        self.best_path = self.dirpath / "best.ckpt"
        self._task_bests: dict[str, dict[str, Any]] = {}
        self.best_model_path: str | None = None
        self.best_model_score: float | None = None

    def task_best_summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for task, payload in self._task_bests.items():
            out[task] = {
                "monitor": payload.get("monitor"),
                "score": payload.get("score"),
                "epoch": payload.get("epoch"),
                "mode": payload.get("mode"),
            }
        return out

    def composite_geomean(self) -> float | None:
        if not self._task_bests:
            return None
        exclude = resolve_composite_exclude_tasks(self.train_cfg)
        task_metrics: dict[str, dict[str, float]] = {}
        for task, payload in self._task_bests.items():
            if task in exclude:
                continue
            metric_name = str(payload.get("selection_metric") or "f1")
            if metric_name in ("mse", "recon_mse"):
                raw = float(payload.get("score", 0.0))
                task_metrics[task] = {"mse": raw}
            elif metric_name in ("nmi_within_domain", "mean_nmi", "macro_nmi", "nmi"):
                raw = float(payload.get("score", 0.0))
                task_metrics[task] = {"nmi_within_domain": raw, "nmi": raw}
            elif metric_name == "f1":
                task_metrics[task] = {"f1": float(payload.get("score", 0.0))}
            else:
                task_metrics[task] = {metric_name: float(payload.get("score", 0.0))}
        try:
            score, _details = specialist_relative_geomean(task_metrics, {name: 1.0 for name in task_metrics})
        except ValueError:
            return None
        return float(score)

    def _active_tasks(self, pl_module: Any) -> list[str]:
        tasks = list(self.train_cfg.get("active_train_tasks") or [])
        if tasks:
            return [str(task) for task in tasks]
        report = getattr(pl_module, "_last_val_report", None) or {}
        return [str(task) for task in report if not str(task).endswith("_z")]

    def _selection_metric_name(self, task: str) -> str:
        catalog = resolve_task_catalog(self.train_cfg)
        spec = catalog.get(task)
        if spec is not None and spec.monitor:
            return str(spec.monitor)
        return KIND_MONITOR.get(spec.kind if spec is not None else "classification", "f1")

    def _save_composite(self, trainer: Any, pl_module: Any) -> None:
        if not self._task_bests:
            return
        self.dirpath.mkdir(parents=True, exist_ok=True)
        live_state = {key: value.detach().cpu() for key, value in pl_module.state_dict().items()}
        merged = merge_stage2_task_states(
            live_state,
            {task: payload["state"] for task, payload in self._task_bests.items()},
        )
        backup = {key: value.detach().cpu().clone() for key, value in pl_module.state_dict().items()}
        pl_module.load_state_dict(merged, strict=False)
        try:
            connector = getattr(trainer, "_checkpoint_connector", None)
            if connector is not None and hasattr(connector, "dump_checkpoint"):
                checkpoint = connector.dump_checkpoint()
            else:
                checkpoint = {
                    "epoch": int(getattr(trainer, "current_epoch", 0)),
                    "global_step": int(getattr(trainer, "global_step", 0)),
                    "state_dict": pl_module.state_dict(),
                }
            if isinstance(checkpoint, dict):
                checkpoint["composite_task_best"] = self.task_best_summary()
                checkpoint["composite_geomean"] = self.composite_geomean()
            torch.save(checkpoint, self.best_path)
        finally:
            pl_module.load_state_dict(backup, strict=False)
        self.best_model_path = str(self.best_path)
        self.best_model_score = self.composite_geomean()
        if rank_zero_info is not None:
            rank_zero_info(
                f"TaskCompositeCheckpoint: wrote {self.best_path} tasks={sorted(self._task_bests)} "
                f"geomean={self.best_model_score}"
            )

    def on_validation_end(self, trainer: Any, pl_module: Any) -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        if self.stage != "stage2" or not self.train_cfg.get("task_schedule"):
            return
        metrics = dict(getattr(trainer, "callback_metrics", {}) or {})
        report = getattr(pl_module, "_last_val_report", None) or {}
        metrics = prune_inactive_task_val_metrics(metrics, report if isinstance(report, dict) else None)
        updated = False
        epoch = int(getattr(trainer, "current_epoch", 0))
        live_state = pl_module.state_dict()
        exclude_tasks = resolve_composite_exclude_tasks(self.train_cfg)
        eval_only = bool(self.train_cfg.get("active_eval_only"))
        for task in self._active_tasks(pl_module):
            if eval_only or task in exclude_tasks:
                continue
            _primary, mode, fallbacks = resolve_task_checkpoint_monitor(
                self.train_cfg,
                stage=self.stage,
                task=task,
            )
            raw = read_task_monitor_value(metrics, fallbacks)
            if raw is None and isinstance(report, dict) and task in report:
                info = report[task]
                if isinstance(info, dict):
                    selection = self._selection_metric_name(task)
                    if selection in ("mse", "recon_mse"):
                        raw = _as_finite_float(info.get("mse", info.get("mean_mse")))
                    elif selection in ("nmi_within_domain", "mean_nmi", "macro_nmi", "nmi"):
                        raw = _as_finite_float(info.get("mean_nmi", info.get("nmi")))
                    elif selection == "f1":
                        raw = _as_finite_float(info.get("f1", info.get("mean_f1")))
                    elif selection == "acc":
                        raw = _as_finite_float(info.get("acc", info.get("mean_acc")))
            if raw is None:
                continue
            prev = self._task_bests.get(task)
            prev_score = None if prev is None else prev.get("score")
            if not metric_is_improved(float(raw), None if prev_score is None else float(prev_score), mode=mode):
                continue
            partial = {
                key: tensor.detach().cpu().clone()
                for key, tensor in filter_stage2_task_state(live_state, task).items()
            }
            self._task_bests[task] = {
                "monitor": fallbacks[0],
                "mode": mode,
                "score": float(raw),
                "selection_metric": self._selection_metric_name(task),
                "epoch": epoch,
                "state": partial,
            }
            updated = True
            if rank_zero_info is not None:
                rank_zero_info(
                    f"TaskCompositeCheckpoint: task={task} improved {fallbacks[0]}={raw} epoch={epoch}"
                )
        if updated:
            self._save_composite(trainer, pl_module)


class TrainStateCallback(Callback):
    """把 epoch / step / best / last 路径写到 ``train_state.json``，便于续训与排查。"""

    def __init__(self, path: Path, *, monitor: str, stage: str) -> None:
        super().__init__()
        self.path = path
        self.monitor = monitor
        self.stage = stage

    def _checkpoint_callback(self, trainer: Any) -> Any | None:
        for callback in getattr(trainer, "callbacks", []):
            if ModelCheckpoint is not None and isinstance(callback, ModelCheckpoint):
                return callback
        return None

    def _composite_callback(self, trainer: Any) -> TaskCompositeCheckpointCallback | None:
        for callback in getattr(trainer, "callbacks", []):
            if isinstance(callback, TaskCompositeCheckpointCallback):
                return callback
        return None

    def write(self, trainer: Any, pl_module: Any, *, extra: dict[str, Any] | None = None) -> None:
        ckpt_cb = self._checkpoint_callback(trainer)
        composite = self._composite_callback(trainer)
        payload: dict[str, Any] = {
            "stage": self.stage,
            "epoch": int(getattr(trainer, "current_epoch", 0)),
            "global_step": int(getattr(trainer, "global_step", 0)),
            "monitor": self.monitor,
            "monitor_value": _metric_float(getattr(trainer, "callback_metrics", {}).get(self.monitor)),
            "best_model_path": getattr(ckpt_cb, "best_model_path", None) or None,
            "best_model_score": _metric_float(getattr(ckpt_cb, "best_model_score", None)),
            "last_model_path": getattr(ckpt_cb, "last_model_path", None) or None,
            "updated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        }
        if composite is not None:
            payload["best_model_path"] = str(composite.best_path) if composite.best_path.is_file() else None
            payload["best_model_score"] = composite.composite_geomean()
            payload["task_best"] = composite.task_best_summary()
        early_cb = None
        for callback in getattr(trainer, "callbacks", []):
            from resmamba_signal_model.training.task_schedule import TaskScheduleEarlyStoppingCallback

            if isinstance(callback, TaskScheduleEarlyStoppingCallback):
                early_cb = callback
                break
        if early_cb is not None and early_cb.stopped_sessions:
            payload["task_schedule_early_stopped"] = list(early_cb.stopped_sessions)
        if extra:
            payload.update(extra)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def on_validation_end(self, trainer: Any, pl_module: Any) -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        self.write(trainer, pl_module)

    def on_train_end(self, trainer: Any, pl_module: Any) -> None:
        self.write(trainer, pl_module)

    def on_exception(self, trainer: Any, pl_module: Any, exception: BaseException) -> None:
        self.write(
            trainer,
            pl_module,
            extra={"exception": type(exception).__name__, "exception_message": str(exception)},
        )
