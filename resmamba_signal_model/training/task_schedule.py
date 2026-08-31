from __future__ import annotations

from typing import Any

from resmamba_signal_model.training.early_stopping import EarlyStopping
from resmamba_signal_model.training.task_catalog import resolve_task_catalog


def resolve_task_schedule(train_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """解析 ``task_schedule``：按任务顺序分阶段训练。

    每项支持 ``task`` / ``tasks`` 与 ``epochs``。缺省时返回空列表（保持多任务混训）。
    """
    raw = train_cfg.get("task_schedule")
    if not isinstance(raw, list) or not raw:
        return []
    sessions: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            sessions.append({"tasks": [str(item)], "epochs": 1})
            continue
        if not isinstance(item, dict):
            raise TypeError(f"task_schedule 条目必须是 dict 或 str，当前为 {type(item)!r}")
        tasks = item.get("tasks")
        if tasks is None and item.get("task") is not None:
            tasks = [item.get("task")]
        if isinstance(tasks, str):
            tasks = [tasks]
        if not tasks:
            raise ValueError(f"task_schedule 条目缺少 task/tasks: {item}")
        names = [str(t).strip() for t in tasks if str(t).strip()]
        if not names:
            raise ValueError(f"task_schedule 条目任务名为空: {item}")
        session: dict[str, Any] = {
            "name": str(item.get("name") or "+".join(names)),
            "tasks": names,
            "epochs": max(1, int(item.get("epochs", 1) or 1)),
        }
        if item.get("eval_only") is not None:
            session["eval_only"] = bool(item.get("eval_only"))
        if item.get("early_stopping_patience") is not None:
            session["early_stopping_patience"] = int(item["early_stopping_patience"])
        if item.get("early_stopping_min_delta") is not None:
            session["early_stopping_min_delta"] = float(item["early_stopping_min_delta"])
        sessions.append(session)
    return sessions


def expand_task_schedule_with_joint(
    sessions: list[dict[str, Any]],
    train_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """在单任务会话后插入联合训练段（``task_joint_epochs``）。

    仅当已完成任务数 >= 2 时插入；单任务 joint 与 solo 等价，跳过。
    """
    joint_epochs = int(train_cfg.get("task_joint_epochs", 0) or 0)
    if joint_epochs <= 0 or not sessions:
        return [dict(item) for item in sessions]
    out: list[dict[str, Any]] = []
    completed: list[str] = []
    for session in sessions:
        solo = dict(session)
        solo.setdefault("joint", False)
        out.append(solo)
        for task in solo.get("tasks") or []:
            name = str(task).strip()
            if name and name not in completed:
                completed.append(name)
        if len(completed) >= 2:
            out.append(
                {
                    "name": f"joint_{'+'.join(completed)}",
                    "tasks": list(completed),
                    "epochs": joint_epochs,
                    "joint": True,
                }
            )
    return out


def total_schedule_epochs(sessions: list[dict[str, Any]], *, default_epochs: int = 1) -> int:
    if not sessions:
        return int(default_epochs)
    return int(sum(int(s.get("epochs", 1) or 1) for s in sessions))


def schedule_session_for_epoch(
    sessions: list[dict[str, Any]],
    epoch: int,
    *,
    default_epochs: int = 1,
) -> dict[str, Any] | None:
    """0-indexed ``epoch`` 落在哪个会话；超出则返回最后一节。"""
    if not sessions:
        return None
    cursor = 0
    last = sessions[-1]
    for session in sessions:
        span = max(1, int(session.get("epochs", default_epochs) or default_epochs))
        if epoch < cursor + span:
            return session
        cursor += span
        last = session
    return last


def slice_task_schedule_from(
    sessions: list[dict[str, Any]],
    start_task: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """从 ``start_task`` 起截断 schedule，并返回此前已完成任务（供 replay）。"""
    key = str(start_task).strip()
    if not key:
        raise ValueError("start_task 不能为空")
    completed: list[str] = []
    sliced: list[dict[str, Any]] = []
    found = False
    for session in sessions:
        names = [str(t).strip() for t in session.get("tasks") or [] if str(t).strip()]
        if not found:
            if key in names:
                found = True
                sliced.append(dict(session))
            else:
                for name in names:
                    if name not in completed:
                        completed.append(name)
            continue
        sliced.append(dict(session))
    if not found:
        raise ValueError(f"start_task={key!r} 不在 task_schedule 中")
    if not sliced:
        raise ValueError(f"start_task={key!r} 截断后 schedule 为空")
    return sliced, completed


def replay_tasks_for_epoch(
    sessions: list[dict[str, Any]],
    epoch: int,
    *,
    default_epochs: int = 1,
) -> list[str]:
    """返回 ``epoch`` 所在会话之前已完成会话的任务名（保序去重）。"""
    if not sessions:
        return []
    cursor = 0
    completed: list[str] = []
    for session in sessions:
        span = max(1, int(session.get("epochs", default_epochs) or default_epochs))
        if int(epoch) < cursor + span:
            break
        for task in session.get("tasks") or []:
            name = str(task).strip()
            if name and name not in completed:
                completed.append(name)
        cursor += span
    return completed


def next_session_start_epoch(
    sessions: list[dict[str, Any]],
    epoch: int,
    *,
    default_epochs: int = 1,
) -> int:
    """返回 ``epoch`` 所在会话结束后的下一 session 起始 epoch（0-indexed）。"""
    cursor = 0
    for session in sessions:
        span = max(1, int(session.get("epochs", default_epochs) or default_epochs))
        end = cursor + span
        if int(epoch) < end:
            return end
        cursor = end
    return cursor


def session_bounds_for_epoch(
    sessions: list[dict[str, Any]],
    epoch: int,
    *,
    default_epochs: int = 1,
) -> tuple[int, int, dict[str, Any] | None]:
    """返回 ``(start, end_exclusive, session)``。"""
    cursor = 0
    for session in sessions:
        span = max(1, int(session.get("epochs", default_epochs) or default_epochs))
        end = cursor + span
        if int(epoch) < end:
            return cursor, end, session
        cursor = end
    last = sessions[-1] if sessions else None
    return cursor, cursor, last


def advance_trainer_to_epoch(trainer: Any, target_epoch: int) -> None:
    """在 ``on_train_epoch_end`` 内调用，使下一轮训练从 ``target_epoch`` 开始。"""
    target = int(target_epoch)
    fit_loop = getattr(trainer, "fit_loop", None)
    if fit_loop is None:
        return
    progress = getattr(fit_loop, "epoch_progress", None)
    if progress is None or getattr(progress, "current", None) is None:
        return
    current = progress.current
    current_epoch = int(getattr(trainer, "current_epoch", 0) or 0)
    skipped = max(0, target - current_epoch - 1)
    current.processed = target
    # ``increment_completed`` 会在 callback 之后执行，因此先设为 target-1。
    current.completed = target - 1
    if skipped > 0:
        total = progress.total
        total.processed += skipped
        total.completed += skipped
        total.ready += skipped
        if hasattr(total, "started"):
            total.started += skipped


def resolve_session_early_stopping_patience(session: dict[str, Any], train_cfg: dict[str, Any]) -> int:
    raw = session.get("early_stopping_patience")
    if raw is not None:
        return int(raw)
    return int(train_cfg.get("early_stopping_patience") or 0)


def resolve_session_early_stopping_min_delta(session: dict[str, Any], train_cfg: dict[str, Any]) -> float:
    raw = session.get("early_stopping_min_delta")
    if raw is not None:
        return float(raw)
    return float(train_cfg.get("early_stopping_min_delta") or 0.0)


def task_schedule_early_stopping_enabled(
    train_cfg: dict[str, Any],
    *,
    stage: str,
    sessions: list[dict[str, Any]],
) -> bool:
    if stage != "stage2" or not sessions:
        return False
    default_patience = int(train_cfg.get("early_stopping_patience") or 0)
    if default_patience > 0:
        return True
    return any(resolve_session_early_stopping_patience(session, train_cfg) > 0 for session in sessions)


def find_task_schedule_callback(trainer: Any) -> "TaskScheduleCallback | None":
    for callback in getattr(trainer, "callbacks", []):
        if isinstance(callback, TaskScheduleCallback):
            return callback
    return None


def sources_for_tasks(train_cfg: dict[str, Any], tasks: list[str]) -> list[str]:
    catalog = resolve_task_catalog(train_cfg)
    sources: list[str] = []
    for name in tasks:
        spec = catalog.get(name)
        if spec is not None:
            sources.append(spec.source)
        else:
            sources.append(name)
    # 保序去重
    seen: set[str] = set()
    out: list[str] = []
    for src in sources:
        if src not in seen:
            seen.add(src)
            out.append(src)
    return out


try:
    from lightning.pytorch.callbacks import Callback
except ImportError:  # pragma: no cover
    Callback = object  # type: ignore[misc, assignment]


class TaskScheduleEarlyStoppingCallback(Callback):
    """``task_schedule`` 段内早停：触发后跳过本段剩余 epoch，进入下一任务。"""

    def __init__(
        self,
        sessions: list[dict[str, Any]],
        train_cfg: dict[str, Any],
        *,
        stage: str = "stage2",
    ) -> None:
        self.sessions = list(sessions)
        self.train_cfg = train_cfg
        self.stage = str(stage)
        self._session_key: tuple[str, ...] | None = None
        self._stopper: EarlyStopping | None = None
        self._monitor_fallbacks: tuple[str, ...] = ()
        self._pending_jump: int | None = None
        self.stopped_sessions: list[str] = []

    def _reset_stopper(self, session: dict[str, Any], task: str) -> None:
        from resmamba_signal_model.training.checkpointing import resolve_task_checkpoint_monitor

        patience = resolve_session_early_stopping_patience(session, self.train_cfg)
        if patience <= 0 or bool(session.get("joint")) or bool(session.get("eval_only")):
            self._stopper = None
            self._monitor_fallbacks = ()
            return
        _primary, mode, fallbacks = resolve_task_checkpoint_monitor(
            self.train_cfg,
            stage=self.stage,
            task=task,
        )
        self._stopper = EarlyStopping(
            patience=patience,
            min_delta=resolve_session_early_stopping_min_delta(session, self.train_cfg),
            mode=mode,
        )
        self._monitor_fallbacks = tuple(fallbacks)

    def _read_score(self, trainer: Any, pl_module: Any, task: str) -> float | None:
        from resmamba_signal_model.training.checkpointing import (
            _as_finite_float,
            prune_inactive_task_val_metrics,
            read_task_monitor_value,
            resolve_task_checkpoint_monitor,
        )
        from resmamba_signal_model.training.task_catalog import KIND_MONITOR

        metrics = dict(getattr(trainer, "callback_metrics", {}) or {})
        report = getattr(pl_module, "_last_val_report", None) or {}
        metrics = prune_inactive_task_val_metrics(metrics, report if isinstance(report, dict) else None)
        raw = read_task_monitor_value(metrics, self._monitor_fallbacks)
        if raw is not None:
            return raw
        if not isinstance(report, dict) or task not in report:
            return None
        info = report[task]
        if not isinstance(info, dict):
            return None
        catalog = resolve_task_catalog(self.train_cfg)
        spec = catalog.get(task)
        selection = (spec.monitor if spec is not None else None) or KIND_MONITOR.get(
            spec.kind if spec is not None else "classification",
            "f1",
        )
        if selection in ("mse", "recon_mse"):
            return _as_finite_float(info.get("mse", info.get("mean_mse")))
        if selection in ("nmi_within_domain", "mean_nmi", "macro_nmi", "nmi"):
            return _as_finite_float(info.get("mean_nmi", info.get("nmi")))
        if selection == "f1":
            return _as_finite_float(info.get("f1", info.get("mean_f1")))
        if selection == "acc":
            return _as_finite_float(info.get("acc", info.get("mean_acc")))
        return None

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        if self.stage != "stage2":
            return
        epoch = int(getattr(trainer, "current_epoch", 0) or 0)
        session = schedule_session_for_epoch(self.sessions, epoch)
        if session is None or bool(session.get("joint")):
            self._session_key = None
            self._stopper = None
            return
        tasks = [str(t) for t in session.get("tasks") or []]
        if not tasks:
            return
        key = (str(session.get("name") or "+".join(tasks)), tuple(tasks))
        if key != self._session_key:
            self._session_key = key
            self._reset_stopper(session, tasks[0])

    def on_validation_end(self, trainer: Any, pl_module: Any) -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        if self.stage != "stage2" or self._stopper is None or not self._stopper.enabled:
            return
        epoch = int(getattr(trainer, "current_epoch", 0) or 0)
        session = schedule_session_for_epoch(self.sessions, epoch)
        if session is None or bool(session.get("joint")):
            return
        tasks = [str(t) for t in session.get("tasks") or []]
        if not tasks:
            return
        score = self._read_score(trainer, pl_module, tasks[0])
        if score is None:
            return
        if not self._stopper.step(float(score)):
            return
        next_start = next_session_start_epoch(self.sessions, epoch)
        self._pending_jump = next_start
        name = str(session.get("name") or tasks[0])
        self.stopped_sessions.append(name)
        import logging

        logging.getLogger("resmamba").info(
            "task_schedule early_stop session=%s epoch=%s jump_to=%s score=%s monitor=%s",
            name,
            epoch,
            next_start,
            score,
            self._monitor_fallbacks[0] if self._monitor_fallbacks else "",
        )
        print(
            f"task_schedule early_stop session={name} epoch={epoch} jump_to={next_start} "
            f"score={score} monitor={self._monitor_fallbacks[0] if self._monitor_fallbacks else ''}",
            flush=True,
        )

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        if self._pending_jump is None:
            return
        target = int(self._pending_jump)
        self._pending_jump = None
        total = total_schedule_epochs(self.sessions)
        if target >= total:
            trainer.should_stop = True
            return
        schedule_cb = find_task_schedule_callback(trainer)
        if schedule_cb is not None:
            schedule_cb._apply(pl_module, trainer, epoch=target)
        advance_trainer_to_epoch(trainer, target)
        self._session_key = None


class TaskScheduleCallback(Callback):
    """按 epoch 切换当前任务的 train/val 数据源，配合 dataloader reload。

    Lightning 在 ``on_train_epoch_start`` 之前就会按 ``reload_dataloaders_every_n_epochs``
    重建 loader，因此必须在上一轮 ``on_train_epoch_end`` 就切好下一 epoch 的 filter，
    否则会出现 allow=emitter 但 batch 仍是 classification。
    """

    def __init__(
        self,
        sessions: list[dict[str, Any]],
        train_cfg: dict[str, Any],
        *,
        stage: str | None = None,
    ) -> None:
        self.sessions = list(sessions)
        self.train_cfg = train_cfg
        self.stage = str(stage or train_cfg.get("stage") or "")
        self._last_key: tuple[str, ...] | None = None
        raw_limit = train_cfg.get("limit_train_batches")
        if raw_limit is None:
            raw_limit = train_cfg.get("steps_per_epoch")
        self._default_limit_train_batches = raw_limit

    def _set_limit_train_batches(self, trainer: Any, limit: int | float | None) -> None:
        if trainer is None:
            return
        value = limit if limit is not None else self._default_limit_train_batches
        if value is not None:
            trainer.limit_train_batches = int(value)
        loop = getattr(trainer, "fit_loop", None)
        if loop is not None and hasattr(loop, "epoch_loop"):
            loop.epoch_loop.limit_train_batches = trainer.limit_train_batches

    def _apply(self, pl_module: Any, trainer: Any, *, epoch: int) -> None:
        session = schedule_session_for_epoch(self.sessions, int(epoch))
        if session is None:
            return
        tasks = [str(t) for t in session.get("tasks") or []]
        sources = sources_for_tasks(self.train_cfg, tasks)
        is_joint = bool(session.get("joint"))
        eval_only = bool(session.get("eval_only"))
        key = ("joint",) + tuple(tasks) if is_joint else tuple(tasks)
        if eval_only:
            key = ("eval_only",) + key
        replay_ratio = float(self.train_cfg.get("replay_mix_ratio", 0.0) or 0.0)
        if is_joint:
            replay_tasks: list[str] = []
            replay_sources: list[str] = []
        else:
            replay_tasks = list(self.train_cfg.get("replay_completed_tasks") or [])
            for name in replay_tasks_for_epoch(self.sessions, int(epoch)):
                if name not in replay_tasks:
                    replay_tasks.append(name)
            replay_sources = (
                sources_for_tasks(self.train_cfg, replay_tasks) if replay_ratio > 0 and replay_tasks else []
            )
        self.train_cfg["active_train_tasks"] = list(tasks)
        self.train_cfg["active_train_sources"] = list(sources)
        self.train_cfg["active_replay_tasks"] = list(replay_tasks)
        self.train_cfg["active_replay_sources"] = list(replay_sources)
        self.train_cfg["active_joint_session"] = is_joint
        self.train_cfg["active_eval_only"] = eval_only
        if eval_only:
            self._set_limit_train_batches(trainer, 1)
        else:
            self._set_limit_train_batches(trainer, self._default_limit_train_batches)
        dm = getattr(trainer, "datamodule", None)
        if dm is not None and hasattr(dm, "set_active_train_filter"):
            dm.set_active_train_filter(sources, replay_sources=replay_sources or None)
        if key != self._last_key:
            import logging

            phase = "eval_only" if eval_only else ("joint" if is_joint else "solo")
            logging.getLogger("resmamba").info(
                "task_schedule epoch=%s session=%s phase=%s tasks=%s sources=%s replay=%s (train+val)",
                epoch,
                session.get("name"),
                phase,
                tasks,
                sources,
                replay_sources,
            )
            print(
                f"task_schedule epoch={epoch} session={session.get('name')} phase={phase} "
                f"tasks={tasks} sources={sources} replay={replay_sources} (train+val)",
                flush=True,
            )
            if (
                not is_joint
                and not eval_only
                and self.stage == "stage2"
                and pl_module is not None
                and tasks
            ):
                from resmamba_signal_model.training.freeze import apply_stage_freeze

                apply_stage_freeze(
                    pl_module.model,
                    "stage2",
                    task=tasks[0] if len(tasks) == 1 else None,
                    train_cfg=self.train_cfg,
                )
                sync = getattr(pl_module, "sync_optimizer_trainable_params", None)
                if callable(sync):
                    sync()
            if (
                not is_joint
                and self._last_key is not None
                and replay_ratio > 0
                and replay_sources
                and pl_module is not None
            ):
                refresh = getattr(pl_module, "refresh_continual_teacher", None)
                if callable(refresh):
                    refresh()
                build = getattr(pl_module, "build_class_center_replay_memory", None)
                if callable(build) and replay_tasks:
                    counts = build(list(replay_tasks), datamodule=getattr(trainer, "datamodule", None))
                    if counts:
                        import logging

                        logging.getLogger("resmamba").info(
                            "replay class_center memory=%s tasks=%s",
                            counts,
                            replay_tasks,
                        )
            self._last_key = key

    def setup(self, trainer: Any, pl_module: Any, stage: str | None = None) -> None:
        del stage
        self._apply(pl_module, trainer, epoch=0)

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        self._apply(pl_module, trainer, epoch=int(getattr(trainer, "current_epoch", 0) or 0))
        completed = [str(t) for t in (self.train_cfg.get("replay_completed_tasks") or [])]
        replay_ratio = float(self.train_cfg.get("replay_mix_ratio", 0.0) or 0.0)
        if completed and replay_ratio > 0 and pl_module is not None:
            refresh = getattr(pl_module, "refresh_continual_teacher", None)
            if callable(refresh):
                refresh()
            build = getattr(pl_module, "build_class_center_replay_memory", None)
            if callable(build):
                counts = build(completed, datamodule=getattr(trainer, "datamodule", None))
                if counts:
                    import logging

                    logging.getLogger("resmamba").info(
                        "replay bootstrap from completed_tasks=%s memory=%s",
                        completed,
                        counts,
                    )

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        # 与当前 epoch 对齐（resume / 首轮）；正常跨 epoch 切任务靠 epoch_end 预切。
        self._apply(pl_module, trainer, epoch=int(getattr(trainer, "current_epoch", 0) or 0))

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        # 验证结束后、下一轮 reload dataloader 之前，切到下一 epoch 的任务。
        nxt = int(getattr(trainer, "current_epoch", 0) or 0) + 1
        self._apply(pl_module, trainer, epoch=nxt)
