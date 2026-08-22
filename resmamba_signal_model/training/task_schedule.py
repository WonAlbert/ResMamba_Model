from __future__ import annotations

from typing import Any

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
        sessions.append(
            {
                "name": str(item.get("name") or "+".join(names)),
                "tasks": names,
                "epochs": max(1, int(item.get("epochs", 1) or 1)),
            }
        )
    return sessions


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


class TaskScheduleCallback(Callback):
    """按 epoch 切换当前任务的 train/val 数据源，配合 dataloader reload。

    Lightning 在 ``on_train_epoch_start`` 之前就会按 ``reload_dataloaders_every_n_epochs``
    重建 loader，因此必须在上一轮 ``on_train_epoch_end`` 就切好下一 epoch 的 filter，
    否则会出现 allow=emitter 但 batch 仍是 classification。
    """

    def __init__(self, sessions: list[dict[str, Any]], train_cfg: dict[str, Any]) -> None:
        self.sessions = list(sessions)
        self.train_cfg = train_cfg
        self._last_key: tuple[str, ...] | None = None

    def _apply(self, pl_module: Any, trainer: Any, *, epoch: int) -> None:
        session = schedule_session_for_epoch(self.sessions, int(epoch))
        if session is None:
            return
        tasks = [str(t) for t in session.get("tasks") or []]
        sources = sources_for_tasks(self.train_cfg, tasks)
        key = tuple(tasks)
        replay_ratio = float(self.train_cfg.get("replay_mix_ratio", 0.0) or 0.0)
        replay_tasks = replay_tasks_for_epoch(self.sessions, int(epoch))
        replay_sources = sources_for_tasks(self.train_cfg, replay_tasks) if replay_ratio > 0 and replay_tasks else []
        self.train_cfg["active_train_tasks"] = list(tasks)
        self.train_cfg["active_train_sources"] = list(sources)
        self.train_cfg["active_replay_tasks"] = list(replay_tasks)
        self.train_cfg["active_replay_sources"] = list(replay_sources)
        dm = getattr(trainer, "datamodule", None)
        if dm is not None and hasattr(dm, "set_active_train_filter"):
            dm.set_active_train_filter(sources, replay_sources=replay_sources or None)
        if key != self._last_key:
            import logging

            logging.getLogger("resmamba").info(
                "task_schedule epoch=%s session=%s tasks=%s sources=%s replay=%s (train+val)",
                epoch,
                session.get("name"),
                tasks,
                sources,
                replay_sources,
            )
            print(
                f"task_schedule epoch={epoch} session={session.get('name')} "
                f"tasks={tasks} sources={sources} replay={replay_sources} (train+val)",
                flush=True,
            )
            if (
                self._last_key is not None
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

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        # 与当前 epoch 对齐（resume / 首轮）；正常跨 epoch 切任务靠 epoch_end 预切。
        self._apply(pl_module, trainer, epoch=int(getattr(trainer, "current_epoch", 0) or 0))

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        # 验证结束后、下一轮 reload dataloader 之前，切到下一 epoch 的任务。
        nxt = int(getattr(trainer, "current_epoch", 0) or 0) + 1
        self._apply(pl_module, trainer, epoch=nxt)
