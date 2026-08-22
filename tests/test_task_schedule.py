from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resmamba_signal_model.training.task_schedule import (
    expand_task_schedule_with_joint,
    replay_tasks_for_epoch,
    resolve_task_schedule,
    schedule_session_for_epoch,
    sources_for_tasks,
    total_schedule_epochs,
)


def test_resolve_task_schedule_and_epoch_mapping() -> None:
    cfg = {
        "tasks": ["modulation", "emitter", "clustering"],
        "task_pools": {
            "classification": ["a", "b"],
            "emitter": ["c", "d"],
            "clustering": ["e", "f"],
        },
        "task_schedule": [
            {"task": "modulation", "epochs": 2},
            {"tasks": ["emitter"], "epochs": 3},
            {"task": "clustering", "epochs": 1},
        ],
    }
    sessions = resolve_task_schedule(cfg)
    assert len(sessions) == 3
    assert total_schedule_epochs(sessions) == 6
    assert schedule_session_for_epoch(sessions, 0)["tasks"] == ["modulation"]
    assert schedule_session_for_epoch(sessions, 1)["tasks"] == ["modulation"]
    assert schedule_session_for_epoch(sessions, 2)["tasks"] == ["emitter"]
    assert schedule_session_for_epoch(sessions, 5)["tasks"] == ["clustering"]
    assert sources_for_tasks(cfg, ["modulation"]) == ["classification"]
    assert sources_for_tasks(cfg, ["emitter"]) == ["emitter"]


def test_expand_task_schedule_inserts_joint_after_each_solo() -> None:
    cfg = {
        "task_joint_epochs": 2,
        "task_schedule": [
            {"task": "modulation", "epochs": 3},
            {"task": "emitter", "epochs": 4},
        ],
    }
    base = resolve_task_schedule(cfg)
    expanded = expand_task_schedule_with_joint(base, cfg)
    assert len(expanded) == 4
    assert expanded[0]["tasks"] == ["modulation"] and expanded[0].get("joint") is False
    assert expanded[1]["joint"] is True and expanded[1]["tasks"] == ["modulation"]
    assert expanded[1]["epochs"] == 2
    assert expanded[2]["tasks"] == ["emitter"]
    assert expanded[3]["joint"] is True and expanded[3]["tasks"] == ["modulation", "emitter"]
    assert total_schedule_epochs(expanded) == 3 + 2 + 4 + 2
    assert schedule_session_for_epoch(expanded, 3)["joint"] is True
    assert schedule_session_for_epoch(expanded, 5)["tasks"] == ["emitter"]


def test_empty_schedule_keeps_mixed_training() -> None:
    assert resolve_task_schedule({}) == []
    assert resolve_task_schedule({"task_schedule": []}) == []


def test_active_filter_limits_train_and_val() -> None:
    from resmamba_signal_model.training.data_module import SignalDataModule

    cfg = {
        "synthetic": True,
        "seed": 0,
        "token_budget": 256,
        "steps_per_epoch": 2,
        "val_batches": 1,
        "num_workers": 0,
        "patch_size": 8,
        "tasks": ["modulation", "emitter"],
        "task_pools": {
            "classification": ["a", "b"],
            "emitter": ["c", "d"],
        },
        "task_schedule": [
            {"task": "modulation", "epochs": 1},
            {"task": "emitter", "epochs": 1},
        ],
        "min_ratio": 0.05,
    }
    dm = SignalDataModule(cfg, stage="stage2")
    dm.setup()
    assert set(dm._train_sets) == {"classification", "emitter"}
    assert set(dm._val_sets) == {"classification", "emitter"}
    dm.set_active_train_filter(["classification"])
    assert list(dm._filtered_train_sets()) == ["classification"]
    assert list(dm._filtered_val_sets()) == ["classification"]
    assert dm.val_source_names == ["classification"]
    _ = dm.val_dataloader()
    dm.set_active_train_filter(["emitter"])
    assert dm.val_source_names == ["emitter"]
    _ = dm.val_dataloader()
    assert getattr(dm, "_cached_val_key", None) == ("emitter",)


def test_replay_tasks_for_epoch_accumulates_prior_sessions() -> None:
    cfg = {
        "tasks": ["modulation", "emitter", "clustering"],
        "task_pools": {
            "classification": ["a"],
            "emitter": ["c"],
            "clustering": ["e"],
        },
        "task_schedule": [
            {"task": "modulation", "epochs": 2},
            {"task": "emitter", "epochs": 2},
            {"task": "clustering", "epochs": 1},
        ],
    }
    sessions = resolve_task_schedule(cfg)
    assert replay_tasks_for_epoch(sessions, 0) == []
    assert replay_tasks_for_epoch(sessions, 1) == []
    assert replay_tasks_for_epoch(sessions, 2) == ["modulation"]
    assert replay_tasks_for_epoch(sessions, 4) == ["modulation", "emitter"]


def test_replay_filter_mixes_train_but_not_val() -> None:
    from resmamba_signal_model.training.data_module import SignalDataModule

    cfg = {
        "synthetic": True,
        "seed": 0,
        "token_budget": 256,
        "steps_per_epoch": 2,
        "val_batches": 1,
        "num_workers": 0,
        "patch_size": 8,
        "tasks": ["modulation", "emitter"],
        "task_pools": {
            "classification": ["a", "b"],
            "emitter": ["c", "d"],
        },
        "task_schedule": [
            {"task": "modulation", "epochs": 1},
            {"task": "emitter", "epochs": 1},
        ],
        "replay_mix_ratio": 0.2,
        "min_ratio": 0.05,
    }
    dm = SignalDataModule(cfg, stage="stage2")
    dm.setup()
    dm.set_active_train_filter(["emitter"], replay_sources=["classification"])
    assert set(dm._filtered_train_sets()) == {"emitter", "classification"}
    assert list(dm._filtered_val_sets()) == ["emitter"]
    shares = dm._token_shares(["emitter", "classification"], train=True)
    assert sum(shares.values()) == dm.token_budget
    assert shares["classification"] < shares["emitter"]


def test_schedule_callback_sets_replay_and_refreshes_teacher() -> None:
    """epoch_end 必须先切下一任务，否则 reload 仍用旧源。"""
    from types import SimpleNamespace

    from resmamba_signal_model.training.data_module import SignalDataModule
    from resmamba_signal_model.training.task_schedule import TaskScheduleCallback

    cfg = {
        "synthetic": True,
        "seed": 0,
        "token_budget": 256,
        "steps_per_epoch": 2,
        "val_batches": 1,
        "num_workers": 0,
        "patch_size": 8,
        "tasks": ["modulation", "emitter"],
        "task_pools": {
            "classification": ["a", "b"],
            "emitter": ["c", "d"],
        },
        "task_schedule": [
            {"task": "modulation", "epochs": 2},
            {"task": "emitter", "epochs": 2},
        ],
        "replay_mix_ratio": 0.15,
        "uti_replay_weight": 0.25,
        "min_ratio": 0.05,
    }
    dm = SignalDataModule(cfg, stage="stage2")
    dm.setup()
    sessions = resolve_task_schedule(cfg)
    cb = TaskScheduleCallback(sessions, cfg)
    trainer = SimpleNamespace(datamodule=dm, current_epoch=0)
    pl_module = SimpleNamespace(refreshed=False)

    def _refresh() -> None:
        pl_module.refreshed = True

    pl_module.refresh_continual_teacher = _refresh
    pl_module.build_class_center_replay_memory = lambda tasks, datamodule=None: (
        datamodule.update_replay_memory({"classification": [0, 1, 2]}) or {"classification": 3}
        if tasks == ["modulation"]
        else {}
    )
    cb.on_fit_start(trainer, pl_module)
    assert cfg["active_train_sources"] == ["classification"]
    assert cfg["active_replay_sources"] == []
    assert list(dm._filtered_train_sets()) == ["classification"]
    # 第 0 轮结束仍属 modulation（epochs=2），预切 epoch=1 仍是 classification
    trainer.current_epoch = 0
    cb.on_train_epoch_end(trainer, pl_module)
    assert cfg["active_train_sources"] == ["classification"]
    # 第 1 轮结束预切 epoch=2 → emitter，随后 reload 应拿到 emitter + replay classification
    trainer.current_epoch = 1
    cb.on_train_epoch_end(trainer, pl_module)
    assert cfg["active_train_tasks"] == ["emitter"]
    assert cfg["active_train_sources"] == ["emitter"]
    assert cfg["active_replay_tasks"] == ["modulation"]
    assert cfg["active_replay_sources"] == ["classification"]
    assert set(dm._filtered_train_sets()) == {"emitter", "classification"}
    assert list(dm._filtered_val_sets()) == ["emitter"]
    assert pl_module.refreshed is True
    assert dm._replay_memory.get("classification") == [0, 1, 2]
    dataset, _ = dm._train_dataset_and_lengths(
        "classification",
        dm._train_sets["classification"],
        train=True,
    )
    from torch.utils.data import Subset

    assert isinstance(dataset, Subset)
    _ = dm.train_dataloader()  # reload 后应能按新 filter 建 loader


def test_joint_session_trains_all_completed_sources_without_replay() -> None:
    from types import SimpleNamespace

    from resmamba_signal_model.training.data_module import SignalDataModule
    from resmamba_signal_model.training.task_schedule import TaskScheduleCallback

    cfg = {
        "synthetic": True,
        "seed": 0,
        "token_budget": 256,
        "steps_per_epoch": 2,
        "val_batches": 1,
        "num_workers": 0,
        "patch_size": 8,
        "tasks": ["modulation", "emitter"],
        "task_pools": {
            "classification": ["a", "b"],
            "emitter": ["c", "d"],
        },
        "task_joint_epochs": 1,
        "replay_mix_ratio": 0.15,
        "min_ratio": 0.05,
    }
    base = [
        {"tasks": ["modulation"], "epochs": 1, "name": "modulation", "joint": False},
        {"name": "joint_modulation", "tasks": ["modulation"], "epochs": 1, "joint": True},
        {"tasks": ["emitter"], "epochs": 1, "name": "emitter", "joint": False},
    ]
    dm = SignalDataModule(cfg, stage="stage2")
    dm.setup()
    cb = TaskScheduleCallback(base, cfg)
    trainer = SimpleNamespace(datamodule=dm, current_epoch=0)
    cb.on_fit_start(trainer, None)
    assert cfg["active_joint_session"] is False
    trainer.current_epoch = 0
    cb.on_train_epoch_end(trainer, None)
    assert cfg["active_joint_session"] is True
    assert cfg["active_train_tasks"] == ["modulation"]
    assert set(cfg["active_train_sources"]) == {"classification"}
    assert cfg["active_replay_sources"] == []
    assert set(dm._filtered_train_sets()) == {"classification"}
    assert set(dm._filtered_val_sets()) == {"classification"}


def test_schedule_callback_prepares_next_epoch_before_reload() -> None:
    """epoch_end 必须先切下一任务，否则 reload 仍用旧源。"""
    from types import SimpleNamespace

    from resmamba_signal_model.training.data_module import SignalDataModule
    from resmamba_signal_model.training.task_schedule import TaskScheduleCallback

    cfg = {
        "synthetic": True,
        "seed": 0,
        "token_budget": 256,
        "steps_per_epoch": 2,
        "val_batches": 1,
        "num_workers": 0,
        "patch_size": 8,
        "tasks": ["modulation", "emitter"],
        "task_pools": {
            "classification": ["a", "b"],
            "emitter": ["c", "d"],
        },
        "task_schedule": [
            {"task": "modulation", "epochs": 2},
            {"task": "emitter", "epochs": 2},
        ],
        "min_ratio": 0.05,
    }
    dm = SignalDataModule(cfg, stage="stage2")
    dm.setup()
    sessions = resolve_task_schedule(cfg)
    cb = TaskScheduleCallback(sessions, cfg)
    trainer = SimpleNamespace(datamodule=dm, current_epoch=0)
    cb.on_fit_start(trainer, None)
    assert cfg["active_train_sources"] == ["classification"]
    assert list(dm._filtered_train_sets()) == ["classification"]
    # 第 0 轮结束仍属 modulation（epochs=2），预切 epoch=1 仍是 classification
    trainer.current_epoch = 0
    cb.on_train_epoch_end(trainer, None)
    assert cfg["active_train_sources"] == ["classification"]
    # 第 1 轮结束预切 epoch=2 → emitter，随后 reload 应拿到 emitter
    trainer.current_epoch = 1
    cb.on_train_epoch_end(trainer, None)
    assert cfg["active_train_tasks"] == ["emitter"]
    assert cfg["active_train_sources"] == ["emitter"]
    assert list(dm._filtered_train_sets()) == ["emitter"]
    _ = dm.train_dataloader()  # reload 后应能按新 filter 建 loader
