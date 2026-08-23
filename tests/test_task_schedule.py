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
        "tasks": ["tx_modulation", "ld_model", "ld_clustering"],
        "task_pools": {
            "tx_modulation": ["a", "b"],
            "ld_model": ["c", "d"],
            "ld_clustering": ["e", "f"],
        },
        "task_schedule": [
            {"task": "tx_modulation", "epochs": 2},
            {"tasks": ["ld_model"], "epochs": 3},
            {"task": "ld_clustering", "epochs": 1},
        ],
    }
    sessions = resolve_task_schedule(cfg)
    assert len(sessions) == 3
    assert total_schedule_epochs(sessions) == 6
    assert schedule_session_for_epoch(sessions, 0)["tasks"] == ["tx_modulation"]
    assert schedule_session_for_epoch(sessions, 1)["tasks"] == ["tx_modulation"]
    assert schedule_session_for_epoch(sessions, 2)["tasks"] == ["ld_model"]
    assert schedule_session_for_epoch(sessions, 5)["tasks"] == ["ld_clustering"]
    assert sources_for_tasks(cfg, ["tx_modulation"]) == ["tx_modulation"]
    assert sources_for_tasks(cfg, ["ld_model"]) == ["ld_model"]


def test_expand_task_schedule_inserts_joint_after_each_solo() -> None:
    cfg = {
        "task_joint_epochs": 2,
        "task_schedule": [
            {"task": "tx_modulation", "epochs": 3},
            {"task": "ld_model", "epochs": 4},
        ],
    }
    base = resolve_task_schedule(cfg)
    expanded = expand_task_schedule_with_joint(base, cfg)
    assert len(expanded) == 3
    assert expanded[0]["tasks"] == ["tx_modulation"] and expanded[0].get("joint") is False
    assert expanded[1]["tasks"] == ["ld_model"] and expanded[1].get("joint") is False
    assert expanded[2]["joint"] is True and expanded[2]["tasks"] == ["tx_modulation", "ld_model"]
    assert expanded[2]["epochs"] == 2
    assert total_schedule_epochs(expanded) == 3 + 4 + 2
    assert schedule_session_for_epoch(expanded, 7)["joint"] is True


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
        "tasks": ["tx_modulation", "ld_model"],
        "task_pools": {
            "tx_modulation": ["a", "b"],
            "ld_model": ["c", "d"],
        },
        "task_schedule": [
            {"task": "tx_modulation", "epochs": 1},
            {"task": "ld_model", "epochs": 1},
        ],
        "min_ratio": 0.05,
    }
    dm = SignalDataModule(cfg, stage="stage2")
    dm.setup()
    assert set(dm._train_sets) == {"tx_modulation", "ld_model"}
    assert set(dm._val_sets) == {"tx_modulation", "ld_model"}
    dm.set_active_train_filter(["tx_modulation"])
    assert list(dm._filtered_train_sets()) == ["tx_modulation"]
    assert list(dm._filtered_val_sets()) == ["tx_modulation"]
    assert dm.val_source_names == ["tx_modulation"]
    _ = dm.val_dataloader()
    dm.set_active_train_filter(["ld_model"])
    assert dm.val_source_names == ["ld_model"]
    _ = dm.val_dataloader()
    assert getattr(dm, "_cached_val_key", None) == ("ld_model",)


def test_replay_tasks_for_epoch_accumulates_prior_sessions() -> None:
    cfg = {
        "tasks": ["tx_modulation", "ld_model", "ld_clustering"],
        "task_pools": {
            "tx_modulation": ["a"],
            "ld_model": ["c"],
            "ld_clustering": ["e"],
        },
        "task_schedule": [
            {"task": "tx_modulation", "epochs": 2},
            {"task": "ld_model", "epochs": 2},
            {"task": "ld_clustering", "epochs": 1},
        ],
    }
    sessions = resolve_task_schedule(cfg)
    assert replay_tasks_for_epoch(sessions, 0) == []
    assert replay_tasks_for_epoch(sessions, 1) == []
    assert replay_tasks_for_epoch(sessions, 2) == ["tx_modulation"]
    assert replay_tasks_for_epoch(sessions, 4) == ["tx_modulation", "ld_model"]


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
        "tasks": ["tx_modulation", "ld_model"],
        "task_pools": {
            "tx_modulation": ["a", "b"],
            "ld_model": ["c", "d"],
        },
        "task_schedule": [
            {"task": "tx_modulation", "epochs": 1},
            {"task": "ld_model", "epochs": 1},
        ],
        "replay_mix_ratio": 0.2,
        "min_ratio": 0.05,
    }
    dm = SignalDataModule(cfg, stage="stage2")
    dm.setup()
    dm.set_active_train_filter(["ld_model"], replay_sources=["tx_modulation"])
    assert set(dm._filtered_train_sets()) == {"ld_model", "tx_modulation"}
    assert list(dm._filtered_val_sets()) == ["ld_model"]
    shares = dm._token_shares(["ld_model", "tx_modulation"], train=True)
    assert sum(shares.values()) == dm.token_budget
    assert shares["tx_modulation"] < shares["ld_model"]


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
        "tasks": ["tx_modulation", "ld_model"],
        "task_pools": {
            "tx_modulation": ["a", "b"],
            "ld_model": ["c", "d"],
        },
        "task_schedule": [
            {"task": "tx_modulation", "epochs": 2},
            {"task": "ld_model", "epochs": 2},
        ],
        "replay_mix_ratio": 0.15,
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
        datamodule.update_replay_memory({"tx_modulation": [0, 1, 2]}) or {"tx_modulation": 3}
        if tasks == ["tx_modulation"]
        else {}
    )
    cb.on_fit_start(trainer, pl_module)
    assert cfg["active_train_sources"] == ["tx_modulation"]
    assert cfg["active_replay_sources"] == []
    assert list(dm._filtered_train_sets()) == ["tx_modulation"]
    # 第 0 轮结束仍属 tx_modulation（epochs=2），预切 epoch=1 仍是 tx_modulation
    trainer.current_epoch = 0
    cb.on_train_epoch_end(trainer, pl_module)
    assert cfg["active_train_sources"] == ["tx_modulation"]
    # 第 1 轮结束预切 epoch=2 → ld_model，随后 reload 应拿到 ld_model + replay tx_modulation
    trainer.current_epoch = 1
    cb.on_train_epoch_end(trainer, pl_module)
    assert cfg["active_train_tasks"] == ["ld_model"]
    assert cfg["active_train_sources"] == ["ld_model"]
    assert cfg["active_replay_tasks"] == ["tx_modulation"]
    assert cfg["active_replay_sources"] == ["tx_modulation"]
    assert set(dm._filtered_train_sets()) == {"ld_model", "tx_modulation"}
    assert list(dm._filtered_val_sets()) == ["ld_model"]
    assert pl_module.refreshed is True
    assert dm._replay_memory.get("tx_modulation") == [0, 1, 2]
    dataset, _ = dm._train_dataset_and_lengths(
        "tx_modulation",
        dm._train_sets["tx_modulation"],
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
        "tasks": ["tx_modulation", "ld_model"],
        "task_pools": {
            "tx_modulation": ["a", "b"],
            "ld_model": ["c", "d"],
        },
        "task_joint_epochs": 1,
        "replay_mix_ratio": 0.15,
        "min_ratio": 0.05,
    }
    base = [
        {"tasks": ["tx_modulation"], "epochs": 1, "name": "tx_modulation", "joint": False},
        {"tasks": ["ld_model"], "epochs": 1, "name": "ld_model", "joint": False},
        {
            "name": "joint_tx_modulation+ld_model",
            "tasks": ["tx_modulation", "ld_model"],
            "epochs": 1,
            "joint": True,
        },
    ]
    dm = SignalDataModule(cfg, stage="stage2")
    dm.setup()
    cb = TaskScheduleCallback(base, cfg)
    trainer = SimpleNamespace(datamodule=dm, current_epoch=0)
    cb.on_fit_start(trainer, None)
    assert cfg["active_joint_session"] is False
    trainer.current_epoch = 1
    cb.on_train_epoch_end(trainer, None)
    assert cfg["active_joint_session"] is True
    assert cfg["active_train_tasks"] == ["tx_modulation", "ld_model"]
    assert set(cfg["active_train_sources"]) == {"tx_modulation", "ld_model"}
    assert cfg["active_replay_sources"] == []
    assert set(dm._filtered_train_sets()) == {"tx_modulation", "ld_model"}
    assert set(dm._filtered_val_sets()) == {"tx_modulation", "ld_model"}


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
        "tasks": ["tx_modulation", "ld_model"],
        "task_pools": {
            "tx_modulation": ["a", "b"],
            "ld_model": ["c", "d"],
        },
        "task_schedule": [
            {"task": "tx_modulation", "epochs": 2},
            {"task": "ld_model", "epochs": 2},
        ],
        "min_ratio": 0.05,
    }
    dm = SignalDataModule(cfg, stage="stage2")
    dm.setup()
    sessions = resolve_task_schedule(cfg)
    cb = TaskScheduleCallback(sessions, cfg)
    trainer = SimpleNamespace(datamodule=dm, current_epoch=0)
    cb.on_fit_start(trainer, None)
    assert cfg["active_train_sources"] == ["tx_modulation"]
    assert list(dm._filtered_train_sets()) == ["tx_modulation"]
    # 第 0 轮结束仍属 tx_modulation（epochs=2），预切 epoch=1 仍是 tx_modulation
    trainer.current_epoch = 0
    cb.on_train_epoch_end(trainer, None)
    assert cfg["active_train_sources"] == ["tx_modulation"]
    # 第 1 轮结束预切 epoch=2 → ld_model，随后 reload 应拿到 ld_model
    trainer.current_epoch = 1
    cb.on_train_epoch_end(trainer, None)
    assert cfg["active_train_tasks"] == ["ld_model"]
    assert cfg["active_train_sources"] == ["ld_model"]
    assert list(dm._filtered_train_sets()) == ["ld_model"]
    _ = dm.train_dataloader()  # reload 后应能按新 filter 建 loader
