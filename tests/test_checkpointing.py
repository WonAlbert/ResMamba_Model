from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from resmamba_signal_model.training.checkpointing import (
    TrainStateCallback,
    aggregate_multitask_geomean,
    aggregate_specialist_geomean,
    aggregate_val_monitor,
    checkpoint_mode_for_monitor,
    extract_task_selection_metrics,
    make_model_checkpoint,
    resolve_lightning_precision,
    resolve_run_and_ckpt,
    resolve_run_seed,
    resolve_train_monitor,
)
from resmamba_signal_model.training.logging_utils import format_val_epoch_metrics
from resmamba_signal_model.training.mix import DynamicRatioScheduler


def test_checkpoint_mode_for_monitor() -> None:
    assert checkpoint_mode_for_monitor("val/monitor") == "min"
    assert checkpoint_mode_for_monitor("val/loss") == "min"
    assert checkpoint_mode_for_monitor("val/f1") == "max"
    assert checkpoint_mode_for_monitor("val/multitask_geomean") == "max"
    assert checkpoint_mode_for_monitor("val/specialist_geomean") == "max"
    assert checkpoint_mode_for_monitor("val/f1", "min") == "min"
    with pytest.raises(ValueError, match="checkpoint_mode"):
        checkpoint_mode_for_monitor("val/loss", "auto")


def test_resolve_train_monitor_stage2_geomean() -> None:
    monitor, mode = resolve_train_monitor({}, stage="stage2", task="all")
    assert monitor == "val/multitask_geomean"
    assert mode == "max"
    monitor, mode = resolve_train_monitor(
        {"checkpoint_monitor": "val/monitor", "checkpoint_mode": "min"},
        stage="pretrain",
        task=None,
    )
    assert monitor == "val/monitor"
    assert mode == "min"


def test_resolve_seed_and_precision() -> None:
    assert resolve_run_seed({}, 7) == 7
    assert resolve_run_seed({"seed": 3}) == 3
    assert resolve_lightning_precision({"amp": False}, cuda_available=True) == "32-true"
    assert resolve_lightning_precision({"amp": True, "amp_dtype": "bfloat16"}, cuda_available=True) == "bf16-mixed"
    assert resolve_lightning_precision({"amp": True, "amp_dtype": "float16"}, cuda_available=True) == "16-mixed"
    assert resolve_lightning_precision({"precision": "16-mixed", "amp_dtype": "bfloat16"}, cuda_available=False) == "16-mixed"
    assert resolve_lightning_precision({"amp": True, "amp_dtype": "bfloat16"}, cuda_available=False) == "32-true"


def test_aggregate_val_monitor_means_per_source_loss() -> None:
    metrics = {
        "val/recon/dataloader_idx_0": torch.tensor(0.2),
        "val/recon/dataloader_idx_1": torch.tensor(0.4),
        "val/loss/dataloader_idx_0": torch.tensor(9.0),
        "val/mae/dataloader_idx_0": torch.tensor(9.0),
        "val/monitor": torch.tensor(99.0),
    }
    assert aggregate_val_monitor(metrics) == pytest.approx(0.3)
    assert aggregate_val_monitor({"val/loss": 0.5}) == pytest.approx(0.5)
    assert aggregate_val_monitor({"val/recon": 0.25}) == pytest.approx(0.25)
    assert aggregate_val_monitor({"train/loss": 1.0}) is None


def test_aggregate_val_monitor_skips_nonfinite() -> None:
    metrics = {
        "val/recon/dataloader_idx_0": torch.tensor(0.2),
        "val/recon/dataloader_idx_1": torch.tensor(float("nan")),
        "val/recon/dataloader_idx_2": torch.tensor(float("inf")),
    }
    assert aggregate_val_monitor(metrics) == pytest.approx(0.2)
    assert aggregate_val_monitor({"val/loss": float("nan")}) is None
    # recon 缺失时回退 loss
    assert aggregate_val_monitor(
        {
            "val/loss/dataloader_idx_0": torch.tensor(0.1),
            "val/loss/dataloader_idx_1": torch.tensor(0.3),
        }
    ) == pytest.approx(0.2)


def test_aggregate_specialist_geomean() -> None:
    metrics = {
        "val/f1_modulation": torch.tensor(0.8),
        "val/acc_emitter": 0.5,
        "val/nmi": 0.6,
        "val/mse_prediction": 0.1,
        "val/mse_imputation": 0.1,
    }
    score = aggregate_specialist_geomean(metrics, {"modulation": 1.0, "emitter": 1.0, "clustering": 1.0, "prediction": 1.0})
    assert score is not None
    assert 0.4 < score < 1.0
    same = aggregate_multitask_geomean(metrics)
    assert same == pytest.approx(score)


def test_extract_task_selection_metrics_per_dataset() -> None:
    metrics = extract_task_selection_metrics(
        {
            "val/acc_emitter": 0.5,
            "val/f1_emitter": 0.4,
            "val/acc_emitter/adsb2": 0.2,
            "val/acc_emitter/wifi150": 0.8,
            "val/macro_acc_emitter": 0.5,
            "val/f1_modulation": 0.9,
            "val/acc_modulation/rml2016_10a": 0.7,
        }
    )
    assert metrics["emitter"]["acc"] == pytest.approx(0.5)
    assert metrics["emitter"]["acc/adsb2"] == pytest.approx(0.2)
    assert metrics["emitter"]["per_dataset_macro_acc"] == pytest.approx(0.5)
    assert metrics["modulation"]["f1"] == pytest.approx(0.9)
    assert metrics["modulation"]["acc/rml2016_10a"] == pytest.approx(0.7)


def test_format_val_epoch_metrics_groups_sources() -> None:
    text = format_val_epoch_metrics(
        {
            "loss/total": 1.0,
            "val/monitor": torch.tensor(0.25),
            "val/loss/dataloader_idx_0": 0.1,
            "val/mae/dataloader_idx_0": 0.02,
            "val/loss/dataloader_idx_1": 0.4,
            "val/impute_mse/dataloader_idx_1": 1.2e5,
        },
        epoch=3,
        source_names=["radcom", "radar"],
    )
    assert "val epoch 3" in text
    assert "monitor=0.25" in text
    assert "[radcom] loss=0.1  mae=0.02" in text
    assert "[radar]" in text
    assert "1.2000e+05" in text
    assert format_val_epoch_metrics({"loss/total": 1.0}, epoch=0) == ""


def test_format_val_epoch_metrics_prints_per_dataset_scores() -> None:
    text = format_val_epoch_metrics(
        {"val/monitor": 0.3, "val/acc_modulation": 0.9},
        epoch=2,
        task_report={
            "modulation": {
                "kind": "classification",
                "acc": 0.8123,
                "f1": 0.7741,
                "mean_acc": 0.8,
                "mean_f1": 0.76,
                "n": 100,
                "datasets": {
                    "rml2016_10a": {"acc": 0.85, "f1": 0.83, "n": 40},
                    "rml2018_1a": {"acc": 0.75, "f1": 0.69, "n": 60},
                },
            }
        },
    )
    assert "modulation  acc=0.8123  f1=0.7741  mean_acc=0.8  mean_f1=0.76  n=100" in text
    assert "rml2016_10a  acc=0.85  f1=0.83  n=40" in text
    assert "rml2018_1a" in text
    assert "monitor=0.3" in text
    assert "acc_modulation" not in text


def test_mix_state_dict_roundtrip() -> None:
    sched = DynamicRatioScheduler(["a", "b"], alpha=0.5, min_ratio=0.05)
    sched.update({"a": 4.0, "b": 1.0}, {"a": 2.0, "b": 2.0})
    restored = DynamicRatioScheduler(["a", "b"], alpha=0.5, min_ratio=0.05)
    restored.load_state_dict(sched.state_dict())
    assert restored.ema == sched.ema


def test_resolve_run_and_ckpt_fresh_and_resume(tmp_path: Path) -> None:
    run_dir, ckpt = resolve_run_and_ckpt(root=tmp_path, stage="pretrain", run_name="exp1", resume=None)
    assert run_dir == tmp_path / "runs" / "experiments" / "exp1"
    assert ckpt is None

    last = tmp_path / "runs" / "experiments" / "exp1" / "ckpts" / "last.ckpt"
    last.parent.mkdir(parents=True)
    last.write_bytes(b"ckpt")
    run_dir, ckpt = resolve_run_and_ckpt(root=tmp_path, stage="pretrain", run_name="exp1", resume="auto")
    assert run_dir == last.parent.parent
    assert ckpt == last.resolve()

    best_only = tmp_path / "runs" / "experiments" / "exp2" / "ckpts" / "best.ckpt"
    best_only.parent.mkdir(parents=True)
    best_only.write_bytes(b"best")
    _, ckpt = resolve_run_and_ckpt(root=tmp_path, stage="pretrain", run_name="exp2", resume="auto")
    assert ckpt == best_only.resolve()

    run_dir, ckpt = resolve_run_and_ckpt(root=tmp_path, stage="pretrain", run_name=None, resume=str(last))
    assert ckpt == last.resolve()
    assert run_dir == last.parent.parent

    with pytest.raises(ValueError, match="--run-name"):
        resolve_run_and_ckpt(root=tmp_path, stage="pretrain", run_name=None, resume="auto")
    with pytest.raises(FileNotFoundError, match="best.ckpt"):
        resolve_run_and_ckpt(root=tmp_path, stage="pretrain", run_name="missing", resume="auto")


def test_pretrain_writes_best_and_resumes(tmp_path: Path) -> None:
    import lightning as L

    from resmamba_signal_model.models.model import SignalFoundationModel, SignalModelConfig
    from resmamba_signal_model.training.data_module import SignalDataModule
    from resmamba_signal_model.training.lit_module import SignalLitModule

    train_cfg = {
        "synthetic": True,
        "synthetic_sources": ["src_a", "src_b"],
        "token_budget": 32,
        "steps_per_epoch": 1,
        "val_batches": 1,
        "epochs": 1,
        "num_workers": 0,
        "patch_size": 8,
        "pin_memory": False,
        "learning_rate": 1e-3,
        "warmup_steps": 0,
        "weight_decay": 0.0,
        "require_mamba_kernel": False,
        "allow_fallback_mamba": True,
    }
    data = SignalDataModule(train_cfg, stage="pretrain")
    data.setup()
    model_cfg = SignalModelConfig(
        d_model=32,
        mamba_d_state=8,
        mamba_headdim=16,
        require_mamba_kernel=False,
        allow_fallback_mamba=True,
        attn_num_heads=4,
        patch_size=8,
        stem_channels=8,
        freq_bands=4,
        dropout=0.0,
        p_trunc=0.0,
        num_datasets=4,
        build_task_heads=False,
    )
    lit = SignalLitModule(SignalFoundationModel(model_cfg), train_cfg, stage="pretrain", mix=data.mix)
    ckpt_dir = tmp_path / "ckpts"
    ckpt_cb = make_model_checkpoint(ckpt_dir, monitor="val/monitor", mode="min")
    state_path = tmp_path / "train_state.json"
    trainer = L.Trainer(
        default_root_dir=str(tmp_path),
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        callbacks=[ckpt_cb, TrainStateCallback(state_path, monitor="val/monitor", stage="pretrain")],
    )
    trainer.fit(lit, datamodule=data)
    last = ckpt_dir / "last.ckpt"
    best = ckpt_dir / "best.ckpt"
    assert not last.is_file()
    assert best.is_file()
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["stage"] == "pretrain"
    assert payload["best_model_path"]
    blob = torch.load(best, map_location="cpu", weights_only=False)
    assert "mix_state" in blob
    assert "optimizer_states" in blob
    assert int(blob.get("global_step", 0)) >= 1

    train_cfg["epochs"] = 2
    data2 = SignalDataModule(train_cfg, stage="pretrain")
    data2.setup()
    lit2 = SignalLitModule(SignalFoundationModel(model_cfg), train_cfg, stage="pretrain", mix=data2.mix)
    trainer2 = L.Trainer(
        default_root_dir=str(tmp_path / "resume"),
        max_epochs=2,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        callbacks=[make_model_checkpoint(tmp_path / "resume_ckpts", monitor="val/monitor", mode="min")],
    )
    trainer2.fit(lit2, datamodule=data2, ckpt_path=str(best))
    assert trainer2.global_step >= trainer.global_step
    assert data2.mix is not None
    assert data2.mix.ema.keys() == data.mix.ema.keys()
