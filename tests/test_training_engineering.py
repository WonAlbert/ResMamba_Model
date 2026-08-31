from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from resmamba_signal_model.models.model import SignalFoundationModel, SignalModelConfig
from resmamba_signal_model.training.checkpointing import resolve_train_monitor
from resmamba_signal_model.training.data_module import (
    SignalDataModule,
    SyntheticIQDataset,
    make_fixed_eval_loader,
    merge_source_batches,
    resolve_dataset_h5,
)
from resmamba_signal_model.training.lit_module import SignalLitModule
from resmamba_signal_model.training.lr_schedule import LightningCompatLRScheduler, WarmupCosineLR
from resmamba_signal_model.training.task_catalog import builtin_spec


def _synthetic_cfg(**overrides) -> dict:
    cfg = {
        "synthetic": True,
        "synthetic_sources": ["src_a", "src_b"],
        "token_budget": 32,
        "steps_per_epoch": 2,
        "val_batches": 2,
        "val_seed": 0,
        "seed": 0,
        "num_workers": 0,
        "patch_size": 8,
        "pin_memory": False,
        "mix_strategy": "token_share",
        "balanced_sampling": False,
        "learning_rate": 1e-3,
        "warmup_steps": 0,
        "epochs": 1,
        "weight_decay": 0.0,
        "lr_schedule": "warmup_cosine",
    }
    cfg.update(overrides)
    return cfg


def _sampler_indices(seed: int) -> list[list[int]]:
    dm = SignalDataModule(_synthetic_cfg(seed=seed), stage="pretrain")
    dm.setup()
    loader = dm._loaders(dm._train_sets, train=True)["src_a"]
    sampler = loader.batch_sampler
    assert getattr(sampler, "_seed", None) is not None
    return [list(batch) for batch in sampler]


def test_train_sampler_seed_is_configurable_and_reproducible() -> None:
    first = _sampler_indices(1)
    second = _sampler_indices(1)
    other = _sampler_indices(2)
    assert first == second
    assert first != other


def test_stage2_and_joint_selection_metric_names() -> None:
    monitor, mode = resolve_train_monitor(
        {"checkpoint_monitor": "val/multitask_geomean", "checkpoint_mode": "max"},
        stage="stage2",
        task="all",
    )
    assert monitor == "val/multitask_geomean"
    assert mode == "max"
    monitor, mode = resolve_train_monitor({}, stage="stage2", task="all")
    assert monitor == "val/multitask_geomean"
    assert mode == "max"
    monitor, mode = resolve_train_monitor({}, stage="joint", task="all")
    assert monitor == "val/specialist_geomean"
    assert mode == "max"


def test_infer_eval_loader_matches_val_token_budget_protocol() -> None:
    dm = SignalDataModule(_synthetic_cfg(val_batches=2, val_seed=11, token_budget=32), stage="pretrain")
    dm.setup()
    val_plan = dm._val_batch_plan["src_a"]
    loader = make_fixed_eval_loader(
        dm._val_sets["src_a"],
        token_budget=dm.token_budget,
        patch_size=dm.patch_size,
        num_batches=dm.val_batches,
        seed=dm.val_seed,
        num_workers=0,
        source_name="src_a",
    )
    infer_plan = [list(batch) for batch in loader.batch_sampler]
    assert infer_plan == val_plan
    val_loader = dm._loaders(dm._val_sets, train=False)["src_a"]
    assert [list(batch) for batch in val_loader.batch_sampler] == infer_plan


def test_resolve_dataset_h5_prefers_test_split(tmp_path: Path) -> None:
    h5_dir = tmp_path / "h5"
    h5_dir.mkdir()
    (h5_dir / "adsb2_val.h5").write_bytes(b"val")
    (h5_dir / "adsb2_test.h5").write_bytes(b"test")
    assert resolve_dataset_h5(tmp_path, "adsb2", "test").name == "adsb2_test.h5"
    assert resolve_dataset_h5(tmp_path, "adsb2", "val").name == "adsb2_val.h5"


def test_label_columns_passthrough_collate_and_merge() -> None:
    sample = SyntheticIQDataset(n=4, lengths=(16,), seed=0)[0]
    assert "canonical_mod_label_id" in sample
    assert "global_emitter_id" in sample
    assert sample["values"] is sample["iq"]

    dm = SignalDataModule(
        {
            "synthetic": True,
            "token_budget": 32,
            "steps_per_epoch": 2,
            "val_batches": 1,
            "seed": 0,
            "num_workers": 0,
            "patch_size": 8,
            "pin_memory": False,
        },
        stage="stage2",
    )
    dm.setup()
    batch = next(iter(dm._loaders(dm._val_sets, train=False)[dm.source_names[0]]))
    assert "canonical_mod_label_id" in batch
    assert "global_emitter_id" in batch
    assert torch.is_tensor(batch["canonical_mod_label_id"])
    assert builtin_spec("tx_modulation").label_field == "canonical_mod_label_id"
    assert builtin_spec("ld_model").label_field == "mod_label_id"

    second_key = dm.source_names[1] if len(dm.source_names) > 1 else dm.source_names[0]
    merged = merge_source_batches(
        {
            dm.source_names[0]: batch,
            second_key: next(iter(dm._loaders(dm._val_sets, train=False)[second_key])),
        }
    )
    assert "canonical_mod_label_id" in merged
    assert "global_emitter_id" in merged
    assert "receiver_id" in merged


def test_infer_task_label_tensor_ld_model_uses_mod_label_id() -> None:
    import infer as infer_script

    lookup = torch.tensor([-1] * 32, dtype=torch.long)
    lookup[10] = 0
    lookup[31] = 15
    batch = {
        "dataset_id": torch.tensor([10, 31]),
        "mod_label_id": torch.tensor([3, 2]),
        "global_emitter_id": torch.tensor([-1, -1]),
    }
    assert infer_script.task_label_tensor(
        "ld_model",
        batch,
        ld_model_offset_lookup=lookup,
    ).tolist() == [3, 17]


def test_infer_task_label_tensor_uses_canonical_and_global_emitter() -> None:
    import infer as infer_script

    batch = {
        "canonical_mod_label_id": torch.tensor([3, 4]),
        "mod_label_id": torch.tensor([1, 2]),
        "global_emitter_id": torch.tensor([10, 11]),
        "emitter_id": torch.tensor([0, 1]),
        "global_label_id": torch.tensor([100, 101]),
    }
    assert infer_script.task_label_tensor("tx_modulation", batch).tolist() == [3, 4]
    assert infer_script.task_label_tensor("ld_model", batch).tolist() == [1, 2]


def test_stage2_task_schedule_optimizer_registers_all_heads() -> None:
    from resmamba_signal_model.models.model import SignalFoundationModel, SignalModelConfig
    from resmamba_signal_model.training.freeze import apply_stage_freeze
    from resmamba_signal_model.training.lit_module import SignalLitModule

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
        num_mod_classes=5,
        num_emitters=6,
        build_task_heads=True,
        build_task_interface=False,
    )
    model_cfg.num_intrapulse_classes = 3
    model_cfg.num_ld_model_classes = 6
    model = SignalFoundationModel(model_cfg)
    train_cfg = {
        "task_schedule": [{"task": "ld_intrapulse", "epochs": 1}, {"task": "ld_model", "epochs": 1}],
        "active_train_tasks": ["ld_intrapulse"],
        "truncate_backward": True,
        "skip_recon": True,
        "learning_rate": 1.0e-3,
        "steps_per_epoch": 2,
        "epochs": 2,
        "warmup_steps": 1,
    }
    apply_stage_freeze(model, "stage2", task="ld_intrapulse", train_cfg=train_cfg)
    lit = SignalLitModule(model, train_cfg, stage="stage2")
    configured = lit.configure_optimizers()
    optimizer = configured["optimizer"]
    opt_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert id(model.ld_intrapulse_head.classifier.weight) in opt_ids
    assert id(model.ld_model_head.classifier.weight) in opt_ids
    assert id(model.z_linear_probes["ld_model"].weight) in opt_ids
    assert not model.ld_model_head.classifier.weight.requires_grad

    apply_stage_freeze(model, "stage2", task="ld_model", train_cfg=train_cfg)
    assert model.ld_model_head.classifier.weight.requires_grad
    before = model.ld_model_head.classifier.weight.detach().clone()
    model.ld_model_head.classifier.weight.grad = torch.ones_like(model.ld_model_head.classifier.weight)
    optimizer.step()
    assert not torch.allclose(model.ld_model_head.classifier.weight, before)


def test_sync_optimizer_adds_newly_unfrozen_head_after_task_switch() -> None:
    from resmamba_signal_model.models.model import SignalFoundationModel, SignalModelConfig
    from resmamba_signal_model.training.freeze import apply_stage_freeze
    from resmamba_signal_model.training.lit_module import SignalLitModule

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
        num_mod_classes=5,
        num_emitters=6,
        build_task_heads=True,
        build_task_interface=False,
    )
    model_cfg.num_intrapulse_classes = 3
    model_cfg.num_ld_model_classes = 6
    model = SignalFoundationModel(model_cfg)
    train_cfg = {
        "task_schedule": [{"task": "ld_intrapulse", "epochs": 1}, {"task": "tx_modulation", "epochs": 1}],
        "active_train_tasks": ["ld_intrapulse"],
        "truncate_backward": True,
        "skip_recon": True,
        "learning_rate": 1.0e-3,
        "steps_per_epoch": 2,
        "epochs": 2,
        "warmup_steps": 1,
    }
    apply_stage_freeze(model, "stage2", task="ld_intrapulse", train_cfg=train_cfg)
    lit = SignalLitModule(model, train_cfg, stage="stage2")
    # 模拟旧 run：优化器只含首个任务头
    old_style = torch.optim.AdamW(
        [p for p in model.ld_intrapulse_head.parameters() if p.requires_grad],
        lr=1.0e-3,
    )
    lit.trainer = type("T", (), {"optimizers": [old_style]})()

    apply_stage_freeze(model, "stage2", task="tx_modulation", train_cfg={**train_cfg, "active_train_tasks": ["tx_modulation"]})
    added = lit.sync_optimizer_trainable_params()
    assert added > 0
    opt_ids = {id(p) for g in old_style.param_groups for p in g["params"]}
    assert id(model.tx_modulation_head.classifier.weight) in opt_ids


def test_configure_optimizers_uses_build_lr_scheduler() -> None:
    train_cfg = _synthetic_cfg()
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
    lit = SignalLitModule(SignalFoundationModel(model_cfg), train_cfg, stage="pretrain")
    configured = lit.configure_optimizers()
    scheduler = configured["lr_scheduler"]["scheduler"]
    assert isinstance(scheduler, LightningCompatLRScheduler)
    assert isinstance(scheduler.wrapped, WarmupCosineLR)
    assert configured["lr_scheduler"]["interval"] == "step"


def test_clustering_train_loader_attaches_physical_view2() -> None:
    dm = SignalDataModule(
        {
            "synthetic": True,
            "token_budget": 32,
            "steps_per_epoch": 1,
            "val_batches": 1,
            "seed": 0,
            "num_workers": 0,
            "patch_size": 8,
            "pin_memory": False,
            "clustering_view2": True,
        },
        stage="stage2",
    )
    dm.setup()
    cluster_key = next((k for k in dm._train_sets if "clustering" in k), dm.source_names[0])
    assert cluster_key in dm._train_sets
    batch = next(iter(dm._loaders(dm._train_sets, train=True)[cluster_key]))
    assert "view2" in batch
    iq = batch["iq"]
    view2 = batch["view2"]
    if torch.is_tensor(iq):
        assert tuple(view2.shape) == tuple(iq.shape)
        assert not torch.equal(view2, iq)
    else:
        assert len(view2) == len(iq)
        assert not torch.equal(view2[0], iq[0])
    val_batch = next(iter(dm._loaders(dm._val_sets, train=False)[cluster_key]))
    assert "view2" not in val_batch


def test_train_sampler_fields_are_label_firewall() -> None:
    from resmamba_signal_model.training.data_module import train_sampler_label_fields

    for task in ("tx_modulation", "ld_model", "ld_clustering", "prediction", None):
        fields = train_sampler_label_fields(task)
        assert "global_label_id" not in fields
    assert train_sampler_label_fields("tx_modulation")[0] == "canonical_mod_label_id"
    assert train_sampler_label_fields("ld_model")[0] == "mod_label_id"


def test_continual_sessions_and_osr_metrics() -> None:
    from resmamba_signal_model.training.continual import resolve_continual_sessions, total_continual_epochs
    from resmamba_signal_model.training.metrics import openset_detection_metrics, openset_four_quadrant_metrics

    sessions = resolve_continual_sessions({"continual": True, "epochs": 3}, stage="continual")
    assert sessions and total_continual_epochs(sessions, default_epochs=3) == 3
    y = torch.tensor([0, 0, 0, 1, 1, 1])
    scores = torch.tensor([0.1, 0.2, 0.15, 0.9, 0.8, 0.95])
    report = openset_detection_metrics(y, scores)
    assert report["auroc"] > 0.9
    assert report["aupr"] > 0.8
    assert 0.0 <= report["fpr95"] <= 1.0
    quad = openset_four_quadrant_metrics(
        known_id_scores=torch.tensor([0.1, 0.2]),
        known_ood_scores=torch.tensor([0.25]),
        unknown_id_scores=torch.tensor([0.8, 0.85]),
        unknown_ood_scores=torch.tensor([0.9]),
    )
    assert quad["quadrants"]["unknown_ood"]["n"] == 1
    assert quad["auroc"] > 0.9


def test_ema_teacher_is_shadow_not_child() -> None:
    train_cfg = _synthetic_cfg(ema_teacher=True, loss_weights={"mse": 1.0, "vicreg": 0.3})
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
    lit = SignalLitModule(SignalFoundationModel(model_cfg), train_cfg, stage="pretrain")
    lit._ema_enabled = lambda: True  # type: ignore[method-assign]
    from resmamba_signal_model.models.ema import EMATeacher

    teacher = EMATeacher(lit.model, momentum=0.996)
    lit._ema_box["teacher"] = teacher
    child_names = {name for name, _ in lit.named_children()}
    assert "teacher" not in child_names
    assert "ema_teacher" not in child_names
    assert lit.ema_teacher is teacher
    assert not any(p.requires_grad for p in teacher.parameters())



def test_sanitize_nonfinite_grads_zeros_nan_and_keeps_finite() -> None:
    from resmamba_signal_model.training.lit_module import sanitize_nonfinite_grads

    layer = torch.nn.Linear(4, 4)
    layer.weight.grad = torch.tensor(
        [[1.0, float("nan"), float("inf"), -2.0]] * 4,
        dtype=layer.weight.dtype,
    )
    layer.bias.grad = torch.ones_like(layer.bias)
    n_bad = sanitize_nonfinite_grads(layer)
    assert n_bad == 1
    assert torch.isfinite(layer.weight.grad).all()
    assert torch.allclose(layer.weight.grad[0], torch.tensor([1.0, 0.0, 0.0, -2.0]))
    assert torch.equal(layer.bias.grad, torch.ones_like(layer.bias))
