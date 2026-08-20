from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from resmamba_signal_model.models.model import SignalFoundationModel, SignalModelConfig
from resmamba_signal_model.models.peft import PeftConfig, inject_hybrid_lora
from resmamba_signal_model.training.data_module import SignalDataModule
from resmamba_signal_model.training.freeze import apply_stage_freeze
from resmamba_signal_model.training.lit_module import SignalLitModule


def _cfg() -> SignalModelConfig:
    return SignalModelConfig(
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
        num_mod_classes=16,
        num_emitters=32,
        num_prototypes=8,
        build_task_heads=True,
    )


def _train_cfg(stage: str) -> dict:
    return {
        "synthetic": True,
        "token_budget": 32,
        "steps_per_epoch": 1,
        "val_batches": 1,
        "epochs": 1,
        "num_workers": 0,
        "patch_size": 8,
        "pin_memory": False,
        "learning_rate": 1e-3,
        "heads_lr": 1e-3,
        "warmup_steps": 0,
        "weight_decay": 0.0,
        "lambda_recon": 0.0,
        "loraplus_lr_ratio": 16,
        "task": "modulation" if stage == "stage3" else None,
        "loss_weights": {"physical": 0.0, "domain": 0.0},
    }


def _fit(stage: str, model: SignalFoundationModel, train_cfg: dict, tmp_path: Path) -> None:
    import lightning as L

    data = SignalDataModule(train_cfg, stage=stage)
    data.setup()
    lit = SignalLitModule(model, train_cfg, stage=stage, mix=data.mix)
    trainer = L.Trainer(
        default_root_dir=str(tmp_path / stage),
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
    )
    trainer.fit(lit, datamodule=data)
    assert trainer.global_step >= 1


def test_stage2_tiny_smoke(tmp_path: Path) -> None:
    model = SignalFoundationModel(_cfg())
    apply_stage_freeze(model, "stage2", train_cfg={"truncate_backward": True, "skip_recon": True})
    _fit("stage2", model, _train_cfg("stage2"), tmp_path)


def test_stage3_tiny_smoke(tmp_path: Path) -> None:
    cfg = _cfg()
    cfg.build_adapters = True
    model = SignalFoundationModel(cfg)
    inject_hybrid_lora(model, ["modulation"], PeftConfig(r_attn=2, r_mamba=2, lora_alpha_attn=2, lora_alpha_mamba=2))
    apply_stage_freeze(model, "stage3", task="modulation", train_cfg={})
    train_cfg = _train_cfg("stage3")
    train_cfg["synthetic_sources"] = ["classification"]
    _fit("stage3", model, train_cfg, tmp_path)


def test_joint_tiny_smoke(tmp_path: Path) -> None:
    cfg = _cfg()
    cfg.build_adapters = True
    cfg.build_shared_adapter = True
    model = SignalFoundationModel(cfg)
    inject_hybrid_lora(
        model,
        ["modulation", "emitter", "clustering", "prediction", "imputation"],
        PeftConfig(r_attn=2, r_mamba=2, lora_alpha_attn=2, lora_alpha_mamba=2, shared_lora=True),
    )
    apply_stage_freeze(model, "joint", train_cfg={})
    _fit("joint", model, _train_cfg("joint"), tmp_path)
    assert any(p.requires_grad for n, p in model.named_parameters() if n.startswith("tokenizer.time_fuse"))
