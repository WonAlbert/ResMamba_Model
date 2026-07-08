from scripts.train_pipeline import build_model, count_params
from resmamba_signal_model.training.stages import configure_stage2_heads


def _any_trainable(module) -> bool:
    return any(p.requires_grad for p in module.parameters())


def test_modulation_unfreezes_mod_path() -> None:
    model = build_model("configs/model_resmamba_400m.yaml")
    configure_stage2_heads(
        model,
        "modulation",
        freeze_backbone=True,
        unfreeze_modulation_backbone=True,
        unfreeze_emitter_backbone=False,
    )
    assert _any_trainable(model.mod_fuse)
    assert _any_trainable(model.space_projs["mod_specific"])
    assert not _any_trainable(model.emitter_fuse)
    assert not _any_trainable(model.space_projs["emitter_specific"])
    assert _any_trainable(model.recognition_heads.modulation)
    assert not _any_trainable(model.encoder)


def test_emitter_unfreezes_emitter_path() -> None:
    model = build_model("configs/model_resmamba_400m.yaml")
    configure_stage2_heads(
        model,
        "emitter",
        freeze_backbone=True,
        unfreeze_modulation_backbone=False,
        unfreeze_emitter_backbone=True,
    )
    assert _any_trainable(model.emitter_fuse)
    assert _any_trainable(model.space_projs["emitter_specific"])
    assert not _any_trainable(model.mod_fuse)
    _, trainable = count_params(model)
    assert trainable > 20_000_000
