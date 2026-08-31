from pathlib import Path

import torch

from resmamba_signal_model.models.model import SignalModelConfig
from resmamba_signal_model.training.compact_labels import (
    apply_compact_task_labels,
    compact_task_labels_enabled,
)
from resmamba_signal_model.training.emitter_labels import (
    build_compact_emitter_dataset_class_mask,
    build_global_emitter_label_map,
    build_global_radar_model_label_map,
    global_emitter_labels,
)
from resmamba_signal_model.training.modulation_labels import (
    build_compact_modulation_label_map,
    build_global_comm_modulation_label_map,
    global_comm_modulation_labels,
    remap_modulation_labels,
)
from resmamba_signal_model.training.losses import downstream_task_loss


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
LEGACY_EMITTER_DATASETS = ("wisig", "adsb2")


def test_compact_task_labels_enabled_by_stage() -> None:
    assert compact_task_labels_enabled({"compact_task_labels": True}, stage="stage2")
    assert not compact_task_labels_enabled({"compact_task_labels": False}, stage="stage2")
    assert not compact_task_labels_enabled({}, stage="pretrain")
    assert not compact_task_labels_enabled({"synthetic": True}, stage="stage2")
    assert compact_task_labels_enabled({}, stage="stage3")


def test_compact_emitter_map_and_mask() -> None:
    if not (DATASET / "label_maps.json").is_file():
        return
    label_map = build_global_emitter_label_map(DATASET, dataset_names=list(LEGACY_EMITTER_DATASETS))
    assert label_map.num_emitters == 250
    assert label_map.class_counts is not None
    assert label_map.class_counts[6] == 100
    assert label_map.class_counts[11] == 150
    mask = build_compact_emitter_dataset_class_mask(label_map, num_datasets=32)
    assert mask is not None
    assert mask.shape == (32, 250)
    assert int(mask[6].sum()) == 100
    assert int(mask[11].sum()) == 150
    assert bool(mask[6, :100].all())
    assert bool(mask[11, 100:250].all())
    assert not bool(mask[6, 100:].any())


def test_compact_modulation_map_rml() -> None:
    if not (DATASET / "label_maps.json").is_file():
        return
    mod_map = build_compact_modulation_label_map(DATASET)
    assert mod_map.num_classes == 11
    assert set(mod_map.old_to_new) == {1, 2, 4, 9, 10, 11, 13, 15, 22, 25, 29}
    lookup = mod_map.lookup()
    labels = torch.tensor([1, 29, 99, -1], dtype=torch.long)
    remapped = remap_modulation_labels(labels, lookup)
    assert remapped.tolist() == [0, 10, -1, -1]


def test_global_comm_modulation_map_scheme_a() -> None:
    if not (DATASET / "label_maps.json").is_file():
        return
    comm_map, canonical = build_global_comm_modulation_label_map(DATASET)
    assert canonical.num_classes == 11
    assert comm_map.num_emitters == 33
    assert comm_map.offsets[2] == 0
    assert comm_map.offsets[3] == 11
    assert comm_map.offsets[4] == 22
    lookup = canonical.lookup()
    offset_lookup = comm_map.offset_lookup()
    # QAM16 on rml2016_10a (dataset_id=3) -> local 7 + offset 11 = 18
    labels = global_comm_modulation_labels(
        torch.tensor([3, 3]),
        torch.tensor([15, 15]),
        lookup,
        offset_lookup,
    )
    assert labels.tolist() == [18, 18]
    # QPSK on 04c (dataset_id=2) -> local 9 + offset 0 = 9
    labels_qpsk = global_comm_modulation_labels(
        torch.tensor([2]),
        torch.tensor([25]),
        lookup,
        offset_lookup,
    )
    assert labels_qpsk.tolist() == [9]


def test_compact_radar_model_map() -> None:
    if not (DATASET / "label_maps.json").is_file():
        return
    label_map = build_global_radar_model_label_map(DATASET)
    assert label_map.num_emitters == 24
    assert label_map.class_counts is not None
    assert sum(label_map.class_counts.values()) == 24
    mask = build_compact_emitter_dataset_class_mask(label_map, num_datasets=32)
    assert mask is not None
    assert mask.shape == (32, 24)


def test_apply_compact_task_labels_sets_model_cfg() -> None:
    if not (DATASET / "label_maps.json").is_file():
        return
    train_cfg = {"rfdata_root": str(DATASET), "compact_task_labels": True}
    model_cfg = SignalModelConfig(num_mod_classes=256, num_emitters=512)
    emitter_map, mod_map = apply_compact_task_labels(train_cfg, model_cfg, stage="stage2")
    assert emitter_map is not None or mod_map is not None
    if emitter_map is not None:
        assert model_cfg.num_emitters >= 0
    if mod_map is not None:
        assert model_cfg.num_mod_classes == 33
        assert train_cfg["model"]["num_mod_classes"] == 33
        assert "compact_tx_modulation" in train_cfg
        assert train_cfg["compact_tx_modulation"]["num_classes"] == 33
    if getattr(model_cfg, "num_ld_model_classes", None):
        assert int(model_cfg.num_ld_model_classes) == 24
        assert "compact_ld_model" in train_cfg
    if getattr(model_cfg, "num_intrapulse_classes", None):
        assert int(model_cfg.num_intrapulse_classes) >= 1
        assert "compact_intrapulse" in train_cfg


def test_downstream_loss_uses_compact_emitter_and_modulation() -> None:
    torch.manual_seed(0)
    batch_size = 4
    d_model = 8
    outputs = {
        "z": torch.randn(batch_size, d_model),
        "task_pooled": torch.randn(batch_size, d_model),
        "task_logits": torch.randn(batch_size, 5),
        "z_probe_logits": torch.randn(batch_size, 5),
    }
    # 局部 emitter_id + offset → 紧凑 0..4（模拟 adsb2 offset=0）
    batch = {
        "dataset_id": torch.tensor([6, 6, 6, 6]),
        "emitter_id": torch.tensor([0, 1, 2, 3]),
        "global_emitter_id": torch.tensor([100, 101, 102, 103]),  # 故意错误的 namespace，不应被使用
    }
    lookup = torch.full((12,), -1, dtype=torch.long)
    lookup[6] = 0
    loss, parts = downstream_task_loss(
        outputs,
        batch,
        "emitter",
        emitter_offset_lookup=lookup,
        z_probe_weight=0.0,
        emitter_contrastive_weight=0.0,
    )
    assert torch.isfinite(loss)
    assert "task_ce" in parts
    # 确认映射到 0..3
    remapped = global_emitter_labels(batch["dataset_id"], batch["emitter_id"], lookup)
    assert remapped.tolist() == [0, 1, 2, 3]

    mod_outputs = {
        "z": torch.randn(batch_size, d_model),
        "task_pooled": torch.randn(batch_size, d_model),
        "task_logits": torch.randn(batch_size, 33),
    }
    mod_batch = {
        "canonical_mod_label_id": torch.tensor([15, 25, 30, -1]),
        "dataset_id": torch.tensor([3, 2, 3, 0]),
    }
    mod_lookup = torch.full((31,), -1, dtype=torch.long)
    mod_lookup[15] = 7
    mod_lookup[25] = 9
    mod_lookup[30] = 2
    offset_lookup = torch.full((8,), -1, dtype=torch.long)
    offset_lookup[2] = 0
    offset_lookup[3] = 11
    loss_m, parts_m = downstream_task_loss(
        mod_outputs,
        mod_batch,
        "tx_modulation",
        modulation_compact_lookup=mod_lookup,
        modulation_offset_lookup=offset_lookup,
        modulation_contrastive_weight=0.0,
        z_probe_weight=0.0,
        label_field="canonical_mod_label_id",
    )
    assert torch.isfinite(loss_m)
    assert "task_ce" in parts_m
    expected = global_comm_modulation_labels(
        mod_batch["dataset_id"],
        mod_batch["canonical_mod_label_id"],
        mod_lookup,
        offset_lookup,
    )
    assert expected.tolist() == [18, 9, 13, -1]


def test_ld_intrapulse_loss_without_comm_compact_lookup_has_grad() -> None:
    torch.manual_seed(0)
    logits = torch.randn(4, 5, requires_grad=True)
    outputs = {
        "z": torch.randn(4, 8),
        "task_pooled": torch.randn(4, 8),
        "task_logits": logits,
    }
    batch = {
        "canonical_mod_label_id": torch.tensor([0, 1, 2, 3]),
        "dataset_id": torch.zeros(4, dtype=torch.long),
    }
    loss, parts = downstream_task_loss(
        outputs,
        batch,
        "ld_intrapulse",
        modulation_compact_lookup=None,
        modulation_contrastive_weight=0.0,
        z_contrastive_weight=0.0,
        z_probe_weight=0.0,
        domain_weight=0.0,
        recon_weight=0.0,
        phys_weight=0.0,
        label_field="canonical_mod_label_id",
        task_kind="classification",
    )
    assert loss.requires_grad
    assert "task_ce" in parts
    assert float(parts["task_ce"].detach()) > 0.0
