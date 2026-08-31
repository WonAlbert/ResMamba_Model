"""H5 v2 schema、7:2:1 划分、stamp、collate、copy_records 无二次归一化。"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_datasets import (  # noqa: E402
    Context,
    PretrainWriter,
    _PRETRAIN_H5_ENV,
    _h5_dataset_id,
    _load_h5_records,
    _skip_eval_refine_for_pretrain,
    copy_records,
    refine_eval_splits,
    stamp_semantic_namespaces,
    stratified_split_indices,
    write_pretrain_stratified_splits,
)
from resmamba_signal_model.data.h5_preprocess import (  # noqa: E402
    H5_SCHEMA_VERSION,
    H5_SCALE_POLICY,
    PretrainH5Config,
    PretrainH5Writer,
)
from resmamba_signal_model.data.rfdata import RFDataH5Dataset, variable_length_collate  # noqa: E402
from resmamba_signal_model.models.model import SignalFoundationModel, SignalModelConfig  # noqa: E402


def test_pretrain_writer_schema(tmp_path: Path) -> None:
    path = tmp_path / "demo_train.h5"
    cfg = PretrainH5Config(precompute_joint_energy=False, device="cpu")
    writer = PretrainWriter(
        path,
        32,
        2,
        0,
        path,
        dataset_name="rml2016_04c",
        cfg=cfg,
    )
    iq = np.random.randn(4, 2, 32).astype(np.float32)
    writer.append_prepared(
        iq,
        mod_label_id=np.arange(4, dtype=np.int16),
        snr=np.zeros(4, dtype=np.float32),
    )
    writer.close()
    with h5py.File(path, "r") as f:
        assert int(f.attrs["h5_schema_version"]) == H5_SCHEMA_VERSION
        assert f.attrs["iq_preprocessed"] == "none"
        assert f.attrs["scale_policy"] == "none"
        assert int(f.attrs["signal_length"]) == 32
        assert float(f.attrs["sampling_rate_hz"]) == pytest.approx(1_000_000.0)
        assert "capture_id" not in f
        assert f["iq"].dtype == np.float32
        assert f["revin_mean"].dtype == np.float16


def test_stratified_split_ratios_and_class_disjoint() -> None:
    labels = np.repeat(np.arange(5), 20).astype(np.int32)
    splits = stratified_split_indices(labels, ratios=(0.7, 0.2, 0.1), seed=0)
    assert set(splits) == {"train", "test", "val"}
    all_idx = np.concatenate([splits["train"], splits["test"], splits["val"]])
    assert all_idx.size == labels.size
    assert np.unique(all_idx).size == labels.size
    for cls in range(5):
        cls_idx = set(np.flatnonzero(labels == cls).tolist())
        seen: set[int] = set()
        for split_name, split_idx in splits.items():
            hit = cls_idx.intersection(split_idx.tolist())
            assert not seen.intersection(hit)
            seen.update(hit)
        assert len(seen) == len(cls_idx)
    train_n = splits["train"].size
    assert 60 <= train_n <= 80


def test_parallel_worker_restores_pretrain_cfg(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = PretrainH5Config(device="cpu", chunk_size=7).to_json()
    monkeypatch.setenv(_PRETRAIN_H5_ENV, payload)
    restored = PretrainH5Config.from_json(os.environ[_PRETRAIN_H5_ENV])
    assert restored.chunk_size == 7
    assert restored.precompute_joint_energy is True


def test_stamp_from_attrs(tmp_path: Path) -> None:
    ctx = Context(tmp_path)
    ctx.maps["datasets"] = {"2": "rml2016_04c"}
    ctx.maps["modulations"] = {"rml2016_04c": {"BPSK": 0, "QPSK": 1}}
    path = ctx.h5 / "rml2016_04c_train.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("iq", data=np.ones((6, 2, 16), dtype=np.float32))
        f.create_dataset("mod_label_id", data=np.array([0, 1, 0, 1, 0, 1], dtype=np.int16))
        f.attrs["h5_schema_version"] = H5_SCHEMA_VERSION
        f.attrs["dataset_id"] = 2
    stats = stamp_semantic_namespaces(ctx)
    assert stats["canonical_mod_label_id"][path.name] == 6
    with h5py.File(path, "r") as f:
        assert "canonical_mod_label_id" in f
        assert int(np.min(f["canonical_mod_label_id"][:])) >= 0
        assert "global_emitter_id" not in f


def test_copy_records_no_double_normalize(tmp_path: Path) -> None:
    src = tmp_path / "src.h5"
    dst = tmp_path / "dst.h5"
    iq = np.random.randn(8, 2, 16).astype(np.float32)
    revin = np.random.randn(8, 2).astype(np.float16)
    with h5py.File(src, "w") as f:
        f.create_dataset("iq", data=iq)
        f.create_dataset("revin_mean", data=revin)
        f.create_dataset("mod_label_id", data=np.arange(8, dtype=np.int16))
        f.create_dataset("snr", data=np.zeros(8, dtype=np.float16))
        f.create_dataset("sample_rate_hz", data=np.full(8, 1e6, dtype=np.float32))
        f.create_dataset("norm_scale", data=np.ones(8, dtype=np.float16))
        f.create_dataset("log_scale", data=np.zeros(8, dtype=np.float16))
        f.create_dataset("log_peak", data=np.zeros(8, dtype=np.float16))
        f.create_dataset("papr_preclip", data=np.ones(8, dtype=np.float16))
        f.create_dataset("scale_gap", data=np.zeros(8, dtype=np.float16))
        f.attrs["h5_schema_version"] = H5_SCHEMA_VERSION
    writer = PretrainWriter(
        dst,
        16,
        2,
        0,
        src,
        dataset_name="rml2016_04c",
        cfg=PretrainH5Config(precompute_joint_energy=True, device="cpu"),
    )
    copy_records([src], np.arange(4, dtype=np.int64), writer)
    writer.close()
    with h5py.File(dst, "r") as f:
        assert np.allclose(f["iq"][:], iq[:4], atol=1e-5)
        assert np.allclose(f["revin_mean"][:], revin[:4], atol=1e-3)


def test_collate_includes_revin_mean(tmp_path: Path) -> None:
    path = tmp_path / "radchar_train.h5"
    cfg = PretrainH5Config(precompute_joint_energy=True, device="cpu", chunk_size=4)
    writer = PretrainWriter(
        path,
        32,
        16,
        0,
        path,
        dataset_name="radchar",
        cfg=cfg,
    )
    iq = np.random.randn(3, 2, 32).astype(np.float32) * 0.5
    writer.append_prepared(
        iq,
        mod_label_id=np.array([0, 1, 2], dtype=np.int16),
        snr=np.full(3, 10.0, dtype=np.float32),
    )
    writer.close()
    ds = RFDataH5Dataset(path, iq_normalize="none", use_labels=True)
    batch = variable_length_collate([ds[i] for i in range(3)])
    assert "revin_mean" in batch
    assert batch["revin_mean"].shape == (3, 2)
    assert batch.get("iq_preprocessed") is True
    for key in ("norm_scale", "log_scale", "log_peak", "papr_preclip", "scale_gap"):
        assert key in batch


def test_model_skip_precomputed_revin(tmp_path: Path) -> None:
    path = tmp_path / "rml2016_04c_train.h5"
    cfg = PretrainH5Config(precompute_joint_energy=True, device="cpu", chunk_size=4)
    writer = PretrainWriter(
        path,
        32,
        2,
        0,
        path,
        dataset_name="rml2016_04c",
        cfg=cfg,
    )
    iq = np.random.randn(2, 2, 32).astype(np.float32) * 0.3
    writer.append_prepared(
        iq,
        mod_label_id=np.array([0, 1], dtype=np.int16),
        snr=np.full(2, 10.0, dtype=np.float32),
    )
    writer.close()
    ds = RFDataH5Dataset(path, iq_normalize="none", use_labels=False)
    batch = variable_length_collate([ds[0], ds[1]])
    model = SignalFoundationModel(
        SignalModelConfig(
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
            build_task_heads=False,
            build_task_interface=False,
            build_prototype_registry=False,
        )
    )
    stats = model._precomputed_revin_from_batch(
        batch,
        device=torch.device("cpu"),
        batch_size=2,
    )
    assert stats is not None
    assert stats.mean.shape == (2, 2)
    batch_no_flag = dict(batch)
    batch_no_flag["iq_preprocessed"] = False
    assert model._precomputed_revin_from_batch(batch_no_flag, device=torch.device("cpu"), batch_size=2) is None


def test_stage2_skip_revin_without_online_normalize(tmp_path: Path) -> None:
    from resmamba_signal_model.training.freeze import apply_stage_freeze

    path = tmp_path / "rml2016_04c_train.h5"
    cfg = PretrainH5Config(precompute_joint_energy=True, device="cpu", chunk_size=4)
    writer = PretrainWriter(
        path,
        32,
        2,
        0,
        path,
        dataset_name="rml2016_04c",
        cfg=cfg,
    )
    iq = np.random.randn(2, 2, 32).astype(np.float32) * 0.3
    writer.append_prepared(
        iq,
        mod_label_id=np.array([0, 1], dtype=np.int16),
        snr=np.full(2, 10.0, dtype=np.float32),
    )
    writer.close()
    ds = RFDataH5Dataset(path, iq_normalize="none", use_labels=False)
    batch = variable_length_collate([ds[0], ds[1]])
    model = SignalFoundationModel(
        SignalModelConfig(
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
            build_task_heads=True,
            build_task_interface=False,
            build_prototype_registry=False,
        )
    )
    apply_stage_freeze(model, "stage2", task="tx_modulation", train_cfg={"skip_revin": True, "skip_recon": True})
    called: list[int] = []
    orig = model.revin.normalize

    def spy(*args, **kwargs):
        called.append(1)
        return orig(*args, **kwargs)

    model.revin.normalize = spy  # type: ignore[method-assign]
    model(batch, mode="task", task="tx_modulation")
    assert called == []
    assert model.skip_revin is True

    raw_batch = {
        "iq": torch.randn(2, 2, 32),
        "values": torch.randn(2, 2, 32),
        "length": torch.tensor([32, 32]),
    }
    called.clear()
    model(raw_batch, mode="task", task="tx_modulation")
    assert called == []


def test_write_pretrain_stratified_splits_three_files(tmp_path: Path) -> None:
    ctx = Context(tmp_path)
    labels_map = {"0": 0, "1": 1}
    labels_all = np.tile(np.arange(2, dtype=np.int32), 30)
    iq_all = np.random.randn(60, 2, 16).astype(np.float32)
    write_pretrain_stratified_splits(
        ctx,
        "radchar",
        16,
        tmp_path,
        16,
        np.float32,
        iq_all,
        labels_all,
        labels_map,
        __import__("collections").Counter(),
        60,
        label_field="mod_label_id",
        extra_meta={"snr": np.zeros(60, dtype=np.float32)},
    )
    for split in ("train", "test", "val"):
        p = ctx.h5 / f"radchar_{split}.h5"
        assert p.is_file()
        with h5py.File(p, "r") as f:
            assert int(f.attrs.get("h5_schema_version", 0)) in (0, H5_SCHEMA_VERSION)
    report = ctx.report["datasets"]["radchar"]
    for split in ("train", "test", "val"):
        assert "removed" in report["splits"][split]


def test_load_h5_records_v2_meta_keys(tmp_path: Path) -> None:
    path = tmp_path / "cjr_mix_val.h5"
    cfg = PretrainH5Config(precompute_joint_energy=True, device="cpu", chunk_size=4)
    writer = PretrainWriter(
        path,
        16,
        31,
        0,
        path,
        dataset_name="cjr_mix",
        cfg=cfg,
    )
    iq = np.random.randn(6, 2, 16).astype(np.float32) * 0.2
    writer.append_prepared(
        iq,
        mod_label_id=np.array([0, 1, 0, 1, 2, 2], dtype=np.int16),
        source_label_id=np.array([0, 1, 0, 1, 2, 2], dtype=np.int16),
        snr=np.full(6, 5.0, dtype=np.float32),
    )
    writer.close()
    with h5py.File(path, "r") as f:
        assert "dataset_id" not in f
        assert int(f.attrs["dataset_id"]) == 31
        assert _h5_dataset_id(f) == 31
    loaded_iq, meta = _load_h5_records(path)
    assert loaded_iq.shape == iq.shape
    assert meta["mod_label_id"] is not None
    assert meta["revin_mean"] is not None
    assert meta["norm_scale"] is not None
    assert int(meta["mod_label_id"][0]) == 0


def test_skip_eval_refine_for_pretrain_stem(tmp_path: Path) -> None:
    ctx = Context(tmp_path)
    ctx.report.setdefault("policy", {})["h5_schema"] = "pretrain_v2_joint_energy"
    ctx.report["datasets"]["cjr_mix"] = {
        "stratified_split": {"label_field": "mod_label_id"},
        "splits": {
            "train": {"file": "cjr_mix_train.h5"},
            "test": {"file": "cjr_mix_test.h5"},
            "val": {"file": "cjr_mix_val.h5"},
        },
    }
    assert _skip_eval_refine_for_pretrain(ctx, "cjr_mix") is True
    assert _skip_eval_refine_for_pretrain(ctx, "xidian14") is False
    refine_eval_splits(ctx, datasets={"cjr_mix", "xidian14"})
    assert "eval_quality_refine" not in ctx.report["datasets"]["cjr_mix"]
