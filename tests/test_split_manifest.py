from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from resmamba_signal_model.data.contracts import MISSING_METADATA
from resmamba_signal_model.data.splits import (
    ImmutableManifestError,
    SplitOverlapError,
    UnverifiableGroupSplitError,
    assert_manifest_files_unchanged,
    build_split_manifest,
    load_manifest,
    write_immutable_manifest,
)


def _write_split(
    path: Path,
    *,
    capture_ids: list[str],
    receiver_ids: list[str] | None = None,
    value: float = 1.0,
) -> None:
    n = len(capture_ids)
    receiver_ids = receiver_ids or [MISSING_METADATA] * n
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as handle:
        handle.attrs["source_path"] = "synthetic"
        handle.attrs["signal_contract_version"] = 1
        handle.create_dataset("iq", data=np.full((n, 2, 8), value, dtype=np.float32))
        handle.create_dataset("mod_label_id", data=np.arange(n, dtype=np.int32) % 2)
        handle.create_dataset("snr", data=np.linspace(0, 10, n, dtype=np.float32))
        handle.create_dataset(
            "capture_id",
            data=np.asarray(capture_ids, dtype=object),
            dtype=string_dtype,
        )
        handle.create_dataset(
            "receiver_id",
            data=np.asarray(receiver_ids, dtype=object),
            dtype=string_dtype,
        )


def test_manifest_hashes_data_and_is_immutable(tmp_path: Path) -> None:
    h5_dir = tmp_path / "h5"
    h5_dir.mkdir()
    _write_split(
        h5_dir / "wisig_train.h5",
        capture_ids=["rx0-day0", "rx0-day0"],
        receiver_ids=["rx0", "rx0"],
        value=1.0,
    )
    _write_split(
        h5_dir / "wisig_val.h5",
        capture_ids=["rx1-day0"],
        receiver_ids=["rx1"],
        value=2.0,
    )
    payload = build_split_manifest(
        h5_dir,
        group_split_claims={"wisig": "receiver_day_group_held_out_v1"},
    )
    assert payload["datasets"]["wisig"]["group_integrity"]["status"] == "verified_group_held_out"
    assert payload["datasets"]["wisig"]["files"][0]["sha256"]
    assert payload["datasets"]["wisig"]["files"][0]["receiver_distribution"] == {"rx0": 2}

    manifest_path = tmp_path / "split_manifest.json"
    digest = write_immutable_manifest(manifest_path, payload)
    assert write_immutable_manifest(manifest_path, payload) == digest
    loaded = load_manifest(manifest_path)
    assert_manifest_files_unchanged(loaded, h5_dir)

    changed = dict(payload)
    changed["hash_algorithm"] = "not-sha256"
    with pytest.raises(ImmutableManifestError):
        write_immutable_manifest(manifest_path, changed)


def test_manifest_rejects_capture_overlap(tmp_path: Path) -> None:
    h5_dir = tmp_path / "h5"
    h5_dir.mkdir()
    _write_split(h5_dir / "wisig_train.h5", capture_ids=["same-capture"], value=1.0)
    _write_split(h5_dir / "wisig_val.h5", capture_ids=["same-capture"], value=2.0)
    with pytest.raises(SplitOverlapError, match="重叠"):
        build_split_manifest(h5_dir)


def test_missing_groups_are_explicit_and_cannot_support_heldout_claim(tmp_path: Path) -> None:
    h5_dir = tmp_path / "h5"
    h5_dir.mkdir()
    _write_split(
        h5_dir / "adsb2_train.h5",
        capture_ids=[MISSING_METADATA],
        value=1.0,
    )
    _write_split(
        h5_dir / "adsb2_val.h5",
        capture_ids=[MISSING_METADATA],
        value=2.0,
    )
    payload = build_split_manifest(h5_dir)
    integrity = payload["datasets"]["adsb2"]["group_integrity"]
    assert integrity["status"] == "unverifiable_missing_metadata"
    assert integrity["claim_group_held_out"] is False
    assert "不能据此宣称" in integrity["reason"]
    with pytest.raises(UnverifiableGroupSplitError):
        build_split_manifest(h5_dir, group_split_claims={"adsb2": "capture_group_held_out"})
