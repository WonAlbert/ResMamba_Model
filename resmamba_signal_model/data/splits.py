from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import h5py
import numpy as np

from resmamba_signal_model.data.contracts import CAPTURE_METADATA_KEYS, MISSING_METADATA


SPLIT_MANIFEST_VERSION = 1
SPLIT_NAMES = ("train", "val", "test")


class SplitManifestError(RuntimeError):
    pass


class SplitOverlapError(SplitManifestError):
    pass


class UnverifiableGroupSplitError(SplitManifestError):
    pass


class ImmutableManifestError(SplitManifestError):
    pass


def parse_split_filename(filename: str) -> tuple[str, str]:
    for split in SPLIT_NAMES:
        suffix = f"_{split}.h5"
        if filename.endswith(suffix):
            return filename[: -len(suffix)], split
    raise ValueError(f"无法从文件名解析 split: {filename!r}")


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _decode(value: Any) -> str:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if value is None:
        return MISSING_METADATA
    text = str(value)
    return text if text else MISSING_METADATA


def summarize_h5_split(path: str | Path) -> tuple[dict[str, Any], set[str], int]:
    path = Path(path)
    dataset_name, split = parse_split_filename(path.name)
    with h5py.File(path, "r") as handle:
        count = int(handle["iq"].shape[0])
        metadata_known: Counter[str] = Counter()
        receiver_counts: Counter[str] = Counter()
        capture_groups: set[str] = set()
        receiver_session_groups: set[str] = set()
        capture_missing = 0
        receiver_session_missing = 0
        label_keys = (
            "mod_label_id",
            "canonical_mod_label_id",
            "emitter_id",
            "global_emitter_id",
            "source_label_id",
            "global_label_id",
        )
        label_counts = {key: Counter() for key in label_keys}
        snr_known = 0
        snr_missing = 0
        snr_min = float("inf")
        snr_max = float("-inf")
        snr_sum = 0.0
        snr_distribution: Counter[float] | None = Counter()
        for start in range(0, count, 65536):
            end = min(start + 65536, count)
            size = end - start
            metadata_columns = {
                key: (
                    np.asarray([_decode(value) for value in handle[key][start:end]], dtype=object)
                    if key in handle and int(handle[key].shape[0]) == count
                    else np.full(size, MISSING_METADATA, dtype=object)
                )
                for key in CAPTURE_METADATA_KEYS
            }
            for key, values in metadata_columns.items():
                metadata_known[key] += int(np.sum(values != MISSING_METADATA))
            capture = metadata_columns["capture_id"]
            capture_missing += int(np.sum(capture == MISSING_METADATA))
            capture_groups.update(str(value) for value in capture if value != MISSING_METADATA)
            receiver = metadata_columns["receiver_id"]
            session = metadata_columns["session_id"]
            receiver_counts.update(str(value) for value in receiver)
            combined_known = (receiver != MISSING_METADATA) & (session != MISSING_METADATA)
            receiver_session_missing += int(np.sum(~combined_known))
            receiver_session_groups.update(
                f"receiver={rx}|session={session_id}"
                for rx, session_id in zip(
                    receiver[combined_known],
                    session[combined_known],
                    strict=True,
                )
            )
            for key, counter in label_counts.items():
                if key not in handle or int(handle[key].shape[0]) != count:
                    continue
                values = np.asarray(handle[key][start:end], dtype=np.int64)
                counter.update(int(value) for value in values[values >= 0])
            if "snr" in handle and int(handle["snr"].shape[0]) == count:
                snr_values = np.asarray(handle["snr"][start:end], dtype=np.float64)
                valid = np.isfinite(snr_values) & (snr_values > -900)
                finite = snr_values[valid]
                snr_known += int(valid.sum())
                snr_missing += int((~valid).sum())
                if finite.size:
                    snr_min = min(snr_min, float(finite.min()))
                    snr_max = max(snr_max, float(finite.max()))
                    snr_sum += float(finite.sum())
                    if snr_distribution is not None:
                        snr_distribution.update(float(value) for value in finite)
                        if len(snr_distribution) > 128:
                            snr_distribution = None
            else:
                snr_missing += size
        if capture_groups:
            known_groups = capture_groups
            group_field = "capture_id"
            missing_groups = capture_missing
        else:
            known_groups = receiver_session_groups
            group_field = "receiver_id+session_id"
            missing_groups = receiver_session_missing
        metadata_availability = {
            key: {
                "known": int(metadata_known[key]),
                "missing": int(count - metadata_known[key]),
            }
            for key in CAPTURE_METADATA_KEYS
        }
        labels = {
            key: {str(label): int(n) for label, n in sorted(counter.items())}
            for key, counter in label_counts.items()
            if counter
        }
        if snr_known:
            snr = {
                "known": snr_known,
                "missing": snr_missing,
                "min": snr_min,
                "max": snr_max,
                "mean": snr_sum / snr_known,
                "distribution": (
                    {str(value): int(n) for value, n in sorted(snr_distribution.items())}
                    if snr_distribution is not None
                    else {}
                ),
            }
        else:
            snr = {"known": 0, "missing": count}
        source_path = _decode(handle.attrs.get("source_path", MISSING_METADATA))
        contract_version = int(handle.attrs.get("signal_contract_version", 0))
    summary = {
        "dataset": dataset_name,
        "split": split,
        "file": path.name,
        "samples": count,
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "source_path": source_path,
        "signal_contract_version": contract_version,
        "group_field": group_field,
        "known_groups": len(known_groups),
        "missing_group_samples": missing_groups,
        "receiver_distribution": dict(sorted(receiver_counts.items())),
        "label_distributions": labels,
        "snr": snr,
        "metadata_availability": metadata_availability,
    }
    return summary, known_groups, missing_groups


def validate_group_disjoint(
    split_groups: Mapping[str, set[str]],
    *,
    dataset_name: str,
    missing_group_samples: int = 0,
    require_verifiable: bool = False,
) -> dict[str, Any]:
    overlaps: dict[str, list[str]] = {}
    present = [split for split in SPLIT_NAMES if split in split_groups]
    for left_index, left in enumerate(present):
        for right in present[left_index + 1 :]:
            shared = split_groups[left] & split_groups[right]
            if shared:
                overlaps[f"{left}:{right}"] = sorted(shared)
    if overlaps:
        preview = {key: values[:10] for key, values in overlaps.items()}
        raise SplitOverlapError(f"{dataset_name} 的 capture/group 跨 split 重叠: {preview}")

    known_group_count = sum(len(groups) for groups in split_groups.values())
    if len(present) < 2:
        status = "not_applicable_single_split"
    elif known_group_count == 0:
        status = "unverifiable_missing_metadata"
    elif missing_group_samples:
        status = "unverifiable_partial_missing_metadata"
    else:
        status = "verified_group_held_out"
    if require_verifiable and status != "verified_group_held_out":
        raise UnverifiableGroupSplitError(
            f"{dataset_name} 声称 group-held-out，但验证状态为 {status}；"
            f"缺失 group 元数据的样本数为 {missing_group_samples}"
        )
    return {
        "status": status,
        "claim_group_held_out": status == "verified_group_held_out",
        "splits_checked": present,
        "known_groups": known_group_count,
        "missing_group_samples": int(missing_group_samples),
        "overlap_count": 0,
    }


def build_split_manifest(
    h5_dir: str | Path,
    *,
    group_split_claims: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    h5_dir = Path(h5_dir)
    claims = {str(key): str(value) for key, value in (group_split_claims or {}).items()}
    paths = []
    for path in sorted(h5_dir.glob("*.h5")):
        if "__balanced_" in path.name:
            continue
        try:
            parse_split_filename(path.name)
        except ValueError:
            continue
        paths.append(path)
    dataset_files: dict[str, list[Path]] = {}
    for path in paths:
        dataset_name, _ = parse_split_filename(path.name)
        dataset_files.setdefault(dataset_name, []).append(path)

    datasets: dict[str, Any] = {}
    seen_file_hashes: dict[str, tuple[str, str]] = {}
    for dataset_name, dataset_paths in sorted(dataset_files.items()):
        file_summaries: list[dict[str, Any]] = []
        summarized: list[tuple[dict[str, Any], set[str], int]] = []
        for path in sorted(dataset_paths):
            summarized.append(summarize_h5_split(path))
        preferred_group_field = (
            "capture_id"
            if any(summary["group_field"] == "capture_id" for summary, _groups, _missing in summarized)
            else "receiver_id+session_id"
        )
        split_groups: dict[str, set[str]] = {}
        missing_groups = 0
        for summary, groups, missing in summarized:
            file_summaries.append(summary)
            split = str(summary["split"])
            if summary["group_field"] != preferred_group_field:
                # 数据集内统一使用最强 group 字段，不能让某个 split 靠较弱字段冒充可验证。
                groups = set()
                missing = int(summary["samples"])
            split_groups.setdefault(split, set()).update(groups)
            missing_groups += missing
            file_hash = str(summary["sha256"])
            previous = seen_file_hashes.get(file_hash)
            if previous is not None and previous != (dataset_name, split) and int(summary["samples"]) > 0:
                raise SplitOverlapError(
                    f"{summary['file']} 与 {previous[0]}:{previous[1]} 的完整 H5 哈希相同，拒绝跨 split 重复"
                )
            seen_file_hashes[file_hash] = (dataset_name, split)
        integrity = validate_group_disjoint(
            split_groups,
            dataset_name=dataset_name,
            missing_group_samples=missing_groups,
            require_verifiable=dataset_name in claims,
        )
        if dataset_name in claims:
            integrity["strategy"] = claims[dataset_name]
        elif integrity["status"].startswith("unverifiable"):
            integrity["reason"] = "capture/receiver/session 元数据缺失，不能据此宣称 group-held-out"
        datasets[dataset_name] = {
            "files": file_summaries,
            "group_field": preferred_group_field,
            "group_integrity": integrity,
        }

    payload: dict[str, Any] = {
        "schema_version": SPLIT_MANIFEST_VERSION,
        "hash_algorithm": "sha256",
        "missing_metadata_marker": MISSING_METADATA,
        "datasets": datasets,
    }
    payload["manifest_sha256"] = manifest_digest(payload)
    return payload


def manifest_digest(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    encoded = json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_manifest(payload: Mapping[str, Any]) -> None:
    expected = str(payload.get("manifest_sha256", ""))
    actual = manifest_digest(payload)
    if not expected or expected != actual:
        raise SplitManifestError(f"split manifest 摘要不匹配: expected={expected!r}, actual={actual}")
    for dataset_name, entry in payload.get("datasets", {}).items():
        integrity = entry.get("group_integrity", {})
        if int(integrity.get("overlap_count", 0)) != 0:
            raise SplitOverlapError(f"{dataset_name} manifest 记录了 group overlap")


def write_immutable_manifest(path: str | Path, payload: Mapping[str, Any]) -> str:
    path = Path(path)
    normalized = dict(payload)
    normalized["manifest_sha256"] = manifest_digest(normalized)
    verify_manifest(normalized)
    text = json.dumps(normalized, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        verify_manifest(existing)
        if existing != normalized:
            raise ImmutableManifestError(
                f"{path} 已存在且内容不同；split manifest 不允许原地覆盖，请使用新的输出目录"
            )
        return str(normalized["manifest_sha256"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
    path.chmod(0o444)
    return str(normalized["manifest_sha256"])


def load_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    verify_manifest(payload)
    return payload


def assert_manifest_files_unchanged(
    manifest: Mapping[str, Any],
    h5_dir: str | Path,
    *,
    datasets: Iterable[str] | None = None,
) -> None:
    selected = set(datasets) if datasets is not None else None
    root = Path(h5_dir)
    for dataset_name, entry in manifest.get("datasets", {}).items():
        if selected is not None and dataset_name not in selected:
            continue
        for file_info in entry.get("files", []):
            path = root / str(file_info["file"])
            if not path.is_file():
                raise SplitManifestError(f"manifest 文件不存在: {path}")
            expected = str(file_info["sha256"])
            actual = sha256_file(path)
            if expected != actual:
                raise SplitManifestError(f"{path.name} 数据哈希已变化: {expected} != {actual}")
