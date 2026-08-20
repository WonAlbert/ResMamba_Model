from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

import numpy as np


MODULATION_ONTOLOGY_VERSION = 1
EMITTER_NAMESPACE_VERSION = 1

# 先固定常见公开基准的顺序；后续未知但具名的类别只追加，不重排已有 ID。
CANONICAL_MODULATION_ORDER = (
    "OOK",
    "BPSK",
    "QPSK",
    "OQPSK",
    "8PSK",
    "16PSK",
    "32PSK",
    "4ASK",
    "8ASK",
    "4PAM",
    "CPFSK",
    "GFSK",
    "GMSK",
    "16QAM",
    "32QAM",
    "64QAM",
    "128QAM",
    "256QAM",
    "16APSK",
    "32APSK",
    "64APSK",
    "128APSK",
    "AM-DSB",
    "AM-DSB-SC",
    "AM-DSB-WC",
    "AM-SSB",
    "AM-SSB-SC",
    "AM-SSB-WC",
    "FM",
    "WBFM",
)

MODULATION_ALIASES = {
    "2PSK": "BPSK",
    "PSK2": "BPSK",
    "B-PSK": "BPSK",
    "4PSK": "QPSK",
    "PSK4": "QPSK",
    "Q-PSK": "QPSK",
    "PSK8": "8PSK",
    "PSK16": "16PSK",
    "PSK32": "32PSK",
    "QAM16": "16QAM",
    "QAM-16": "16QAM",
    "16-QAM": "16QAM",
    "QAM32": "32QAM",
    "QAM-32": "32QAM",
    "32-QAM": "32QAM",
    "QAM64": "64QAM",
    "QAM-64": "64QAM",
    "64-QAM": "64QAM",
    "QAM128": "128QAM",
    "QAM-128": "128QAM",
    "128-QAM": "128QAM",
    "QAM256": "256QAM",
    "QAM-256": "256QAM",
    "256-QAM": "256QAM",
    "PAM4": "4PAM",
    "PAM-4": "4PAM",
    "AMDSB": "AM-DSB",
    "AMSSB": "AM-SSB",
}


def _alias_key(name: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(name).strip().upper())


_ALIASES_BY_KEY = {
    _alias_key(alias): canonical
    for alias, canonical in {
        **{name: name for name in CANONICAL_MODULATION_ORDER},
        **MODULATION_ALIASES,
    }.items()
}


def modulation_name_is_recoverable(name: object) -> bool:
    text = str(name).strip()
    if not text or text.isdigit():
        return False
    return re.fullmatch(r"(?:CLASS|LABEL|MOD)[-_ ]?\d+", text, flags=re.IGNORECASE) is None


def canonicalize_modulation_name(name: object) -> str:
    """将跨数据集别名归一为语义名称，不把纯局部数字伪装成全局类别。"""
    text = str(name).strip()
    if not modulation_name_is_recoverable(text):
        raise ValueError(f"局部标签 {text!r} 不含可恢复的调制语义")
    key = _alias_key(text)
    known = _ALIASES_BY_KEY.get(key)
    if known is not None:
        return known
    qam = re.fullmatch(r"(?:QAM(\d+)|(\d+)QAM)", key)
    if qam:
        return f"{qam.group(1) or qam.group(2)}QAM"
    psk = re.fullmatch(r"(?:PSK(\d+)|(\d+)PSK)", key)
    if psk:
        order = int(psk.group(1) or psk.group(2))
        return {2: "BPSK", 4: "QPSK"}.get(order, f"{order}PSK")
    apsk = re.fullmatch(r"(?:APSK(\d+)|(\d+)APSK)", key)
    if apsk:
        return f"{apsk.group(1) or apsk.group(2)}APSK"
    ask = re.fullmatch(r"(?:ASK(\d+)|(\d+)ASK)", key)
    if ask:
        return f"{ask.group(1) or ask.group(2)}ASK"
    # 对不在内置表但具名的公开类别保留可读、稳定的规范形式。
    return re.sub(r"[-_ ]+", "-", text.upper()).strip("-")


def _existing_ids(payload: Mapping[str, Any] | None, key: str) -> dict[str, int]:
    if not payload:
        return {}
    raw = payload.get(key, {})
    if not isinstance(raw, Mapping):
        return {}
    out = {str(name): int(idx) for name, idx in raw.items()}
    if len(set(out.values())) != len(out):
        raise ValueError(f"{key} 中存在重复 ID")
    if sorted(out.values()) != list(range(len(out))):
        raise ValueError(f"{key} 必须是从 0 开始的连续命名空间")
    return out


@dataclass(frozen=True)
class ModulationOntology:
    canonical_to_id: dict[str, int]
    dataset_local_to_canonical: dict[str, dict[int, int]]
    dataset_local_names: dict[str, dict[int, str]]
    unresolved: dict[str, tuple[str, ...]]
    version: int = MODULATION_ONTOLOGY_VERSION

    def map_local(self, dataset_name: str, local_ids: np.ndarray) -> np.ndarray:
        mapping = self.dataset_local_to_canonical.get(str(dataset_name), {})
        return remap_local_ids(local_ids, mapping)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "canonical_to_id": dict(sorted(self.canonical_to_id.items(), key=lambda item: item[1])),
            "aliases": dict(sorted(MODULATION_ALIASES.items())),
            "dataset_local_to_canonical": {
                dataset: {str(local): canonical for local, canonical in sorted(mapping.items())}
                for dataset, mapping in sorted(self.dataset_local_to_canonical.items())
            },
            "dataset_local_names": {
                dataset: {str(local): name for local, name in sorted(mapping.items())}
                for dataset, mapping in sorted(self.dataset_local_names.items())
            },
            "unresolved": {
                dataset: list(names)
                for dataset, names in sorted(self.unresolved.items())
                if names
            },
        }


def build_modulation_ontology(
    modulation_tables: Mapping[str, Mapping[str, int]],
    *,
    existing: Mapping[str, Any] | None = None,
) -> ModulationOntology:
    canonical_to_id = _existing_ids(existing, "canonical_to_id")
    discovered: set[str] = set()
    names_by_dataset: dict[str, dict[int, str]] = {}
    canonical_names_by_dataset: dict[str, dict[int, str]] = {}
    unresolved: dict[str, tuple[str, ...]] = {}

    for dataset_name, table in sorted(modulation_tables.items()):
        local_names: dict[int, str] = {}
        local_canonical: dict[int, str] = {}
        missing: list[str] = []
        for raw_name, raw_local_id in table.items():
            local_id = int(raw_local_id)
            name = str(raw_name)
            previous = local_names.get(local_id)
            if previous is not None and previous != name:
                raise ValueError(
                    f"{dataset_name} 的局部调制 ID {local_id} 同时对应 {previous!r} 和 {name!r}"
                )
            local_names[local_id] = name
            if not modulation_name_is_recoverable(name):
                missing.append(name)
                continue
            canonical = canonicalize_modulation_name(name)
            local_canonical[local_id] = canonical
            discovered.add(canonical)
        names_by_dataset[str(dataset_name)] = local_names
        canonical_names_by_dataset[str(dataset_name)] = local_canonical
        unresolved[str(dataset_name)] = tuple(sorted(set(missing)))

    next_id = max(canonical_to_id.values(), default=-1) + 1
    ordered = [name for name in CANONICAL_MODULATION_ORDER if name in discovered]
    ordered.extend(sorted(discovered - set(ordered)))
    for name in ordered:
        if name not in canonical_to_id:
            canonical_to_id[name] = next_id
            next_id += 1

    local_to_global = {
        dataset: {
            local_id: canonical_to_id[canonical_name]
            for local_id, canonical_name in mapping.items()
        }
        for dataset, mapping in canonical_names_by_dataset.items()
    }
    return ModulationOntology(
        canonical_to_id=canonical_to_id,
        dataset_local_to_canonical=local_to_global,
        dataset_local_names=names_by_dataset,
        unresolved=unresolved,
    )


@dataclass(frozen=True)
class EmitterNamespace:
    namespaced_to_id: dict[str, int]
    dataset_local_to_global: dict[str, dict[int, int]]
    dataset_local_names: dict[str, dict[int, str]]
    version: int = EMITTER_NAMESPACE_VERSION

    def map_local(self, dataset_name: str, local_ids: np.ndarray) -> np.ndarray:
        return remap_local_ids(local_ids, self.dataset_local_to_global.get(str(dataset_name), {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "namespaced_to_id": dict(sorted(self.namespaced_to_id.items(), key=lambda item: item[1])),
            "dataset_local_to_global": {
                dataset: {str(local): global_id for local, global_id in sorted(mapping.items())}
                for dataset, mapping in sorted(self.dataset_local_to_global.items())
            },
            "dataset_local_names": {
                dataset: {str(local): name for local, name in sorted(mapping.items())}
                for dataset, mapping in sorted(self.dataset_local_names.items())
            },
            "num_emitters": len(self.namespaced_to_id),
        }


def emitter_namespace_key(dataset_name: str, emitter_name: object) -> str:
    return f"{str(dataset_name)}::{str(emitter_name)}"


def build_emitter_namespace(
    emitter_tables: Mapping[str, Mapping[str, int]],
    *,
    existing: Mapping[str, Any] | None = None,
) -> EmitterNamespace:
    namespaced_to_id = _existing_ids(existing, "namespaced_to_id")
    local_names_by_dataset: dict[str, dict[int, str]] = {}
    local_keys_by_dataset: dict[str, dict[int, str]] = {}
    all_keys: list[str] = []
    for dataset_name, table in sorted(emitter_tables.items()):
        local_names: dict[int, str] = {}
        local_keys: dict[int, str] = {}
        for raw_name, raw_local_id in table.items():
            local_id = int(raw_local_id)
            name = str(raw_name)
            previous = local_names.get(local_id)
            if previous is not None and previous != name:
                raise ValueError(
                    f"{dataset_name} 的局部 emitter ID {local_id} 同时对应 {previous!r} 和 {name!r}"
                )
            key = emitter_namespace_key(str(dataset_name), name)
            local_names[local_id] = name
            local_keys[local_id] = key
            all_keys.append(key)
        local_names_by_dataset[str(dataset_name)] = local_names
        local_keys_by_dataset[str(dataset_name)] = local_keys

    next_id = max(namespaced_to_id.values(), default=-1) + 1
    for key in sorted(set(all_keys)):
        if key not in namespaced_to_id:
            namespaced_to_id[key] = next_id
            next_id += 1
    local_to_global = {
        dataset: {local_id: namespaced_to_id[key] for local_id, key in mapping.items()}
        for dataset, mapping in local_keys_by_dataset.items()
    }
    return EmitterNamespace(
        namespaced_to_id=namespaced_to_id,
        dataset_local_to_global=local_to_global,
        dataset_local_names=local_names_by_dataset,
    )


def remap_local_ids(
    local_ids: np.ndarray,
    mapping: Mapping[int, int],
    *,
    missing_value: int = -1,
) -> np.ndarray:
    values = np.asarray(local_ids)
    out = np.full(values.shape, int(missing_value), dtype=np.int32)
    for local_id, global_id in mapping.items():
        out[values == int(local_id)] = int(global_id)
    return out
