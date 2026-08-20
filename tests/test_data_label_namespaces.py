from __future__ import annotations

from pathlib import Path
import sys

import h5py
import numpy as np

from resmamba_signal_model.data.labels import (
    build_emitter_namespace,
    build_modulation_ontology,
    canonicalize_modulation_name,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_datasets import Context, Writer, stamp_semantic_namespaces


def test_modulation_aliases_share_one_canonical_id() -> None:
    ontology = build_modulation_ontology({
        "rml2016": {"QAM16": 0, "B-PSK": 1, "QPSK": 2},
        "rml2018": {"16QAM": 7, "2PSK": 8, "4PSK": 9},
        "opaque": {"0": 0, "99": 99},
    })
    qam_id = ontology.canonical_to_id["16QAM"]
    assert ontology.dataset_local_to_canonical["rml2016"][0] == qam_id
    assert ontology.dataset_local_to_canonical["rml2018"][7] == qam_id
    assert ontology.dataset_local_to_canonical["rml2016"][1] == ontology.canonical_to_id["BPSK"]
    assert ontology.dataset_local_to_canonical["rml2018"][9] == ontology.canonical_to_id["QPSK"]
    assert ontology.dataset_local_to_canonical["opaque"] == {}
    assert ontology.unresolved["opaque"] == ("0", "99")
    assert canonicalize_modulation_name("qam-64") == "64QAM"


def test_existing_ontology_ids_are_not_reassigned() -> None:
    first = build_modulation_ontology({"a": {"QAM16": 0}})
    second = build_modulation_ontology(
        {"a": {"QAM16": 0}, "b": {"BPSK": 4}},
        existing=first.to_dict(),
    )
    assert second.canonical_to_id["16QAM"] == first.canonical_to_id["16QAM"]


def test_emitter_namespace_prevents_cross_dataset_collisions() -> None:
    namespace = build_emitter_namespace({
        "adsb": {"device-a": 10},
        "wisig": {"device-a": 10, "device-b": 30},
    })
    adsb_id = namespace.dataset_local_to_global["adsb"][10]
    wisig_id = namespace.dataset_local_to_global["wisig"][10]
    assert adsb_id != wisig_id
    assert sorted(namespace.namespaced_to_id.values()) == [0, 1, 2]
    values = namespace.map_local("wisig", np.asarray([10, 30, 999]))
    assert values[0] != values[1]
    assert values[2] == -1


def test_stamp_semantic_namespaces_keeps_local_labels(tmp_path: Path) -> None:
    ctx = Context(tmp_path)
    ctx.maps["datasets"] = {"2": "rml", "11": "wisig"}
    ctx.maps["modulations"] = {"rml": {"QAM16": 4}}
    ctx.maps["emitters"] = {"wisig": {"tx-a": 5}}

    mod_path = ctx.h5 / "rml_train.h5"
    mod_writer = Writer(mod_path, 8, np.float32, 2, 0, mod_path)
    mod_writer.append_raw(
        np.ones((2, 2, 8), dtype=np.float32),
        mod_label_id=np.asarray([4, 4], dtype=np.int32),
    )
    mod_writer.close()

    emitter_path = ctx.h5 / "wisig_train.h5"
    emitter_writer = Writer(emitter_path, 8, np.float32, 11, 1, emitter_path)
    emitter_writer.append_raw(
        np.ones((2, 2, 8), dtype=np.float32) * 2,
        emitter_id=np.asarray([5, 5], dtype=np.int32),
    )
    emitter_writer.close()

    stats = stamp_semantic_namespaces(ctx)
    assert stats["canonical_mod_label_id"]["rml_train.h5"] == 2
    assert stats["global_emitter_id"]["wisig_train.h5"] == 2
    with h5py.File(mod_path, "r") as handle:
        assert handle["mod_label_id"][:].tolist() == [4, 4]
        assert handle["canonical_mod_label_id"][:].tolist() == [
            ctx.maps["canonical_modulations"]["16QAM"],
        ] * 2
    with h5py.File(emitter_path, "r") as handle:
        assert handle["emitter_id"][:].tolist() == [5, 5]
        assert np.all(handle["global_emitter_id"][:] >= 0)
