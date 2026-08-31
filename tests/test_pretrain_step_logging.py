from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resmamba_signal_model.training.lit_module import _resolve_batch_stem


def test_resolve_batch_stem_from_moe_route() -> None:
    batch = {"moe_route_stem": ["rml2016_10a", "rml2016_10a"]}
    assert _resolve_batch_stem(batch) == "rml2016_10a"

    assert _resolve_batch_stem({"moe_route_stem": "radchar"}) == "radchar"


def test_resolve_batch_stem_missing() -> None:
    assert _resolve_batch_stem({"length": 128}) is None
