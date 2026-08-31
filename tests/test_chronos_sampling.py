from __future__ import annotations

import torch

from resmamba_signal_model.data.chronos_sampling import apply_iq_mixup, parse_chronos_sampling_cfg
from resmamba_signal_model.data.sampling import HomogeneousTokenBudgetSampler, pool_segments
from resmamba_signal_model.data.rfdata import RFDataPoolDataset


def test_parse_chronos_sampling_cfg() -> None:
    cfg = parse_chronos_sampling_cfg(
        {
            "chronos_sampling": {
                "enabled": True,
                "length_tier_pool": True,
                "stem_sticky_batches": 0,
                "shuffle_buffer_batches": 32,
                "iq_mixup": {"enabled": True, "k": 2, "alpha": 0.4},
            }
        }
    )
    assert cfg["enabled"] is True
    assert cfg["length_tier_pool"] is True
    assert cfg["shuffle_buffer_batches"] == 32
    assert cfg["iq_mixup_k"] == 2


def test_apply_iq_mixup_changes_tensor() -> None:
    iq = torch.randn(4, 2, 16)
    batch = {"iq": iq.clone(), "values": iq.clone(), "moe_route_stem": ["a", "b", "c", "d"]}
    out = apply_iq_mixup(batch, k=2, alpha=0.5)
    assert not torch.allclose(out["iq"], iq)
    assert len(out["moe_route_stem"]) == 4


def test_chronos_tier_pool_can_mix_stems_in_batch(tmp_path) -> None:
    """128 长度 tier 内可混 rml2016_10a / 10b（Chronos 式跨库同 batch）。"""
    from tests.test_homogeneous_sampler import _write_h5
    from resmamba_signal_model.data.rfdata import RFDataH5Dataset

    path_a = tmp_path / "rml2016_10a_train.h5"
    path_b = tmp_path / "rml2016_10b_train.h5"
    _write_h5(path_a, n=40, length=128, dataset_id=1)
    _write_h5(path_b, n=40, length=128, dataset_id=2)
    pool = RFDataPoolDataset(
        [RFDataH5Dataset(p, use_labels=False) for p in (path_a, path_b)],
        pool_name="pretrain_train",
    )
    lengths = [128] * 80
    sampler = HomogeneousTokenBudgetSampler(
        pool,
        token_budget=512,
        patch_size=8,
        num_batches=30,
        seed=0,
        lengths=lengths,
        homogeneous_length_tier=True,
        chronos_length_tier_pool=True,
    )
    segments = pool_segments(pool)
    seg_a = next(i for i, s in enumerate(segments) if "10a" in s.h5_name)
    seg_b = next(i for i, s in enumerate(segments) if "10b" in s.h5_name)
    mixed = False
    for batch in sampler:
        seg_ids = set()
        for idx in batch:
            for si, seg in enumerate(segments):
                if seg.offset <= idx < seg.offset + seg.size:
                    seg_ids.add(si)
                    break
        if seg_a in seg_ids and seg_b in seg_ids:
            mixed = True
            break
    assert mixed, "tier pool 应在同 batch 内混合 10a/10b"
