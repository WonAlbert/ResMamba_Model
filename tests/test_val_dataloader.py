from pathlib import Path

import pytest
import torch

from resmamba_signal_model.data.rfdata import build_rfdata_pool, variable_length_collate
from resmamba_signal_model.data.sampling import TokenBudgetSampler, pool_sample_lengths


@pytest.mark.skipif(not Path("dataset/label_maps.json").exists(), reason="no rfdata")
def test_val_token_budget_dataloader_fetch() -> None:
    pool = build_rfdata_pool("dataset", "downstream_prediction_val")
    lengths = pool_sample_lengths(pool)
    sampler = TokenBudgetSampler(lengths, token_budget=64, patch_size=8, num_batches=1, seed=1)
    loader = torch.utils.data.DataLoader(pool, batch_sampler=sampler, collate_fn=variable_length_collate, num_workers=0)
    batch = next(iter(loader))
    assert "iq" in batch
    iq = batch["iq"]
    if torch.is_tensor(iq):
        assert iq.ndim == 3
        assert iq.shape[0] >= 1
    else:
        assert isinstance(iq, list)
        assert len(iq) >= 1
