from resmamba_signal_model.data.rfdata import build_rfdata_pool, pad_iq_collate
from scripts.train_pipeline import build_val_dataloader


def test_val_length_bucket_dataloader_fetch() -> None:
    pool = build_rfdata_pool("dataset", "downstream_prediction_val")
    loader = build_val_dataloader(
        pool,
        batch_size=4,
        collate_fn=pad_iq_collate,
        num_workers=0,
        pin_memory=False,
        stage="stage2",
        task="prediction",
        subset_fraction=0.01,
        subset_seed=1,
        epoch=1,
        resample_each_epoch=False,
        max_per_dataset=20,
        length_bucket_batching=True,
    )
    batch = next(iter(loader))
    assert "iq" in batch
    assert batch["iq"].shape[0] <= 4
