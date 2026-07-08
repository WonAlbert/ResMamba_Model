from resmamba_signal_model.training.selection import compute_selection_score, resolve_selection_metric_name


def test_emitter_default_selection_metric() -> None:
    assert resolve_selection_metric_name({}, stage="stage2", task="emitter") == "per_dataset_macro_acc"
    assert resolve_selection_metric_name({}, stage="stage2", task="modulation") == "val_loss"


def test_per_dataset_macro_acc() -> None:
    metrics = {
        "loss": 2.5,
        "acc": 0.16,
        "acc/adsb2": 0.20,
        "acc/wifi150": 0.18,
        "acc/radar_emitters": 0.40,
    }
    score = compute_selection_score(metrics, "per_dataset_macro_acc")
    assert abs(score - (0.20 + 0.18 + 0.40) / 3) < 1e-6


def test_val_loss_selection_is_negated() -> None:
    metrics = {"loss": 2.0, "acc": 0.1}
    assert compute_selection_score(metrics, "val_loss") == -2.0
