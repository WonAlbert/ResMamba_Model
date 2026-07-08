from resmamba_signal_model.training.early_stopping import EarlyStopping


def test_early_stopping_triggers_after_patience() -> None:
    es = EarlyStopping(patience=2, min_delta=0.0)
    assert not es.step(1.0)
    assert not es.step(1.1)
    assert es.step(1.2)


def test_early_stopping_resets_on_improvement() -> None:
    es = EarlyStopping(patience=2, min_delta=0.01)
    assert not es.step(1.0)
    assert not es.step(1.05)
    assert not es.step(0.98)
    assert not es.step(1.0)
    assert es.step(1.02)


def test_early_stopping_disabled_when_patience_zero() -> None:
    es = EarlyStopping(patience=0)
    assert not es.enabled
    for score in [1.0, 2.0, 3.0]:
        assert not es.step(score)


def test_early_stopping_max_mode() -> None:
    stopper = EarlyStopping(patience=2, min_delta=0.01, mode="max")
    assert stopper.step(0.10) is False
    assert stopper.step(0.20) is False
    assert stopper.step(0.19) is False
    assert stopper.step(0.18) is False
    assert stopper.step(0.17) is True
