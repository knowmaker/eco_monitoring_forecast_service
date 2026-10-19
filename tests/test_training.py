from app.training.train import _should_activate


def _metrics(model_mae: float, persistence_mae: float) -> dict[str, object]:
    return {
        "overall": {"mae": model_mae},
        "baselines": {"persistence": {"mae": persistence_mae}},
    }


def test_initial_model_is_activated():
    assert _should_activate(None, _metrics(0.2, 0.1)) == (True, "initial_model")


def test_candidate_must_not_be_worse_than_current_model():
    current = {"metrics": {"overall": {"mae": 0.12}}}

    assert _should_activate(current, _metrics(0.11, 0.10))[0] is True
    assert _should_activate(current, _metrics(0.13, 0.10))[0] is False
