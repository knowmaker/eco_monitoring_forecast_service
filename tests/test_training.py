import pandas as pd

from app.training.evaluate import _compare_by_gas
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


def test_temporal_comparison_uses_same_rows_for_model_and_persistence():
    frame = pd.DataFrame({
        "substance_code": ["CO", "CO"],
        "target_value": [1.0, 2.0],
        "gas_CO_lag_0": [1.0, 2.0],
        "physical_baseline": [0.5, 1.5],
        "catboost": [1.5, 2.0],
    })

    result = _compare_by_gas(frame)["CO"]

    assert result["rows"] == 2
    assert result["persistence"]["mae"] == 0.0
    assert result["catboost"]["mae"] == 0.25
    assert result["physical_only"]["mae"] == 0.5
