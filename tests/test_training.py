import numpy as np
import pandas as pd

from app.training.train import _score_holdout


def test_holdout_metrics_are_calculated_separately_for_five_hours():
    frame = pd.DataFrame({
        "horizon_hours": [1, 2, 3, 4, 5],
        "target_value": [1.0] * 5,
        "physical_baseline": [0.5] * 5,
        "gas_CO_lag_0": [0.8] * 5,
    })
    scores = _score_holdout(frame, np.array([0.9] * 5), "CO")

    assert set(scores) == {"1", "2", "3", "4", "5"}
    assert all(item["rows"] == 1 for item in scores.values())
    assert scores["5"]["model"]["mae"] < scores["5"]["physical_only"]["mae"]
