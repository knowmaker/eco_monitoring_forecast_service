from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from app.inference.predict import _eligible_prediction_rows


def test_prediction_requires_latest_target_measurement_only():
    cutoff = datetime(2026, 9, 27, 12, tzinfo=timezone.utc)
    history = pd.DataFrame(
        [
            {
                "monitoring_post_id": 1,
                "data_cutoff": cutoff - timedelta(hours=1),
                "gas_NO2_lag_0": np.nan,
                "gas_NO2_lag_1": 0.03,
            },
            {
                "monitoring_post_id": 1,
                "data_cutoff": cutoff,
                "gas_NO2_lag_0": 0.04,
                "gas_NO2_lag_1": np.nan,
            },
            {
                "monitoring_post_id": 2,
                "data_cutoff": cutoff,
                "gas_NO2_lag_0": np.nan,
                "gas_NO2_lag_1": 0.05,
            },
        ]
    )

    rows = _eligible_prediction_rows(history, "NO2", cutoff)

    assert rows["monitoring_post_id"].tolist() == [1]
