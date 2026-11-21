from __future__ import annotations

import pandas as pd

from app.features.horizons import expand_horizons


def test_expanded_hours_use_the_correct_target_measurement_and_weather():
    hours = pd.date_range("2026-11-01T00:00:00Z", periods=6, freq="h")
    source = pd.DataFrame({
        "monitoring_post_id": [10] * 6,
        "source_bucket": hours,
        "data_cutoff": hours + pd.Timedelta(hours=1),
        "substance_code": ["NO2"] * 6,
        "gas_NO2_lag_0": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        "target_start": hours + pd.Timedelta(hours=2),
        "target_value": [0.03, 0.04, 0.05, 0.06, None, None],
    })
    weather = pd.DataFrame({
        "monitoring_post_id": [10] * 6,
        "bucket_start": hours,
        "air_temp": [1, 2, 3, 4, 5, 6],
    })
    expanded = expand_horizons(source, weather, include_target=True)
    first = expanded[(expanded.source_bucket == hours[0]) & (expanded.horizon_hours == 5)].iloc[0]
    assert first.target_start == hours[5]
    assert first.target_value == 0.06
    assert first.target_weather_air_temp == 6
    assert first.data_cutoff == hours[1]
    assert not expanded.target_value.isna().any()
