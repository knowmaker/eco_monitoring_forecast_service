import pandas as pd
import pytest

from app.features.hourly_gas import build_hourly_features


def test_causal_median_is_applied_before_hourly_mean():
    raw = pd.DataFrame(
        {
            "monitoring_post_id": [1, 1, 1],
            "substance_code": ["NO2", "NO2", "NO2"],
            "timestamp": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-01T00:05:00Z", "2026-01-01T00:10:00Z"]
            ),
            "value": [1.0, 100.0, 1.0],
        }
    )

    hourly = build_hourly_features(raw, "UTC").iloc[0]

    assert hourly.raw_hourly_mean == pytest.approx(34.0)
    assert hourly.raw_hourly_median == pytest.approx(1.0)
    assert hourly.filtered_hourly_mean == pytest.approx(17.5)
    assert hourly.samples_count == 3


def test_median_does_not_bridge_large_data_gap():
    raw = pd.DataFrame(
        {
            "monitoring_post_id": [1, 1, 1, 1],
            "substance_code": ["CO"] * 4,
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:05:00Z",
                    "2026-01-01T00:10:00Z",
                    "2026-01-01T02:00:00Z",
                ]
            ),
            "value": [1.0, 2.0, 3.0, 100.0],
        }
    )

    hourly = build_hourly_features(raw, "UTC")

    assert hourly.loc[hourly.bucket_start == pd.Timestamp("2026-01-01T02:00:00Z"), "filtered_hourly_mean"].item() == 100.0

