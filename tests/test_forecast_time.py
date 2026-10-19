from datetime import datetime, timedelta, timezone

from app.inference.runner import current_cutoff
from app.worker import _cutoffs_to_rebuild


def test_current_cutoff_is_top_of_current_local_hour():
    now = datetime(2026, 9, 25, 11, 37, tzinfo=timezone.utc)
    assert current_cutoff(now) == datetime(2026, 9, 25, 11, 0, tzinfo=timezone.utc)


def test_rebuild_continues_from_first_missing_grid_hour():
    first_source = datetime(2026, 9, 25, 9, 0, tzinfo=timezone.utc)
    second_source = datetime(2026, 9, 25, 10, 0, tzinfo=timezone.utc)
    expected = [
        {"bucket_start": first_source, "substance_code": "NO2", "forecast_expected": True},
        {"bucket_start": second_source, "substance_code": "NO2", "forecast_expected": True},
    ]
    existing = [
        {"hour_start": second_source, "data_kind": "forecast", "substance_code": "NO2"},
        {"hour_start": second_source, "data_kind": "analysis", "substance_code": "NO2"},
        {
            "hour_start": second_source + timedelta(hours=2),
            "data_kind": "forecast",
            "substance_code": "NO2",
        },
    ]

    assert _cutoffs_to_rebuild(expected, existing) == [
        first_source + timedelta(hours=1),
        second_source + timedelta(hours=1),
    ]


def test_missing_historical_forecast_is_ignored_without_prediction():
    source_hour = datetime(2026, 9, 25, 9, 0, tzinfo=timezone.utc)
    expected = [
        {"bucket_start": source_hour, "substance_code": "H2S", "forecast_expected": False}
    ]
    existing = [
        {"hour_start": source_hour, "data_kind": "analysis", "substance_code": "H2S"}
    ]

    assert _cutoffs_to_rebuild(expected, existing) == []
