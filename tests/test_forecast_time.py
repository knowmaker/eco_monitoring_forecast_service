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
            "hour_start": second_source + timedelta(hours=1),
            "data_kind": "forecast",
            "substance_code": "NO2",
        },
        {
            "hour_start": second_source + timedelta(hours=2),
            "data_kind": "forecast",
            "substance_code": "NO2",
        },
    ]

    assert _cutoffs_to_rebuild(expected, existing, second_source + timedelta(hours=1)) == [
        first_source + timedelta(hours=1),
        second_source + timedelta(hours=1),
    ]


def test_existing_nowcast_needs_no_station_prediction():
    source_hour = datetime(2026, 9, 25, 9, 0, tzinfo=timezone.utc)
    expected = [
        {"bucket_start": source_hour, "substance_code": "H2S", "forecast_expected": False}
    ]
    existing = [
        {"hour_start": source_hour, "data_kind": "analysis", "substance_code": "H2S"},
        {
            "hour_start": source_hour + timedelta(hours=1),
            "data_kind": "forecast",
            "substance_code": "H2S",
        },
    ]

    assert _cutoffs_to_rebuild(expected, existing, source_hour + timedelta(hours=1)) == []


def test_stale_forecast_for_current_hour_is_rebuilt_as_nowcast():
    source_hour = datetime(2026, 9, 25, 9, 0, tzinfo=timezone.utc)
    expected = [
        {"bucket_start": source_hour, "substance_code": "NO2", "forecast_expected": True}
    ]
    existing = [
        {
            "hour_start": source_hour,
            "data_kind": "analysis",
            "substance_code": "NO2",
            "generated_at": source_hour + timedelta(hours=2),
        },
        {
            "hour_start": source_hour + timedelta(hours=1),
            "data_kind": "forecast",
            "substance_code": "NO2",
            "generated_at": source_hour + timedelta(hours=1),
        },
        {
            "hour_start": source_hour + timedelta(hours=2),
            "data_kind": "forecast",
            "substance_code": "NO2",
            "generated_at": source_hour + timedelta(hours=2),
        },
    ]

    assert _cutoffs_to_rebuild(
        expected, existing, source_hour + timedelta(hours=1)
    ) == [source_hour + timedelta(hours=1)]


def test_old_historical_forecast_does_not_trigger_full_backfill():
    source_hour = datetime(2026, 9, 25, 9, 0, tzinfo=timezone.utc)
    expected = [
        {"bucket_start": source_hour, "substance_code": "NO2", "forecast_expected": True}
    ]
    existing = [
        {
            "hour_start": source_hour,
            "data_kind": "analysis",
            "substance_code": "NO2",
            "generated_at": source_hour + timedelta(hours=2),
        },
        {
            "hour_start": source_hour + timedelta(hours=1),
            "data_kind": "forecast",
            "substance_code": "NO2",
            "generated_at": source_hour + timedelta(hours=1),
        },
        {
            "hour_start": source_hour + timedelta(hours=2),
            "data_kind": "forecast",
            "substance_code": "NO2",
        },
    ]

    assert _cutoffs_to_rebuild(expected, existing, source_hour + timedelta(hours=2)) == []
