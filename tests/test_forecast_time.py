from datetime import datetime, timedelta, timezone

from app.inference.runner import current_cutoff
from app.worker import _cutoffs_to_rebuild, latest_due_cutoff, next_run_time


def test_current_cutoff_is_top_of_current_local_hour():
    now = datetime(2026, 9, 25, 11, 37, tzinfo=timezone.utc)
    assert current_cutoff(now) == datetime(2026, 9, 25, 11, 0, tzinfo=timezone.utc)


def test_next_run_is_minute_ten_of_next_hour_after_minute_ten():
    now = datetime(2026, 9, 25, 11, 37, tzinfo=timezone.utc)
    assert next_run_time(now) == datetime(2026, 9, 25, 12, 10, tzinfo=timezone.utc)


def test_latest_due_cutoff_waits_until_run_minute():
    before_run = datetime(2026, 9, 25, 11, 9, tzinfo=timezone.utc)
    at_run = datetime(2026, 9, 25, 11, 10, tzinfo=timezone.utc)

    assert latest_due_cutoff(before_run) == datetime(2026, 9, 25, 10, 0, tzinfo=timezone.utc)
    assert latest_due_cutoff(at_run) == datetime(2026, 9, 25, 11, 0, tzinfo=timezone.utc)


def test_rebuild_continues_from_first_missing_grid_hour():
    first_source = datetime(2026, 9, 25, 9, 0, tzinfo=timezone.utc)
    second_source = datetime(2026, 9, 25, 10, 0, tzinfo=timezone.utc)
    expected = [
        {"bucket_start": first_source, "substance_code": "NO2"},
        {"bucket_start": second_source, "substance_code": "NO2"},
    ]
    existing = [
        {"hour_start": second_source, "data_kind": "forecast", "substance_code": "NO2"},
        {"hour_start": second_source, "data_kind": "observed", "substance_code": "NO2"},
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
