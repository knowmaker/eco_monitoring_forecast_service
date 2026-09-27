from datetime import datetime, timezone

from app.inference.runner import current_cutoff
from app.worker import next_run_time


def test_current_cutoff_is_top_of_current_local_hour():
    now = datetime(2026, 9, 25, 11, 37, tzinfo=timezone.utc)
    assert current_cutoff(now) == datetime(2026, 9, 25, 11, 0, tzinfo=timezone.utc)


def test_next_run_is_minute_ten_of_next_hour_after_minute_ten():
    now = datetime(2026, 9, 25, 11, 37, tzinfo=timezone.utc)
    assert next_run_time(now) == datetime(2026, 9, 25, 12, 10, tzinfo=timezone.utc)

