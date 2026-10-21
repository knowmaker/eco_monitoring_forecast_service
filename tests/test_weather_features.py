from datetime import datetime, timezone

import pytest

from app.features import weather_features
from app.features.weather_features import _weather_rows


def test_open_meteo_payload_is_mapped_to_internal_units():
    payload = {
        "latitude": 55.9375,
        "longitude": 38.125,
        "hourly": {
            "time": ["2026-09-25T10:00"],
            "temperature_2m": [12.5],
            "relative_humidity_2m": [73],
            "surface_pressure": [1002.4],
            "wind_speed_10m": [3.2],
            "wind_direction_10m": [225],
            "precipitation": [0.1],
            "cloud_cover": [80],
        },
    }

    row = _weather_rows(622, payload, "historical_forecast")[0]

    assert row[0] == 622
    assert row[1] == datetime(2026, 9, 25, 10, tzinfo=timezone.utc)
    assert row[2] == "historical_forecast"
    assert row[7] == 1002.4
    assert row[8] == 3.2
    assert row[9] == 225.0


def test_historical_weather_is_synced_only_when_missing(monkeypatch):
    hour = datetime(2026, 10, 21, 9, tzinfo=timezone.utc)
    missing = iter([{"missing": 2}, {"missing": 0}, {"missing": 0}])
    synced = []
    monkeypatch.setattr(weather_features, "fetch_one", lambda *_: next(missing))
    monkeypatch.setattr(
        weather_features,
        "sync_historical_weather",
        lambda _, start, end: synced.append((start, end)) or 24,
    )

    assert weather_features.ensure_historical_weather(object(), hour) == 24
    assert weather_features.ensure_historical_weather(object(), hour) == 0
    assert synced == [(hour.date(), hour.date())]


def test_historical_weather_remains_missing_after_sync(monkeypatch):
    hour = datetime(2026, 10, 21, 9, tzinfo=timezone.utc)
    monkeypatch.setattr(weather_features, "fetch_one", lambda *_: {"missing": 1})
    monkeypatch.setattr(weather_features, "sync_historical_weather", lambda *_: 0)

    with pytest.raises(RuntimeError, match="Historical weather is incomplete"):
        weather_features.ensure_historical_weather(object(), hour)
