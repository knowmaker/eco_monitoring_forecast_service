from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
from psycopg import Connection

from app.config import get_settings
from app.db import db_connection, fetch_all, fetch_one


WEATHER_COLUMNS = [
    "atm_press",
    "air_temp",
    "air_hum",
    "hor_win_dir",
    "hor_win_spd",
    "precipitation",
    "cloud_cover",
    "cloud_cover_low",
    "wind_speed_100m",
    "wind_direction_100m",
    "wind_gusts_10m",
    "boundary_layer_height",
    "shortwave_radiation",
]
OPEN_METEO_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "cloud_cover",
    "cloud_cover_low",
    "wind_speed_100m",
    "wind_direction_100m",
    "wind_gusts_10m",
    "boundary_layer_height",
    "shortwave_radiation",
]


def load_posts_with_coordinates(connection: Connection) -> list[dict[str, Any]]:
    return fetch_all(
        connection,
        """
        SELECT id AS monitoring_post_id, latitude, longitude
        FROM public.monitoring_posts
        WHERE active_to IS NULL AND latitude IS NOT NULL AND longitude IS NOT NULL
        ORDER BY id
        """,
    )


def _request_json(url: str, params: dict[str, Any], retries: int = 3) -> dict[str, Any]:
    request = Request(
        f"{url}?{urlencode(params)}",
        headers={"User-Agent": "eco-monitoring-forecast-service/1.0 (Open-Meteo weather features)"},
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=get_settings().OPEN_METEO_TIMEOUT_SECONDS) as response:
                return json.load(response)
        except Exception as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Open-Meteo request failed after {retries} attempts: {last_error}") from last_error


def _weather_rows(post_id: int, payload: dict[str, Any], data_kind: str) -> list[tuple[Any, ...]]:
    hourly = payload.get("hourly") or {}
    timestamps = hourly.get("time") or []
    rows: list[tuple[Any, ...]] = []
    for index, timestamp in enumerate(timestamps):
        def value(name: str) -> float | None:
            values = hourly.get(name) or []
            item = values[index] if index < len(values) else None
            return float(item) if item is not None else None

        rows.append(
            (
                post_id,
                datetime.fromisoformat(timestamp).replace(tzinfo=timezone.utc),
                data_kind,
                float(payload["latitude"]),
                float(payload["longitude"]),
                value("temperature_2m"),
                value("relative_humidity_2m"),
                value("surface_pressure"),
                value("wind_speed_10m"),
                value("wind_direction_10m"),
                value("precipitation"),
                value("cloud_cover"),
                value("cloud_cover_low"),
                value("wind_speed_100m"),
                value("wind_direction_100m"),
                value("wind_gusts_10m"),
                value("boundary_layer_height"),
                value("shortwave_radiation"),
            )
        )
    return rows


def _upsert_weather(connection: Connection, rows: list[tuple[Any, ...]]) -> int:
    if not rows:
        return 0
    with connection.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO public.external_weather_hourly (
                monitoring_post_id, bucket_start, data_kind, provider,
                model_latitude, model_longitude, air_temp, air_hum, atm_press,
                hor_win_spd, hor_win_dir, precipitation, cloud_cover, fetched_at
                , cloud_cover_low, wind_speed_100m, wind_direction_100m,
                wind_gusts_10m, boundary_layer_height, shortwave_radiation
            )
            VALUES (
                %s, %s, %s, 'open-meteo', %s, %s, %s, %s, %s, %s, %s, %s, %s,
                NOW(), %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (monitoring_post_id, bucket_start, data_kind)
            DO UPDATE SET
                provider = EXCLUDED.provider,
                model_latitude = EXCLUDED.model_latitude,
                model_longitude = EXCLUDED.model_longitude,
                air_temp = EXCLUDED.air_temp,
                air_hum = EXCLUDED.air_hum,
                atm_press = EXCLUDED.atm_press,
                hor_win_spd = EXCLUDED.hor_win_spd,
                hor_win_dir = EXCLUDED.hor_win_dir,
                precipitation = EXCLUDED.precipitation,
                cloud_cover = EXCLUDED.cloud_cover,
                cloud_cover_low = EXCLUDED.cloud_cover_low,
                wind_speed_100m = EXCLUDED.wind_speed_100m,
                wind_direction_100m = EXCLUDED.wind_direction_100m,
                wind_gusts_10m = EXCLUDED.wind_gusts_10m,
                boundary_layer_height = EXCLUDED.boundary_layer_height,
                shortwave_radiation = EXCLUDED.shortwave_radiation,
                fetched_at = NOW()
            """,
            rows,
        )
    return len(rows)


def sync_historical_weather(
    connection: Connection,
    start_date: date | None = None,
    end_date: date | None = None,
) -> int:
    if start_date is None or end_date is None:
        bounds = fetch_one(
            connection,
            "SELECT min(bucket_start) AS first, max(bucket_start) AS last FROM public.gas_hourly_features",
        )
        if not bounds or bounds["first"] is None:
            return 0
        start_date = start_date or bounds["first"].date()
        end_date = end_date or (bounds["last"] + timedelta(hours=3)).date()

    total = 0
    settings = get_settings()
    for post in load_posts_with_coordinates(connection):
        payload = _request_json(
            settings.OPEN_METEO_HISTORICAL_FORECAST_URL,
            {
                "latitude": float(post["latitude"]),
                "longitude": float(post["longitude"]),
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "hourly": ",".join(OPEN_METEO_VARIABLES),
                "wind_speed_unit": "ms",
                "timezone": "UTC",
            },
        )
        total += _upsert_weather(
            connection,
            _weather_rows(int(post["monitoring_post_id"]), payload, "historical_forecast"),
        )
    return total


def sync_live_weather(connection: Connection) -> int:
    total = 0
    settings = get_settings()
    for post in load_posts_with_coordinates(connection):
        payload = _request_json(
            settings.OPEN_METEO_FORECAST_URL,
            {
                "latitude": float(post["latitude"]),
                "longitude": float(post["longitude"]),
                "past_days": 2,
                "forecast_days": 3,
                "hourly": ",".join(OPEN_METEO_VARIABLES),
                "wind_speed_unit": "ms",
                "timezone": "UTC",
            },
        )
        total += _upsert_weather(
            connection,
            _weather_rows(int(post["monitoring_post_id"]), payload, "live_forecast"),
        )
    return total


def load_hourly_weather(
    connection: Connection,
    start: object | None = None,
    end: object | None = None,
    *,
    data_kind: str,
) -> pd.DataFrame:
    conditions = ["data_kind = %s"]
    params: list[object] = [data_kind]
    if start is not None:
        conditions.append("bucket_start >= %s")
        params.append(pd.Timestamp(start).to_pydatetime())
    if end is not None:
        conditions.append("bucket_start < %s")
        params.append(pd.Timestamp(end).to_pydatetime())
    rows = fetch_all(
        connection,
        f"""
        SELECT monitoring_post_id, bucket_start, {', '.join(WEATHER_COLUMNS)}
        FROM public.external_weather_hourly
        WHERE {' AND '.join(conditions)}
        ORDER BY monitoring_post_id, bucket_start
        """,
        params,
    )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["monitoring_post_id", "bucket_start", *WEATHER_COLUMNS])
    frame["bucket_start"] = pd.to_datetime(frame["bucket_start"], utc=True)
    for column in WEATHER_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronise Open-Meteo weather features.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--history", action="store_true")
    group.add_argument("--live", action="store_true")
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat)
    args = parser.parse_args()
    with db_connection() as connection:
        count = (
            sync_historical_weather(connection, args.start_date, args.end_date)
            if args.history
            else sync_live_weather(connection)
        )
    print(f"Upserted {count} Open-Meteo hourly weather rows.")


if __name__ == "__main__":
    main()
