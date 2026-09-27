from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from math import atan2, degrees, radians
from typing import Any

import numpy as np
import pandas as pd
from psycopg import Connection

from app.config import get_settings
from app.db import db_connection, fetch_all, fetch_one
from app.features.concentration_grid import build_concentration_grid


GRID_COPY_SQL = """
    COPY public.gas_concentration_grid (
        substance_code, hour_start, data_kind, cluster_id, grid_x, grid_y,
        latitude, longitude, south, west, north, east, value,
        lower_bound, upper_bound, confidence, source_station_count,
        wind_speed, wind_direction
    )
    FROM STDIN
"""
GRID_STAGE_COPY_SQL = GRID_COPY_SQL.replace(
    "public.gas_concentration_grid",
    "gas_concentration_grid_stage",
)


def _anchor_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "monitoring_post_id",
                "latitude",
                "longitude",
                "value",
                "lower_bound",
                "upper_bound",
            ]
        )
    return frame


def _observed_anchors(connection: Connection, substance_code: str, hour_start: datetime) -> pd.DataFrame:
    return _anchor_frame(
        fetch_all(
            connection,
            """
            SELECT h.monitoring_post_id, p.latitude, p.longitude,
                   ABS(h.filtered_hourly_mean) AS value,
                   NULL::double precision AS lower_bound,
                   NULL::double precision AS upper_bound
            FROM public.gas_hourly_features h
            JOIN public.monitoring_posts p ON p.id = h.monitoring_post_id
            WHERE h.substance_code = %s AND h.bucket_start = %s
              AND h.filtered_hourly_mean IS NOT NULL
              AND p.latitude IS NOT NULL AND p.longitude IS NOT NULL
            ORDER BY h.monitoring_post_id
            """,
            (substance_code, hour_start),
        )
    )


def _forecast_anchors(connection: Connection, substance_code: str, hour_start: datetime) -> pd.DataFrame:
    return _anchor_frame(
        fetch_all(
            connection,
            """
            SELECT f.monitoring_post_id, p.latitude, p.longitude,
                   f.predicted_value AS value, f.lower_bound, f.upper_bound
            FROM public.gas_predictions f
            JOIN public.monitoring_posts p ON p.id = f.monitoring_post_id
            JOIN public.gas_hourly_features h
              ON h.monitoring_post_id = f.monitoring_post_id
             AND h.substance_code = f.substance_code
             AND h.bucket_start = f.data_cutoff - INTERVAL '1 hour'
             AND h.filtered_hourly_mean IS NOT NULL
            WHERE f.substance_code = %s AND f.target_start = %s
              AND f.status = 'ready' AND f.predicted_value IS NOT NULL
              AND p.latitude IS NOT NULL AND p.longitude IS NOT NULL
            ORDER BY f.monitoring_post_id
            """,
            (substance_code, hour_start),
        )
    )


def _mean_wind(
    connection: Connection,
    hour_start: datetime,
    data_kind: str,
    station_ids: list[int],
) -> tuple[float | None, float | None]:
    if not station_ids:
        return None, None
    rows = fetch_all(
        connection,
        """
        SELECT hor_win_spd, hor_win_dir
        FROM public.external_weather_hourly
        WHERE bucket_start = %s AND data_kind = %s
          AND monitoring_post_id = ANY(%s)
          AND hor_win_spd IS NOT NULL AND hor_win_dir IS NOT NULL
        """,
        (hour_start, data_kind, station_ids),
    )
    if not rows:
        return None, None
    speeds = np.asarray([row["hor_win_spd"] for row in rows], dtype=float)
    directions = np.deg2rad(np.asarray([row["hor_win_dir"] for row in rows], dtype=float))
    direction = (degrees(atan2(float(np.sin(directions).mean()), float(np.cos(directions).mean()))) + 360.0) % 360.0
    return float(speeds.mean()), direction


def _replace_grid(
    connection: Connection,
    substance_code: str,
    hour_start: datetime,
    data_kind: str,
    grid: pd.DataFrame,
) -> int:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            DELETE FROM public.gas_concentration_grid
            WHERE substance_code = %s AND hour_start = %s AND data_kind = %s
            """,
            (substance_code, hour_start, data_kind),
        )
        if grid.empty:
            return 0
        with cursor.copy(GRID_COPY_SQL) as copy:
            _write_grid_rows(copy, substance_code, hour_start, data_kind, grid)
    return len(grid)


def _write_grid_rows(
    copy: Any,
    substance_code: str,
    hour_start: datetime,
    data_kind: str,
    grid: pd.DataFrame,
) -> None:
    for row in grid.itertuples(index=False):
        copy.write_row(
            (
                substance_code,
                hour_start,
                data_kind,
                int(row.cluster_id),
                int(row.grid_x),
                int(row.grid_y),
                float(row.latitude),
                float(row.longitude),
                float(row.south),
                float(row.west),
                float(row.north),
                float(row.east),
                float(row.value),
                float(row.lower_bound) if np.isfinite(row.lower_bound) else None,
                float(row.upper_bound) if np.isfinite(row.upper_bound) else None,
                float(row.confidence),
                int(row.source_station_count),
                float(row.wind_speed) if row.wind_speed is not None and np.isfinite(row.wind_speed) else None,
                float(row.wind_direction) if row.wind_direction is not None and np.isfinite(row.wind_direction) else None,
            )
        )


def _insert_grid_batch(
    connection: Connection,
    batch: list[tuple[str, datetime, str, pd.DataFrame]],
    copy_sql: str = GRID_COPY_SQL,
) -> None:
    if not batch:
        return
    with connection.cursor() as cursor:
        with cursor.copy(copy_sql) as copy:
            for substance_code, hour_start, data_kind, grid in batch:
                _write_grid_rows(copy, substance_code, hour_start, data_kind, grid)
    connection.commit()


def _mean_wind_frame(
    weather_by_hour: dict[tuple[str, datetime], pd.DataFrame],
    hour_start: datetime,
    data_kind: str,
    station_ids: list[int],
) -> tuple[float | None, float | None]:
    weather = weather_by_hour.get((data_kind, hour_start))
    if weather is None or weather.empty:
        return None, None
    selected = weather[weather["monitoring_post_id"].isin(station_ids)].dropna(
        subset=["hor_win_spd", "hor_win_dir"]
    )
    if selected.empty:
        return None, None
    speeds = selected["hor_win_spd"].to_numpy(dtype=float)
    directions = np.deg2rad(selected["hor_win_dir"].to_numpy(dtype=float))
    direction = (degrees(atan2(float(np.sin(directions).mean()), float(np.cos(directions).mean()))) + 360.0) % 360.0
    return float(speeds.mean()), direction


def build_observed_grid(connection: Connection, substance_code: str, hour_start: datetime) -> int:
    anchors = _observed_anchors(connection, substance_code, hour_start)
    station_ids = anchors["monitoring_post_id"].astype(int).tolist() if not anchors.empty else []
    wind_speed, wind_direction = _mean_wind(
        connection,
        hour_start,
        "historical_forecast",
        station_ids,
    )
    grid = build_concentration_grid(
        anchors,
        data_kind="observed",
        wind_speed=wind_speed,
        wind_from_degrees=wind_direction,
    )
    return _replace_grid(connection, substance_code, hour_start, "observed", grid)


def build_forecast_grid(connection: Connection, substance_code: str, hour_start: datetime) -> int:
    anchors = _forecast_anchors(connection, substance_code, hour_start)
    station_ids = anchors["monitoring_post_id"].astype(int).tolist() if not anchors.empty else []
    wind_speed, wind_direction = _mean_wind(
        connection,
        hour_start,
        "live_forecast",
        station_ids,
    )
    cutoff = fetch_one(
        connection,
        """
        SELECT max(data_cutoff) AS data_cutoff
        FROM public.gas_predictions
        WHERE substance_code = %s AND target_start = %s AND status = 'ready'
        """,
        (substance_code, hour_start),
    )
    source_hour = cutoff["data_cutoff"] - timedelta(hours=1) if cutoff and cutoff["data_cutoff"] else None
    source_anchors = (
        _observed_anchors(connection, substance_code, source_hour)
        if source_hour is not None
        else pd.DataFrame()
    )
    grid = build_concentration_grid(
        anchors,
        data_kind="forecast",
        wind_speed=wind_speed,
        wind_from_degrees=wind_direction,
        source_anchors=source_anchors,
    )
    return _replace_grid(connection, substance_code, hour_start, "forecast", grid)


def refresh_concentration_grids(cutoff: datetime) -> dict[str, int]:
    observed_hour = cutoff - timedelta(hours=1)
    forecast_hour = cutoff + timedelta(hours=1)
    counts = {"observed": 0, "forecast": 0}
    with db_connection() as connection:
        for substance_code in get_settings().active_gases:
            counts["observed"] += build_observed_grid(connection, substance_code, observed_hour)
            counts["forecast"] += build_forecast_grid(connection, substance_code, forecast_hour)
        retention_start = cutoff - timedelta(days=get_settings().GRID_RETENTION_DAYS)
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM public.gas_concentration_grid WHERE hour_start < %s",
                (retention_start,),
            )
    return counts


def backfill_concentration_grids() -> dict[str, int]:
    counts = {"observed": 0, "forecast": 0}
    settings = get_settings()
    retention_start = datetime.now(timezone.utc).replace(
        minute=0,
        second=0,
        microsecond=0,
    ) - timedelta(days=settings.GRID_RETENTION_DAYS)
    with db_connection() as connection:
        observed_rows = fetch_all(
            connection,
            """
            SELECT h.substance_code, h.bucket_start AS hour_start,
                   h.monitoring_post_id, p.latitude, p.longitude,
                   ABS(h.filtered_hourly_mean) AS value,
                   NULL::double precision AS lower_bound,
                   NULL::double precision AS upper_bound
            FROM public.gas_hourly_features h
            JOIN public.monitoring_posts p ON p.id = h.monitoring_post_id
            WHERE h.substance_code = ANY(%s)
              AND h.bucket_start >= %s
              AND h.filtered_hourly_mean IS NOT NULL
              AND p.latitude IS NOT NULL AND p.longitude IS NOT NULL
            ORDER BY h.bucket_start, h.substance_code, h.monitoring_post_id
            """,
            (list(settings.active_gases), retention_start),
        )
        forecast_rows = fetch_all(
            connection,
            """
            SELECT f.substance_code, f.target_start AS hour_start, f.data_cutoff,
                   f.monitoring_post_id, p.latitude, p.longitude,
                   f.predicted_value AS value, f.lower_bound, f.upper_bound
            FROM public.gas_predictions f
            JOIN public.monitoring_posts p ON p.id = f.monitoring_post_id
            JOIN public.gas_hourly_features h
              ON h.monitoring_post_id = f.monitoring_post_id
             AND h.substance_code = f.substance_code
             AND h.bucket_start = f.data_cutoff - INTERVAL '1 hour'
             AND h.filtered_hourly_mean IS NOT NULL
            WHERE f.substance_code = ANY(%s)
              AND f.target_start >= %s
              AND f.status = 'ready' AND f.predicted_value IS NOT NULL
              AND p.latitude IS NOT NULL AND p.longitude IS NOT NULL
            ORDER BY f.target_start, f.substance_code, f.monitoring_post_id
            """,
            (list(settings.active_gases), retention_start),
        )
        weather_rows = fetch_all(
            connection,
            """
            SELECT monitoring_post_id, bucket_start, data_kind, hor_win_spd, hor_win_dir
            FROM public.external_weather_hourly
            WHERE data_kind IN ('historical_forecast', 'live_forecast')
            """,
        )
        observed = pd.DataFrame(observed_rows)
        forecast = pd.DataFrame(forecast_rows)
        weather = pd.DataFrame(weather_rows)
        weather_by_hour = (
            {
                (data_kind, hour_start): group
                for (data_kind, hour_start), group in weather.groupby(
                    ["data_kind", "bucket_start"],
                    sort=False,
                )
            }
            if not weather.empty
            else {}
        )
        observed_by_hour = {
            (substance_code, hour_start): group
            for (substance_code, hour_start), group in observed.groupby(
                ["substance_code", "hour_start"],
                sort=False,
            )
        } if not observed.empty else {}

        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TEMP TABLE gas_concentration_grid_stage
                (LIKE public.gas_concentration_grid INCLUDING DEFAULTS)
                ON COMMIT PRESERVE ROWS
                """
            )
        connection.commit()

        batch: list[tuple[str, datetime, str, pd.DataFrame]] = []
        for (substance_code, hour_start), anchors in observed_by_hour.items():
            if anchors["monitoring_post_id"].nunique() < settings.GRID_MIN_STATIONS:
                continue
            station_ids = anchors["monitoring_post_id"].astype(int).tolist()
            wind_speed, wind_direction = _mean_wind_frame(
                weather_by_hour,
                hour_start,
                "historical_forecast",
                station_ids,
            )
            grid = build_concentration_grid(
                anchors,
                data_kind="observed",
                wind_speed=wind_speed,
                wind_from_degrees=wind_direction,
            )
            if not grid.empty:
                batch.append((substance_code, hour_start, "observed", grid))
                counts["observed"] += len(grid)
            if len(batch) >= 25:
                _insert_grid_batch(connection, batch, GRID_STAGE_COPY_SQL)
                batch.clear()
        _insert_grid_batch(connection, batch, GRID_STAGE_COPY_SQL)
        batch.clear()

        if not forecast.empty:
            for (substance_code, hour_start), anchors in forecast.groupby(
                ["substance_code", "hour_start"],
                sort=False,
            ):
                if anchors["monitoring_post_id"].nunique() < settings.GRID_MIN_STATIONS:
                    continue
                station_ids = anchors["monitoring_post_id"].astype(int).tolist()
                wind_speed, wind_direction = _mean_wind_frame(
                    weather_by_hour,
                    hour_start,
                    "live_forecast",
                    station_ids,
                )
                cutoff = anchors["data_cutoff"].max()
                source_anchors = observed_by_hour.get(
                    (substance_code, cutoff - timedelta(hours=1)),
                    pd.DataFrame(),
                )
                grid = build_concentration_grid(
                    anchors,
                    data_kind="forecast",
                    wind_speed=wind_speed,
                    wind_from_degrees=wind_direction,
                    source_anchors=source_anchors,
                )
                if not grid.empty:
                    batch.append((substance_code, hour_start, "forecast", grid))
                    counts["forecast"] += len(grid)
                if len(batch) >= 25:
                    _insert_grid_batch(connection, batch, GRID_STAGE_COPY_SQL)
                    batch.clear()
            _insert_grid_batch(connection, batch, GRID_STAGE_COPY_SQL)

        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM public.gas_concentration_grid WHERE substance_code = ANY(%s)",
                (list(settings.active_gases),),
            )
            cursor.execute(
                "INSERT INTO public.gas_concentration_grid SELECT * FROM gas_concentration_grid_stage"
            )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build gas concentration grids for the heat map.")
    parser.add_argument("--backfill", action="store_true")
    parser.add_argument("--cutoff", help="Optional ISO-8601 data cutoff.")
    args = parser.parse_args()
    if args.backfill:
        print(backfill_concentration_grids())
        return
    cutoff = datetime.fromisoformat(args.cutoff) if args.cutoff else datetime.now(timezone.utc)
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=timezone.utc)
    cutoff = cutoff.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    print(refresh_concentration_grids(cutoff))


if __name__ == "__main__":
    main()
