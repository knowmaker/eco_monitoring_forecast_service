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
from app.features.concentration_grid import (
    assimilate_observations,
    build_semiphysical_forecast_grid,
)


GRID_COPY_SQL = """
    COPY public.gas_concentration_grid (
        substance_code, hour_start, data_kind, cluster_id, grid_x, grid_y,
        latitude, longitude, south, west, north, east, value,
        analysis_value, physical_forecast, correction_value,
        lower_bound, upper_bound, confidence, source_station_count,
        wind_speed, wind_direction, wind_u, wind_v, boundary_layer_height,
        diffusion_coefficient, decay_coefficient
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
                   GREATEST(h.filtered_hourly_mean, 0.0) AS value,
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
                   f.predicted_value AS value, f.lower_bound, f.upper_bound,
                   w.wind_speed_100m AS weather_wind_speed,
                   w.wind_direction_100m AS weather_wind_direction,
                   w.boundary_layer_height AS weather_boundary_layer_height,
                   w.precipitation AS weather_precipitation
            FROM public.gas_predictions f
            JOIN public.monitoring_posts p ON p.id = f.monitoring_post_id
            JOIN public.gas_hourly_features h
              ON h.monitoring_post_id = f.monitoring_post_id
             AND h.substance_code = f.substance_code
             AND h.bucket_start = f.data_cutoff - INTERVAL '1 hour'
             AND h.filtered_hourly_mean IS NOT NULL
            LEFT JOIN public.external_weather_hourly w
              ON w.monitoring_post_id = f.monitoring_post_id
             AND w.bucket_start = f.data_cutoff
             AND w.data_kind = 'live_forecast'
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


def _mean_weather(
    connection: Connection,
    hour_start: datetime,
    data_kind: str,
    station_ids: list[int],
) -> dict[str, float | None]:
    if not station_ids:
        return {
            "wind_speed": None,
            "wind_direction": None,
            "boundary_layer_height": None,
            "precipitation": None,
        }
    rows = fetch_all(
        connection,
        """
        SELECT hor_win_spd, hor_win_dir, wind_speed_100m, wind_direction_100m,
               boundary_layer_height, precipitation
        FROM public.external_weather_hourly
        WHERE bucket_start = %s AND data_kind = %s
          AND monitoring_post_id = ANY(%s)
        """,
        (hour_start, data_kind, station_ids),
    )
    if not rows:
        return {
            "wind_speed": None,
            "wind_direction": None,
            "boundary_layer_height": None,
            "precipitation": None,
        }
    speeds = np.asarray([
        row["wind_speed_100m"] if row["wind_speed_100m"] is not None else row["hor_win_spd"]
        for row in rows
    ], dtype=float)
    directions = np.asarray([
        row["wind_direction_100m"] if row["wind_direction_100m"] is not None else row["hor_win_dir"]
        for row in rows
    ], dtype=float)
    usable_wind = np.isfinite(speeds) & np.isfinite(directions)
    direction = None
    if usable_wind.any():
        radians_values = np.deg2rad(directions[usable_wind])
        direction = (
            degrees(atan2(float(np.sin(radians_values).mean()), float(np.cos(radians_values).mean())))
            + 360.0
        ) % 360.0
    pbl = np.asarray([row["boundary_layer_height"] for row in rows], dtype=float)
    rain = np.asarray([row["precipitation"] for row in rows], dtype=float)
    return {
        "wind_speed": float(speeds[usable_wind].mean()) if usable_wind.any() else None,
        "wind_direction": direction,
        "boundary_layer_height": float(np.nanmean(pbl)) if np.isfinite(pbl).any() else None,
        "precipitation": float(np.nanmean(rain)) if np.isfinite(rain).any() else None,
    }


def _stored_grid(
    connection: Connection,
    substance_code: str,
    hour_start: datetime,
    data_kind: str,
) -> pd.DataFrame:
    return pd.DataFrame(fetch_all(
        connection,
        """
        SELECT cluster_id, grid_x, grid_y, latitude, longitude, south, west, north, east,
               value, analysis_value, physical_forecast, correction_value,
               lower_bound, upper_bound, confidence, source_station_count,
               wind_speed, wind_direction, wind_u, wind_v, boundary_layer_height,
               diffusion_coefficient, decay_coefficient
        FROM public.gas_concentration_grid
        WHERE substance_code = %s AND hour_start = %s AND data_kind = %s
        ORDER BY cluster_id, grid_y, grid_x
        """,
        (substance_code, hour_start, data_kind),
    ))


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
    def optional_float(row: Any, name: str) -> float | None:
        value = getattr(row, name, np.nan)
        return float(value) if value is not None and np.isfinite(value) else None

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
                optional_float(row, "analysis_value"),
                optional_float(row, "physical_forecast"),
                optional_float(row, "correction_value"),
                optional_float(row, "lower_bound"),
                optional_float(row, "upper_bound"),
                float(row.confidence),
                int(row.source_station_count),
                optional_float(row, "wind_speed"),
                optional_float(row, "wind_direction"),
                optional_float(row, "wind_u"),
                optional_float(row, "wind_v"),
                optional_float(row, "boundary_layer_height"),
                optional_float(row, "diffusion_coefficient"),
                optional_float(row, "decay_coefficient"),
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
    direction = (
        degrees(atan2(float(np.sin(directions).mean()), float(np.cos(directions).mean())))
        + 360.0
    ) % 360.0
    return float(speeds.mean()), direction


def _mean_weather_frame(
    weather_by_hour: dict[tuple[str, datetime], pd.DataFrame],
    hour_start: datetime,
    data_kind: str,
    station_ids: list[int],
) -> dict[str, float | None]:
    weather = weather_by_hour.get((data_kind, hour_start))
    if weather is None or weather.empty:
        return {
            "wind_speed": None,
            "wind_direction": None,
            "boundary_layer_height": None,
            "precipitation": None,
        }
    selected = weather[weather["monitoring_post_id"].isin(station_ids)]
    if selected.empty:
        return {
            "wind_speed": None,
            "wind_direction": None,
            "boundary_layer_height": None,
            "precipitation": None,
        }
    speed = selected["wind_speed_100m"].fillna(selected["hor_win_spd"]).to_numpy(dtype=float)
    direction_values = selected["wind_direction_100m"].fillna(
        selected["hor_win_dir"]
    ).to_numpy(dtype=float)
    usable = np.isfinite(speed) & np.isfinite(direction_values)
    direction = None
    if usable.any():
        radians_values = np.deg2rad(direction_values[usable])
        direction = (
            degrees(atan2(
                float(np.sin(radians_values).mean()),
                float(np.cos(radians_values).mean()),
            ))
            + 360.0
        ) % 360.0
    pbl = selected["boundary_layer_height"].to_numpy(dtype=float)
    rain = selected["precipitation"].to_numpy(dtype=float)
    return {
        "wind_speed": float(speed[usable].mean()) if usable.any() else None,
        "wind_direction": direction,
        "boundary_layer_height": float(np.nanmean(pbl)) if np.isfinite(pbl).any() else None,
        "precipitation": float(np.nanmean(rain)) if np.isfinite(rain).any() else None,
    }


def build_observed_grid(connection: Connection, substance_code: str, hour_start: datetime) -> int:
    anchors = _observed_anchors(connection, substance_code, hour_start)
    station_ids = anchors["monitoring_post_id"].astype(int).tolist() if not anchors.empty else []
    wind_speed, wind_direction = _mean_wind(
        connection,
        hour_start,
        "historical_forecast",
        station_ids,
    )
    prior = _stored_grid(connection, substance_code, hour_start, "forecast")
    grid = assimilate_observations(
        anchors,
        prior,
        wind_speed=wind_speed,
        wind_from_degrees=wind_direction,
    )
    return _replace_grid(connection, substance_code, hour_start, "observed", grid)


def build_forecast_grid(connection: Connection, substance_code: str, hour_start: datetime) -> int:
    anchors = _forecast_anchors(connection, substance_code, hour_start)
    station_ids = anchors["monitoring_post_id"].astype(int).tolist() if not anchors.empty else []
    cutoff = fetch_one(
        connection,
        """
        SELECT max(data_cutoff) AS data_cutoff
        FROM public.gas_predictions
        WHERE substance_code = %s AND target_start = %s AND status = 'ready'
        """,
        (substance_code, hour_start),
    )
    data_cutoff = cutoff["data_cutoff"] if cutoff else None
    weather = _mean_weather(
        connection,
        data_cutoff if data_cutoff is not None else hour_start,
        "live_forecast",
        station_ids,
    )
    source_hour = data_cutoff - timedelta(hours=1) if data_cutoff is not None else None
    analysis = (
        _stored_grid(connection, substance_code, source_hour, "observed")
        if source_hour is not None
        else pd.DataFrame()
    )
    grid = build_semiphysical_forecast_grid(
        analysis,
        anchors,
        substance_code=substance_code,
        wind_speed=weather["wind_speed"],
        wind_from_degrees=weather["wind_direction"],
        boundary_layer_height=weather["boundary_layer_height"],
        precipitation=weather["precipitation"],
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
                   GREATEST(h.filtered_hourly_mean, 0.0) AS value,
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
            SELECT monitoring_post_id, bucket_start, data_kind, hor_win_spd, hor_win_dir,
                   wind_speed_100m, wind_direction_100m, boundary_layer_height, precipitation
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
        if not forecast.empty:
            forecast["source_hour"] = forecast["data_cutoff"] - pd.Timedelta(hours=1)
        forecast_by_source = {
            (substance_code, source_hour): group
            for (substance_code, source_hour), group in forecast.groupby(
                ["substance_code", "source_hour"],
                sort=False,
            )
        } if not forecast.empty else {}

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
        forecast_grids: dict[tuple[str, datetime], pd.DataFrame] = {}
        for substance_code, hour_start in sorted(observed_by_hour, key=lambda item: (item[1], item[0])):
            anchors = observed_by_hour[(substance_code, hour_start)]
            if anchors["monitoring_post_id"].nunique() < settings.GRID_MIN_STATIONS:
                continue
            station_ids = anchors["monitoring_post_id"].astype(int).tolist()
            wind_speed, wind_direction = _mean_wind_frame(
                weather_by_hour,
                hour_start,
                "historical_forecast",
                station_ids,
            )
            analysis_grid = assimilate_observations(
                anchors,
                forecast_grids.get((substance_code, hour_start)),
                wind_speed=wind_speed,
                wind_from_degrees=wind_direction,
            )
            if not analysis_grid.empty:
                batch.append((substance_code, hour_start, "observed", analysis_grid))
                counts["observed"] += len(analysis_grid)
            forecast_anchors = forecast_by_source.get((substance_code, hour_start))
            if forecast_anchors is not None and not forecast_anchors.empty and not analysis_grid.empty:
                forecast_anchors = forecast_anchors.copy()
                target_hour = forecast_anchors["hour_start"].max()
                station_ids = forecast_anchors["monitoring_post_id"].astype(int).tolist()
                cutoff = forecast_anchors["data_cutoff"].max()
                weather_values = _mean_weather_frame(
                    weather_by_hour,
                    cutoff,
                    "live_forecast",
                    station_ids,
                )
                cluster_weather = weather_by_hour.get(("live_forecast", cutoff))
                if cluster_weather is not None and not cluster_weather.empty:
                    forecast_anchors = forecast_anchors.merge(
                        cluster_weather[[
                            "monitoring_post_id", "wind_speed_100m", "wind_direction_100m",
                            "boundary_layer_height", "precipitation",
                        ]].rename(columns={
                            "wind_speed_100m": "weather_wind_speed",
                            "wind_direction_100m": "weather_wind_direction",
                            "boundary_layer_height": "weather_boundary_layer_height",
                            "precipitation": "weather_precipitation",
                        }),
                        on="monitoring_post_id",
                        how="left",
                    )
                forecast_grid = build_semiphysical_forecast_grid(
                    analysis_grid,
                    forecast_anchors,
                    substance_code=substance_code,
                    wind_speed=weather_values["wind_speed"],
                    wind_from_degrees=weather_values["wind_direction"],
                    boundary_layer_height=weather_values["boundary_layer_height"],
                    precipitation=weather_values["precipitation"],
                )
                if not forecast_grid.empty:
                    forecast_grids[(substance_code, target_hour)] = forecast_grid
                    batch.append((substance_code, target_hour, "forecast", forecast_grid))
                    counts["forecast"] += len(forecast_grid)
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
