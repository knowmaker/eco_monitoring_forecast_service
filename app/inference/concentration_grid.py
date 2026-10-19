from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from math import atan2, degrees, radians
from typing import Any

import numpy as np
import pandas as pd
from psycopg import Connection

from app.concentration_frame import decode_frame, encode_frame
from app.config import get_settings
from app.db import db_connection, fetch_all, fetch_one
from app.features.concentration_grid import (
    assimilate_observations,
    build_semiphysical_forecast_grid,
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


def _analysis_anchors(connection: Connection, substance_code: str, hour_start: datetime) -> pd.DataFrame:
    return _anchor_frame(
        fetch_all(
            connection,
            """
            SELECT h.monitoring_post_id, p.latitude, p.longitude, p.source,
                   GREATEST(h.filtered_hourly_mean, 0.0) AS value,
                   h.samples_count,
                   NULL::double precision AS lower_bound,
                   NULL::double precision AS upper_bound,
                   w.wind_speed_100m AS weather_wind_speed,
                   w.wind_direction_100m AS weather_wind_direction,
                   w.boundary_layer_height AS weather_boundary_layer_height,
                   w.precipitation AS weather_precipitation
            FROM public.gas_hourly_features h
            JOIN public.monitoring_posts p ON p.id = h.monitoring_post_id
            LEFT JOIN public.external_weather_hourly w
              ON w.monitoring_post_id = h.monitoring_post_id
             AND w.bucket_start = h.bucket_start
             AND w.data_kind = 'historical_forecast'
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


def _nowcast_weather_anchors(
    connection: Connection,
    substance_code: str,
    source_hour: datetime,
    hour_start: datetime,
) -> pd.DataFrame:
    return _anchor_frame(
        fetch_all(
            connection,
            """
            SELECT h.monitoring_post_id, p.latitude, p.longitude,
                   GREATEST(h.filtered_hourly_mean, 0.0) AS value,
                   COALESCE(w.wind_speed_100m, old_w.wind_speed_100m) AS weather_wind_speed,
                   COALESCE(w.wind_direction_100m, old_w.wind_direction_100m) AS weather_wind_direction,
                   COALESCE(w.boundary_layer_height, old_w.boundary_layer_height) AS weather_boundary_layer_height,
                   COALESCE(w.precipitation, old_w.precipitation) AS weather_precipitation
            FROM public.gas_hourly_features h
            JOIN public.monitoring_posts p ON p.id = h.monitoring_post_id
            LEFT JOIN public.external_weather_hourly w
              ON w.monitoring_post_id = h.monitoring_post_id
             AND w.bucket_start = %s
             AND w.data_kind = 'live_forecast'
            LEFT JOIN public.external_weather_hourly old_w
              ON old_w.monitoring_post_id = h.monitoring_post_id
             AND old_w.bucket_start = h.bucket_start
             AND old_w.data_kind = 'historical_forecast'
            WHERE h.substance_code = %s AND h.bucket_start = %s
              AND h.filtered_hourly_mean IS NOT NULL
              AND p.latitude IS NOT NULL AND p.longitude IS NOT NULL
            ORDER BY h.monitoring_post_id
            """,
            (hour_start, substance_code, source_hour),
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
    row = fetch_one(
        connection,
        """
        SELECT payload, content_sha256
        FROM public.gas_concentration_frames
        WHERE substance_code = %s AND hour_start = %s AND data_kind = %s
        """,
        (substance_code, hour_start, data_kind),
    )
    if row is None:
        return pd.DataFrame()
    return decode_frame(bytes(row["payload"]), row["content_sha256"])


def _replace_grid(
    connection: Connection,
    substance_code: str,
    hour_start: datetime,
    data_kind: str,
    grid: pd.DataFrame,
) -> int:
    with connection.cursor() as cursor:
        if grid.empty:
            cursor.execute(
                """
                DELETE FROM public.gas_concentration_frames
                WHERE substance_code = %s AND hour_start = %s AND data_kind = %s
                """,
                (substance_code, hour_start, data_kind),
            )
            connection.commit()
            return 0
        content, content_hash = encode_frame(grid)
        station_count = int(grid["source_station_count"].max())
        mean_wind_u = float(grid["wind_u"].dropna().mean())
        mean_wind_v = float(grid["wind_v"].dropna().mean())
        mean_wind_speed = float(np.hypot(mean_wind_u, mean_wind_v))
        mean_wind_direction = (
            (degrees(atan2(mean_wind_u, mean_wind_v)) + 540.0) % 360.0
            if mean_wind_speed > 0.0
            else None
        )
        cursor.execute(
            """
            INSERT INTO public.gas_concentration_frames (
                substance_code, hour_start, data_kind, payload, content_sha256,
                cell_count, cell_size_meters,
                south, west, north, east, value_min, value_max,
                source_station_count, wind_speed, wind_direction, generated_at
            )
            VALUES (
                %s, %s, %s, %s, %s,
                %s, %s,
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, NOW()
            )
            ON CONFLICT (substance_code, hour_start, data_kind) DO UPDATE SET
                payload = EXCLUDED.payload,
                content_sha256 = EXCLUDED.content_sha256,
                cell_count = EXCLUDED.cell_count,
                cell_size_meters = EXCLUDED.cell_size_meters,
                south = EXCLUDED.south,
                west = EXCLUDED.west,
                north = EXCLUDED.north,
                east = EXCLUDED.east,
                value_min = EXCLUDED.value_min,
                value_max = EXCLUDED.value_max,
                source_station_count = EXCLUDED.source_station_count,
                wind_speed = EXCLUDED.wind_speed,
                wind_direction = EXCLUDED.wind_direction,
                generated_at = EXCLUDED.generated_at
            """,
            (
                substance_code,
                hour_start,
                data_kind,
                content,
                content_hash,
                len(grid),
                get_settings().GRID_CELL_METERS,
                float(grid["south"].min()),
                float(grid["west"].min()),
                float(grid["north"].max()),
                float(grid["east"].max()),
                float(grid["value"].min()),
                float(grid["value"].max()),
                station_count,
                mean_wind_speed,
                mean_wind_direction,
            ),
        )
    connection.commit()
    return len(grid)


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


def _attach_live_weather(anchors: pd.DataFrame, weather: pd.DataFrame | None) -> pd.DataFrame:
    if weather is None or weather.empty:
        return anchors
    weather_columns = (
        "weather_wind_speed", "weather_wind_direction",
        "weather_boundary_layer_height", "weather_precipitation",
    )
    sources = anchors.copy()
    for column in weather_columns:
        if column not in sources:
            sources[column] = np.nan
    joined = sources.merge(
        weather[[
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
        suffixes=("", "_live"),
    )
    for column in weather_columns:
        joined[column] = joined[f"{column}_live"].combine_first(joined[column])
    return joined.drop(columns=[f"{column}_live" for column in weather_columns])


def build_analysis_grid(connection: Connection, substance_code: str, hour_start: datetime) -> int:
    anchors = _analysis_anchors(connection, substance_code, hour_start)
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
    return _replace_grid(connection, substance_code, hour_start, "analysis", grid)


def build_nowcast_grid(connection: Connection, substance_code: str, hour_start: datetime) -> int:
    source_hour = hour_start - timedelta(hours=1)
    analysis = _stored_grid(connection, substance_code, source_hour, "analysis")
    if analysis.empty:
        return 0
    weather_anchors = _nowcast_weather_anchors(
        connection, substance_code, source_hour, hour_start
    )
    station_ids = (
        weather_anchors["monitoring_post_id"].astype(int).tolist()
        if not weather_anchors.empty else []
    )
    weather = _mean_weather(connection, hour_start, "live_forecast", station_ids)
    grid = build_semiphysical_forecast_grid(
        analysis,
        pd.DataFrame(),
        substance_code=substance_code,
        wind_speed=weather["wind_speed"],
        wind_from_degrees=weather["wind_direction"],
        weather_anchors=weather_anchors,
        boundary_layer_height=weather["boundary_layer_height"],
        precipitation=weather["precipitation"],
    )
    if grid.empty:
        return 0
    return _replace_grid(connection, substance_code, hour_start, "forecast", grid)


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
    nowcast = (
        _stored_grid(connection, substance_code, data_cutoff, "forecast")
        if data_cutoff is not None
        else pd.DataFrame()
    )
    grid = build_semiphysical_forecast_grid(
        nowcast,
        anchors,
        substance_code=substance_code,
        wind_speed=weather["wind_speed"],
        wind_from_degrees=weather["wind_direction"],
        boundary_layer_height=weather["boundary_layer_height"],
        precipitation=weather["precipitation"],
        duration_seconds=(hour_start - data_cutoff).total_seconds() if data_cutoff else 3600.0,
    )
    return _replace_grid(connection, substance_code, hour_start, "forecast", grid)


def refresh_concentration_grids(cutoff: datetime) -> dict[str, int]:
    analysis_hour = cutoff - timedelta(hours=1)
    forecast_hour = cutoff + timedelta(hours=1)
    counts = {"analysis": 0, "forecast": 0}
    with db_connection() as connection:
        for substance_code in get_settings().active_gases:
            counts["analysis"] += build_analysis_grid(connection, substance_code, analysis_hour)
            counts["forecast"] += build_nowcast_grid(connection, substance_code, cutoff)
            counts["forecast"] += build_forecast_grid(connection, substance_code, forecast_hour)
        retention_start = cutoff - timedelta(hours=get_settings().GRID_RETENTION_HOURS)
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM public.gas_concentration_frames WHERE hour_start < %s",
                (retention_start,),
            )
        connection.commit()
    return counts


def backfill_concentration_grids() -> dict[str, int]:
    counts = {"analysis": 0, "forecast": 0}
    settings = get_settings()
    retention_start = datetime.now(timezone.utc).replace(
        minute=0,
        second=0,
        microsecond=0,
    ) - timedelta(hours=settings.GRID_RETENTION_HOURS)
    source_start = retention_start - timedelta(hours=2)
    with db_connection() as connection:
        analysis_source_rows = fetch_all(
            connection,
            """
            SELECT h.substance_code, h.bucket_start AS hour_start,
                   h.monitoring_post_id, p.latitude, p.longitude, p.source,
                   GREATEST(h.filtered_hourly_mean, 0.0) AS value,
                   h.samples_count,
                   NULL::double precision AS lower_bound,
                   NULL::double precision AS upper_bound,
                   w.wind_speed_100m AS weather_wind_speed,
                   w.wind_direction_100m AS weather_wind_direction,
                   w.boundary_layer_height AS weather_boundary_layer_height,
                   w.precipitation AS weather_precipitation
            FROM public.gas_hourly_features h
            JOIN public.monitoring_posts p ON p.id = h.monitoring_post_id
            LEFT JOIN public.external_weather_hourly w
              ON w.monitoring_post_id = h.monitoring_post_id
             AND w.bucket_start = h.bucket_start
             AND w.data_kind = 'historical_forecast'
            WHERE h.substance_code = ANY(%s)
              AND h.bucket_start >= %s
              AND h.filtered_hourly_mean IS NOT NULL
              AND p.latitude IS NOT NULL AND p.longitude IS NOT NULL
            ORDER BY h.bucket_start, h.substance_code, h.monitoring_post_id
            """,
            (list(settings.active_gases), source_start),
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
        analysis_sources = pd.DataFrame(analysis_source_rows)
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
        analysis_sources_by_hour = {
            (substance_code, hour_start): group
            for (substance_code, hour_start), group in analysis_sources.groupby(
                ["substance_code", "hour_start"],
                sort=False,
            )
        } if not analysis_sources.empty else {}
        if not forecast.empty:
            forecast["source_hour"] = forecast["data_cutoff"] - pd.Timedelta(hours=1)
        forecast_by_source = {
            (substance_code, source_hour): group
            for (substance_code, source_hour), group in forecast.groupby(
                ["substance_code", "source_hour"],
                sort=False,
            )
        } if not forecast.empty else {}

        forecast_grids: dict[tuple[str, datetime], pd.DataFrame] = {}
        for substance_code, hour_start in sorted(
            analysis_sources_by_hour,
            key=lambda item: (item[1], item[0]),
        ):
            anchors = analysis_sources_by_hour[(substance_code, hour_start)]
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
            if not analysis_grid.empty and hour_start >= retention_start:
                counts["analysis"] += _replace_grid(
                    connection,
                    substance_code,
                    hour_start,
                    "analysis",
                    analysis_grid,
                )
            if analysis_grid.empty:
                continue
            cutoff = hour_start + timedelta(hours=1)
            weather_at_cutoff = weather_by_hour.get(("live_forecast", cutoff))
            weather_sources = _attach_live_weather(anchors, weather_at_cutoff)
            weather_values = _mean_weather_frame(
                weather_by_hour,
                cutoff,
                "live_forecast",
                station_ids,
            )
            nowcast_grid = build_semiphysical_forecast_grid(
                analysis_grid,
                pd.DataFrame(),
                substance_code=substance_code,
                wind_speed=weather_values["wind_speed"],
                wind_from_degrees=weather_values["wind_direction"],
                weather_anchors=weather_sources,
                boundary_layer_height=weather_values["boundary_layer_height"],
                precipitation=weather_values["precipitation"],
            )
            if nowcast_grid.empty:
                continue
            forecast_grids[(substance_code, cutoff)] = nowcast_grid
            if cutoff >= retention_start:
                counts["forecast"] += _replace_grid(
                    connection, substance_code, cutoff, "forecast", nowcast_grid
                )

            forecast_anchors = forecast_by_source.get((substance_code, hour_start))
            if forecast_anchors is not None and not forecast_anchors.empty:
                forecast_anchors = _attach_live_weather(
                    forecast_anchors.copy(), weather_at_cutoff
                )
                target_hour = forecast_anchors["hour_start"].max()
                forecast_station_ids = forecast_anchors["monitoring_post_id"].astype(int).tolist()
                forecast_weather = _mean_weather_frame(
                    weather_by_hour,
                    cutoff,
                    "live_forecast",
                    forecast_station_ids,
                )
                forecast_grid = build_semiphysical_forecast_grid(
                    nowcast_grid,
                    forecast_anchors,
                    substance_code=substance_code,
                    wind_speed=forecast_weather["wind_speed"],
                    wind_from_degrees=forecast_weather["wind_direction"],
                    boundary_layer_height=forecast_weather["boundary_layer_height"],
                    precipitation=forecast_weather["precipitation"],
                    duration_seconds=(target_hour - cutoff).total_seconds(),
                )
                if not forecast_grid.empty:
                    forecast_grids[(substance_code, target_hour)] = forecast_grid
                    counts["forecast"] += _replace_grid(
                        connection,
                        substance_code,
                        target_hour,
                        "forecast",
                        forecast_grid,
                    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and store gas concentration frames.")
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
