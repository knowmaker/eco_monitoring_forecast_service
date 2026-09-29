from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable

import numpy as np
import pandas as pd
from psycopg import Connection

from app.config import get_settings
from app.db import fetch_all
from app.features.weather_features import WEATHER_COLUMNS, load_hourly_weather


GAS_LAGS = (0, 1, 2, 3, 6, 12, 24)
ROLLING_WINDOWS = (3, 6, 12, 24)
META_COLUMNS = {
    "source_bucket",
    "data_cutoff",
    "target_start",
    "target_end",
    "target_value",
    "substance_code",
    "correction_target",
}
ABSOLUTE_GAS_COLUMNS = (
    "raw_hourly_mean",
    "raw_hourly_median",
    "filtered_hourly_mean",
    "hourly_p95",
    "hourly_std",
)


def load_hourly_gas_features(
    connection: Connection,
    start: object | None = None,
    end: object | None = None,
) -> pd.DataFrame:
    conditions: list[str] = []
    params: list[object] = []
    if start is not None:
        conditions.append("bucket_start >= %s")
        params.append(pd.Timestamp(start).to_pydatetime())
    if end is not None:
        conditions.append("bucket_start < %s")
        params.append(pd.Timestamp(end).to_pydatetime())
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    rows = fetch_all(
        connection,
        f"""
        SELECT monitoring_post_id,
               substance_code,
               bucket_start,
               raw_hourly_mean,
               raw_hourly_median,
               filtered_hourly_mean,
               hourly_min,
               hourly_max,
               hourly_p95,
               hourly_std,
               samples_count
        FROM public.gas_hourly_features
        {where}
        ORDER BY monitoring_post_id, bucket_start, substance_code
        """,
        params,
    )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["bucket_start"] = pd.to_datetime(frame["bucket_start"], utc=True)
    for column in ABSOLUTE_GAS_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").clip(lower=0.0)
    stored_minimum = pd.to_numeric(frame["hourly_min"], errors="coerce")
    stored_maximum = pd.to_numeric(frame["hourly_max"], errors="coerce")
    frame["hourly_min"] = stored_minimum.clip(lower=0.0)
    frame["hourly_max"] = stored_maximum.clip(lower=0.0)
    return frame


def load_posts(connection: Connection) -> pd.DataFrame:
    rows = fetch_all(
        connection,
        """
        SELECT id AS monitoring_post_id, latitude, longitude
        FROM public.monitoring_posts
        WHERE active_to IS NULL
        ORDER BY id
        """,
    )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["monitoring_post_id", "latitude", "longitude"])
    frame["monitoring_post_id"] = frame["monitoring_post_id"].astype(int)
    return frame


def _dense_post_grid(
    gas: pd.DataFrame,
    active_gases: Iterable[str],
    *,
    post_ids: Iterable[int] | None = None,
    grid_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    grids: list[pd.DataFrame] = []
    grouped = {int(post_id): group for post_id, group in gas.groupby("monitoring_post_id", sort=False)}
    selected_post_ids = list(post_ids) if post_ids is not None else list(grouped)
    fallback_start = gas["bucket_start"].min()
    for post_id in selected_post_ids:
        group = grouped.get(int(post_id))
        start = group["bucket_start"].min() if group is not None else fallback_start
        end = grid_end if grid_end is not None else group["bucket_start"].max()
        grids.append(
            pd.DataFrame(
                {
                    "monitoring_post_id": int(post_id),
                    "source_bucket": pd.date_range(start, end, freq="1h"),
                }
            )
        )
    return pd.concat(grids, ignore_index=True) if grids else pd.DataFrame()


def _add_time_features(frame: pd.DataFrame) -> None:
    target_local = frame["target_start"].dt.tz_convert(get_settings().APP_TIMEZONE)
    hour = target_local.dt.hour
    weekday = target_local.dt.dayofweek
    frame["target_hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    frame["target_hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    frame["target_weekday_sin"] = np.sin(2 * np.pi * weekday / 7.0)
    frame["target_weekday_cos"] = np.cos(2 * np.pi * weekday / 7.0)


def build_local_frame(
    connection: Connection,
    target_substance: str,
    *,
    include_target: bool,
    history_start: object | None = None,
    history_end: object | None = None,
    weather_data_kind: str = "historical_forecast",
) -> pd.DataFrame:
    target_substance = target_substance.upper()
    settings = get_settings()
    gas = load_hourly_gas_features(connection, history_start, history_end)
    if gas.empty:
        return pd.DataFrame()
    gas = gas[gas["substance_code"].isin(settings.active_gases)].copy()
    posts = load_posts(connection)
    inference_grid_end = None
    inference_post_ids = None
    if not include_target and history_end is not None:
        inference_grid_end = pd.Timestamp(history_end) - pd.Timedelta(hours=1)
        inference_post_ids = posts["monitoring_post_id"].tolist()
    grid = _dense_post_grid(
        gas,
        settings.active_gases,
        post_ids=inference_post_ids,
        grid_end=inference_grid_end,
    )
    if grid.empty:
        return grid

    filtered_pivot = gas.pivot_table(
        index=["monitoring_post_id", "bucket_start"],
        columns="substance_code",
        values="filtered_hourly_mean",
        aggfunc="last",
    ).reset_index()
    filtered_pivot = filtered_pivot.rename(columns={code: f"gas_{code}" for code in settings.active_gases})
    frame = grid.merge(
        filtered_pivot,
        left_on=["monitoring_post_id", "source_bucket"],
        right_on=["monitoring_post_id", "bucket_start"],
        how="left",
    ).drop(columns=["bucket_start"], errors="ignore")

    target_stats = gas[gas["substance_code"] == target_substance][
        [
            "monitoring_post_id",
            "bucket_start",
            "raw_hourly_mean",
            "raw_hourly_median",
            "hourly_min",
            "hourly_max",
            "hourly_p95",
            "hourly_std",
            "samples_count",
        ]
    ]
    frame = frame.merge(
        target_stats,
        left_on=["monitoring_post_id", "source_bucket"],
        right_on=["monitoring_post_id", "bucket_start"],
        how="left",
    ).drop(columns=["bucket_start"], errors="ignore")

    for post_id, indices in frame.groupby("monitoring_post_id", sort=False).groups.items():
        ordered_indices = list(indices)
        for gas_code in settings.active_gases:
            source_column = f"gas_{gas_code}"
            if source_column not in frame:
                frame[source_column] = np.nan
            values = frame.loc[ordered_indices, source_column]
            for lag in GAS_LAGS:
                frame.loc[ordered_indices, f"{source_column}_lag_{lag}"] = values.shift(lag).to_numpy()
        target_column = f"gas_{target_substance}"
        values = frame.loc[ordered_indices, target_column]
        for window in ROLLING_WINDOWS:
            frame.loc[ordered_indices, f"target_rolling_mean_{window}"] = (
                values.rolling(window=window, min_periods=1).mean().to_numpy()
            )
            frame.loc[ordered_indices, f"target_rolling_std_{window}"] = (
                values.rolling(window=window, min_periods=2).std(ddof=0).to_numpy()
            )

    source_gas_columns = [f"gas_{code}" for code in settings.active_gases]
    frame = frame.drop(columns=source_gas_columns, errors="ignore")
    frame["data_cutoff"] = frame["source_bucket"] + pd.Timedelta(hours=1)
    frame["target_start"] = frame["source_bucket"] + pd.Timedelta(hours=2)
    frame["target_end"] = frame["target_start"] + pd.Timedelta(hours=1)

    frame = frame.merge(posts, on="monitoring_post_id", how="left")
    for column in ("latitude", "longitude"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    weather_end = pd.Timestamp(history_end) + pd.Timedelta(hours=3) if history_end is not None else None
    weather = load_hourly_weather(
        connection,
        history_start,
        weather_end,
        data_kind=weather_data_kind,
    )
    source_weather = weather.rename(
        columns={
            "bucket_start": "source_bucket",
            **{column: f"source_weather_{column}" for column in WEATHER_COLUMNS},
        }
    )
    target_weather = weather.rename(
        columns={
            "bucket_start": "target_start",
            **{column: f"target_weather_{column}" for column in WEATHER_COLUMNS},
        }
    )
    transport_weather = weather.rename(
        columns={
            "bucket_start": "data_cutoff",
            **{column: f"transport_weather_{column}" for column in WEATHER_COLUMNS},
        }
    )
    frame = frame.merge(source_weather, on=["monitoring_post_id", "source_bucket"], how="left")
    frame = frame.merge(transport_weather, on=["monitoring_post_id", "data_cutoff"], how="left")
    frame = frame.merge(target_weather, on=["monitoring_post_id", "target_start"], how="left")
    frame["substance_code"] = target_substance
    _add_time_features(frame)

    if include_target:
        labels = gas[gas["substance_code"] == target_substance][
            ["monitoring_post_id", "bucket_start", "filtered_hourly_mean"]
        ].rename(columns={"bucket_start": "target_start", "filtered_hourly_mean": "target_value"})
        frame = frame.merge(labels, on=["monitoring_post_id", "target_start"], how="left")

    current_column = f"gas_{target_substance}_lag_0"
    if include_target:
        frame = frame[frame[current_column].notna()]
    frame = frame.reset_index(drop=True)
    return frame


def feature_columns(frame: pd.DataFrame, *, include_post_id: bool = True) -> list[str]:
    excluded = set(META_COLUMNS)
    if not include_post_id:
        excluded.add("monitoring_post_id")
    return [column for column in frame.columns if column not in excluded]


def inference_rows_for_cutoff(
    connection: Connection,
    substance_code: str,
    cutoff: datetime,
) -> pd.DataFrame:
    cutoff_timestamp = pd.Timestamp(cutoff)
    if cutoff_timestamp.tzinfo is None:
        cutoff_timestamp = cutoff_timestamp.tz_localize("UTC")
    else:
        cutoff_timestamp = cutoff_timestamp.tz_convert("UTC")
    frame = build_local_frame(
        connection,
        substance_code,
        include_target=False,
        history_start=cutoff_timestamp - pd.Timedelta(hours=get_settings().FEATURE_LOOKBACK_HOURS),
        history_end=cutoff_timestamp,
        weather_data_kind="live_forecast",
    )
    if frame.empty:
        return frame
    # Keep the history until spatial features are calculated: neighbour travel
    # time may require a value from one to three earlier hours.
    return frame[frame["data_cutoff"] <= cutoff_timestamp].reset_index(drop=True)
