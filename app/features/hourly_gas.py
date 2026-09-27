from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from psycopg import Connection

from app.config import get_settings
from app.db import db_connection, fetch_all


UPSERT_SQL = """
INSERT INTO public.gas_hourly_features (
    monitoring_post_id,
    substance_code,
    bucket_start,
    raw_hourly_mean,
    raw_hourly_median,
    filtered_hourly_mean,
    hourly_min,
    hourly_max,
    hourly_p95,
    hourly_std,
    samples_count,
    refreshed_at
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
ON CONFLICT (monitoring_post_id, substance_code, bucket_start)
DO UPDATE SET
    raw_hourly_mean = EXCLUDED.raw_hourly_mean,
    raw_hourly_median = EXCLUDED.raw_hourly_median,
    filtered_hourly_mean = EXCLUDED.filtered_hourly_mean,
    hourly_min = EXCLUDED.hourly_min,
    hourly_max = EXCLUDED.hourly_max,
    hourly_p95 = EXCLUDED.hourly_p95,
    hourly_std = EXCLUDED.hourly_std,
    samples_count = EXCLUDED.samples_count,
    refreshed_at = NOW()
"""
NORMALIZE_STORED_VALUES_SQL = """
UPDATE public.gas_hourly_features
SET raw_hourly_mean = ABS(raw_hourly_mean),
    raw_hourly_median = ABS(raw_hourly_median),
    filtered_hourly_mean = ABS(filtered_hourly_mean),
    hourly_min = CASE
        WHEN hourly_min <= 0 AND hourly_max >= 0 THEN 0
        ELSE LEAST(ABS(hourly_min), ABS(hourly_max))
    END,
    hourly_max = GREATEST(ABS(hourly_min), ABS(hourly_max)),
    hourly_p95 = ABS(hourly_p95),
    hourly_std = ABS(hourly_std)
WHERE raw_hourly_mean < 0
   OR raw_hourly_median < 0
   OR filtered_hourly_mean < 0
   OR hourly_min < 0
   OR hourly_max < 0
   OR hourly_p95 < 0
   OR hourly_std < 0
"""


def _raw_gas_frame(
    connection: Connection,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    conditions = ["g.value IS NOT NULL", "g.substance_code IS NOT NULL"]
    params: list[object] = []
    if start is not None:
        conditions.append("g.device_timestamp_ms >= %s")
        params.append(int(start.timestamp() * 1000))
    if end is not None:
        conditions.append("g.device_timestamp_ms < %s")
        params.append(int(end.timestamp() * 1000))

    rows = fetch_all(
        connection,
        f"""
        SELECT p.monitoring_post_id,
               upper(g.substance_code) AS substance_code,
               g.device_timestamp_ms,
               g.value
        FROM public.gas_sensors g
        JOIN public.device_state ds ON ds.id = g.device_state_id
        JOIN public.plc_state p ON p.id = ds.plc_state_id
        WHERE {' AND '.join(conditions)}
        ORDER BY p.monitoring_post_id, upper(g.substance_code), g.device_timestamp_ms
        """,
        params,
    )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["device_timestamp_ms"], unit="ms", utc=True)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.dropna(subset=["value"])


def _causal_median_by_continuous_segment(group: pd.DataFrame) -> pd.Series:
    ordered = group.sort_values("timestamp")
    deltas = ordered["timestamp"].diff().dt.total_seconds()
    cadence = float(deltas[deltas > 0].median()) if (deltas > 0).any() else np.nan
    if np.isfinite(cadence) and cadence > 0:
        segment_ids = (deltas > cadence * 3).cumsum()
    else:
        segment_ids = pd.Series(0, index=ordered.index)
    result = ordered.groupby(segment_ids, sort=False)["value"].transform(
        lambda values: values.rolling(window=3, min_periods=1).median()
    )
    return result.reindex(group.index)


def build_hourly_features(raw: pd.DataFrame, timezone_name: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()

    frame = raw.copy().sort_values(["monitoring_post_id", "substance_code", "timestamp"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce").abs()
    frame = frame.dropna(subset=["value"])
    filtered = pd.Series(index=frame.index, dtype=float)
    for _, indices in frame.groupby(["monitoring_post_id", "substance_code"], sort=False).groups.items():
        filtered.loc[indices] = _causal_median_by_continuous_segment(frame.loc[indices])
    frame["filtered_value"] = filtered

    local_zone = ZoneInfo(timezone_name)
    frame["bucket_start"] = frame["timestamp"].dt.tz_convert(local_zone).dt.floor("h").dt.tz_convert("UTC")
    grouped = frame.groupby(["monitoring_post_id", "substance_code", "bucket_start"], as_index=False)
    hourly = grouped.agg(
        raw_hourly_mean=("value", "mean"),
        raw_hourly_median=("value", "median"),
        filtered_hourly_mean=("filtered_value", "mean"),
        hourly_min=("value", "min"),
        hourly_max=("value", "max"),
        hourly_p95=("value", lambda values: float(values.quantile(0.95))),
        hourly_std=("value", lambda values: float(values.std(ddof=0)) if len(values) > 1 else 0.0),
        samples_count=("value", "count"),
    )
    return hourly


def _database_rows(hourly: pd.DataFrame) -> Iterable[tuple[object, ...]]:
    for row in hourly.itertuples(index=False):
        yield (
            int(row.monitoring_post_id),
            str(row.substance_code),
            row.bucket_start.to_pydatetime(),
            float(row.raw_hourly_mean),
            float(row.raw_hourly_median),
            float(row.filtered_hourly_mean),
            float(row.hourly_min),
            float(row.hourly_max),
            float(row.hourly_p95),
            float(row.hourly_std),
            int(row.samples_count),
        )


def refresh_hourly_gas_features(
    connection: Connection,
    *,
    full: bool = False,
    cutoff: datetime | None = None,
) -> int:
    settings = get_settings()
    end = cutoff or datetime.now(timezone.utc)
    start = None if full else end - timedelta(hours=settings.FEATURE_LOOKBACK_HOURS)
    # Include two extra raw samples before the requested window for causal median continuity.
    query_start = start - timedelta(hours=1) if start is not None else None
    raw = _raw_gas_frame(connection, query_start, end)
    hourly = build_hourly_features(raw, settings.APP_TIMEZONE)
    if start is not None and not hourly.empty:
        hourly = hourly[hourly["bucket_start"] >= pd.Timestamp(start)]
    if hourly.empty:
        return 0
    with connection.cursor() as cursor:
        cursor.executemany(UPSERT_SQL, list(_database_rows(hourly)))
        cursor.execute(NORMALIZE_STORED_VALUES_SQL)
    return len(hourly)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build filtered hourly gas features.")
    parser.add_argument("--full", action="store_true", help="Rebuild all available history.")
    args = parser.parse_args()
    with db_connection() as connection:
        count = refresh_hourly_gas_features(connection, full=args.full)
    print(f"Upserted {count} hourly gas feature rows.")


if __name__ == "__main__":
    main()
