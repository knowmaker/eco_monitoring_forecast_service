from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from app.config import get_settings
from app.db import db_connection, fetch_one
from app.features.local_features import feature_columns
from app.models.spatial import create_estimator
from app.training.backtest import regression_metrics
from app.training.train import _training_frame


def _forecast(estimator, frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    correction = estimator.predict(frame[columns])
    return np.maximum(frame["physical_baseline"].to_numpy(float) + correction, 0.0)


def _metrics(frame: pd.DataFrame, predicted: np.ndarray) -> dict[str, object]:
    result: dict[str, object] = {
        "overall": regression_metrics(frame["target_value"], predicted),
        "by_station": {},
        "by_gas": {},
    }
    for station_id in sorted(frame["monitoring_post_id"].unique()):
        mask = frame["monitoring_post_id"] == station_id
        result["by_station"][str(station_id)] = regression_metrics(
            frame.loc[mask, "target_value"], predicted[mask.to_numpy()]
        )
    for substance_code in sorted(frame["substance_code"].unique()):
        mask = frame["substance_code"] == substance_code
        result["by_gas"][str(substance_code)] = regression_metrics(
            frame.loc[mask, "target_value"], predicted[mask.to_numpy()]
        )
    return result


def _compare_by_gas(frame: pd.DataFrame) -> dict[str, object]:
    comparison = {}
    for code, group in frame.groupby("substance_code", sort=True):
        actual = group["target_value"]
        persistence = group[f"gas_{code}_lag_0"].to_numpy(float)
        comparison[str(code)] = {
            "rows": len(group),
            "catboost": regression_metrics(actual, group["catboost"].to_numpy(float)),
            "persistence": regression_metrics(actual, persistence),
            "physical_only": regression_metrics(actual, group["physical_baseline"].to_numpy(float)),
        }
    return comparison


def compare_persistence_rolling() -> dict[str, object]:
    """Test three successive days with models fitted only on earlier hours."""
    with db_connection(readonly=True) as connection:
        bounds = fetch_one(
            connection,
            """
            SELECT LEAST(
                (SELECT max(bucket_start) FROM public.external_weather_hourly
                 WHERE data_kind = 'historical_forecast'),
                (SELECT max(bucket_start) FROM public.gas_hourly_features)
            ) AS last_hour
            """,
        )
        if not bounds or bounds["last_hour"] is None:
            raise ValueError("No historical weather and gas observations are available")
        last_hour = pd.Timestamp(bounds["last_hour"])
        frame = _training_frame(
            connection,
            history_start=last_hour - pd.Timedelta(days=10),
            history_end=last_hour - pd.Timedelta(hours=1),
        )

    frame = frame.dropna(subset=["target_value", "physical_baseline", "correction_target"]).copy()
    frame = frame[frame["target_start"] <= last_hour].copy()
    frame["substance_code"] = frame["substance_code"].astype(str)
    columns = feature_columns(frame, include_post_id=False) + ["substance_code"]
    end = last_hour + pd.Timedelta(hours=1)
    windows = []
    tested = []
    for index in range(3):
        start = end - pd.Timedelta(hours=24 * (3 - index))
        stop = start + pd.Timedelta(hours=24)
        train = frame[frame["target_start"] < start - pd.Timedelta(hours=1)]
        test = frame[(frame["target_start"] >= start) & (frame["target_start"] < stop)].copy()
        if len(train) < get_settings().MIN_TRAIN_ROWS or test.empty:
            raise ValueError(f"Insufficient training or test rows for window {start}")
        estimator = create_estimator(["substance_code"])
        estimator.fit(train[columns], train["correction_target"])
        test["catboost"] = _forecast(estimator, test, columns)
        tested.append(test)
        windows.append({
            "start": start.isoformat(),
            "end": stop.isoformat(),
            "train_rows": len(train),
            "test_rows": len(test),
            "by_gas": _compare_by_gas(test),
        })
    return {
        "historical_weather_through": last_hour.isoformat(),
        "windows": windows,
        "combined_by_gas": _compare_by_gas(pd.concat(tested, ignore_index=True)),
    }


def evaluate_temporal_and_station_holdouts() -> dict[str, object]:
    """Run several time-origin tests plus leave-one-station-out tests."""
    settings = get_settings()
    with db_connection(readonly=True) as connection:
        frame = _training_frame(connection)
    frame = frame.dropna(subset=["target_value", "physical_baseline", "correction_target"]).copy()
    columns = feature_columns(frame, include_post_id=False) + ["substance_code"]
    categorical = ["substance_code"]
    frame["monitoring_post_id"] = frame["monitoring_post_id"].astype(str)
    frame["substance_code"] = frame["substance_code"].astype(str)
    distinct_hours = np.asarray(sorted(frame["target_start"].unique()))
    report: dict[str, object] = {"rolling_windows": [], "station_holdouts": {}}

    for fraction in (0.45, 0.60, 0.72):
        split_index = min(len(distinct_hours) - 2, max(1, int(len(distinct_hours) * fraction)))
        split_time = distinct_hours[split_index]
        end_index = min(len(distinct_hours), split_index + 24)
        test_end = distinct_hours[end_index - 1] + pd.Timedelta(hours=1)
        train = frame[frame["target_start"] < split_time]
        test = frame[(frame["target_start"] >= split_time) & (frame["target_start"] < test_end)]
        if len(train) < settings.MIN_TRAIN_ROWS or test.empty:
            continue
        estimator = create_estimator(categorical)
        estimator.fit(train[columns], train["correction_target"])
        report["rolling_windows"].append({
            "start": pd.Timestamp(split_time).isoformat(),
            "end": pd.Timestamp(test_end).isoformat(),
            "train_rows": len(train),
            "test_rows": len(test),
            **_metrics(test, _forecast(estimator, test, columns)),
        })

    for station_id, station_frame in frame.groupby("monitoring_post_id", sort=True):
        station_hours = np.asarray(sorted(station_frame["target_start"].unique()))
        if len(station_frame) < 100 or len(station_hours) < 12:
            continue
        split_time = station_hours[max(1, int(len(station_hours) * 0.8))]
        train = frame[
            (frame["monitoring_post_id"] != station_id)
            & (frame["target_start"] < split_time)
        ]
        test = station_frame[station_frame["target_start"] >= split_time]
        if len(train) < settings.MIN_TRAIN_ROWS or test.empty:
            continue
        estimator = create_estimator(categorical)
        estimator.fit(train[columns], train["correction_target"])
        report["station_holdouts"][station_id] = {
            "start": pd.Timestamp(split_time).isoformat(),
            "train_rows": len(train),
            "test_rows": len(test),
            **_metrics(test, _forecast(estimator, test, columns)),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate forecast models without writing to the database.")
    parser.add_argument("--compare-persistence", action="store_true")
    args = parser.parse_args()
    result = compare_persistence_rolling() if args.compare_persistence else evaluate_temporal_and_station_holdouts()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
