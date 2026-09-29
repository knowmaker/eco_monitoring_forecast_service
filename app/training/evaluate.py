from __future__ import annotations

import json

import numpy as np
import pandas as pd

from app.config import get_settings
from app.db import db_connection
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


def evaluate_temporal_and_station_holdouts() -> dict[str, object]:
    """Run several time-origin tests plus leave-one-station-out tests."""
    settings = get_settings()
    with db_connection(readonly=True) as connection:
        frame = _training_frame(connection)
    frame = frame.dropna(subset=["target_value", "physical_baseline", "correction_target"]).copy()
    columns = feature_columns(frame, include_post_id=True) + ["substance_code"]
    categorical = ["monitoring_post_id", "substance_code"]
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
    print(json.dumps(evaluate_temporal_and_station_holdouts(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
