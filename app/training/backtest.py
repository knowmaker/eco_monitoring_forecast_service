from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class TimeSplit:
    train: pd.DataFrame
    test: pd.DataFrame


def chronological_split(frame: pd.DataFrame, test_fraction: float) -> TimeSplit:
    ordered_times = np.array(sorted(frame["target_start"].dropna().unique()))
    if len(ordered_times) < 2:
        raise ValueError("Not enough distinct hours for chronological split.")
    split_index = max(1, min(len(ordered_times) - 1, int(len(ordered_times) * (1.0 - test_fraction))))
    split_time = ordered_times[split_index]
    train = frame[frame["target_start"] < split_time].copy()
    test = frame[frame["target_start"] >= split_time].copy()
    if train.empty or test.empty:
        raise ValueError("Chronological split produced an empty partition.")
    return TimeSplit(train=train, test=test)


def regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else float("nan"),
    }


def residual_interval(actual: pd.Series, predicted: np.ndarray) -> tuple[float, float]:
    residuals = actual.to_numpy(dtype=float) - np.asarray(predicted, dtype=float)
    return float(np.quantile(residuals, 0.05)), float(np.quantile(residuals, 0.95))

