"""Fit one residual CatBoost estimator per gas for five target hours."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from app.config import EXCLUDED_FORECAST_POST_IDS, get_settings
from app.db import db_connection
from app.features.horizons import FORECAST_HORIZONS, expand_horizons
from app.features.local_features import build_local_frame, feature_columns
from app.features.physical_baseline import add_physical_baseline
from app.features.spatial_features import add_spatial_features
from app.features.weather_features import load_hourly_weather
from app.models.base import ModelArtifact, serialize_artifact
from app.models.spatial import PerGasEstimator, create_estimator
from app.training.backtest import regression_metrics, residual_interval
from app.training.model_store import replace_current_model


def _score_holdout(frame: pd.DataFrame, predictions: np.ndarray, gas: str) -> dict:
    physical = frame["physical_baseline"].to_numpy(float)
    persistence = frame[f"gas_{gas}_lag_0"].to_numpy(float)
    scores: dict[str, object] = {}
    for horizon in FORECAST_HORIZONS:
        mask = frame["horizon_hours"].eq(horizon).to_numpy()
        actual = frame.loc[mask, "target_value"]
        scores[str(horizon)] = {
            "rows": int(mask.sum()),
            "model": regression_metrics(actual, predictions[mask]),
            "physical_only": regression_metrics(actual, physical[mask]),
            "persistence": regression_metrics(actual, persistence[mask]),
        }
    return scores


def train_and_replace() -> dict[str, object]:
    settings = get_settings()
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=settings.TRAIN_HISTORY_DAYS)
    holdout_start = end - timedelta(days=settings.TRAIN_HOLDOUT_DAYS)
    if holdout_start <= start + timedelta(days=1):
        raise ValueError("Holdout period leaves no useful training interval")

    models = {}
    columns = None
    metrics = {"holdout_start_utc": holdout_start.isoformat(), "by_gas": {}}
    intervals = {}
    rows_count = 0
    first_target = None
    last_target = None

    with db_connection(readonly=True) as connection:
        weather = load_hourly_weather(
            connection, start, end + timedelta(hours=max(FORECAST_HORIZONS)),
            data_kind="historical_forecast",
        )
        for gas in settings.active_gases:
            source = build_local_frame(
                connection, gas, include_target=True,
                history_start=start, history_end=end,
                exclude_post_ids=EXCLUDED_FORECAST_POST_IDS,
            )
            if source.empty:
                raise ValueError(f"No eligible source measurements for {gas}")
            source = add_spatial_features(source, gas)
            frame = add_physical_baseline(
                expand_horizons(source, weather, include_target=True)
            )
            frame = frame.dropna(subset=["target_value", "physical_baseline"])
            frame = frame[
                (frame["target_start"] >= pd.Timestamp(start))
                & (frame["target_start"] < pd.Timestamp(end))
            ].copy()
            if frame.empty:
                raise ValueError(f"No training targets for {gas}")
            train = frame[frame["target_start"] < pd.Timestamp(holdout_start)]
            test = frame[frame["target_start"] >= pd.Timestamp(holdout_start)]
            if len(train) < settings.MIN_TRAIN_ROWS or len(test) < settings.MIN_TRAIN_ROWS:
                raise ValueError(f"Insufficient train or holdout rows for {gas}")

            gas_columns = [
                name for name in feature_columns(frame, include_post_id=False)
                if name != "substance_code"
            ]
            if columns is None:
                columns = gas_columns
            elif gas_columns != columns:
                raise ValueError(f"Feature columns differ for {gas}")

            candidate = create_estimator()
            candidate.fit(
                train[columns], train["target_value"] - train["physical_baseline"]
            )
            prediction = np.maximum(
                test["physical_baseline"].to_numpy(float)
                + candidate.predict(test[columns]), 0.0,
            )
            metrics["by_gas"][gas] = {
                "train_rows": len(train), "holdout_rows": len(test),
                "by_horizon": _score_holdout(test, prediction, gas),
            }
            low, high = residual_interval(test["target_value"], prediction)
            intervals[gas] = {"lower": low, "upper": high}

            final = create_estimator()
            final.fit(frame[columns], frame["target_value"] - frame["physical_baseline"])
            models[gas] = final
            rows_count += len(frame)
            first = frame["target_start"].min()
            last = frame["target_start"].max()
            first_target = first if first_target is None else min(first_target, first)
            last_target = last if last_target is None else max(last_target, last)

    if columns is None or first_target is None or last_target is None:
        raise ValueError("No forecast models were trained")
    artifact = ModelArtifact(
        estimator=PerGasEstimator(models, columns),
        feature_columns=[*columns, "substance_code"],
        categorical_features=["substance_code"],
        residual_lower=None, residual_upper=None,
        metadata={
            "model_type": "per_gas_five_hour",
            "prediction_mode": "physical_residual",
            "active_gases": list(settings.active_gases),
            "forecast_horizons": list(FORECAST_HORIZONS),
            "excluded_training_post_ids": sorted(EXCLUDED_FORECAST_POST_IDS),
            "residual_intervals": intervals,
            "train_end_utc": last_target.isoformat(),
        },
    )
    with db_connection() as connection:
        replace_current_model(
            connection, artifact=serialize_artifact(artifact),
            feature_columns=artifact.feature_columns,
            categorical_features=artifact.categorical_features,
            train_start=first_target.to_pydatetime(),
            train_end=last_target.to_pydatetime(),
            rows_count=rows_count, metrics=metrics,
            residual_intervals=intervals,
        )
    return {"rows_count": rows_count, "metrics": metrics}


def main() -> None:
    print(train_and_replace())


if __name__ == "__main__":
    main()
