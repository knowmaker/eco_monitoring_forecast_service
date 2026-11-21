from __future__ import annotations

from datetime import datetime

import pandas as pd

from app.config import EXCLUDED_FORECAST_POST_IDS, get_settings
from app.db import db_connection
from app.features.horizons import FORECAST_HORIZONS, expand_horizons
from app.features.local_features import inference_rows_for_cutoff
from app.features.physical_baseline import add_physical_baseline
from app.features.spatial_features import add_spatial_features
from app.features.weather_features import load_hourly_weather
from app.inference.store import (
    remove_predictions_for_excluded_posts,
    remove_predictions_without_latest_measurement,
    store_prediction_rows,
)
from app.models.base import deserialize_artifact, prediction_results
from app.training.model_store import get_current_model


def _eligible_prediction_rows(
    history: pd.DataFrame,
    substance_code: str,
    cutoff: datetime,
) -> pd.DataFrame:
    current_value_column = f"gas_{substance_code}_lag_0"
    if history.empty or current_value_column not in history:
        return history.iloc[0:0].copy()
    return history[
        (history["data_cutoff"] == cutoff)
        & history[current_value_column].notna()
    ].reset_index(drop=True)


def predict_all_stations(cutoff: datetime) -> dict[str, int]:
    """Forecast all available gases and target hours from one data cutoff."""
    counts = {substance_code: 0 for substance_code in get_settings().active_gases}
    with db_connection() as connection:
        model = get_current_model(connection)
        if model is None:
            raise RuntimeError("No current forecast model in the database")
        artifact = deserialize_artifact(model["artifact"])
        if artifact.metadata.get("model_type") != "per_gas_five_hour":
            raise ValueError("The current model does not support gas-specific five-hour forecasts")
        horizons = FORECAST_HORIZONS
        target_weather = load_hourly_weather(
            connection, cutoff, cutoff + pd.Timedelta(hours=max(horizons)),
            data_kind="live_forecast",
        )
        excluded = EXCLUDED_FORECAST_POST_IDS
        remove_predictions_for_excluded_posts(connection, cutoff, excluded, horizons=horizons)
        frames: list[pd.DataFrame] = []
        for substance_code in get_settings().active_gases:
            remove_predictions_without_latest_measurement(
                connection, substance_code, cutoff, horizons=horizons,
            )
            history = add_spatial_features(
                inference_rows_for_cutoff(
                    connection, substance_code, cutoff, exclude_post_ids=excluded,
                ), substance_code,
            )
            rows = _eligible_prediction_rows(history, substance_code, cutoff)
            if not rows.empty:
                rows = expand_horizons(
                    rows, target_weather, horizons=horizons, include_target=False,
                )
                rows = add_physical_baseline(rows)
                frames.append(rows)
        if not frames:
            return counts

        rows = pd.concat(frames, ignore_index=True, sort=False)
        try:
            prepared = rows.copy()
            prepared["substance_code"] = prepared["substance_code"].astype(str)
            results = prediction_results(artifact, prepared)
            store_prediction_rows(connection, rows, "ready", results)
            for substance_code, count in rows["substance_code"].value_counts().items():
                counts[str(substance_code)] = int(count)
        except Exception as error:
            connection.rollback()
            store_prediction_rows(connection, rows, "failed", error_message=str(error))
            connection.commit()
            raise
        return counts
