from __future__ import annotations

from datetime import datetime

import pandas as pd

from app.config import get_settings
from app.db import db_connection
from app.features.local_features import inference_rows_for_cutoff
from app.features.physical_baseline import add_physical_baseline
from app.features.spatial_features import add_spatial_features
from app.inference.store import (
    mark_rows,
    remove_predictions_without_latest_measurement,
    set_prediction_status,
)
from app.models.base import load_artifact, prediction_results
from app.training.model_store import get_current_model, resolve_artifact_path


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
    """Run the single current model for every gas/station row in one batch."""
    counts = {substance_code: 0 for substance_code in get_settings().active_gases}
    with db_connection() as connection:
        frames: list[pd.DataFrame] = []
        for substance_code in get_settings().active_gases:
            remove_predictions_without_latest_measurement(connection, substance_code, cutoff)
            history = add_physical_baseline(
                add_spatial_features(
                    inference_rows_for_cutoff(connection, substance_code, cutoff), substance_code
                )
            )
            rows = _eligible_prediction_rows(history, substance_code, cutoff)
            if not rows.empty:
                frames.append(rows)
        if not frames:
            return counts

        rows = pd.concat(frames, ignore_index=True, sort=False)
        mark_rows(connection, rows, "running")
        model = get_current_model(connection)
        if model is None:
            mark_rows(connection, rows, "unavailable", "Обученная модель отсутствует.")
            return counts
        try:
            artifact = load_artifact(resolve_artifact_path(model["artifact_path"]))
            prepared = rows.copy()
            prepared["substance_code"] = prepared["substance_code"].astype(str)
            results = prediction_results(artifact, prepared)
            for row, result in zip(rows.itertuples(index=False), results, strict=True):
                substance_code = str(row.substance_code)
                set_prediction_status(
                    connection,
                    monitoring_post_id=int(row.monitoring_post_id),
                    substance_code=substance_code,
                    data_cutoff=row.data_cutoff.to_pydatetime(),
                    target_start=row.target_start.to_pydatetime(),
                    target_end=row.target_end.to_pydatetime(),
                    status="ready",
                    result=result,
                )
                counts[substance_code] += 1
        except Exception as error:
            mark_rows(connection, rows, "failed", str(error))
        return counts
