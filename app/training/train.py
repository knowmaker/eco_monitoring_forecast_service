from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import get_settings
from app.db import db_connection
from app.features.local_features import build_local_frame, feature_columns
from app.features.physical_baseline import add_physical_baseline
from app.features.spatial_features import add_spatial_features
from app.models.base import ModelArtifact, save_artifact
from app.models.spatial import create_estimator
from app.training.backtest import chronological_split, regression_metrics, residual_interval
from app.training.model_store import replace_current_model


def _training_frame(connection) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for substance_code in get_settings().active_gases:
        frame = add_physical_baseline(
            add_spatial_features(
                build_local_frame(connection, substance_code, include_target=True), substance_code
            )
        )
        frame["correction_target"] = frame["target_value"] - frame["physical_baseline"]
        frames.append(frame.dropna(subset=["target_value"]))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False).sort_values("target_start").reset_index(drop=True)


def train_and_replace() -> dict[str, object]:
    settings = get_settings()
    with db_connection() as connection:
        clean = _training_frame(connection)
        if len(clean) < settings.MIN_TRAIN_ROWS:
            raise ValueError(f"{len(clean)} training rows; at least {settings.MIN_TRAIN_ROWS} required.")

        columns = feature_columns(clean, include_post_id=True) + ["substance_code"]
        categorical = ["monitoring_post_id", "substance_code"]
        clean["monitoring_post_id"] = clean["monitoring_post_id"].astype(str)
        clean["substance_code"] = clean["substance_code"].astype(str)
        split = chronological_split(clean, settings.TEST_FRACTION)

        estimator = create_estimator(categorical)
        estimator.fit(split.train[columns], split.train["correction_target"])
        correction = estimator.predict(split.test[columns])
        predicted = split.test["physical_baseline"].to_numpy(dtype=float) + correction
        predicted = predicted.clip(min=0.0)
        persistence = np.asarray([
            row.get(f"gas_{row['substance_code']}_lag_0", np.nan)
            for _, row in split.test.iterrows()
        ], dtype=float)
        metrics: dict[str, object] = {
            "overall": regression_metrics(split.test["target_value"], predicted),
            "test_rows": len(split.test),
            "by_gas": {},
            "by_station": {},
            "baselines": {
                "physical_only": regression_metrics(
                    split.test["target_value"],
                    split.test["physical_baseline"].to_numpy(dtype=float),
                ),
                "persistence": regression_metrics(split.test["target_value"], persistence),
            },
        }
        residual_intervals: dict[str, dict[str, float]] = {}
        for substance_code in settings.active_gases:
            mask = split.test["substance_code"] == substance_code
            if not mask.any():
                continue
            actual = split.test.loc[mask, "target_value"]
            gas_predicted = predicted[mask.to_numpy()]
            metrics["by_gas"][substance_code] = regression_metrics(actual, gas_predicted)
            lower, upper = residual_interval(actual, gas_predicted)
            residual_intervals[substance_code] = {"lower": lower, "upper": upper}

        for monitoring_post_id in sorted(split.test["monitoring_post_id"].unique()):
            mask = split.test["monitoring_post_id"] == monitoring_post_id
            if mask.any():
                metrics["by_station"][str(monitoring_post_id)] = regression_metrics(
                    split.test.loc[mask, "target_value"], predicted[mask.to_numpy()]
                )

        final_estimator = create_estimator(categorical)
        final_estimator.fit(clean[columns], clean["correction_target"])
        artifact_path = settings.ARTIFACTS_DIR / "current" / "gas_forecast.joblib"
        artifact = ModelArtifact(
            estimator=final_estimator,
            feature_columns=columns,
            categorical_features=categorical,
            residual_lower=None,
            residual_upper=None,
            metadata={
                "metrics": metrics,
                "residual_intervals": residual_intervals,
                "prediction_mode": "physical_residual",
            },
        )
        save_artifact(artifact_path, artifact)
        replace_current_model(
            connection,
            artifact_path=artifact_path,
            feature_columns=columns,
            categorical_features=categorical,
            train_start=clean["target_start"].min().to_pydatetime(),
            train_end=clean["target_start"].max().to_pydatetime(),
            rows_count=len(clean),
            metrics=metrics,
            residual_intervals=residual_intervals,
        )
        return {"rows": len(clean), "metrics": metrics}


def main() -> None:
    print(train_and_replace())


if __name__ == "__main__":
    main()
