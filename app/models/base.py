from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastResult:
    value: float
    lower_bound: float | None = None
    upper_bound: float | None = None


@dataclass
class ModelArtifact:
    estimator: Any
    feature_columns: list[str]
    categorical_features: list[str]
    residual_lower: float | None
    residual_upper: float | None
    metadata: dict[str, Any]

    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = frame.reindex(columns=self.feature_columns).copy()
        for column in self.categorical_features:
            prepared[column] = prepared[column].fillna("__MISSING__").astype(str)
        for column in set(self.feature_columns) - set(self.categorical_features):
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        return prepared

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict(self.prepare(frame)), dtype=float)


def save_artifact(path: Path, artifact: ModelArtifact) -> None:
    """Replace the current artifact without exposing a partially written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    joblib.dump(artifact, temporary_path)
    temporary_path.replace(path)


def load_artifact(path: Path) -> ModelArtifact:
    artifact = joblib.load(path)
    if not isinstance(artifact, ModelArtifact):
        raise TypeError(f"Unsupported artifact in {path}")
    return artifact


def prediction_results(artifact: ModelArtifact, frame: pd.DataFrame) -> list[ForecastResult]:
    results: list[ForecastResult] = []
    intervals = artifact.metadata.get("residual_intervals", {})
    for position, value in enumerate(artifact.predict(frame)):
        code = str(frame.iloc[position].get("substance_code", ""))
        interval = intervals.get(code, {})
        residual_lower = interval.get("lower", artifact.residual_lower)
        residual_upper = interval.get("upper", artifact.residual_upper)
        lower = value + residual_lower if residual_lower is not None else None
        upper = value + residual_upper if residual_upper is not None else None
        results.append(
            ForecastResult(
                value=max(0.0, float(value)),
                lower_bound=max(0.0, float(lower)) if lower is not None else None,
                upper_bound=max(0.0, float(upper)) if upper is not None else None,
            )
        )
    return results
