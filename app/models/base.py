from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
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


def serialize_artifact(artifact: ModelArtifact) -> bytes:
    buffer = BytesIO()
    joblib.dump(artifact, buffer)
    return buffer.getvalue()


def deserialize_artifact(payload: bytes | memoryview) -> ModelArtifact:
    artifact = joblib.load(BytesIO(bytes(payload)))
    if not isinstance(artifact, ModelArtifact):
        raise TypeError("Unsupported model artifact")
    return artifact


def prediction_results(artifact: ModelArtifact, frame: pd.DataFrame) -> list[ForecastResult]:
    results: list[ForecastResult] = []
    intervals = artifact.metadata.get("residual_intervals", {})
    predicted = artifact.predict(frame)
    if artifact.metadata.get("prediction_mode") == "physical_residual":
        baseline = pd.to_numeric(frame.get("physical_baseline"), errors="coerce").to_numpy(dtype=float)
        predicted = baseline + predicted
    for position, value in enumerate(predicted):
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
