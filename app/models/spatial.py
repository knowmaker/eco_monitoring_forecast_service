from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from app.config import get_settings


def create_estimator() -> CatBoostRegressor:
    settings = get_settings()
    return CatBoostRegressor(
        iterations=settings.CATBOOST_ITERATIONS,
        depth=settings.CATBOOST_DEPTH,
        learning_rate=settings.CATBOOST_LEARNING_RATE,
        loss_function="MAE",
        random_seed=settings.RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=4,
    )


@dataclass
class PerGasEstimator:
    """One fitted regressor per substance, with horizon as a numeric feature."""

    models: dict[str, CatBoostRegressor]
    feature_columns: list[str]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if "substance_code" not in frame:
            raise ValueError("The gas code is required to route predictions")
        codes = frame["substance_code"].astype(str).to_numpy()
        unknown = set(codes) - set(self.models)
        if unknown:
            raise ValueError(f"No model for gases: {sorted(unknown)}")
        result = np.empty(len(frame), dtype=float)
        for gas, model in self.models.items():
            mask = codes == gas
            if mask.any():
                result[mask] = np.asarray(
                    model.predict(frame.loc[mask, self.feature_columns]), dtype=float
                )
        return result
