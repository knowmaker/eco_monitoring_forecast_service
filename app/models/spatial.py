from __future__ import annotations

from catboost import CatBoostRegressor

from app.config import get_settings


def create_estimator(categorical_features: list[str]) -> CatBoostRegressor:
    settings = get_settings()
    return CatBoostRegressor(
        iterations=settings.CATBOOST_ITERATIONS,
        depth=settings.CATBOOST_DEPTH,
        learning_rate=settings.CATBOOST_LEARNING_RATE,
        loss_function="RMSE",
        random_seed=settings.RANDOM_SEED,
        cat_features=categorical_features,
        verbose=False,
        allow_writing_files=False,
    )
