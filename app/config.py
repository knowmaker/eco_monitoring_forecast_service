from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


SERVICE_ROOT = Path(__file__).resolve().parents[1]
FORECAST_GASES = ("CO", "NO", "NO2", "O3", "SO2", "H2S")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=SERVICE_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    DATABASE_URL: str = Field(validation_alias=AliasChoices("DATABASE_URL", "DB_DSN"))
    APP_TIMEZONE: str = "Europe/Moscow"
    FORECAST_CATCHUP_HOURS: int = Field(default=72, ge=1, le=168)
    FEATURE_LOOKBACK_HOURS: int = Field(default=72, ge=24)
    CATBOOST_ITERATIONS: int = Field(default=300, ge=20)
    CATBOOST_DEPTH: int = Field(default=6, ge=2, le=12)
    CATBOOST_LEARNING_RATE: float = Field(default=0.05, gt=0, le=1)
    RANDOM_SEED: int = 42
    TEST_FRACTION: float = Field(default=0.2, gt=0.05, lt=0.5)
    MIN_TRAIN_ROWS: int = Field(default=240, ge=48)
    OPEN_METEO_FORECAST_URL: str = "https://api.open-meteo.com/v1/forecast"
    OPEN_METEO_HISTORICAL_FORECAST_URL: str = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    OPEN_METEO_TIMEOUT_SECONDS: int = Field(default=30, ge=5, le=120)
    GRID_CELL_METERS: int = Field(default=500, ge=25, le=1000)
    MAX_OBSERVATION_INFLUENCE_METERS: int = Field(default=10000, ge=500, le=50000)
    GRID_MIN_CONFIDENCE: float = Field(default=0.02, ge=0, le=1)
    GRID_RETENTION_HOURS: int = Field(default=72, ge=1, le=24 * 366)
    PHYSICS_MIN_DIFFUSIVITY_M2_S: float = Field(default=4.0, gt=0, le=500)
    PHYSICS_MAX_DIFFUSIVITY_M2_S: float = Field(default=40.0, gt=0, le=1000)
    PHYSICS_CONFIDENCE_RETENTION_PER_HOUR: float = Field(default=0.65, gt=0, le=1)
    PHYSICS_MIN_CONCENTRATION: float = Field(default=0.000001, ge=0)

    @property
    def active_gases(self) -> tuple[str, ...]:
        return FORECAST_GASES


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
