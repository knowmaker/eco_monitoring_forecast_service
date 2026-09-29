from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


SERVICE_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=SERVICE_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    DATABASE_URL: str = Field(validation_alias=AliasChoices("DATABASE_URL", "DB_DSN"))
    APP_TIMEZONE: str = "Europe/Moscow"
    FORECAST_RUN_MINUTE: int = Field(default=10, ge=0, le=59)
    FEATURE_LOOKBACK_HOURS: int = Field(default=72, ge=24)
    ACTIVE_GASES: str = "CO,NO,NO2,O3,SO2"
    ARTIFACTS_DIR: Path = SERVICE_ROOT / "artifacts"
    CATBOOST_ITERATIONS: int = Field(default=300, ge=20)
    CATBOOST_DEPTH: int = Field(default=6, ge=2, le=12)
    CATBOOST_LEARNING_RATE: float = Field(default=0.05, gt=0, le=1)
    RANDOM_SEED: int = 42
    TEST_FRACTION: float = Field(default=0.2, gt=0.05, lt=0.5)
    MIN_TRAIN_ROWS: int = Field(default=240, ge=48)
    OPEN_METEO_FORECAST_URL: str = "https://api.open-meteo.com/v1/forecast"
    OPEN_METEO_HISTORICAL_FORECAST_URL: str = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    OPEN_METEO_TIMEOUT_SECONDS: int = Field(default=30, ge=5, le=120)
    GRID_CELL_METERS: int = Field(default=100, ge=25, le=500)
    GRID_BUFFER_METERS: int = Field(default=1000, ge=100, le=5000)
    GRID_CLUSTER_DISTANCE_METERS: int = Field(default=5000, ge=500, le=50000)
    GRID_MIN_STATIONS: int = Field(default=1, ge=1, le=20)
    GRID_MAX_ADVECTION_METERS: int = Field(default=1000, ge=0, le=10000)
    GRID_TRANSPORT_BLEND: float = Field(default=0.25, ge=0, le=1)
    GRID_RETENTION_DAYS: int = Field(default=7, ge=1, le=366)
    PHYSICS_TIME_STEP_SECONDS: int = Field(default=3600, ge=60, le=3600)
    PHYSICS_MAX_ADVECTION_METERS: int = Field(default=1000, ge=100, le=20000)
    PHYSICS_ASSIMILATION_RADIUS_METERS: int = Field(default=600, ge=100, le=5000)
    PHYSICS_MIN_DIFFUSIVITY_M2_S: float = Field(default=4.0, gt=0, le=500)
    PHYSICS_MAX_DIFFUSIVITY_M2_S: float = Field(default=40.0, gt=0, le=1000)
    PHYSICS_CORRECTION_RADIUS_METERS: int = Field(default=700, ge=100, le=5000)

    @property
    def active_gases(self) -> tuple[str, ...]:
        return tuple(code.strip().upper() for code in self.ACTIVE_GASES.split(",") if code.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
