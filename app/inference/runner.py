from __future__ import annotations

from datetime import datetime, timezone
import logging
from zoneinfo import ZoneInfo

from app.config import get_settings
from app.db import db_connection
from app.features.hourly_gas import refresh_hourly_gas_features
from app.features.weather_features import sync_live_weather
from app.inference.concentration_grid import refresh_concentration_grids
from app.inference.predict import predict_all_stations


logger = logging.getLogger(__name__)


def current_cutoff(now: datetime | None = None) -> datetime:
    settings = get_settings()
    instant = now or datetime.now(timezone.utc)
    local = instant.astimezone(ZoneInfo(settings.APP_TIMEZONE))
    return local.replace(minute=0, second=0, microsecond=0).astimezone(timezone.utc)


def run_forecast(cutoff: datetime | None = None) -> dict[str, object]:
    resolved_cutoff = cutoff or current_cutoff()
    with db_connection() as connection:
        hourly_rows = refresh_hourly_gas_features(connection, cutoff=resolved_cutoff)
        try:
            weather_rows = sync_live_weather(connection)
        except Exception as error:
            logger.warning("Open-Meteo synchronisation failed: %s", error)
            weather_rows = 0
    predictions = predict_all_stations(resolved_cutoff)
    grid_cells = refresh_concentration_grids(resolved_cutoff)
    return {
        "status": "completed", "data_cutoff": resolved_cutoff.isoformat(),
        "hourly_rows": hourly_rows, "weather_rows": weather_rows,
        "predictions": predictions,
        "grid_cells": grid_cells,
    }
