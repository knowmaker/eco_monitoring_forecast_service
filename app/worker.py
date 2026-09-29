from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from app.config import get_settings
from app.db import db_connection, fetch_all
from app.features.hourly_gas import refresh_hourly_gas_features
from app.features.weather_features import sync_live_weather
from app.inference.runner import run_forecast


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def next_run_time(now: datetime | None = None) -> datetime:
    settings = get_settings()
    zone = ZoneInfo(settings.APP_TIMEZONE)
    local_now = (now or datetime.now(timezone.utc)).astimezone(zone)
    candidate = local_now.replace(minute=settings.FORECAST_RUN_MINUTE, second=0, microsecond=0)
    if candidate <= local_now:
        candidate += timedelta(hours=1)
    return candidate.astimezone(timezone.utc)


def latest_due_cutoff(now: datetime | None = None) -> datetime:
    settings = get_settings()
    zone = ZoneInfo(settings.APP_TIMEZONE)
    local_now = (now or datetime.now(timezone.utc)).astimezone(zone)
    cutoff = local_now.replace(minute=0, second=0, microsecond=0)
    if local_now.minute < settings.FORECAST_RUN_MINUTE:
        cutoff -= timedelta(hours=1)
    return cutoff.astimezone(timezone.utc)


def _cutoffs_to_rebuild(
    expected: list[dict[str, object]],
    existing: list[dict[str, object]],
) -> list[datetime]:
    existing_keys = {
        (row["hour_start"], row["data_kind"], row["substance_code"])
        for row in existing
    }
    eligible_cutoffs: set[datetime] = set()
    missing: set[datetime] = set()
    for row in expected:
        source_hour = row["bucket_start"]
        substance_code = row["substance_code"]
        if not isinstance(source_hour, datetime) or not isinstance(substance_code, str):
            continue
        cutoff = source_hour + timedelta(hours=1)
        eligible_cutoffs.add(cutoff)
        observed_key = (source_hour, "observed", substance_code)
        forecast_key = (source_hour + timedelta(hours=2), "forecast", substance_code)
        if observed_key not in existing_keys or forecast_key not in existing_keys:
            missing.add(cutoff)

    if not missing:
        return []
    first_missing = min(missing)
    return sorted(cutoff for cutoff in eligible_cutoffs if cutoff >= first_missing)


def missing_grid_cutoffs(due_cutoff: datetime) -> list[datetime]:
    settings = get_settings()
    earliest_cutoff = due_cutoff - timedelta(hours=settings.FORECAST_CATCHUP_HOURS - 1)
    source_start = earliest_cutoff - timedelta(hours=1)
    with db_connection(readonly=True) as connection:
        expected = fetch_all(
            connection,
            """
            SELECT DISTINCT h.bucket_start, h.substance_code
            FROM public.gas_hourly_features h
            JOIN public.monitoring_posts p ON p.id = h.monitoring_post_id
            WHERE h.substance_code = ANY(%s)
              AND h.bucket_start >= %s
              AND h.bucket_start < %s
              AND h.filtered_hourly_mean IS NOT NULL
              AND p.latitude IS NOT NULL
              AND p.longitude IS NOT NULL
            ORDER BY h.bucket_start, h.substance_code
            """,
            (list(settings.active_gases), source_start, due_cutoff),
        )
        existing = fetch_all(
            connection,
            """
            SELECT hour_start, data_kind, substance_code
            FROM public.gas_concentration_grid
            WHERE substance_code = ANY(%s)
              AND hour_start >= %s
              AND hour_start <= %s
            GROUP BY hour_start, data_kind, substance_code
            """,
            (
                list(settings.active_gases),
                source_start,
                due_cutoff + timedelta(hours=1),
            ),
        )

    return _cutoffs_to_rebuild(expected, existing)


def run_due_forecasts(now: datetime | None = None) -> dict[str, object]:
    due_cutoff = latest_due_cutoff(now)
    with db_connection() as connection:
        refresh_hourly_gas_features(connection, cutoff=due_cutoff)
    cutoffs = missing_grid_cutoffs(due_cutoff)
    if not cutoffs:
        return {
            "status": "up_to_date",
            "due_cutoff": due_cutoff.isoformat(),
            "rebuilt_cutoffs": [],
        }

    try:
        with db_connection() as connection:
            sync_live_weather(connection)
    except Exception as error:
        logging.warning("Open-Meteo synchronisation failed during catch-up: %s", error)

    results: list[dict[str, object]] = []
    for cutoff in cutoffs:
        logging.info("Building missing forecast and grid for cutoff %s", cutoff.isoformat())
        results.append(run_forecast(cutoff, sync_weather=False))
    return {
        "status": "completed",
        "due_cutoff": due_cutoff.isoformat(),
        "rebuilt_cutoffs": [cutoff.isoformat() for cutoff in cutoffs],
        "results": results,
    }


def run_forever() -> None:
    while True:
        result = run_due_forecasts()
        logging.info("Forecast catch-up result: %s", json.dumps(result, ensure_ascii=False))
        if latest_due_cutoff() > datetime.fromisoformat(str(result["due_cutoff"])):
            continue
        run_at = next_run_time()
        delay = max((run_at - datetime.now(timezone.utc)).total_seconds(), 0.0)
        logging.info("Next forecast run at %s", run_at.isoformat())
        time.sleep(delay)


def ensure_model_artifact() -> None:
    artifact_path = get_settings().ARTIFACTS_DIR / "current" / "gas_forecast.joblib"
    if artifact_path.is_file():
        try:
            from app.models.base import load_artifact

            artifact = load_artifact(artifact_path)
            if (
                artifact.metadata.get("prediction_mode") == "physical_residual"
                and "monitoring_post_id" not in artifact.feature_columns
            ):
                return
            logging.info("Existing model uses an outdated feature contract; retraining")
        except Exception as error:
            logging.warning("Current model artifact cannot be loaded and will be replaced: %s", error)

    logging.info("Training the current semi-physical correction model at %s", artifact_path)
    from app.training.train import train_and_replace

    result = train_and_replace()
    logging.info("Initial training result: %s", json.dumps(result, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Hourly gas forecast worker.")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--train-if-missing", action="store_true")
    parser.add_argument("--cutoff", help="Optional ISO-8601 data cutoff for a one-shot run.")
    args = parser.parse_args()
    if args.train_if_missing:
        ensure_model_artifact()
    if args.once:
        cutoff = datetime.fromisoformat(args.cutoff) if args.cutoff else None
        if cutoff is not None and cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)
        print(json.dumps(run_forecast(cutoff), ensure_ascii=False, indent=2))
    else:
        run_forever()


if __name__ == "__main__":
    main()
