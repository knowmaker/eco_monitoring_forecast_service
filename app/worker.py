from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone

from app.config import get_settings
from app.db import db_connection, fetch_all
from app.features.hourly_gas import refresh_hourly_gas_features
from app.features.weather_features import sync_live_weather
from app.inference.runner import current_cutoff, run_forecast
from app.models.base import deserialize_artifact
from app.training.model_store import get_current_model


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _cutoffs_to_rebuild(
    expected: list[dict[str, object]],
    existing: list[dict[str, object]],
    due_cutoff: datetime,
) -> list[datetime]:
    existing_frames = {
        (row["hour_start"], row["data_kind"], row["substance_code"]): row.get("generated_at")
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
        analysis_key = (source_hour, "analysis", substance_code)
        nowcast_key = (cutoff, "forecast", substance_code)
        forecast_key = (source_hour + timedelta(hours=2), "forecast", substance_code)
        analysis_generated = existing_frames.get(analysis_key)
        nowcast_generated = existing_frames.get(nowcast_key)
        if (
            analysis_key not in existing_frames
            or (row["forecast_expected"] and forecast_key not in existing_frames)
            or (
                cutoff == due_cutoff
                and (
                    nowcast_key not in existing_frames
                    or (
                        analysis_generated is not None
                        and nowcast_generated is not None
                        and nowcast_generated < analysis_generated
                    )
                )
            )
        ):
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
            SELECT DISTINCT h.bucket_start, h.substance_code,
                   EXISTS (
                       SELECT 1
                       FROM public.gas_predictions f
                       WHERE f.substance_code = h.substance_code
                         AND f.data_cutoff = h.bucket_start + INTERVAL '1 hour'
                         AND f.target_start = h.bucket_start + INTERVAL '2 hours'
                         AND f.status = 'ready'
                   ) AS forecast_expected
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
            SELECT hour_start, data_kind, substance_code, max(generated_at) AS generated_at
            FROM public.gas_concentration_frames
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

    return _cutoffs_to_rebuild(expected, existing, due_cutoff)


def run_due_forecasts(now: datetime | None = None) -> dict[str, object]:
    due_cutoff = current_cutoff(now)
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


def ensure_model() -> None:
    with db_connection(readonly=True) as connection:
        model = get_current_model(connection)
    if model is not None:
        try:
            artifact = deserialize_artifact(model["artifact"])
            if (
                artifact.metadata.get("prediction_mode") == "physical_residual"
                and "monitoring_post_id" not in artifact.feature_columns
                and artifact.metadata.get("active_gases") == list(get_settings().active_gases)
            ):
                return
            logging.info("Existing model uses an outdated feature contract; retraining")
        except Exception as error:
            logging.warning("Current model cannot be loaded and will be replaced: %s", error)

    logging.info("Training the current semi-physical correction model")
    from app.training.train import train_and_replace

    result = train_and_replace()
    logging.info("Initial training result: %s", json.dumps(result, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one hourly gas forecast job.")
    parser.add_argument("--train-if-missing", action="store_true")
    parser.add_argument("--cutoff", help="Optional ISO-8601 data cutoff for a one-shot run.")
    args = parser.parse_args()
    if args.train_if_missing:
        ensure_model()
    if args.cutoff:
        cutoff = datetime.fromisoformat(args.cutoff)
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)
        result = run_forecast(cutoff)
    else:
        result = run_due_forecasts()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
