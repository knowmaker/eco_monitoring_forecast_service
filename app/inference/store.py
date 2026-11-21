from __future__ import annotations

from datetime import datetime, timedelta

from psycopg import Connection

from app.models.base import ForecastResult


_UPSERT_PREDICTION = """
    INSERT INTO public.gas_predictions (
        monitoring_post_id, substance_code, generated_at, data_cutoff,
        target_start, target_end, predicted_value, lower_bound,
        upper_bound, status, error_message, updated_at
    )
    VALUES (%s, %s, NOW(), %s, %s, %s, %s, %s, %s, %s, %s, NOW())
    ON CONFLICT (monitoring_post_id, substance_code, target_start)
    DO UPDATE SET
        generated_at = NOW(), data_cutoff = EXCLUDED.data_cutoff,
        target_end = EXCLUDED.target_end, predicted_value = EXCLUDED.predicted_value,
        lower_bound = EXCLUDED.lower_bound, upper_bound = EXCLUDED.upper_bound,
        status = EXCLUDED.status, error_message = EXCLUDED.error_message,
        updated_at = NOW()
"""


def store_prediction_rows(
    connection: Connection, rows, status: str,
    results: list[ForecastResult] | None = None,
    error_message: str | None = None,
) -> None:
    if results is not None and len(results) != len(rows):
        raise ValueError("Prediction result count does not match row count")
    parameters = []
    for index, row in enumerate(rows.itertuples(index=False)):
        result = results[index] if results is not None else None
        parameters.append((
            int(row.monitoring_post_id), str(row.substance_code),
            row.data_cutoff.to_pydatetime(), row.target_start.to_pydatetime(),
            row.target_end.to_pydatetime(),
            result.value if result else None,
            result.lower_bound if result else None,
            result.upper_bound if result else None,
            status, error_message[:4000] if error_message else None,
        ))
    if parameters:
        with connection.cursor() as cursor:
            cursor.executemany(_UPSERT_PREDICTION, parameters)


def remove_predictions_without_latest_measurement(
    connection: Connection,
    substance_code: str,
    data_cutoff: datetime,
    *,
    horizons: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> int:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            DELETE FROM public.gas_predictions AS prediction
            WHERE prediction.substance_code = %s
              AND prediction.target_start >= %s
              AND prediction.target_start <= %s
              AND NOT EXISTS (
                  SELECT 1
                  FROM public.gas_hourly_features AS feature
                  WHERE feature.monitoring_post_id = prediction.monitoring_post_id
                    AND feature.substance_code = prediction.substance_code
                    AND feature.bucket_start = %s
                    AND feature.filtered_hourly_mean IS NOT NULL
              )
            """,
            (
                substance_code,
                data_cutoff + timedelta(hours=min(horizons) - 1),
                data_cutoff + timedelta(hours=max(horizons) - 1),
                data_cutoff - timedelta(hours=1),
            ),
        )
        return cursor.rowcount


def remove_predictions_for_excluded_posts(
    connection: Connection, data_cutoff: datetime, post_ids: frozenset[int],
    *, horizons: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> int:
    if not post_ids:
        return 0
    with connection.cursor() as cursor:
        cursor.execute(
            """
            DELETE FROM public.gas_predictions
            WHERE monitoring_post_id = ANY(%s)
              AND target_start >= %s AND target_start <= %s
            """,
            (
                sorted(post_ids),
                data_cutoff + timedelta(hours=min(horizons) - 1),
                data_cutoff + timedelta(hours=max(horizons) - 1),
            ),
        )
        return cursor.rowcount
