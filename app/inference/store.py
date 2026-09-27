from __future__ import annotations

from datetime import datetime

from psycopg import Connection

from app.models.base import ForecastResult


def set_prediction_status(
    connection: Connection,
    *,
    monitoring_post_id: int,
    substance_code: str,
    data_cutoff: datetime,
    target_start: datetime,
    target_end: datetime,
    status: str,
    result: ForecastResult | None = None,
    error_message: str | None = None,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
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
            """,
            (
                monitoring_post_id, substance_code, data_cutoff, target_start, target_end,
                result.value if result else None,
                result.lower_bound if result else None,
                result.upper_bound if result else None,
                status, error_message[:4000] if error_message else None,
            ),
        )


def mark_rows(connection: Connection, rows, status: str, error_message: str | None = None) -> None:
    for row in rows.itertuples(index=False):
        set_prediction_status(
            connection,
            monitoring_post_id=int(row.monitoring_post_id),
            substance_code=str(row.substance_code),
            data_cutoff=row.data_cutoff.to_pydatetime(),
            target_start=row.target_start.to_pydatetime(),
            target_end=row.target_end.to_pydatetime(),
            status=status,
            error_message=error_message,
        )
