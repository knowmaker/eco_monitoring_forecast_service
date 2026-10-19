from __future__ import annotations

from typing import Any

from psycopg import Connection
from psycopg.types.json import Jsonb

from app.db import fetch_one


def replace_current_model(
    connection: Connection,
    *,
    artifact: bytes,
    feature_columns: list[str],
    categorical_features: list[str],
    train_start,
    train_end,
    rows_count: int,
    metrics: dict[str, Any],
    residual_intervals: dict[str, dict[str, float]],
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO public.current_model (
                singleton, artifact, feature_columns, categorical_features,
                train_start, train_end, rows_count, metrics,
                residual_intervals, trained_at
            )
            VALUES (TRUE, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (singleton) DO UPDATE SET
                artifact = EXCLUDED.artifact,
                feature_columns = EXCLUDED.feature_columns,
                categorical_features = EXCLUDED.categorical_features,
                train_start = EXCLUDED.train_start,
                train_end = EXCLUDED.train_end,
                rows_count = EXCLUDED.rows_count,
                metrics = EXCLUDED.metrics,
                residual_intervals = EXCLUDED.residual_intervals,
                trained_at = NOW()
            """,
            (
                artifact, Jsonb(feature_columns), Jsonb(categorical_features),
                train_start, train_end, rows_count, Jsonb(metrics), Jsonb(residual_intervals),
            ),
        )


def get_current_model(connection: Connection) -> dict[str, Any] | None:
    return fetch_one(connection, "SELECT * FROM public.current_model WHERE singleton = TRUE")
