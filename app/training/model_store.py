from __future__ import annotations

from pathlib import Path
from typing import Any

from psycopg import Connection
from psycopg.types.json import Jsonb

from app.config import SERVICE_ROOT
from app.db import fetch_one


def replace_current_model(
    connection: Connection,
    *,
    artifact_path: Path,
    feature_columns: list[str],
    categorical_features: list[str],
    train_start,
    train_end,
    rows_count: int,
    metrics: dict[str, Any],
    residual_intervals: dict[str, dict[str, float]],
) -> None:
    try:
        stored_path = artifact_path.resolve().relative_to(SERVICE_ROOT.resolve()).as_posix()
    except ValueError:
        stored_path = artifact_path.resolve().as_posix()
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO public.current_model (
                singleton, artifact_path, feature_columns, categorical_features,
                train_start, train_end, rows_count, metrics,
                residual_intervals, trained_at
            )
            VALUES (TRUE, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (singleton) DO UPDATE SET
                artifact_path = EXCLUDED.artifact_path,
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
                stored_path, Jsonb(feature_columns), Jsonb(categorical_features),
                train_start, train_end, rows_count, Jsonb(metrics), Jsonb(residual_intervals),
            ),
        )


def get_current_model(connection: Connection) -> dict[str, Any] | None:
    return fetch_one(connection, "SELECT * FROM public.current_model WHERE singleton = TRUE")


def resolve_artifact_path(stored_path: str) -> Path:
    path = Path(stored_path.replace("\\", "/"))
    return path if path.is_absolute() else SERVICE_ROOT / path
