from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Sequence

import psycopg
from psycopg import Connection
from psycopg.rows import dict_row

from app.config import get_settings


@contextmanager
def db_connection(*, readonly: bool = False) -> Iterator[Connection]:
    settings = get_settings()
    options = "-c statement_timeout=120000"
    if readonly:
        options += " -c default_transaction_read_only=on"
    database_url = settings.DATABASE_URL.replace("postgresql+psycopg://", "postgresql://", 1)
    connection = psycopg.connect(database_url, row_factory=dict_row, options=options)
    try:
        yield connection
        if not readonly:
            connection.commit()
    except Exception:
        if not connection.closed:
            connection.rollback()
        raise
    finally:
        connection.close()


def fetch_all(connection: Connection, query: str, params: Sequence[Any] | None = None) -> list[dict[str, Any]]:
    with connection.cursor() as cursor:
        cursor.execute(query, params)
        return list(cursor.fetchall())


def fetch_one(connection: Connection, query: str, params: Sequence[Any] | None = None) -> dict[str, Any] | None:
    with connection.cursor() as cursor:
        cursor.execute(query, params)
        return cursor.fetchone()
