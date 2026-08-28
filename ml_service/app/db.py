"""PostgreSQL persistence for prediction audit logs."""

import psycopg
from psycopg.rows import dict_row

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS prediction_logs (
    id BIGSERIAL PRIMARY KEY,
    listing_id VARCHAR(100) NOT NULL,
    risk_score DOUBLE PRECISION NOT NULL CHECK (risk_score BETWEEN 0 AND 1),
    risk_level VARCHAR(20) NOT NULL,
    recommended_action VARCHAR(30) NOT NULL,
    model_version VARCHAR(100) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""


class DatabaseUnavailableError(RuntimeError):
    """Raised when prediction logs cannot be stored or read."""


def _require_database_url(database_url: str | None) -> str:
    if not database_url:
        raise DatabaseUnavailableError("DATABASE_URL is not configured")
    return database_url


def initialize_database(database_url: str | None) -> None:
    url = _require_database_url(database_url)
    try:
        with psycopg.connect(url, connect_timeout=3) as connection:
            with connection.cursor() as cursor:
                cursor.execute(CREATE_TABLE_SQL)
    except psycopg.Error as exc:
        raise DatabaseUnavailableError("Could not initialize PostgreSQL") from exc


def check_database(database_url: str | None) -> bool:
    if not database_url:
        return False
    try:
        with psycopg.connect(database_url, connect_timeout=2) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
    except psycopg.Error:
        return False
    return True


def insert_prediction_log(database_url: str | None, record: dict) -> None:
    url = _require_database_url(database_url)
    try:
        with psycopg.connect(url, connect_timeout=3) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO prediction_logs (
                        listing_id,
                        risk_score,
                        risk_level,
                        recommended_action,
                        model_version
                    )
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        record["listing_id"],
                        record["risk_score"],
                        record["risk_level"],
                        record["recommended_action"],
                        record["model_version"],
                    ),
                )
    except psycopg.Error as exc:
        raise DatabaseUnavailableError("Could not write prediction log") from exc


def read_recent_prediction_logs(
    database_url: str | None,
    limit: int = 10,
) -> list[dict]:
    url = _require_database_url(database_url)
    try:
        with psycopg.connect(
            url,
            connect_timeout=3,
            row_factory=dict_row,
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        id,
                        listing_id,
                        risk_score,
                        risk_level,
                        recommended_action,
                        model_version,
                        created_at
                    FROM prediction_logs
                    ORDER BY id DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                return [dict(row) for row in cursor.fetchall()]
    except psycopg.Error as exc:
        raise DatabaseUnavailableError("Could not read prediction logs") from exc
