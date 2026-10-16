import sqlite3
from pathlib import Path
from typing import Any, Sequence

from flask import current_app, g

POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id BIGSERIAL PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS analyses (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    source_type TEXT NOT NULL CHECK (source_type IN ('csv', 'manual')),
    filename TEXT,
    total_records INTEGER NOT NULL,
    normal_count INTEGER NOT NULL,
    attack_count INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    analysis_id BIGINT NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    record_index INTEGER NOT NULL,
    predicted_label TEXT NOT NULL,
    attack_category TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    source_type TEXT NOT NULL CHECK (source_type IN ('csv', 'manual')),
    filename TEXT,
    total_records INTEGER NOT NULL,
    normal_count INTEGER NOT NULL,
    attack_count INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    record_index INTEGER NOT NULL,
    predicted_label TEXT NOT NULL,
    attack_category TEXT NOT NULL,
    confidence REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


class DatabaseConnection:
    def __init__(self, connection: Any, postgres: bool) -> None:
        self.connection = connection
        self.postgres = postgres

    def execute(self, statement: str, parameters: Sequence[Any] = ()) -> Any:
        if self.postgres:
            statement = statement.replace("?", "%s")
        return self.connection.execute(statement, parameters)

    def executemany(self, statement: str, parameters: Sequence[Sequence[Any]]) -> Any:
        if self.postgres:
            statement = statement.replace("?", "%s")
            return self.connection.cursor().executemany(statement, parameters)
        return self.connection.executemany(statement, parameters)

    def executescript(self, script: str) -> None:
        if self.postgres:
            for statement in script.split(";"):
                if statement.strip():
                    self.execute(statement)
        else:
            self.connection.executescript(script)

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


def get_db() -> DatabaseConnection:
    if "db" not in g:
        database = Path(current_app.config["DATABASE"])
        database_url = current_app.config["DATABASE"]
        if str(database_url).startswith(("postgres://", "postgresql://")):
            import psycopg
            from psycopg.rows import dict_row

            connection = psycopg.connect(database_url, row_factory=dict_row)
            g.db = DatabaseConnection(connection, postgres=True)
        else:
            database.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(database)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            g.db = DatabaseConnection(connection, postgres=False)
    return g.db


def close_db(_error: BaseException | None = None) -> None:
    connection = g.pop("db", None)
    if connection is not None:
        connection.close()


def init_db() -> None:
    connection = get_db()
    schema = POSTGRES_SCHEMA if connection.postgres else SCHEMA
    connection.executescript(schema)
    connection.commit()


def is_unique_violation(error: Exception) -> bool:
    if isinstance(error, sqlite3.IntegrityError):
        return "UNIQUE constraint failed" in str(error)
    return error.__class__.__name__ == "UniqueViolation"


def save_analysis(
    user_id: int,
    source_type: str,
    filename: str | None,
    predictions: list[dict[str, object]],
) -> int:
    normal_count = sum(item["predicted_label"] == "Normal" for item in predictions)
    connection = get_db()
    cursor = connection.execute(
        """INSERT INTO analyses
           (user_id, source_type, filename, total_records, normal_count, attack_count)
           VALUES (?, ?, ?, ?, ?, ?)""" + (" RETURNING id" if connection.postgres else ""),
        (
            user_id,
            source_type,
            filename,
            len(predictions),
            normal_count,
            len(predictions) - normal_count,
        ),
    )
    analysis_id = int(
        cursor.fetchone()["id"] if connection.postgres else cursor.lastrowid
    )
    connection.executemany(
        """INSERT INTO predictions
           (analysis_id, record_index, predicted_label, attack_category, confidence)
           VALUES (?, ?, ?, ?, ?)""",
        [
            (
                analysis_id,
                index,
                str(item["predicted_label"]),
                str(item["attack_category"]),
                float(item["confidence"]),
            )
            for index, item in enumerate(predictions, start=1)
        ],
    )
    connection.commit()
    return analysis_id
