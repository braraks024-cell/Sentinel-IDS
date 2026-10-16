import sqlite3
from pathlib import Path

from flask import current_app, g


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


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        database = Path(current_app.config["DATABASE"])
        database.parent.mkdir(parents=True, exist_ok=True)
        g.db = sqlite3.connect(database)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


def close_db(_error: BaseException | None = None) -> None:
    connection = g.pop("db", None)
    if connection is not None:
        connection.close()


def init_db() -> None:
    connection = get_db()
    connection.executescript(SCHEMA)
    connection.commit()


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
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            user_id,
            source_type,
            filename,
            len(predictions),
            normal_count,
            len(predictions) - normal_count,
        ),
    )
    analysis_id = int(cursor.lastrowid)
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
