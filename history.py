"""
Research session history backed by SQLite.
Stores queries, reports, and metadata for revisiting past research.
"""

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = os.environ.get("HISTORY_DB", str(Path(__file__).parent / "history.db"))
_local = threading.local()


def _conn():
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(DB_PATH)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                report TEXT,
                model TEXT,
                sources TEXT,
                created_at TEXT NOT NULL
            )
        """)
        _local.conn.commit()
    return _local.conn


def save(query: str, report: str, model: str = "", sources: dict = None) -> int:
    db = _conn()
    now = datetime.now(timezone.utc).isoformat()
    cur = db.execute(
        "INSERT INTO sessions (query, report, model, sources, created_at) VALUES (?, ?, ?, ?, ?)",
        (query, report, model, json.dumps(sources or {}), now),
    )
    db.commit()
    return cur.lastrowid


def list_sessions(limit: int = 50, offset: int = 0) -> list[dict]:
    db = _conn()
    rows = db.execute(
        "SELECT id, query, model, created_at FROM sessions ORDER BY id DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    return [dict(r) for r in rows]


def get_session(session_id: int) -> dict | None:
    db = _conn()
    row = db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return dict(row) if row else None


def delete_session(session_id: int) -> bool:
    db = _conn()
    cur = db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    db.commit()
    return cur.rowcount > 0
