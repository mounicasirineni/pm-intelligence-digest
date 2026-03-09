from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..config import load_settings


@dataclass(frozen=True)
class DigestRecord:
    synthesis: Dict[str, Any]
    items_by_theme: Dict[str, Any]
    generated_at: datetime


def _get_db_path() -> Path:
    settings = load_settings()
    db_path = settings.database_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def _get_connection() -> sqlite3.Connection:
    db_path = _get_db_path()
    conn = sqlite3.connect(db_path)
    return conn


def init_db() -> None:
    """Ensure the digests table exists."""
    conn = _get_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS digests (
                date TEXT PRIMARY KEY,
                synthesis_json TEXT NOT NULL,
                items_by_theme_json TEXT NOT NULL,
                generated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def get_digest_for_today() -> Optional[DigestRecord]:
    """Return today's cached digest if present, else None."""
    today = date.today().isoformat()
    conn = _get_connection()
    try:
        cur = conn.execute(
            "SELECT synthesis_json, items_by_theme_json, generated_at FROM digests WHERE date = ?",
            (today,),
        )
        row = cur.fetchone()
    finally:
        conn.close()

    if row is None:
        return None

    synthesis_json, items_json, generated_at_str = row
    try:
        synthesis = json.loads(synthesis_json)
        items_by_theme = json.loads(items_json)
        generated_at = datetime.fromisoformat(generated_at_str)
    except Exception:
        return None

    return DigestRecord(
        synthesis=synthesis,
        items_by_theme=items_by_theme,
        generated_at=generated_at,
    )


def save_digest(
    synthesis: Dict[str, Any],
    items_by_theme: Dict[str, Any],
    generated_at: datetime,
) -> None:
    """Insert or replace today's digest."""
    digest_date = generated_at.date().isoformat()
    conn = _get_connection()
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO digests (date, synthesis_json, items_by_theme_json, generated_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                digest_date,
                json.dumps(synthesis or {}),
                json.dumps(items_by_theme or {}),
                generated_at.isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()

