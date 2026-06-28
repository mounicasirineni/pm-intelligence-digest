from __future__ import annotations

import hashlib
import logging
import sqlite3
from datetime import date
from typing import Any, Dict, List

from ..config import load_settings

logger = logging.getLogger(__name__)


def _get_connection() -> sqlite3.Connection:
    settings = load_settings()
    return sqlite3.connect(str(settings.database_path))


def _ensure_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prompt_versions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            active_from  TEXT NOT NULL,
            active_to    TEXT,
            call_name    TEXT NOT NULL,
            prompt_hash  TEXT NOT NULL,
            change_reason TEXT,
            proposed_by  TEXT DEFAULT 'manual'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prompt_patches (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at      TEXT NOT NULL,
            call_name       TEXT NOT NULL,
            trigger_type    TEXT NOT NULL,
            trigger_detail  TEXT,
            proposed_patch  TEXT NOT NULL,
            status          TEXT DEFAULT 'proposed',
            reviewed_at     TEXT,
            reviewer_notes  TEXT,
            applied_from    TEXT
        )
        """
    )
    conn.commit()


def prompt_hash(prompt_text: str) -> str:
    """Return first 12 chars of SHA256 of prompt text — enough to detect changes."""
    return hashlib.sha256(prompt_text.encode()).hexdigest()[:12]


def register_prompt(
    call_name: str,
    prompt_text: str,
    change_reason: str = "",
    proposed_by: str = "manual",
) -> bool:
    """
    Register a prompt version if it differs from the currently active version.
    Returns True if a new version was registered, False if unchanged.
    """
    h = prompt_hash(prompt_text)
    today = date.today().isoformat()

    conn = _get_connection()
    try:
        _ensure_tables(conn)

        cur = conn.execute(
            """
            SELECT id, prompt_hash FROM prompt_versions
            WHERE call_name = ? AND active_to IS NULL
            ORDER BY active_from DESC LIMIT 1
            """,
            (call_name,),
        )
        row = cur.fetchone()

        if row and row[1] == h:
            return False

        if row:
            conn.execute(
                "UPDATE prompt_versions SET active_to = ? WHERE id = ?",
                (today, row[0]),
            )

        conn.execute(
            """
            INSERT INTO prompt_versions
                (active_from, call_name, prompt_hash, change_reason, proposed_by)
            VALUES (?, ?, ?, ?, ?)
            """,
            (today, call_name, h, change_reason, proposed_by),
        )
        conn.commit()
        logger.info(
            "PROMPT VERSION: %s registered new version (hash=%s, reason=%s)",
            call_name, h, change_reason or "unspecified",
        )
        return True

    finally:
        conn.close()


def propose_patch(
    call_name: str,
    trigger_type: str,
    trigger_detail: str,
    proposed_patch: str,
) -> int:
    """
    Store a proposed prompt patch for human review.
    Returns the patch id.
    """
    conn = _get_connection()
    try:
        _ensure_tables(conn)
        cur = conn.execute(
            """
            INSERT INTO prompt_patches
                (created_at, call_name, trigger_type, trigger_detail, proposed_patch)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                date.today().isoformat(),
                call_name,
                trigger_type,
                trigger_detail,
                proposed_patch,
            ),
        )
        conn.commit()
        patch_id = cur.lastrowid
        logger.info(
            "PROMPT PATCH PROPOSED: id=%d call=%s trigger=%s",
            patch_id, call_name, trigger_type,
        )
        return patch_id or 0
    finally:
        conn.close()


def get_pending_patches() -> List[Dict[str, Any]]:
    """Return all proposed patches awaiting human review."""
    conn = _get_connection()
    try:
        _ensure_tables(conn)
        cur = conn.execute(
            """
            SELECT id, created_at, call_name, trigger_type,
                   trigger_detail, proposed_patch, status
            FROM prompt_patches
            WHERE status = 'proposed'
            ORDER BY created_at DESC
            """
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    return [
        {
            "id": r[0], "created_at": r[1], "call_name": r[2],
            "trigger_type": r[3], "trigger_detail": r[4],
            "proposed_patch": r[5], "status": r[6],
        }
        for r in rows
    ]


def accept_patch(patch_id: int, reviewer_notes: str = "") -> None:
    """Mark a patch as accepted and record when it was applied."""
    conn = _get_connection()
    try:
        _ensure_tables(conn)
        conn.execute(
            """
            UPDATE prompt_patches
            SET status = 'accepted', reviewed_at = ?, reviewer_notes = ?, applied_from = ?
            WHERE id = ?
            """,
            (date.today().isoformat(), reviewer_notes, date.today().isoformat(), patch_id),
        )
        conn.commit()
        logger.info("PROMPT PATCH ACCEPTED: id=%d", patch_id)
    finally:
        conn.close()


def reject_patch(patch_id: int, reviewer_notes: str = "") -> None:
    """Mark a patch as rejected."""
    conn = _get_connection()
    try:
        _ensure_tables(conn)
        conn.execute(
            """
            UPDATE prompt_patches
            SET status = 'rejected', reviewed_at = ?, reviewer_notes = ?
            WHERE id = ?
            """,
            (date.today().isoformat(), reviewer_notes, patch_id),
        )
        conn.commit()
        logger.info("PROMPT PATCH REJECTED: id=%d", patch_id)
    finally:
        conn.close()


def get_version_history(call_name: str) -> List[Dict[str, Any]]:
    """Return full prompt version history for a given call."""
    conn = _get_connection()
    try:
        _ensure_tables(conn)
        cur = conn.execute(
            """
            SELECT id, active_from, active_to, prompt_hash, change_reason, proposed_by
            FROM prompt_versions
            WHERE call_name = ?
            ORDER BY active_from DESC
            """,
            (call_name,),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    return [
        {
            "id": r[0], "active_from": r[1], "active_to": r[2],
            "prompt_hash": r[3], "change_reason": r[4], "proposed_by": r[5],
        }
        for r in rows
    ]
