from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import load_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DigestRecord:
    synthesis: Dict[str, Any]
    items_by_theme: Dict[str, Any]
    generated_at: datetime
    fetch_metadata: Dict[str, Any]


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
    """Ensure the digests table exists with all required columns."""
    conn = _get_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS digests (
                date TEXT PRIMARY KEY,
                synthesis_json TEXT NOT NULL,
                items_by_theme_json TEXT NOT NULL,
                generated_at TEXT NOT NULL,
                fetch_metadata_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS synthesizer_inputs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date TEXT NOT NULL,
                created_at TEXT NOT NULL,
                grouped_summaries TEXT NOT NULL
            )
            """
        )
        # Migrate existing tables that lack fetch_metadata_json
        try:
            conn.execute("ALTER TABLE digests ADD COLUMN fetch_metadata_json TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
        conn.commit()
    finally:
        conn.close()


def get_digest_for_today() -> Optional[DigestRecord]:
    """Return today's cached digest if present, else None."""
    today = date.today().isoformat()
    conn = _get_connection()
    try:
        cur = conn.execute(
            """
            SELECT synthesis_json, items_by_theme_json, generated_at, fetch_metadata_json
            FROM digests WHERE date = ?
            """,
            (today,),
        )
        row = cur.fetchone()
    finally:
        conn.close()

    if row is None:
        return None

    synthesis_json, items_json, generated_at_str, fetch_metadata_json = row
    try:
        synthesis = json.loads(synthesis_json)
        items_by_theme = json.loads(items_json)
        generated_at = datetime.fromisoformat(generated_at_str)
        fetch_metadata = json.loads(fetch_metadata_json) if fetch_metadata_json else {}
    except Exception:
        return None

    return DigestRecord(
        synthesis=synthesis,
        items_by_theme=items_by_theme,
        generated_at=generated_at,
        fetch_metadata=fetch_metadata,
    )


def save_digest(
    synthesis: Dict[str, Any],
    items_by_theme: Dict[str, Any],
    generated_at: datetime,
    fetch_metadata: Dict[str, Any] | None = None,
) -> None:
    """Insert or replace today's digest."""
    digest_date = generated_at.date().isoformat()
    conn = _get_connection()
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO digests
                (date, synthesis_json, items_by_theme_json, generated_at, fetch_metadata_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                digest_date,
                json.dumps(synthesis or {}),
                json.dumps(items_by_theme or {}),
                generated_at.isoformat(),
                json.dumps(fetch_metadata or {}),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_digest_by_date(target_date: str) -> Optional[DigestRecord]:
    """Return digest for a specific date (YYYY-MM-DD), else None."""
    conn = _get_connection()
    try:
        cur = conn.execute(
            """
            SELECT synthesis_json, items_by_theme_json, generated_at, fetch_metadata_json
            FROM digests WHERE date = ?
            """,
            (target_date,),
        )
        row = cur.fetchone()
    finally:
        conn.close()

    if row is None:
        return None

    synthesis_json, items_json, generated_at_str, fetch_metadata_json = row
    try:
        synthesis = json.loads(synthesis_json)
        items_by_theme = json.loads(items_json)
        generated_at = datetime.fromisoformat(generated_at_str)
        fetch_metadata = json.loads(fetch_metadata_json) if fetch_metadata_json else {}
    except Exception:
        logger.warning("Failed to parse digest record for date %s", target_date)
        return None

    return DigestRecord(
        synthesis=synthesis,
        items_by_theme=items_by_theme,
        generated_at=generated_at,
        fetch_metadata=fetch_metadata,
    )


# ---------------------------------------------------------------------------
# Eval query functions — migrated from main.py (Fix 4b)
# ---------------------------------------------------------------------------

def get_eval_summary_for_date(date_str: str) -> Optional[Dict[str, Any]]:
    """Fetch a compact eval summary for a specific YYYY-MM-DD date, or None."""
    conn = _get_connection()
    try:
        cur = conn.execute(
            """
            SELECT
              pipeline_funnel_json,
              llm_judge_json,
              overall_score,
              flags_json
            FROM evals
            WHERE date = ?
            """,
            (date_str,),
        )
        row = cur.fetchone()
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()

    if row is None:
        return None

    pipeline_funnel_json, llm_judge_json, overall_score, flags_json = row
    try:
        llm_judge = json.loads(llm_judge_json) if llm_judge_json else {}
        flags = json.loads(flags_json) if flags_json else {}
    except Exception:
        return None

    return {
        "overall_score": float(overall_score or 0.0),
        "avg_coherence": float(llm_judge.get("ws_avg_coherence") or llm_judge.get("avg_coherence") or 0.0),
        "avg_insight_depth": float(llm_judge.get("ws_avg_insight_depth") or llm_judge.get("avg_insight_depth") or 0.0),
        "avg_citation_support": float(llm_judge.get("ws_avg_citation_support") or llm_judge.get("avg_citation_support") or 0.0),
        "has_flags": bool(flags.get("flagged_paragraphs")),
    }


def get_all_evals() -> List[Dict[str, Any]]:
    """Fetch all eval rows, newest first."""
    conn = _get_connection()
    try:
        cur = conn.execute(
            """
            SELECT
              date, pipeline_funnel_json, pm_relevance_json,
              llm_judge_json, pm_craft_json, interview_angle_json,
              overall_score, flags_json
            FROM evals
            ORDER BY date DESC
            """
        )
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()

    result = []
    for row in rows:
        (date_str, pf_json, pm_json, llm_json,
         pc_json, ia_json, overall_score, flags_json) = row

        try:
            d = datetime.strptime(date_str, "%Y-%m-%d")
            label = d.strftime("%B %d, %Y")
        except ValueError:
            logger.debug("Skipping eval row with non-standard date string: %s", date_str)
            continue

        pf    = json.loads(pf_json)    if pf_json    else {}
        pm    = json.loads(pm_json)    if pm_json    else {}
        llm   = json.loads(llm_json)   if llm_json   else {}
        pc    = json.loads(pc_json)    if pc_json    else {}
        ia    = json.loads(ia_json)    if ia_json    else {}
        flags = json.loads(flags_json) if flags_json else {}

        # Fallback to legacy key names for rows written before Fix 2b.
        result.append({
            "date":          date_str,
            "label":         label,
            "overall_score": float(overall_score or 0.0),
            "ws_coherence":        float(llm.get("ws_avg_coherence")  or llm.get("avg_coherence")  or 0.0),
            "ws_coherence_reason": str(llm.get("ws_coherence_reason") or ""),
            "ws_insight":          float(llm.get("ws_avg_insight_depth") or llm.get("avg_insight_depth") or 0.0),
            "ws_insight_reason":   str(llm.get("ws_insight_reason") or ""),
            "ws_grounding":        float(llm.get("ws_avg_citation_support") or llm.get("avg_citation_support") or 0.0),
            "ws_grounding_reason": str(llm.get("ws_grounding_reason") or ""),
            "ws_breadth":          float(llm.get("ws_topical_breadth") or 0.0),
            "ws_breadth_reason":   str(llm.get("ws_topical_breadth_reason") or ""),
            "cw_coherence":        float(llm.get("cw_avg_coherence")  or 0.0),
            "cw_coherence_reason": str(llm.get("cw_coherence_reason") or ""),
            "cw_insight":          float(llm.get("cw_avg_insight_depth")  or 0.0),
            "cw_insight_reason":   str(llm.get("cw_insight_reason") or ""),
            "cw_grounding":        float(llm.get("cw_avg_citation_support") or 0.0),
            "cw_grounding_reason": str(llm.get("cw_grounding_reason") or ""),
            "sr_coherence":        float(llm.get("sr_avg_coherence")  or 0.0),
            "sr_coherence_reason": str(llm.get("sr_coherence_reason") or ""),
            "sr_insight":          float(llm.get("sr_avg_insight_depth")  or 0.0),
            "sr_insight_reason":   str(llm.get("sr_insight_reason") or ""),
            "sr_grounding":        float(llm.get("sr_avg_citation_support") or 0.0),
            "sr_grounding_reason": str(llm.get("sr_grounding_reason") or ""),
            "pc_insight":         float(pc.get("insight_depth") or 0.0),
            "pc_insight_reason":  str(pc.get("insight_depth_reason") or ""),
            "ia_relevance":        float(ia.get("relevance") or 0.0),
            "ia_relevance_reason": str(ia.get("relevance_reason") or ""),
            "sources_configured": int(pf.get("sources_configured") or 0),
            "sources_active":     int(pf.get("sources_active") or 0),
            "sources_active_pct": float(pf.get("sources_active_pct") or 0.0),
            "empty_source_names": [str(n) for n in (pf.get("empty_source_names") or []) if n],
            "fetched":            int(pf.get("fetched") or 0),
            "confident":          int(pf.get("confident") or 0),
            "confident_pct":      float(pf.get("confident_pct") or 0.0),
            "relevant":           int(pf.get("relevant") or 0),
            "relevant_pct":       float(pf.get("relevant_pct") or 0.0),
            "utilized":           int(pf.get("utilized") or 0),
            "utilized_pct":       float(pf.get("utilized_pct") or 0.0),
            "theme_funnel":       pf.get("theme_funnel") or {},
            "pm_high":   float(pm.get("high_pct") or 0.0),
            "pm_med":    float(pm.get("medium_pct") or 0.0),
            "pm_low":    float(pm.get("low_pct") or 0.0),
            "weak_pct": float(flags.get("weak_pct") or 0.0),
        })

    return result


def get_pipeline_health(days: int = 14) -> List[Dict[str, Any]]:
    """Fetch pipeline funnel data for the last N days."""
    conn = _get_connection()
    try:
        cur = conn.execute(
            """
            SELECT e.date, e.pipeline_funnel_json
            FROM evals e
            ORDER BY e.date DESC
            LIMIT ?
            """,
            (days,),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    result = []
    for date_str, pf_json in rows:
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d")
            pf = json.loads(pf_json) if pf_json else {}
        except Exception:
            continue
        result.append({
            "date": date_str,
            "label": d.strftime("%b %d"),
            "sources_configured": int(pf.get("sources_configured") or 0),
            "sources_active":     int(pf.get("sources_active") or 0),
            "fetched":            int(pf.get("fetched") or 0),
            "confident":          int(pf.get("confident") or 0),
            "relevant":           int(pf.get("relevant") or 0),
            "utilized":           int(pf.get("utilized") or 0),
            "sources_active_pct": float(pf.get("sources_active_pct") or 0),
            "confident_pct":      float(pf.get("confident_pct") or 0),
            "relevant_pct":       float(pf.get("relevant_pct") or 0),
            "utilized_pct":       float(pf.get("utilized_pct") or 0),
            "empty_source_names": pf.get("empty_source_names") or [],
            "theme_funnel":       pf.get("theme_funnel") or {},
        })
    return result


def get_warning_history(days: int = 30) -> List[Dict[str, Any]]:
    """Fetch warning_counts rows for the last N days."""
    conn = _get_connection()
    try:
        cur = conn.execute(
            """
            SELECT DISTINCT date FROM warning_counts
            ORDER BY date DESC LIMIT ?
            """,
            (days,),
        )
        dates = [r[0] for r in cur.fetchall()]

        result = []
        for date_str in dates:
            cur2 = conn.execute(
                """
                SELECT warning_type, count, consecutive_days
                FROM warning_counts WHERE date = ?
                """,
                (date_str,),
            )
            warnings = {
                r[0]: {"count": r[1], "consecutive_days": r[2]}
                for r in cur2.fetchall()
            }
            try:
                d = datetime.strptime(date_str, "%Y-%m-%d")
                label = d.strftime("%b %d")
            except Exception:
                label = date_str
            result.append({"date": date_str, "label": label, "warnings": warnings})
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
    return result


def get_quality_scores(days: int = 30) -> List[Dict[str, Any]]:
    """Fetch all quality score dimensions for the last N days."""
    conn = _get_connection()
    try:
        cur = conn.execute(
            """
            SELECT date, llm_judge_json, pm_craft_json,
                   interview_angle_json, overall_score, flags_json
            FROM evals ORDER BY date DESC LIMIT ?
            """,
            (days,),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    result = []
    for date_str, llm_j, pc_j, ia_j, overall, flags_j in rows:
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d")
            label = d.strftime("%b %d")
        except Exception:
            continue
        llm   = json.loads(llm_j)   if llm_j   else {}
        pc    = json.loads(pc_j)    if pc_j    else {}
        ia    = json.loads(ia_j)    if ia_j    else {}
        flags = json.loads(flags_j) if flags_j else {}
        # Fallback to legacy key names for rows written before Fix 2b.
        result.append({
            "date":         date_str,
            "label":        label,
            "overall_score": float(overall or 0),
            "ws_coherence": float(llm.get("ws_avg_coherence") or 0),
            "ws_insight":   float(llm.get("ws_avg_insight_depth") or 0),
            "ws_grounding": float(llm.get("ws_avg_citation_support") or 0),
            "ws_breadth":   float(llm.get("ws_topical_breadth") or 0),
            "cw_coherence": float(llm.get("cw_avg_coherence") or 0),
            "cw_insight":   float(llm.get("cw_avg_insight_depth") or 0),
            "cw_grounding": float(llm.get("cw_avg_citation_support") or 0),
            "sr_coherence": float(llm.get("sr_avg_coherence") or 0),
            "sr_insight":   float(llm.get("sr_avg_insight_depth") or 0),
            "sr_grounding": float(llm.get("sr_avg_citation_support") or 0),
            "pc_insight":   float(pc.get("insight_depth") or 0),
            "ia_relevance": float(ia.get("relevance") or 0),
            "sections_scored":      flags.get("sections_scored") or [],
            "weak_pct":             float(flags.get("weak_pct") or 0),
            "ws_breadth_reason":    str(llm.get("ws_topical_breadth_reason") or ""),
            "ws_coherence_reason":  str(llm.get("ws_coherence_reason") or ""),
            "ws_insight_reason":    str(llm.get("ws_insight_reason") or ""),
            "ws_grounding_reason":  str(llm.get("ws_grounding_reason") or ""),
            "cw_coherence_reason":  str(llm.get("cw_coherence_reason") or ""),
            "sr_coherence_reason":  str(llm.get("sr_coherence_reason") or ""),
            "pc_insight_reason":    str(pc.get("insight_depth_reason") or ""),
            "ia_relevance_reason":  str(ia.get("relevance_reason") or ""),
        })
    return result


def get_digest_history() -> List[Dict[str, Any]]:
    """Fetch date and generated_at for all digests, newest first."""
    conn = _get_connection()
    try:
        cur = conn.execute(
            "SELECT date, generated_at FROM digests ORDER BY date DESC"
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    result = []
    for date_str, generated_at_str in rows:
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue
        try:
            gen_dt = datetime.fromisoformat(generated_at_str)
        except Exception:
            gen_dt = None
        result.append({
            "date":         date_str,
            "label":        d.strftime("%B %d, %Y"),
            "generated_at": gen_dt.strftime("%H:%M") if gen_dt else "",
        })
    return result