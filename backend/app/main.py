from __future__ import annotations

from datetime import datetime
import json
import sqlite3
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import atexit
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, redirect, render_template, url_for, abort
import pytz

from .config import load_settings
from .services.cache import get_digest_for_today, init_db, save_digest
from .services.rss import fetch_items_grouped_by_theme
from .services.summarizer import summarize_item
from .services.synthesizer import synthesize_trends

BASE_DIR = Path(__file__).parent
logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

# Ensure the digests table exists before serving requests.
init_db()

_scheduler: BackgroundScheduler | None = None


def _bold_md(value: str) -> str:
    """Convert **bold** markdown to <strong>bold</strong> for inline emphasis."""
    if not isinstance(value, str):
        return value
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", value)


app.jinja_env.filters["bold_md"] = _bold_md


_CACHE = {
    "synthesis": None,
    "items_by_theme": None,
    "generated_at": None,
    "fetch_metadata": None,
}


def _run_pipeline():
    """
    Fetch -> summarize -> synthesize.
    Returns (synthesis, items_by_theme, generated_at, fetch_metadata).
    """
    grouped_raw, fetch_metadata = fetch_items_grouped_by_theme()
 
    items_by_theme = {}
 
    for theme, items in grouped_raw.items():
        summarized_items = []
        for item in items:
            try:
                summary = summarize_item(item)
            except Exception as exc:
                print(f"Summarizer failed for item '{item.get('title')}' in theme '{theme}': {exc}")
                continue
 
            summarized_items.append(
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "source_name": item.get("source_name"),
                    "theme": theme,
                    "insights": summary.get("insights", []),
                    "pm_interview_relevance": summary.get("pm_interview_relevance"),
                    "pm_relevance_score": summary.get("pm_relevance_score", "medium"),
                    "confidence": summary.get("confidence", "medium"),
                }
            )
 
        if summarized_items:
            items_by_theme[theme] = summarized_items
 
    synthesis = synthesize_trends(items_by_theme)
    generated_at = datetime.now()
 
    return synthesis, items_by_theme, generated_at, fetch_metadata


def _start_scheduler_if_needed() -> None:
    """Start the daily digest scheduler, avoiding duplicate schedulers in debug mode."""
    global _scheduler
    print("_start_scheduler_if_needed called", flush=True)

    settings = load_settings()
    if settings.app_env.lower() == "testing":
        return

    # In Flask debug, the reloader spawns two processes. Only start the scheduler in the main one.
    if os.environ.get("WERKZEUG_RUN_MAIN") not in {None, "true"}:
        return

    if _scheduler is not None:
        return

    tz = pytz.timezone(settings.digest_timezone)
    _scheduler = BackgroundScheduler(timezone=tz)

    _scheduler.add_job(
        lambda: _get_or_run_pipeline(force_refresh=True),
        "cron",
        hour=settings.digest_schedule_hour,
        minute=settings.digest_schedule_minute,
        id="daily_digest_refresh",
        replace_existing=True,
    )

    _scheduler.start()

    print(
        f"Scheduler started: digest will refresh daily at "
        f"{settings.digest_schedule_hour:02d}:{settings.digest_schedule_minute:02d} "
        f"({settings.digest_timezone})",
        flush=True,
    )

    def _shutdown_scheduler() -> None:
        global _scheduler
        if _scheduler is not None:
            _scheduler.shutdown(wait=False)
            _scheduler = None

    atexit.register(_shutdown_scheduler)


def _get_or_run_pipeline(force_refresh: bool = False):
    if not force_refresh:
        if _CACHE["synthesis"] is not None:
            return (
                _CACHE["synthesis"],
                _CACHE["items_by_theme"],
                _CACHE["generated_at"],
                _CACHE.get("fetch_metadata") or {},
            )
 
        record = get_digest_for_today()
        if record is not None:
            _CACHE["synthesis"] = record.synthesis
            _CACHE["items_by_theme"] = record.items_by_theme
            _CACHE["generated_at"] = record.generated_at
            _CACHE["fetch_metadata"] = record.fetch_metadata
            return record.synthesis, record.items_by_theme, record.generated_at, record.fetch_metadata
 
    synthesis, items_by_theme, generated_at, fetch_metadata = _run_pipeline()
    _CACHE["synthesis"] = synthesis
    _CACHE["items_by_theme"] = items_by_theme
    _CACHE["generated_at"] = generated_at
    _CACHE["fetch_metadata"] = fetch_metadata
 
    save_digest(synthesis, items_by_theme, generated_at, fetch_metadata=fetch_metadata)
 
    try:
        from .services import evaluator
        date_str = generated_at.date().isoformat() if generated_at else None
        print(f"[evals] Starting eval for {date_str}", flush=True)
        evaluator.run(date_str, synthesis, items_by_theme, fetch_metadata=fetch_metadata)
        print(f"[evals] Completed eval for {date_str}", flush=True)
    except Exception as e:
        print(f"[evals] FAILED: {e}", flush=True)
 
    return synthesis, items_by_theme, generated_at, fetch_metadata


def _get_digest_for_date(date_str: str):
    """Fetch a digest for a specific YYYY-MM-DD date from SQLite."""
    settings = load_settings()
    conn = sqlite3.connect(str(settings.database_path))
    try:
        cur = conn.execute(
            "SELECT synthesis_json, items_by_theme_json, generated_at, fetch_metadata_json "
            "FROM digests WHERE date = ?",
            (date_str,),
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

    return synthesis, items_by_theme, generated_at, fetch_metadata


def _get_eval_summary_for_date(date_str: str):
    """Fetch a compact eval summary for a specific YYYY-MM-DD date, or None."""
    settings = load_settings()
    conn = sqlite3.connect(str(settings.database_path))
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


def _get_all_evals():
    """Fetch all eval rows for the /evals page, newest first."""
    settings = load_settings()
    conn = sqlite3.connect(str(settings.database_path))
    try:
        cur = conn.execute(
            """
            SELECT
              date,
              pipeline_funnel_json,
              pm_relevance_json,
              llm_judge_json,
              pm_craft_json,
              interview_angle_json,
              overall_score,
              flags_json
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
        (date_str, pf_json, pm_json, llm_json, pc_json, ia_json, overall_score, flags_json) = row
 
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d")
            label = d.strftime("%B %d, %Y")
        except ValueError:
            label = date_str
 
        pf    = json.loads(pf_json)    if pf_json    else {}
        pm    = json.loads(pm_json)    if pm_json    else {}
        llm   = json.loads(llm_json)   if llm_json   else {}
        pc    = json.loads(pc_json)    if pc_json    else {}
        ia    = json.loads(ia_json)    if ia_json    else {}
        flags = json.loads(flags_json) if flags_json else {}
 
        result.append({
            "date":          date_str,
            "label":         label,
            "overall_score": float(overall_score or 0.0),
            # Quality — WS
            "ws_coherence":  float(llm.get("ws_avg_coherence")  or llm.get("avg_coherence")  or 0.0),
            "ws_insight":    float(llm.get("ws_avg_insight_depth") or llm.get("avg_insight_depth") or 0.0),
            "ws_grounding":  float(llm.get("ws_avg_citation_support") or llm.get("avg_citation_support") or 0.0),
            "ws_breadth":    float(llm.get("ws_topical_breadth") or 0.0),
            # Quality — CW
            "cw_coherence":  float(llm.get("cw_avg_coherence")  or 0.0),
            "cw_insight":    float(llm.get("cw_avg_insight_depth")  or 0.0),
            "cw_grounding":  float(llm.get("cw_avg_citation_support") or 0.0),
            # Quality — SR
            "sr_coherence":  float(llm.get("sr_avg_coherence")  or 0.0),
            "sr_insight":    float(llm.get("sr_avg_insight_depth")  or 0.0),
            "sr_grounding":  float(llm.get("sr_avg_citation_support") or 0.0),
            # Quality — PM Craft
            "pc_insight":    float(pc.get("insight_depth") or 0.0),
            # Quality — IA
            "ia_relevance":  float(ia.get("relevance") or 0.0),
            # Guardrails — pipeline funnel (5 stages)
            "sources_configured": int(pf.get("sources_configured") or 0),
            "sources_active":     int(pf.get("sources_active") or 0),
            "sources_active_pct": float(pf.get("sources_active_pct") or 0.0),
            "fetched":            int(pf.get("fetched") or 0),
            "confident":          int(pf.get("confident") or 0),
            "confident_pct":      float(pf.get("confident_pct") or 0.0),
            "relevant":           int(pf.get("relevant") or 0),
            "relevant_pct":       float(pf.get("relevant_pct") or 0.0),
            "utilized":           int(pf.get("utilized") or 0),
            "utilized_pct":       float(pf.get("utilized_pct") or 0.0),
            # Guardrails — PM relevance distribution
            "pm_high":   float(pm.get("high_pct") or 0.0),
            "pm_med":    float(pm.get("medium_pct") or 0.0),
            "pm_low":    float(pm.get("low_pct") or 0.0),
            "weak_pct": float(flags.get("weak_pct") or 0.0),
        })
 
    return result


@app.route("/")
def index():
    synthesis, items_by_theme, generated_at, fetch_metadata = _get_or_run_pipeline(force_refresh=False)
    eval_summary = None
    if generated_at:
        eval_summary = _get_eval_summary_for_date(generated_at.date().isoformat())
    source_index_lookup = (synthesis or {}).get("source_index_lookup") or {}
    citation_index_map = {
        (v["source_name"], v["title"]): k
        for k, v in source_index_lookup.items()
    }
    return render_template(
        "index.html",
        synthesis=synthesis or {},
        items_by_theme=items_by_theme or {},
        generated_at=generated_at,
        eval_summary=eval_summary,
        citation_index_map=citation_index_map,
    )


@app.route("/refresh")
def refresh():
    synthesis, items_by_theme, generated_at, fetch_metadata = _get_or_run_pipeline(force_refresh=True)
    return redirect(url_for("index"))


@app.route("/history")
def history():
    settings = load_settings()
    conn = sqlite3.connect(str(settings.database_path))
    try:
        cur = conn.execute(
            "SELECT date, generated_at FROM digests ORDER BY date DESC"
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    history_rows: List[Dict[str, Any]] = []
    for date_str, generated_at_str in rows:
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue
        try:
            gen_dt = datetime.fromisoformat(generated_at_str)
        except Exception:
            gen_dt = None

        history_rows.append(
            {
                "date": date_str,
                "label": d.strftime("%B %d, %Y"),
                "generated_at": gen_dt.strftime("%H:%M") if gen_dt else "",
            }
        )

    return render_template(
        "history.html",
        history_rows=history_rows,
    )


@app.route("/evals")
def evals_page():
    eval_rows = _get_all_evals()
    return render_template("evals.html", eval_rows=eval_rows)


@app.route("/<date_str>")
def digest_by_date(date_str: str):
    # Expect YYYY-MM-DD
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        abort(404)

    result = _get_digest_for_date(date_str)
    if result is None:
        abort(404)

    synthesis, items_by_theme, generated_at, fetch_metadata = result
    eval_summary = _get_eval_summary_for_date(date_str)
    source_index_lookup = (synthesis or {}).get("source_index_lookup") or {}
    citation_index_map = {
        (v["source_name"], v["title"]): k
        for k, v in source_index_lookup.items()
    }
    return render_template(
        "index.html",
        synthesis=synthesis or {},
        items_by_theme=items_by_theme or {},
        generated_at=generated_at,
        eval_summary=eval_summary,
        citation_index_map=citation_index_map,
    )


def create_app() -> Flask:
    """Factory for external runners if needed."""
    _start_scheduler_if_needed()
    return app


# When running this module directly via Flask/WSGI, ensure scheduler is considered.
_start_scheduler_if_needed()

