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


_CACHE: Dict[str, Any] = {
    "synthesis": None,
    "items_by_theme": None,
    "generated_at": None,
}


def _run_pipeline(max_items_per_theme: int = 3) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]], datetime]:
    """
    Fetch → summarize → synthesize for the current moment.

    Only the first `max_items_per_theme` items per theme are summarized
    to keep API usage manageable.
    """
    grouped_raw = fetch_items_grouped_by_theme()

    items_by_theme: Dict[str, List[Dict[str, Any]]] = {}

    for theme, items in grouped_raw.items():
        summarized_items: List[Dict[str, Any]] = []
        for item in items[:max_items_per_theme]:
            try:
                summary = summarize_item(item)
            except Exception as exc:  # pragma: no cover - defensive
                # In this early stage, we simply skip failing items and continue.
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
                    "confidence": summary.get("confidence", "medium"),
                }
            )

        if summarized_items:
            items_by_theme[theme] = summarized_items

    synthesis = synthesize_trends(items_by_theme)
    generated_at = datetime.now()

    return synthesis, items_by_theme, generated_at


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
        # First try the in-memory cache.
        if _CACHE["synthesis"] is not None:
            return _CACHE["synthesis"], _CACHE["items_by_theme"], _CACHE["generated_at"]

        # Then try today's persisted digest in SQLite.
        record = get_digest_for_today()
        if record is not None:
            _CACHE["synthesis"] = record.synthesis
            _CACHE["items_by_theme"] = record.items_by_theme
            _CACHE["generated_at"] = record.generated_at
            return record.synthesis, record.items_by_theme, record.generated_at

    synthesis, items_by_theme, generated_at = _run_pipeline()
    _CACHE["synthesis"] = synthesis
    _CACHE["items_by_theme"] = items_by_theme
    _CACHE["generated_at"] = generated_at

    # Persist today's digest so subsequent processes can reuse it.
    save_digest(synthesis, items_by_theme, generated_at)

    return synthesis, items_by_theme, generated_at


def _get_digest_for_date(date_str: str):
    """Fetch a digest for a specific YYYY-MM-DD date from SQLite."""
    settings = load_settings()
    conn = sqlite3.connect(str(settings.database_path))
    try:
        cur = conn.execute(
            "SELECT synthesis_json, items_by_theme_json, generated_at "
            "FROM digests WHERE date = ?",
            (date_str,),
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

    return synthesis, items_by_theme, generated_at


@app.route("/")
def index():
    synthesis, items_by_theme, generated_at = _get_or_run_pipeline(force_refresh=False)
    return render_template(
        "index.html",
        synthesis=synthesis or {},
        items_by_theme=items_by_theme or {},
        generated_at=generated_at,
    )


@app.route("/refresh")
def refresh():
    _get_or_run_pipeline(force_refresh=True)
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

    synthesis, items_by_theme, generated_at = result
    return render_template(
        "index.html",
        synthesis=synthesis or {},
        items_by_theme=items_by_theme or {},
        generated_at=generated_at,
    )


def create_app() -> Flask:
    """Factory for external runners if needed."""
    _start_scheduler_if_needed()
    return app


# When running this module directly via Flask/WSGI, ensure scheduler is considered.
_start_scheduler_if_needed()

