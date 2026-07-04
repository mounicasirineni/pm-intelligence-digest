from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple

import atexit
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, redirect, render_template, url_for, abort, request, send_from_directory
import pytz

from .config import load_settings
from .digest_utils import get_used_indices
from .services.cache import (
    get_digest_for_today,
    get_digest_by_date,
    get_eval_summary_for_date,
    get_all_evals,
    get_pipeline_health,
    get_warning_history,
    get_quality_scores,
    get_digest_history,
    init_db,
    save_digest,
    DigestRecord,
)
from .services.evaluator import get_consecutive_warning_types, get_score_trend
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


@app.after_request
def _add_noindex_headers(response):
    response.headers.setdefault("X-Robots-Tag", "noindex, nofollow, noarchive")
    return response


@app.route("/robots.txt")
def robots_txt():
    return send_from_directory(app.static_folder, "robots.txt")


init_db()

_scheduler: BackgroundScheduler | None = None

# FIX: thread-safe cache — APScheduler background job and Flask request threads
# both write _CACHE; without a lock, concurrent /refresh requests could both
# pass the None check and launch duplicate pipeline runs.
_CACHE_LOCK = threading.Lock()

_CACHE = {
    "synthesis": None,
    "items_by_theme": None,
    "generated_at": None,
    "fetch_metadata": None,
}


def _bold_md(value: str) -> str:
    """Convert **bold** markdown to <strong>bold</strong> for inline emphasis."""
    if not isinstance(value, str):
        return value
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", value)


app.jinja_env.filters["bold_md"] = _bold_md


def _build_utilized_keys(synthesis: dict) -> set:
    """
    Return the set of (source_name, title) pairs for articles actually
    cited in synthesis output via source_indices.

    FIX: previously duplicated the used_indices extraction logic from
    evaluator.pipeline_funnel.  Now uses the shared get_used_indices()
    utility from digest_utils so adding a new digest section only
    requires updating one place.
    """
    source_index_lookup = (synthesis or {}).get("source_index_lookup") or {}
    used_indices = get_used_indices(synthesis or {})

    return {
        (v["source_name"], v["title"])
        for k, v in source_index_lookup.items()
        if k in used_indices and isinstance(v, dict)
    }


def _run_pipeline():
    """
    Fetch -> summarize (parallel) -> synthesize.
    Returns (synthesis, items_by_theme, generated_at, fetch_metadata).
    """
    grouped_raw, fetch_metadata = fetch_items_grouped_by_theme()

    items_by_theme = {}

    all_items = [
        {**item, "_theme": theme}
        for theme, items in grouped_raw.items()
        for item in items
    ]

    total_items = len(all_items)
    logger.info("Summarizer: %d items to process (parallel, max_workers=10)", total_items)

    def _summarize_with_theme(item: dict) -> tuple[str, dict, dict]:
        theme = item["_theme"]
        try:
            summary = summarize_item(item)
            return theme, item, summary
        except Exception as exc:
            logger.warning(
                "Summarizer failed for item '%s' in theme '%s': %s",
                item.get("title"), theme, exc,
            )
            return theme, item, None

    theme_results: dict[str, list] = {theme: [] for theme in grouped_raw}
    theme_failed: dict[str, int] = {theme: 0 for theme in grouped_raw}
    theme_total: dict[str, int] = {
        theme: len(items) for theme, items in grouped_raw.items()
    }

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(_summarize_with_theme, item): item
            for item in all_items
        }
        for future in as_completed(futures):
            theme, item, summary = future.result()
            if summary is None:
                theme_failed[theme] = theme_failed.get(theme, 0) + 1
                continue
            theme_results[theme].append(
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "source_name": item.get("source_name"),
                    "company_id": item.get("company_id"),
                    "theme": theme,
                    "insights": summary.get("insights", []),
                    "pm_interview_relevance": summary.get("pm_interview_relevance"),
                    "pm_relevance_score": summary.get("pm_relevance_score", "medium"),
                    "confidence": summary.get("confidence", "medium"),
                }
            )

    for theme in grouped_raw:
        total = theme_total[theme]
        failed = theme_failed[theme]
        succeeded = len(theme_results[theme])

        if total > 0:
            logger.info(
                "Summarization for theme '%s': %d/%d items succeeded, %d failed",
                theme, succeeded, total, failed,
            )
            if failed > 0 and failed / total > 0.5:
                logger.warning(
                    "Summarization high failure rate in theme '%s': %d/%d items failed — "
                    "continuing with %d succeeded items. "
                    "Check API health if this persists across themes.",
                    theme, failed, total, succeeded,
                )

        if theme_results[theme]:
            items_by_theme[theme] = theme_results[theme]

    synthesis = synthesize_trends(items_by_theme)
    generated_at = datetime.now()

    return synthesis, items_by_theme, generated_at, fetch_metadata


def _start_scheduler_if_needed() -> None:
    global _scheduler
    logger.debug("_start_scheduler_if_needed called")

    settings = load_settings()
    if settings.app_env.lower() == "testing":
        return

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

    logger.info(
        "Scheduler started: digest will refresh daily at %02d:%02d (%s)",
        settings.digest_schedule_hour,
        settings.digest_schedule_minute,
        settings.digest_timezone,
    )

    def _shutdown_scheduler() -> None:
        global _scheduler
        if _scheduler is not None:
            _scheduler.shutdown(wait=False)
            _scheduler = None

    atexit.register(_shutdown_scheduler)


def _get_or_run_pipeline(force_refresh: bool = False) -> DigestRecord:
    with _CACHE_LOCK:
        if not force_refresh:
            if _CACHE["synthesis"] is not None:
                return DigestRecord(
                    synthesis=_CACHE["synthesis"],
                    items_by_theme=_CACHE["items_by_theme"],
                    generated_at=_CACHE["generated_at"],
                    fetch_metadata=_CACHE.get("fetch_metadata") or {},
                )

            record = get_digest_for_today()
            if record is not None:
                _CACHE["synthesis"] = record.synthesis
                _CACHE["items_by_theme"] = record.items_by_theme
                _CACHE["generated_at"] = record.generated_at
                _CACHE["fetch_metadata"] = record.fetch_metadata
                return record

        synthesis, items_by_theme, generated_at, fetch_metadata = _run_pipeline()
        _CACHE["synthesis"] = synthesis
        _CACHE["items_by_theme"] = items_by_theme
        _CACHE["generated_at"] = generated_at
        _CACHE["fetch_metadata"] = fetch_metadata

    save_digest(synthesis, items_by_theme, generated_at, fetch_metadata=fetch_metadata)

    try:
        from .services import evaluator
        date_str = generated_at.date().isoformat() if generated_at else None
        logger.info("[evals] Starting eval for %s", date_str)
        evaluator.run(date_str, synthesis, items_by_theme, fetch_metadata=fetch_metadata)
        logger.info("[evals] Completed eval for %s", date_str)
    except Exception:
        logger.exception("[evals] FAILED for %s — eval not written to DB", date_str)

    return DigestRecord(
        synthesis=synthesis,
        items_by_theme=items_by_theme,
        generated_at=generated_at,
        fetch_metadata=fetch_metadata,
    )



@app.route("/")
def index():
    record = _get_or_run_pipeline(force_refresh=False)
    eval_summary = None
    if record.generated_at:
        eval_summary = get_eval_summary_for_date(record.generated_at.date().isoformat())
    source_index_lookup = (record.synthesis or {}).get("source_index_lookup") or {}
    citation_index_map = {
        (v["source_name"], v["title"]): k
        for k, v in source_index_lookup.items()
    }
    citation_sort_map = {
        (v["source_name"], v["title"]): int(k)
        for k, v in source_index_lookup.items()
    }
    utilized_keys = _build_utilized_keys(record.synthesis or {})
    return render_template(
        "index.html",
        synthesis=record.synthesis or {},
        items_by_theme=record.items_by_theme or {},
        generated_at=record.generated_at,
        eval_summary=eval_summary,
        citation_index_map=citation_index_map,
        citation_sort_map=citation_sort_map,
        utilized_keys=utilized_keys,
    )


@app.route("/refresh")
def refresh():
    # FIX: require a secret token to prevent unauthenticated pipeline runs.
    # Set REFRESH_TOKEN in Railway environment variables.
    # Usage: GET /refresh?token=<your_secret>
    settings = load_settings()
    refresh_token = getattr(settings, "refresh_token", None) or os.environ.get("REFRESH_TOKEN", "")
    if refresh_token and request.args.get("token") != refresh_token:
        abort(403)
    _get_or_run_pipeline(force_refresh=True)
    return redirect(url_for("index"))


@app.route("/digest-health")
def digest_health():
    signals = get_consecutive_warning_types()
    trend = get_score_trend(lookback_days=7)
    pipeline = get_pipeline_health(days=1)
    today_str = date.today().isoformat()
    today_pipeline = pipeline[0] if pipeline and pipeline[0]["date"] == today_str else {}

    # Today's eval for Output Quality card
    today_eval = None
    quality_scores = get_quality_scores(days=1)
    if quality_scores:
        row = quality_scores[0]
        dimensions = {
            "Breadth":   row.get("ws_breadth"),
            "Coherence": row.get("ws_coherence"),
            "Insight":   row.get("ws_insight"),
            "Grounding": row.get("ws_grounding"),
            "PM Craft":  row.get("pc_insight"),
        }
        scored = {k: v for k, v in dimensions.items() if v}
        weakest_key = min(scored, key=scored.get) if scored else None
        today_eval = {
            "overall_score": row.get("overall_score"),
            "weakest_dimension": {
                "name": weakest_key,
                "score": scored[weakest_key],
            } if weakest_key else None,
        }

    return render_template(
        "digest_health.html",
        signals=signals,
        trend=trend,
        today_pipeline=today_pipeline,
        today_eval=today_eval,
    )


@app.route("/digest-health/pipeline")
def digest_health_pipeline():
    rows = get_pipeline_health(days=14)
    return render_template(
        "source_pipeline.html",
        rows=rows,
        today_date=date.today().isoformat(),
    )


@app.route("/digest-health/deviations")
def digest_health_deviations():
    warnings = get_warning_history(days=30)
    warning_types = sorted({
        wt for entry in warnings for wt in entry.get("warnings", {})
    })
    return render_template(
        "prompt_deviations.html",
        warnings=warnings,
        warning_types=warning_types,
    )


@app.route("/digest-health/quality")
def digest_health_quality():
    scores = get_quality_scores(days=30)
    return render_template(
        "output_quality.html",
        scores=scores,
    )


@app.route("/history")
def history():
    history_rows = get_digest_history()
    return render_template(
        "history.html",
        history_rows=history_rows,
    )


# RETIRED: /evals — superseded by /digest-health/quality

@app.route("/debug-eval/<date_str>")
def debug_eval(date_str: str):
    """
    Return the raw synthesis JSON for a given date for debugging purposes.

    FIX: previously called evaluator.run(date_str + '-debug', ...) which wrote
    a malformed-date row to the evals table and polluted the eval history page.
    Now returns the synthesis as JSON without any DB write.
    """
    record = get_digest_by_date(date_str)
    if record is None:
        return "No digest found for this date", 404
    return app.response_class(
        json.dumps({"synthesis": record.synthesis, "generated_at": record.generated_at.isoformat()}, indent=2),
        mimetype="application/json",
    )


@app.route("/<date_str>")
def digest_by_date(date_str: str):
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        abort(404)

    record = get_digest_by_date(date_str)
    if record is None:
        abort(404)

    eval_summary = get_eval_summary_for_date(date_str)
    source_index_lookup = (record.synthesis or {}).get("source_index_lookup") or {}
    citation_index_map = {
        (v["source_name"], v["title"]): k
        for k, v in source_index_lookup.items()
    }
    citation_sort_map = {
        (v["source_name"], v["title"]): int(k)
        for k, v in source_index_lookup.items()
    }
    utilized_keys = _build_utilized_keys(record.synthesis or {})
    return render_template(
        "index.html",
        synthesis=record.synthesis or {},
        items_by_theme=record.items_by_theme or {},
        generated_at=record.generated_at,
        eval_summary=eval_summary,
        citation_index_map=citation_index_map,
        citation_sort_map=citation_sort_map,
        utilized_keys=utilized_keys,
    )


def create_app() -> Flask:
    """Factory for external runners if needed."""
    _start_scheduler_if_needed()
    _register_all_prompts()
    return app


def _register_all_prompts() -> None:
    """
    Register all prompt system strings at startup.
    Auto-detects changes via content hash — no manual version bumping needed.
    Only the static system prompt strings are registered (not user prompt
    templates, which embed dynamic values like today's date and context blocks).
    """
    try:
        from .services.prompt_registry import register_prompt
        from .services.summarizer import (
            _CALL_A_SYSTEM,
            _CALL_B_SYSTEM,
            _CALL_C_SYSTEM,
        )
        from .services.synthesizer import (
            SYSTEM_PROMPT,
            _CALL_1A_SYSTEM,
            _CALL_3_SYSTEM,
            _CALL_4A_SYSTEM,
            _CALL_4B_SYSTEM,
        )

        register_prompt("summarizer.call_a.system", _CALL_A_SYSTEM)
        register_prompt("summarizer.call_b.system", _CALL_B_SYSTEM)
        register_prompt("summarizer.call_c.system", _CALL_C_SYSTEM)
        register_prompt("synthesizer.shared.system", SYSTEM_PROMPT)
        register_prompt("synthesizer.call_1a.system", _CALL_1A_SYSTEM)
        register_prompt("synthesizer.call_2_pm_craft.system", _CALL_4B_SYSTEM)
        register_prompt("synthesizer.call_3_cw.system", _CALL_3_SYSTEM)
        register_prompt("synthesizer.call_4a.system", _CALL_4A_SYSTEM)
        register_prompt("synthesizer.call_4b_sr.system", SYSTEM_PROMPT)

        logger.info("Prompt versions registered at startup.")
    except Exception:
        logger.exception("Failed to register prompt versions — continuing without version tracking.")


_start_scheduler_if_needed()
_register_all_prompts()