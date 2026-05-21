from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone, timedelta
from http.client import IncompleteRead
from time import mktime
from typing import Any, Dict, List, Tuple

import feedparser

from ..config import load_settings, load_sources_config
from .fetcher import fetch_article_text, MIN_WORD_THRESHOLD

logger = logging.getLogger(__name__)

MAX_FEED_RETRIES = 2

# Network errors worth retrying beyond IncompleteRead.
_RETRYABLE_ERRORS = (IncompleteRead, OSError, ConnectionError, TimeoutError)


def _parse_published(entry: Any) -> datetime | None:
    """Best-effort parsing of a feed entry's published date (UTC-aware)."""
    struct_time = getattr(entry, "published_parsed", None) or getattr(
        entry, "updated_parsed", None
    )
    if not struct_time:
        return None
    try:
        return datetime.fromtimestamp(mktime(struct_time), tz=timezone.utc)
    except Exception:
        return None


def _resolve_env_url(url: str) -> str | None:
    """Resolve a URL that may be of the form 'env:VAR_NAME'."""
    if not url.startswith("env:"):
        return url
    var_name = url.split(":", 1)[1]
    resolved = os.getenv(var_name)
    if not resolved:
        logger.warning("Environment variable %s not set for source feed.", var_name)
    return resolved


def _fetch_rss_items(
    source: Dict[str, Any],
    max_items: int = 5,
    lookback_hours: int = 24,
) -> List[Dict[str, Any]]:
    # --- FIX: resolve env: URLs for RSS sources (previously only done for podcasts) ---
    url = _resolve_env_url(source["url"])
    if not url:
        return []

    # Per-source overrides (optional fields in sources.json)
    max_items = source.get("max_items", max_items)
    lookback_hours = source.get("lookback_hours", lookback_hours)

    is_thin_feed = source.get("thin_feed", False)
    is_fetch_blocked = source.get("fetch_blocked", False)

    parsed = None
    for attempt in range(MAX_FEED_RETRIES + 1):
        try:
            parsed = feedparser.parse(url)
            break
        except _RETRYABLE_ERRORS as exc:
            # --- FIX: retry on OSError/TimeoutError in addition to IncompleteRead ---
            if attempt < MAX_FEED_RETRIES:
                logger.warning(
                    "RSS source %s: %s on attempt %d, retrying — %s",
                    source.get("id"),
                    type(exc).__name__,
                    attempt + 1,
                    exc,
                )
                time.sleep(2 ** attempt)
            else:
                raise

    assert parsed is not None

    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    total_entries = 0
    kept_entries = 0
    enriched_entries = 0

    if is_fetch_blocked:
        logger.warning(
            "RSS source %s: full fetch disabled (fetch_blocked=true) — using RSS summary only",
            source.get("id"),
        )
    elif is_thin_feed:
        logger.info(
            "RSS source %s: thin_feed=true — full fetch when entry text is below %d words",
            source.get("id"),
            MIN_WORD_THRESHOLD,
        )

    items: List[Dict[str, Any]] = []
    for entry in parsed.entries[:max_items]:
        total_entries += 1

        content_text = ""
        if getattr(entry, "content", None):
            try:
                content_text = entry.content[0].value or ""
            except Exception:
                content_text = ""
        if not content_text:
            content_text = getattr(entry, "summary", "") or getattr(
                entry, "description", ""
            )

        published_at = _parse_published(entry)

        # --- FIX: treat undated items as stale rather than always keeping them ---
        if published_at is None:
            logger.debug(
                "RSS source %s: no date for '%s' — skipping (treat as stale)",
                source.get("id"),
                getattr(entry, "title", "")[:80],
            )
            continue

        if published_at < cutoff:
            continue

        kept_entries += 1

        entry_url = getattr(entry, "link", "")
        word_count = len(content_text.split())
        needs_fetch = (is_thin_feed or word_count < MIN_WORD_THRESHOLD) \
                      and not is_fetch_blocked

        if (
            is_fetch_blocked
            and not is_thin_feed
            and entry_url
            and word_count < MIN_WORD_THRESHOLD
        ):
            logger.info(
                "RSS source %s: fetch_blocked, short RSS text (%d words) for '%s' — "
                "skipping full fetch, using RSS summary only",
                source.get("id"),
                word_count,
                getattr(entry, "title", "")[:80],
            )

        original_rss_summary = content_text

        if needs_fetch and entry_url:
            logger.info(
                "RSS source %s: %s for '%s' — attempting full fetch",
                source.get("id"),
                "thin_feed=true" if is_thin_feed
                else f"thin content ({word_count} words)",
                getattr(entry, "title", "")[:80],
            )
            fetched_text = fetch_article_text(entry_url)
            if fetched_text:
                content_text = fetched_text
                enriched_entries += 1
                logger.info(
                    "RSS source %s: full fetch succeeded (%d words) for '%s'",
                    source.get("id"),
                    len(fetched_text.split()),
                    getattr(entry, "title", "")[:80],
                )
            else:
                logger.warning(
                    "RSS source %s: full fetch failed for '%s' — url=%s",
                    source.get("id"),
                    getattr(entry, "title", "")[:80],
                    entry_url,
                )

        items.append(
            {
                "source_id": source["id"],
                "source_name": source["name"],
                "company_id": source.get("company_id"),
                "theme": source.get("theme"),
                "type": source.get("type"),
                "title": getattr(entry, "title", ""),
                "url": entry_url,
                "published_at": published_at.isoformat() if published_at else None,
                "summary": content_text,
                "rss_summary": original_rss_summary,
            }
        )

    logger.info(
        "RSS source %s: %d/%d items within last %d hours, %d enriched via full fetch",
        source.get("id"),
        kept_entries,
        total_entries,
        lookback_hours,
        enriched_entries,
    )
    return items


def _fetch_podcast_items(
    source: Dict[str, Any],
    max_items: int = 3,
    lookback_hours: int = 24,
) -> List[Dict[str, Any]]:
    raw_url = source["url"]
    url = _resolve_env_url(raw_url)
    if not url:
        return []

    # Per-source overrides
    max_items = source.get("max_items", max_items)
    lookback_hours = source.get("lookback_hours", lookback_hours)

    parsed = feedparser.parse(url)

    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    total_entries = 0
    kept_entries = 0

    items: List[Dict[str, Any]] = []
    for entry in parsed.entries[:max_items]:
        total_entries += 1

        transcript_or_description = getattr(entry, "summary", "") or getattr(
            entry, "description", ""
        )
        if not transcript_or_description and getattr(entry, "content", None):
            try:
                transcript_or_description = entry.content[0].value or ""
            except Exception:
                transcript_or_description = ""

        published_at = _parse_published(entry)

        # Treat undated podcast entries as stale (consistent with RSS handling)
        if published_at is None:
            logger.debug(
                "Podcast source %s: no date for '%s' — skipping",
                source.get("id"),
                getattr(entry, "title", "")[:80],
            )
            continue

        if published_at < cutoff:
            continue

        kept_entries += 1

        items.append(
            {
                "source_id": source["id"],
                "source_name": source["name"],
                "company_id": source.get("company_id"),
                "theme": source.get("theme"),
                "type": source.get("type"),
                "title": getattr(entry, "title", ""),
                "url": getattr(entry, "link", ""),
                "published_at": published_at.isoformat() if published_at else None,
                "summary": transcript_or_description,
                # --- FIX: add rss_summary so downstream refusal-retry path works
                #     identically for podcast items and RSS items ---
                "rss_summary": transcript_or_description,
            }
        )

    logger.info(
        "Podcast source %s: %d/%d items within last %d hours",
        source.get("id"),
        kept_entries,
        total_entries,
        lookback_hours,
    )
    return items


def fetch_items_grouped_by_theme() -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """
    Fetch items from all configured sources and group them by theme.

    Returns:
        Tuple of:
          - grouped items: {theme: [item, ...], ...}
          - fetch_metadata: {
              sources_configured: int,
              sources_active: int,
              sources_empty: int,
              empty_source_names: [...],
            }
    """
    settings = load_settings()
    cfg = load_sources_config(settings.sources_config_path)
    sources = cfg.get("sources", [])

    by_theme: Dict[str, List[Dict[str, Any]]] = {}

    sources_configured = len(sources)
    sources_active = 0
    sources_empty = 0
    empty_source_names: List[str] = []

    for source in sources:
        theme = source.get("theme", "unknown")
        source_type = source.get("type")
        source_name = source.get("name", source.get("id", "unknown"))

        try:
            if source_type == "rss":
                fetched = _fetch_rss_items(source, lookback_hours=settings.lookback_hours)
            elif source_type == "podcast":
                fetched = _fetch_podcast_items(source, lookback_hours=settings.lookback_hours)
            else:
                logger.warning(
                    "Unsupported source type '%s' for id=%s", source_type, source.get("id")
                )
                sources_empty += 1
                empty_source_names.append(source_name)
                continue
        except Exception as exc:
            logger.exception(
                "Failed to fetch source id=%s url=%s: %s",
                source.get("id"),
                source.get("url"),
                exc,
            )
            sources_empty += 1
            empty_source_names.append(source_name)
            continue

        if not fetched:
            sources_empty += 1
            empty_source_names.append(source_name)
            continue

        sources_active += 1
        bucket = by_theme.setdefault(theme, [])
        bucket.extend(fetched)

    fetch_metadata: Dict[str, Any] = {
        "sources_configured": sources_configured,
        "sources_active": sources_active,
        "sources_empty": sources_empty,
        "empty_source_names": empty_source_names,
    }

    logger.info(
        "Fetch complete: %d/%d sources active, %d empty",
        sources_active,
        sources_configured,
        sources_empty,
    )

    return by_theme, fetch_metadata