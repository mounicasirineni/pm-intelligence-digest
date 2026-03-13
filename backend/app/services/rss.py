from __future__ import annotations

import logging
import os
from datetime import datetime, timezone, timedelta
from time import mktime
from typing import Any, Dict, List, Tuple

import feedparser

from ..config import load_settings, load_sources_config

logger = logging.getLogger(__name__)


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


def _fetch_rss_items(
    source: Dict[str, Any],
    max_items: int = 5,
    lookback_hours: int = 24,
) -> List[Dict[str, Any]]:
    url = source["url"]
    parsed = feedparser.parse(url)

    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    total_entries = 0
    kept_entries = 0

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

        if published_at is not None and published_at < cutoff:
            continue

        kept_entries += 1

        items.append(
            {
                "source_id": source["id"],
                "source_name": source["name"],
                "theme": source.get("theme"),
                "type": source.get("type"),
                "title": getattr(entry, "title", ""),
                "url": getattr(entry, "link", ""),
                "published_at": published_at.isoformat() if published_at else None,
                "summary": content_text,
            }
        )

    logger.info(
        "RSS source %s: %d/%d items within last %d hours",
        source.get("id"),
        kept_entries,
        total_entries,
        lookback_hours,
    )
    return items


def _resolve_env_url(url: str) -> str | None:
    """Resolve a URL that may be of the form 'env:VAR_NAME'."""
    if not url.startswith("env:"):
        return url
    var_name = url.split(":", 1)[1]
    resolved = os.getenv(var_name)
    if not resolved:
        logger.warning("Environment variable %s not set for source feed.", var_name)
    return resolved


def _fetch_podcast_items(
    source: Dict[str, Any],
    max_items: int = 3,
    lookback_hours: int = 24,
) -> List[Dict[str, Any]]:
    raw_url = source["url"]
    url = _resolve_env_url(raw_url)
    if not url:
        return []

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

        if published_at is not None and published_at < cutoff:
            continue

        kept_entries += 1

        items.append(
            {
                "source_id": source["id"],
                "source_name": source["name"],
                "theme": source.get("theme"),
                "type": source.get("type"),
                "title": getattr(entry, "title", ""),
                "url": getattr(entry, "link", ""),
                "published_at": published_at.isoformat() if published_at else None,
                "summary": transcript_or_description,
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
              sources_active: int,       # returned ≥1 item in lookback window
              sources_empty: int,        # returned 0 items in lookback window
              empty_source_names: [...], # names of empty sources
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