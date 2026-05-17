"""
Standalone fetch test for all sources in sources.json.

Tests the full fetch tier chain (direct → Jina → og:description) for every
RSS source, and RSS-only for podcast sources. Reports word counts, fetch tier
used, and whether content meets the summarizer's thresholds.

Usage:
    python test_fetch_sources.py
    python test_fetch_sources.py --sources-path /path/to/sources.json
    python test_fetch_sources.py --lookback-hours 72
    python test_fetch_sources.py --source-id openai_news politico_tech

Does NOT touch the database, summarizer, or synthesizer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from time import mktime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import feedparser
import httpx
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Thresholds (mirror summarizer.py and fetcher.py)
# ---------------------------------------------------------------------------
MIN_WORD_THRESHOLD = 100          # fetcher.py: discard below this
JINA_MIN_WORD_THRESHOLD = 150     # fetcher.py: discard Jina results below this
MINIMUM_CONTENT_WORDS = 200       # summarizer.py: hard skip below this
CONFIDENCE_FLOOR_WORDS = 400      # summarizer.py: cap confidence at medium below this
OG_DESCRIPTION_MIN_WORDS = 20     # fetcher.py: minimum for og:description fallback

JINA_PREFERRED_DOMAINS = {
    "www.theverge.com",
    "techcrunch.com",
    "www.wired.com",
    "www.nytimes.com",
    "www.wsj.com",
    "www.ft.com",
}

PAYWALL_SIGNALS = [
    "subscribe to continue",
    "create a free account",
    "sign in to read",
    "already a subscriber",
    "get unlimited access",
    "this article is for subscribers",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; PMDigestBot/1.0; "
        "+https://github.com/mounicasirineni/pm-ai-agents-2026)"
    )
}

JINA_HEADERS = {
    "User-Agent": "PMDigestBot/1.0",
    "Accept": "text/plain",
    "X-Return-Format": "text",
}

# ---------------------------------------------------------------------------
# ANSI colors for terminal output
# ---------------------------------------------------------------------------
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
CYAN   = "\033[36m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def _get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _is_paywalled(text: str) -> bool:
    lower = text.lower()
    return any(signal in lower for signal in PAYWALL_SIGNALS)


def _extract_og_description(soup: BeautifulSoup) -> str:
    for attr_name, attr_value in [
        ("property", "og:description"),
        ("name", "twitter:description"),
        ("name", "description"),
    ]:
        tag = soup.find("meta", attrs={attr_name: attr_value})
        if tag and tag.get("content"):
            text = tag["content"].strip()
            if len(text.split()) >= OG_DESCRIPTION_MIN_WORDS:
                return text
    return ""


def _fetch_via_jina(url: str, timeout: int = 20) -> tuple[str, str]:
    """Returns (text, status_note)."""
    jina_url = f"https://r.jina.ai/{url}"
    try:
        response = httpx.get(
            jina_url,
            headers=JINA_HEADERS,
            timeout=timeout,
            follow_redirects=True,
        )
        response.raise_for_status()
        text = response.text.strip()
        word_count = len(text.split())
        if word_count < JINA_MIN_WORD_THRESHOLD:
            return "", f"jina_too_short ({word_count} words)"
        return text, "jina"
    except httpx.HTTPStatusError as exc:
        return "", f"jina_http_{exc.response.status_code}"
    except Exception as exc:
        return "", f"jina_error: {str(exc)[:60]}"


def fetch_article_text_with_tier(url: str, timeout: int = 10) -> tuple[str, str]:
    """
    Full fetch tier chain. Returns (text, tier_used).
    tier_used values: 'primary', 'jina', 'og_description', 'blocked', 'failed'
    """
    if not url:
        return "", "no_url"

    domain = _get_domain(url)

    og_description = ""

    if domain not in JINA_PREFERRED_DOMAINS:
        try:
            response = httpx.get(
                url, headers=HEADERS, timeout=timeout, follow_redirects=True
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            og_description = _extract_og_description(soup)

            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                tag.decompose()

            for selector in ["article", "main", "[role='main']"]:
                container = soup.select_one(selector)
                if container:
                    text = container.get_text(separator=" ", strip=True)
                    if len(text.split()) >= MIN_WORD_THRESHOLD:
                        if _is_paywalled(text):
                            break
                        return text, "primary"

        except httpx.HTTPStatusError:
            pass
        except Exception:
            pass

    jina_text, jina_status = _fetch_via_jina(url)
    if jina_text:
        return jina_text, "jina"

    if domain in JINA_PREFERRED_DOMAINS and not og_description:
        try:
            response = httpx.get(
                url, headers=HEADERS, timeout=min(5, timeout), follow_redirects=True
            )
            soup = BeautifulSoup(response.text, "html.parser")
            og_description = _extract_og_description(soup)
        except Exception:
            pass

    if og_description:
        return og_description, "og_description"

    return "", f"failed ({jina_status})"


def _confidence_label(word_count: int, tier: str) -> str:
    if tier == "blocked_domain":
        return "blocked"
    if tier == "og_description":
        return "low (og:description)"
    if word_count == 0:
        return "failed"
    if word_count < MINIMUM_CONTENT_WORDS:
        return "skip (below summarizer minimum)"
    if word_count < CONFIDENCE_FLOOR_WORDS:
        return "medium (below floor)"
    return "high"


def _color_for_confidence(label: str) -> str:
    if "high" in label:
        return GREEN
    if "medium" in label:
        return YELLOW
    return RED


def _parse_published(entry: Any) -> Optional[datetime]:
    struct_time = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if not struct_time:
        return None
    try:
        return datetime.fromtimestamp(mktime(struct_time), tz=timezone.utc)
    except Exception:
        return None


def _resolve_env_url(url: str) -> Optional[str]:
    if not url.startswith("env:"):
        return url
    var_name = url.split(":", 1)[1]
    return os.getenv(var_name)


# ---------------------------------------------------------------------------
# Per-source test
# ---------------------------------------------------------------------------

def test_source(
    source: Dict[str, Any],
    lookback_hours: int = 48,
    max_items: int = 3,
) -> Dict[str, Any]:
    source_id = source.get("id", "unknown")
    source_name = source.get("name", source_id)
    raw_url = source.get("url", "")
    is_thin = source.get("thin_feed", False)
    is_blocked = source.get("fetch_blocked", False)

    url = _resolve_env_url(raw_url)
    if not url:
        return {
            "source_id": source_id,
            "source_name": source_name,
            "error": "URL not resolved (env var not set)",
            "items": [],
        }

    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)

    # Parse feed
    try:
        parsed = feedparser.parse(url)
    except Exception as exc:
        return {
            "source_id": source_id,
            "source_name": source_name,
            "error": f"Feed parse failed: {exc}",
            "items": [],
        }

    entries = parsed.entries[:max_items]
    results = []

    for entry in entries:
        title = getattr(entry, "title", "(no title)")
        entry_url = getattr(entry, "link", "")
        published = _parse_published(entry)

        if published and published < cutoff:
            results.append({
                "title": title,
                "url": entry_url,
                "skipped": "outside lookback window",
            })
            continue

        # RSS content
        rss_text = ""
        if getattr(entry, "content", None):
            try:
                rss_text = entry.content[0].value or ""
            except Exception:
                pass
        if not rss_text:
            rss_text = getattr(entry, "summary", "") or getattr(entry, "description", "")

        rss_words = len(rss_text.split()) if rss_text else 0

        # Decide whether to attempt full fetch
        needs_fetch = (
            not is_blocked
            and entry_url
            and (is_thin or rss_words < MIN_WORD_THRESHOLD)
        )

        if is_blocked:
            fetch_tier = "fetch_blocked (config)"
            fetch_words = 0
            fetch_text = ""
        elif needs_fetch:
            time.sleep(0.5)  # polite delay between fetches
            fetch_text, fetch_tier = fetch_article_text_with_tier(entry_url)
            fetch_words = len(fetch_text.split()) if fetch_text else 0
        else:
            fetch_tier = "rss_sufficient"
            fetch_words = rss_words
            fetch_text = rss_text

        final_words = fetch_words if needs_fetch and fetch_text else rss_words
        final_tier = fetch_tier if needs_fetch else "rss"
        confidence = _confidence_label(final_words, fetch_tier if is_blocked else final_tier)

        results.append({
            "title": title,
            "url": entry_url,
            "published": published.isoformat() if published else None,
            "rss_words": rss_words,
            "fetch_tier": final_tier,
            "fetch_words": final_words,
            "confidence": confidence,
            "is_thin_feed": is_thin,
            "is_fetch_blocked": is_blocked,
            "preview": (fetch_text or rss_text)[:200].replace("\n", " ") if (fetch_text or rss_text) else "",
        })

    return {
        "source_id": source_id,
        "source_name": source_name,
        "theme": source.get("theme"),
        "type": source.get("type", "rss"),
        "thin_feed": is_thin,
        "fetch_blocked": is_blocked,
        "feed_url": url,
        "items_checked": len(results),
        "items": results,
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def render_report(results: List[Dict[str, Any]]) -> None:
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}PM Digest — Source Fetch Test Report{RESET}")
    print(f"{'='*70}\n")

    # Summary table
    print(f"{BOLD}{'SOURCE':<35} {'THEME':<22} {'ITEMS':<6} {'STATUS'}{RESET}")
    print("-" * 80)

    for r in results:
        if r.get("error"):
            status = f"{RED}ERROR: {r['error']}{RESET}"
            items_str = "-"
        else:
            items = [i for i in r.get("items", []) if not i.get("skipped")]
            if not items:
                status = f"{YELLOW}no items in window{RESET}"
                items_str = "0"
            else:
                confidences = [i.get("confidence", "") for i in items]
                if all("high" in c for c in confidences):
                    status = f"{GREEN}all high{RESET}"
                elif any("failed" in c or "blocked" in c or "skip" in c for c in confidences):
                    status = f"{RED}some failed/blocked{RESET}"
                else:
                    status = f"{YELLOW}mixed/medium{RESET}"
                items_str = str(len(items))

        name = r.get("source_name", r.get("source_id", "?"))[:34]
        theme = (r.get("theme") or "")[:21]
        print(f"{name:<35} {theme:<22} {items_str:<6} {status}")

    # Detailed breakdown
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}Detailed Breakdown{RESET}")
    print(f"{'='*70}\n")

    for r in results:
        name = r.get("source_name", r.get("source_id", "?"))
        flags = []
        if r.get("thin_feed"):
            flags.append("thin_feed")
        if r.get("fetch_blocked"):
            flags.append("fetch_blocked")
        flag_str = f" [{', '.join(flags)}]" if flags else ""

        print(f"{BOLD}{CYAN}{name}{flag_str}{RESET}  ({r.get('theme', '')})")

        if r.get("error"):
            print(f"  {RED}ERROR: {r['error']}{RESET}\n")
            continue

        items = r.get("items", [])
        if not items:
            print(f"  {YELLOW}No entries returned from feed{RESET}\n")
            continue

        for item in items:
            if item.get("skipped"):
                print(f"  ↷ {item['title'][:70]} — {YELLOW}skipped: {item['skipped']}{RESET}")
                continue

            title = item.get("title", "")[:70]
            rss_words = item.get("rss_words", 0)
            fetch_tier = item.get("fetch_tier", "")
            fetch_words = item.get("fetch_words", 0)
            confidence = item.get("confidence", "")
            color = _color_for_confidence(confidence)
            preview = item.get("preview", "")

            print(f"  • {title}")
            print(f"    RSS: {rss_words}w  |  Fetch tier: {fetch_tier}  |  Final: {fetch_words}w")
            print(f"    Confidence: {color}{confidence}{RESET}")
            if preview:
                print(f"    Preview: {preview[:120]}…")
            print()

        print()

    # Recommendations
    print(f"{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}Recommendations{RESET}")
    print(f"{'='*70}\n")

    for r in results:
        if r.get("error"):
            continue
        items = [i for i in r.get("items", []) if not i.get("skipped")]
        if not items:
            continue

        name = r.get("source_name", "?")
        is_blocked = r.get("fetch_blocked", False)
        is_thin = r.get("thin_feed", False)

        all_failed = all(
            "failed" in i.get("confidence", "") or "blocked" in i.get("confidence", "")
            for i in items
        )
        all_high = all("high" in i.get("confidence", "") for i in items)
        any_jina_ok = any(
            i.get("fetch_tier") == "jina" and "high" in i.get("confidence", "")
            for i in items
        )
        any_og_only = all(i.get("fetch_tier") == "og_description" for i in items)

        if is_blocked and any_jina_ok:
            print(
                f"  {GREEN}✓ {name}{RESET} — Jina succeeds. Consider removing fetch_blocked "
                "+ domain block to unlock full content."
            )
        elif is_blocked and all_failed:
            print(f"  {RED}✗ {name}{RESET} — fetch_blocked and Jina also fails. Keep blocked.")
        elif not is_thin and all(i.get("rss_words", 0) < MIN_WORD_THRESHOLD for i in items):
            print(
                f"  {YELLOW}⚠ {name}{RESET} — RSS is thin and thin_feed not set. "
                "Consider adding thin_feed: true."
            )
        elif is_thin and all_high:
            print(
                f"  {GREEN}✓ {name}{RESET} — thin_feed=true and Jina producing "
                "high-confidence content. Working well."
            )
        elif any_og_only:
            print(
                f"  {RED}✗ {name}{RESET} — falling back to og:description only. "
                "Jina not helping. Review source."
            )
        elif all_high:
            print(f"  {GREEN}✓ {name}{RESET} — all items high confidence. No changes needed.")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Test fetch pipeline for all PM Digest sources.")
    parser.add_argument(
        "--sources-path",
        default="config/sources.json",
        help="Path to sources.json (default: config/sources.json)",
    )
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=48,
        help="How many hours back to look for feed entries (default: 48)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=2,
        help="Max items to test per source (default: 2)",
    )
    parser.add_argument(
        "--source-id",
        nargs="*",
        help="Test only specific source IDs (e.g. --source-id openai_news politico_tech)",
    )
    parser.add_argument(
        "--json-out",
        help="Optional path to write full results as JSON",
    )
    args = parser.parse_args()

    sources_path = Path(args.sources_path)
    if not sources_path.exists():
        print(f"{RED}sources.json not found at {sources_path}{RESET}")
        sys.exit(1)

    with open(sources_path) as f:
        cfg = json.load(f)

    sources = cfg.get("sources", [])

    if args.source_id:
        sources = [s for s in sources if s.get("id") in args.source_id]
        if not sources:
            print(f"{RED}No sources matched: {args.source_id}{RESET}")
            sys.exit(1)

    print(
        f"\nTesting {len(sources)} source(s) — lookback {args.lookback_hours}h, "
        f"max {args.max_items} items each"
    )
    print("This may take a few minutes depending on source count and fetch timeouts.\n")

    all_results = []
    for i, source in enumerate(sources, 1):
        name = source.get("name", source.get("id", "?"))
        print(f"[{i}/{len(sources)}] Testing: {name}...", end=" ", flush=True)
        result = test_source(
            source,
            lookback_hours=args.lookback_hours,
            max_items=args.max_items,
        )
        all_results.append(result)
        items_ok = sum(
            1
            for item in result.get("items", [])
            if not item.get("skipped") and "high" in item.get("confidence", "")
        )
        print(f"{GREEN}✓{RESET}" if items_ok else f"{YELLOW}~{RESET}")
        time.sleep(0.25)

    render_report(all_results)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"Full results written to {args.json_out}\n")


if __name__ == "__main__":
    main()
