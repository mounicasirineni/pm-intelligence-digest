from __future__ import annotations

import logging
from typing import Tuple
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from ..constants import OG_DESCRIPTION_PREFIX

logger = logging.getLogger(__name__)

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

MIN_WORD_THRESHOLD = 100

# Floor for og:description to be considered usable.
OG_DESCRIPTION_MIN_WORDS = 20

# If Jina returns fewer words than this, treat it as a failed fetch.
JINA_MIN_WORD_THRESHOLD = 150

# Domains where the primary fetcher reliably fails (JS-rendered or hard paywalls).
# Skip tier 1 entirely and go straight to Jina.
# --- FIX: added politico.com, uxdesign.cc, qz.com — consistent 403/robots/451 ---
JINA_PREFERRED_DOMAINS = {
    "www.theverge.com",
    "techcrunch.com",
    "www.wired.com",
    "www.nytimes.com",
    "www.wsj.com",
    "www.ft.com",
    "www.politico.com",   # consistent 403 on tier 1
    "uxdesign.cc",        # robots.txt disallows all fetching
    "qz.com",             # consistent 451 geo-block
}

PAYWALL_SIGNALS = [
    "subscribe to continue",
    "create a free account",
    "sign in to read",
    "already a subscriber",
    "get unlimited access",
    "this article is for subscribers",
]

# Jina HTTP status codes that indicate a hard block at the CDN/host level.
# When Jina returns one of these, tier 3 (og:description fallback) will also
# fail with the same error — skip tier 3 entirely to avoid a wasted HTTP call.
_JINA_HARD_BLOCK_STATUSES = {403, 451}


def _get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _is_paywalled(text: str) -> bool:
    lower = text.lower()
    return any(signal in lower for signal in PAYWALL_SIGNALS)


def _extract_og_description(soup: BeautifulSoup) -> str:
    """
    Extract og:description or twitter:description from a parsed page.
    Returns empty string if not found or below minimum word count.
    """
    for attr_name, attr_value in [
        ("property", "og:description"),
        ("name", "twitter:description"),
        ("name", "description"),
    ]:
        tag = soup.find("meta", attrs={attr_name: attr_value})
        if tag and tag.get("content"):  # type: ignore[union-attr]
            text = tag["content"].strip()  # type: ignore[index]
            if len(text.split()) >= OG_DESCRIPTION_MIN_WORDS:
                return text
    return ""


def _fetch_via_jina(url: str, timeout: int = 20) -> Tuple[str, str, str]:
    """
    Fetch article content via Jina Reader (r.jina.ai).

    Returns:
        (text, status, og_description) where:
          text            — extracted article text, or "" on failure
          status          — "ok" | "thin" | "hard_block" | "error"
          og_description  — og:description extracted from Jina's response HTML,
                            or "" if not available / not applicable
    """
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

        # --- FIX: extract og:description from Jina response while we have it,
        #     so tier 3 doesn't need a second HTTP call for JINA_PREFERRED domains ---
        og_description = ""
        try:
            soup = BeautifulSoup(response.text, "html.parser")
            og_description = _extract_og_description(soup)
        except Exception:
            pass

        word_count = len(text.split())
        if word_count < JINA_MIN_WORD_THRESHOLD:
            logger.warning(
                "Jina fetch returned too little content for %s (%d words) — discarding",
                url,
                word_count,
            )
            return "", "thin", og_description

        logger.info("Jina fetch succeeded for %s (%d words)", url, word_count)
        return text, "ok", og_description

    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        logger.warning("Jina fetch HTTP error for %s: %s", url, exc)
        # --- FIX: surface hard block status so caller can skip tier 3 ---
        if status_code in _JINA_HARD_BLOCK_STATUSES:
            return "", "hard_block", ""
        return "", "error", ""
    except Exception as exc:
        logger.warning("Jina fetch failed for %s: %s", url, exc)
        return "", "error", ""


def fetch_article_text(url: str, timeout: int = 10) -> str:
    """
    Fetch full article body from URL using a three-tier fallback chain:

      1. Primary: httpx + BeautifulSoup (fast, no external dependency)
         Skipped for domains in JINA_PREFERRED_DOMAINS.
      2. Secondary: Jina Reader (handles JS-rendered pages and soft paywalls)
      3. Tertiary: og:description meta tag (always low-confidence, 20–80 words)
         Skipped when Jina returned a hard block (403/451) — the same block
         would apply to a direct GET, making tier 3 a wasted HTTP call.

    Returns extracted text, or empty string if all tiers fail.

    When content came from og:description, the return value is prefixed with
    OG_DESCRIPTION_PREFIX (imported from constants.py) so summarizer.py can
    detect it and cap confidence at "low".
    """
    if not url:
        return ""

    domain = _get_domain(url)
    og_description = ""

    # --- Tier 1: Primary httpx fetch (skip for known JS-heavy/blocked domains) ---
    if domain not in JINA_PREFERRED_DOMAINS:
        try:
            response = httpx.get(
                url,
                headers=HEADERS,
                timeout=timeout,
                follow_redirects=True,
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            for tag in soup(["script", "style", "nav", "footer",
                              "header", "aside", "form"]):
                tag.decompose()

            # Always extract og:description while we have the soup — available
            # as tier-3 fallback if tiers 1 and 2 both fail.
            og_description = _extract_og_description(soup)

            # --- FIX: use continue instead of break so remaining selectors
            #     are tried if the first one returns a paywalled block ---
            for selector in ["article", "main", "[role='main']"]:
                container = soup.select_one(selector)
                if container:
                    text = container.get_text(separator=" ", strip=True)
                    if len(text.split()) >= MIN_WORD_THRESHOLD:
                        if _is_paywalled(text):
                            logger.warning(
                                "Paywall detected in '%s' selector for %s — trying next selector",
                                selector,
                                url,
                            )
                            continue  # try next selector instead of breaking to Jina
                        logger.info(
                            "Primary fetch succeeded for %s (%d words)",
                            url,
                            len(text.split()),
                        )
                        return text
            # All selectors exhausted (or paywalled) — fall through to Jina
            logger.warning("Primary fetch: no usable content found for %s — falling through to Jina", url)

        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 403:
                logger.warning(
                    "Article fetch blocked (403) for %s — falling through to Jina", url,
                )
            else:
                logger.warning(
                    "Article fetch HTTP error for %s: %s — falling through to Jina", url, exc,
                )
        except Exception as exc:
            logger.warning(
                "Article fetch failed for %s: %s — falling through to Jina", url, exc,
            )

    # --- Tier 2: Jina Reader ---
    jina_text, jina_status, jina_og = _fetch_via_jina(url)

    # Use og:description from Jina response if we don't already have one
    # (JINA_PREFERRED_DOMAINS skip tier 1, so og_description may still be "")
    if not og_description and jina_og:
        og_description = jina_og

    if jina_text:
        # --- FIX: apply paywall detection to Jina output ---
        if _is_paywalled(jina_text):
            logger.warning(
                "Paywall detected in Jina output for %s — falling through to og:description", url,
            )
            # fall through to tier 3 below
        else:
            return jina_text

    # --- FIX: skip tier 3 entirely when Jina returned a hard block ---
    # A 403 or 451 at the CDN level applies equally to a direct GET.
    # Firing tier 3 would only waste an HTTP call.
    if jina_status == "hard_block":
        logger.warning("All fetch tiers failed for %s — returning empty (hard block)", url)
        return ""

    # --- Tier 3: og:description ---
    # For non-JINA_PREFERRED domains: extracted during tier 1.
    # For JINA_PREFERRED domains:     extracted from Jina response body above.
    if og_description:
        logger.info(
            "Falling back to og:description for %s (%d words) — confidence will be low",
            url,
            len(og_description.split()),
        )
        return f"{OG_DESCRIPTION_PREFIX}{og_description}"

    logger.warning("All fetch tiers failed for %s — returning empty", url)
    return ""