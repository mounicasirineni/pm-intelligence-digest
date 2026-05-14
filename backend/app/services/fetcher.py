from __future__ import annotations

import logging
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; PMDigestBot/1.0; "
        "+https://github.com/mounicasirineni/pm-ai-agents-2026)"
    )
}

# Jina Reader doesn't need auth for basic extraction — uses their public endpoint.
JINA_HEADERS = {
    "User-Agent": "PMDigestBot/1.0",
    "Accept": "text/plain",
    # Return clean plain text, not markdown with link noise
    "X-Return-Format": "text",
}

MIN_WORD_THRESHOLD = 100

# Floor for og:description to be considered usable.
# og:description is always low-confidence but better than nothing.
OG_DESCRIPTION_MIN_WORDS = 20

# If Jina returns fewer words than this, treat it as a failed fetch.
JINA_MIN_WORD_THRESHOLD = 150

BLOCKED_DOMAINS = {
    "blogs.microsoft.com",
    "openai.com",
    "qz.com",
    "www.politico.com",
}

# Domains where the primary fetcher reliably fails (JS-rendered or hard paywalls)
# and we should skip straight to Jina rather than wasting a round-trip.
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
    These are populated even on JS-rendered or paywalled pages.
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


def _fetch_via_jina(url: str, timeout: int = 20) -> str:
    """
    Fetch article content via Jina Reader (r.jina.ai).
    Jina renders JS, strips nav/footer/ads, and returns clean prose.
    Returns extracted text or empty string on failure.
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
        word_count = len(text.split())
        if word_count < JINA_MIN_WORD_THRESHOLD:
            logger.warning(
                "Jina fetch returned too little content for %s (%d words) — discarding",
                url,
                word_count,
            )
            return ""
        logger.info("Jina fetch succeeded for %s (%d words)", url, word_count)
        return text
    except httpx.HTTPStatusError as exc:
        logger.warning("Jina fetch HTTP error for %s: %s", url, exc)
        return ""
    except Exception as exc:
        logger.warning("Jina fetch failed for %s: %s", url, exc)
        return ""


def fetch_article_text(url: str, timeout: int = 10) -> str:
    """
    Fetch full article body from URL using a three-tier fallback chain:

      1. Primary: httpx + BeautifulSoup (fast, no external dependency)
      2. Secondary: Jina Reader (handles JS-rendered pages and soft paywalls)
      3. Tertiary: og:description meta tag (always low-confidence, 20–80 words)

    Returns extracted text or empty string if all tiers fail.

    Callers can inspect word count to determine confidence level:
      >= 400 words  → high confidence
      100–399 words → medium confidence (or og:description if flagged)
      < 100 words   → should be skipped upstream (MIN_WORD_THRESHOLD)

    The returned string is tagged with a prefix when sourced from og:description
    so the summarizer can detect it and force confidence=low:
      "OG_DESCRIPTION: <text>"
    """
    if not url:
        return ""

    domain = _get_domain(url)

    if domain in BLOCKED_DOMAINS:
        logger.warning(
            "Skipping full fetch for blocked domain %s — will use RSS summary only",
            domain,
        )
        return ""

    # --- Tier 1: Primary httpx fetch (skip for known JS-heavy domains) ---
    og_description = ""

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

            # Always attempt og:description extraction while we have the soup,
            # so we have it ready if tiers 1 and 2 both fail.
            og_description = _extract_og_description(soup)

            for selector in ["article", "main", "[role='main']"]:
                container = soup.select_one(selector)
                if container:
                    text = container.get_text(separator=" ", strip=True)
                    if len(text.split()) >= MIN_WORD_THRESHOLD:
                        if _is_paywalled(text):
                            logger.warning(
                                "Paywall detected for %s — falling through to Jina",
                                url,
                            )
                            break
                        logger.info(
                            "Primary fetch succeeded for %s (%d words)",
                            url,
                            len(text.split()),
                        )
                        return text

        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 403:
                logger.warning(
                    "Article fetch blocked (403) for %s — falling through to Jina",
                    url,
                )
            else:
                logger.warning(
                    "Article fetch HTTP error for %s: %s — falling through to Jina",
                    url,
                    exc,
                )
        except Exception as exc:
            logger.warning(
                "Article fetch failed for %s: %s — falling through to Jina", url, exc
            )

    # --- Tier 2: Jina Reader ---
    jina_result = _fetch_via_jina(url)
    if jina_result:
        return jina_result

    # JINA-preferred domains skip tier 1, so og:description was never extracted.
    # Small HTML GET (HEAD has no body) to read meta tags for tier 3 only.
    if domain in JINA_PREFERRED_DOMAINS and not og_description:
        try:
            response = httpx.get(
                url,
                headers=HEADERS,
                timeout=min(5, timeout),
                follow_redirects=True,
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            og_description = _extract_og_description(soup)
        except Exception as exc:
            logger.debug(
                "Og meta-only fetch failed for JINA-preferred URL %s: %s", url, exc
            )

    # --- Tier 3: og:description ---
    # Non-JINA-preferred: usually extracted during tier 1. JINA-preferred may
    # have been filled by the small meta fetch above.
    if og_description:
        logger.info(
            "Falling back to og:description for %s (%d words) — confidence will be low",
            url,
            len(og_description.split()),
        )
        # Prefix so summarizer.py can detect this and force confidence=low.
        return f"OG_DESCRIPTION: {og_description}"

    logger.warning("All fetch tiers failed for %s — returning empty", url)
    return ""