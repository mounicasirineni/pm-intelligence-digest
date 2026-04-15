from __future__ import annotations

import logging

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; PMDigestBot/1.0; "
        "+https://github.com/mounicasirineni/pm-ai-agents-2026)"
    )
}

MIN_WORD_THRESHOLD = 50


def fetch_article_text(url: str, timeout: int = 10) -> str:
    """
    Fetch full article body from URL.
    Returns extracted text or empty string on failure.
    """
    if not url:
        return ""
    try:
        response = httpx.get(url, headers=HEADERS, timeout=timeout, follow_redirects=True)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Try article tag first, fall back to main, then body
        for selector in ["article", "main", "[role='main']", "body"]:
            container = soup.select_one(selector)
            if container:
                text = container.get_text(separator=" ", strip=True)
                if len(text.split()) >= MIN_WORD_THRESHOLD:
                    return text

        return ""

    except Exception as exc:
        logger.warning("Article fetch failed for %s: %s", url, exc)
        return ""
