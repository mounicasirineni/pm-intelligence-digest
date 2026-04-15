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

MIN_WORD_THRESHOLD = 50

BLOCKED_DOMAINS = {
    "blogs.microsoft.com",
    "openai.com",
    "qz.com",
    "www.politico.com",
}


def _get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def fetch_article_text(url: str, timeout: int = 10) -> str:
    """
    Fetch full article body from URL.
    Returns extracted text or empty string on failure.
    """
    if not url:
        return ""

    domain = _get_domain(url)
    if domain in BLOCKED_DOMAINS:
        logger.warning(
            "Skipping full fetch for blocked domain %s — "
            "will use RSS summary only", domain
        )
        return ""

    try:
        response = httpx.get(
            url,
            headers=HEADERS,
            timeout=timeout,
            follow_redirects=True
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer",
                         "header", "aside", "form"]):
            tag.decompose()

        for selector in ["article", "main", "[role='main']", "body"]:
            container = soup.select_one(selector)
            if container:
                text = container.get_text(separator=" ", strip=True)
                if len(text.split()) >= MIN_WORD_THRESHOLD:
                    return text

        return ""

    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 403:
            logger.warning(
                "Article fetch blocked (403) for %s — "
                "consider adding to BLOCKED_DOMAINS", url
            )
        else:
            logger.warning(
                "Article fetch HTTP error for %s: %s", url, exc
            )
        return ""

    except Exception as exc:
        logger.warning("Article fetch failed for %s: %s", url, exc)
        return ""
