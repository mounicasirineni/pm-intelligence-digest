import httpx
import feedparser
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; PMDigestBot/1.0; "
        "+https://github.com/mounicasirineni/pm-ai-agents-2026)"
    )
}

MIN_WORD_THRESHOLD = 50

thin_feed_urls = {
    "openai_news": "https://openai.com/blog/rss.xml",
    "apple_newsroom": "https://www.apple.com/newsroom/rss-feed.rss",
    "anthropic_news": "https://www.anthropic.com/rss.xml",
    "techcrunch": "https://techcrunch.com/feed/",
    "rest_of_world": "https://restofworld.org/feed/",
    "sifted": "https://sifted.eu/feed",
    "quartz": "https://qz.com/feed",
    "politico_tech": "https://www.politico.com/rss/technology.xml",
}


def test_fetch(url):
    try:
        response = httpx.get(
            url, headers=HEADERS, timeout=10, follow_redirects=True
        )
        if response.status_code != 200:
            return response.status_code, 0, "none"

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer",
                         "header", "aside", "form"]):
            tag.decompose()

        for selector in ["article", "main", "[role='main']", "body"]:
            container = soup.select_one(selector)
            if container:
                text = container.get_text(separator=" ", strip=True)
                words = len(text.split())
                if words >= MIN_WORD_THRESHOLD:
                    return response.status_code, words, selector

        return response.status_code, 0, "no selector matched"
    except Exception as e:
        return "ERROR", 0, str(e)


for source_id, feed_url in thin_feed_urls.items():
    # Get first article URL from feed
    parsed = feedparser.parse(feed_url)
    if not parsed.entries:
        print(f"{source_id}: no entries in feed")
        continue

    article_url = parsed.entries[0].get("link", "")
    if not article_url:
        print(f"{source_id}: no URL in first entry")
        continue

    status, words, selector = test_fetch(article_url)

    if isinstance(status, int) and status == 200 and words >= MIN_WORD_THRESHOLD:
        result = "FETCHABLE"
        action = "thin_feed: true"
    elif status == 403:
        result = "BLOCKED (403)"
        action = "thin_feed: true, fetch_blocked: true"
    elif status == 200 and words < MIN_WORD_THRESHOLD:
        result = "FETCHABLE BUT THIN"
        action = "may need Playwright"
    else:
        result = f"FAILED ({status})"
        action = "investigate"

    print(f"{source_id:20} {result:30} | {words:5} words | {action}")
    print(f"  URL tested: {article_url}")
