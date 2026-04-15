import feedparser
import httpx
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; PMDigestBot/1.0; "
        "+https://github.com/mounicasirineni/pm-ai-agents-2026)"
    )
}

MIN_WORD_THRESHOLD = 50

# Step 1 — check RSS feed content
feed_url = "https://blog.google/rss/"
parsed = feedparser.parse(feed_url)

print(f"Feed title: {parsed.feed.get('title', 'unknown')}")
print(f"Entries found: {len(parsed.entries)}")

article_urls = []
for entry in parsed.entries[:5]:
    content = ""
    if getattr(entry, "content", None):
        try:
            content = entry.content[0].value or ""
        except Exception:
            content = ""
    if not content:
        content = getattr(entry, "summary", "") or getattr(entry, "description", "")

    url = entry.get("link", "")
    article_urls.append(url)
    print(f"\nTitle: {entry.get('title', '')}")
    print(f"URL: {url}")
    print(f"RSS content words: {len(content.split())}")

# Step 2 — test full fetch on first 3 URLs
print("\n--- Full fetch test ---")
for url in article_urls[:3]:
    try:
        response = httpx.get(url, headers=HEADERS, timeout=10, follow_redirects=True)
        print(f"\nURL: {url}")
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
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
                        print(f"Words: {words}, Selector: {selector}")
                        print(f"Preview: {text[:200]}")
                        break
        else:
            print(f"Blocked/error: {response.status_code}")

    except Exception as e:
        print(f"FAILED: {e}")
