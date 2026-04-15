import feedparser
import httpx
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; PMDigestBot/1.0; "
        "+https://github.com/mounicasirineni/pm-ai-agents-2026)"
    )
}

feed_url = "https://rss.politico.com/technology.xml"
parsed = feedparser.parse(feed_url)

# Check RSS word counts
print("RSS entry word counts:")
for entry in parsed.entries[:3]:
    content = ""
    if getattr(entry, "content", None):
        try:
            content = entry.content[0].value or ""
        except Exception:
            content = ""
    if not content:
        content = getattr(entry, "summary", "") or getattr(entry, "description", "")
    url = entry.get("link", "")
    print(f"  {len(content.split()):4} words | {entry.get('title', '')[:60]}")
    print(f"  {url}")

# Test full fetch on first entry
first_url = parsed.entries[0].get("link", "")
print(f"\nFull fetch test: {first_url}")
try:
    response = httpx.get(first_url, headers=HEADERS, timeout=10, follow_redirects=True)
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
                if words >= 50:
                    print(f"Words: {words}, Selector: {selector}")
                    print(f"Preview: {text[:200]}")
                    break
except Exception as e:
    print(f"FAILED: {e}")
