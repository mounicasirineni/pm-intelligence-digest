import httpx
import feedparser

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; PMDigestBot/1.0; "
        "+https://github.com/mounicasirineni/pm-ai-agents-2026)"
    )
}

# Common RSS URL patterns to try for each base URL
RSS_SUFFIXES = [
    "/rss.xml",
    "/feed.xml",
    "/feed",
    "/rss",
    "/atom.xml",
]

base_urls = [
    "https://www.anthropic.com",
    "https://www.anthropic.com/research",
    "https://www.anthropic.com/engineering",
    "https://www.anthropic.com/news",
]

for base in base_urls:
    print(f"\n--- {base} ---")

    # Try common RSS suffixes
    found = False
    for suffix in RSS_SUFFIXES:
        url = base.rstrip("/") + suffix
        try:
            response = httpx.get(
                url, headers=HEADERS, timeout=5, follow_redirects=True
            )
            if response.status_code == 200:
                parsed = feedparser.parse(response.text)
                if parsed.entries:
                    print(f"FOUND: {url}")
                    print(f"   Title: {parsed.feed.get('title', 'unknown')}")
                    print(f"   Entries: {len(parsed.entries)}")
                    # Check word counts
                    for entry in parsed.entries[:3]:
                        content = ""
                        if getattr(entry, "content", None):
                            try:
                                content = entry.content[0].value or ""
                            except Exception:
                                content = ""
                        if not content:
                            content = getattr(entry, "summary", "") or \
                                      getattr(entry, "description", "")
                        print(f"   {len(content.split()):4} words | "
                              f"{entry.get('title', '')[:60]}")
                    found = True
                    break
        except Exception:
            continue

    if not found:
        # Check if page itself links to RSS
        try:
            response = httpx.get(
                base, headers=HEADERS, timeout=5, follow_redirects=True
            )
            if response.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.text, "html.parser")
                rss_links = soup.find_all(
                    "link",
                    type=lambda t: t and "rss" in t.lower() or
                                   t and "atom" in t.lower()
                ) if soup else []
                if rss_links:
                    for link in rss_links:
                        print(f"RSS link tag found: {link.get('href', '')}")
                else:
                    print(f"No RSS found at {base}")
        except Exception as e:
            print(f"Page fetch failed: {e}")
