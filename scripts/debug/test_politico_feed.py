import feedparser

url = "https://rss.politico.com/technology.xml"
parsed = feedparser.parse(url)

print(f"Status: {parsed.get('status', 'unknown')}")
print(f"Feed title: {parsed.feed.get('title', 'unknown')}")
print(f"Entries found: {len(parsed.entries)}")
print(f"Bozo: {parsed.bozo}")
if parsed.bozo:
    print(f"Bozo exception: {parsed.bozo_exception}")

for entry in parsed.entries[:3]:
    print(f"\nTitle: {entry.get('title', '')}")
    print(f"URL: {entry.get('link', '')}")
    published = entry.get('published', 'no date')
    print(f"Published: {published}")
