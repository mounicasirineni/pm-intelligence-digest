import feedparser
import json
import os

# Adjust path to your actual sources.json location
SOURCES_PATH = "config/sources.json"

with open(SOURCES_PATH) as f:
    config = json.load(f)

sources = config.get("sources", [])
print(f"Total sources configured: {len(sources)}\n")

thin_feeds = []
ok_feeds = []
empty_feeds = []
error_feeds = []

for source in sources:
    source_id = source.get("id", "unknown")
    name = source.get("name", source_id)
    url = source.get("url", "")
    theme = source.get("theme", "unknown")
    source_type = source.get("type", "rss")

    # Skip env: URLs (private podcast feeds)
    if url.startswith("env:"):
        print(f"SKIP (private): {name} [{source_id}]")
        continue

    # Skip non-RSS for now
    if source_type != "rss":
        print(f"SKIP (type={source_type}): {name} [{source_id}]")
        continue

    try:
        parsed = feedparser.parse(url)
        entries = parsed.entries[:5]

        if not entries:
            empty_feeds.append(source_id)
            print(f"EMPTY FEED:  {name} [{source_id}] — no entries returned")
            continue

        word_counts = []
        for entry in entries:
            content = ""
            if getattr(entry, "content", None):
                try:
                    content = entry.content[0].value or ""
                except Exception:
                    content = ""
            if not content:
                content = (
                    getattr(entry, "summary", "")
                    or getattr(entry, "description", "")
                )
            word_counts.append(len(content.split()))

        avg = sum(word_counts) / len(word_counts) if word_counts else 0
        min_wc = min(word_counts)
        max_wc = max(word_counts)
        thin = avg < 50

        status = "THIN_FEED" if thin else "OK"
        if thin:
            thin_feeds.append(source_id)
        else:
            ok_feeds.append(source_id)

        print(
            f"{status:10} {name} [{source_id}] "
            f"| theme={theme} "
            f"| avg={avg:.0f} min={min_wc} max={max_wc} words"
        )

    except Exception as e:
        error_feeds.append(source_id)
        print(f"ERROR:      {name} [{source_id}] — {e}")

print(f"\n--- Summary ---")
print(f"Total scanned: {len(sources)}")
print(f"Thin feeds (avg <50 words): {len(thin_feeds)}")
print(f"  {thin_feeds}")
print(f"OK feeds: {len(ok_feeds)}")
print(f"Empty feeds: {len(empty_feeds)}")
print(f"  {empty_feeds}")
print(f"Errors: {len(error_feeds)}")
print(f"  {error_feeds}")
