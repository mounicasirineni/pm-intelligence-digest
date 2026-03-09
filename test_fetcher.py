from __future__ import annotations

from pprint import pprint

from backend.app.services.rss import fetch_items_grouped_by_theme
from backend.app.services.summarizer import summarize_item
from backend.app.services.synthesizer import synthesize_trends


def main() -> None:
    print("Step 1: Fetching raw items from RSS/podcast sources...")
    grouped_raw = fetch_items_grouped_by_theme()

    if not grouped_raw:
        print("No items were fetched from any theme.")
        return

    print("=== Item counts by theme (raw) ===")
    all_themes = sorted(grouped_raw.keys())
    for theme in all_themes:
        items = grouped_raw.get(theme, [])
        print(f"- {theme}: {len(items)} items")

    print("\nStep 2: Running summarizer on first 2 items per theme...")
    grouped_summaries: dict[str, list[dict]] = {}
    for theme in all_themes:
        items = grouped_raw.get(theme, [])
        if not items:
            continue

        grouped_summaries[theme] = []
        for item in items[:2]:
            print(f"  Summarizing [{theme}] '{item.get('title', '<no title>')}'...")
            try:
                summary = summarize_item(item)
            except Exception as exc:
                print(f"    Summarizer failed for this item: {exc}")
                continue

            summarized_item = {
                "title": item.get("title"),
                "url": item.get("url"),
                "source_name": item.get("source_name"),
                "theme": theme,
                "insights": summary.get("insights", []),
                "pm_interview_relevance": summary.get("pm_interview_relevance"),
                "confidence": summary.get("confidence", "medium"),
            }
            grouped_summaries[theme].append(summarized_item)

    print("\nStep 3: Running synthesizer across all summarized items...")
    synthesis = synthesize_trends(grouped_summaries)

    print("\n=== Full synthesis output ===")
    pprint(synthesis)


if __name__ == "__main__":
    main()

