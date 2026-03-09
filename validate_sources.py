from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import feedparser
import httpx

from backend.app.config import load_settings, load_sources_config


def _resolve_url(source: Dict[str, Any]) -> tuple[str | None, str | None]:
    """
    Resolve the effective URL for a source.

    Returns (url, skip_reason). If url is None and skip_reason is not, the caller
    should treat the source as skipped.
    """
    raw_url = source.get("url") or ""
    source_type = source.get("type")

    if source_type == "podcast" and raw_url.startswith("env:"):
        var_name = raw_url.split(":", 1)[1]
        value = os.getenv(var_name)
        if not value:
            return None, f"environment variable {var_name} not set"
        return value, None

    return raw_url, None


def main() -> None:
    settings = load_settings()
    config_path: Path = settings.sources_config_path
    cfg = load_sources_config(config_path)
    sources = cfg.get("sources", [])

    passed = 0
    failed = 0
    skipped = 0

    print(f"Validating {len(sources)} sources from {config_path}...\n")

    for source in sources:
        name = source.get("name", "<unnamed>")
        src_type = source.get("type", "")

        url, skip_reason = _resolve_url(source)
        if skip_reason:
            skipped += 1
            print(f"SKIP  {name:30} — {skip_reason}")
            continue

        if not url:
            failed += 1
            print(f"FAIL  {name:30} — missing URL")
            continue

        try:
            # Follow redirects and keep a short timeout so dead feeds don't hang.
            with httpx.Client(timeout=2.0, follow_redirects=True) as client:
                resp = client.get(url)
                parsed = feedparser.parse(resp.text)
        except Exception as exc:
            failed += 1
            print(f"FAIL  {name:30} — request error: {exc}")
            continue

        status = getattr(parsed, "status", getattr(resp, "status_code", None))
        entry_count = len(parsed.entries or [])

        if entry_count > 0 or status in {200, 301, 302}:
            passed += 1
            print(f"PASS  {name:30} — {entry_count} entries (status={status})")
        else:
            failed += 1
            print(f"FAIL  {name:30} — 0 entries and status={status} (possible dead or empty feed)")

    print("\nSummary:")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {skipped}")


if __name__ == "__main__":
    main()

