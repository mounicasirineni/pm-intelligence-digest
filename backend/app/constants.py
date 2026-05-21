from __future__ import annotations

# ---------------------------------------------------------------------------
# Shared string constants
# ---------------------------------------------------------------------------

# Prefix written by fetcher.py when article content came from the og:description
# meta tag rather than the full article body.  summarizer.py detects this prefix
# and caps confidence at "low" regardless of what Claude returns.
#
# IMPORTANT: both fetcher.py and summarizer.py import this constant.  Never
# write this string as a literal in either file — a mismatch between the
# writer and the reader would silently miscalibrate confidence scores.
OG_DESCRIPTION_PREFIX: str = "OG_DESCRIPTION: "
