from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date


@dataclass(frozen=True)
class Source:
    id: str
    type: str
    name: str
    url: str


@dataclass(frozen=True)
class ContentItem:
    source_id: str
    external_id: str
    title: str
    url: str
    published_at: datetime | None
    raw_text: str


@dataclass(frozen=True)
class ItemSummary:
    source_id: str
    external_id: str
    title: str
    url: str
    published_at: datetime | None
    summary: str


@dataclass(frozen=True)
class DailyDigest:
    digest_date: date
    synthesized_insights_md: str
    items: list[ItemSummary]

