from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    app_env: str
    host: str
    port: int
    database_path: Path
    sources_config_path: Path
    anthropic_api_key: str | None
    claude_model: str
    lookback_hours: int
    digest_schedule_hour: int
    digest_schedule_minute: int
    digest_timezone: str


_settings: Settings | None = None


def load_settings() -> Settings:
    global _settings
    if _settings is not None:
        return _settings

    load_dotenv()

    _settings = Settings(
        app_env=os.getenv("APP_ENV", "dev"),
        host=os.getenv("APP_HOST", "127.0.0.1"),
        port=int(os.getenv("APP_PORT", "8000")),
        database_path=Path(os.getenv("DATABASE_PATH", "data/digest.sqlite3")),
        sources_config_path=Path(os.getenv("SOURCES_CONFIG_PATH", "config/sources.json")),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        claude_model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5"),
        lookback_hours=int(os.getenv("LOOKBACK_HOURS", "24")),
        digest_schedule_hour=int(os.getenv("DIGEST_SCHEDULE_HOUR", "7")),
        digest_schedule_minute=int(os.getenv("DIGEST_SCHEDULE_MINUTE", "0")),
        digest_timezone=os.getenv("DIGEST_TIMEZONE", "Asia/Kolkata"),
    )
    return _settings


def load_sources_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Sources config not found at {path}. "
            f"Copy config/sources.example.json → {path} and edit it."
        )
    raw = path.read_text(encoding="utf-8")
    return json.loads(raw)

