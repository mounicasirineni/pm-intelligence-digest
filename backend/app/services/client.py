from __future__ import annotations

from anthropic import Anthropic

from ..config import load_settings

_client: Anthropic | None = None


def get_client() -> Anthropic:
    """Return the shared Anthropic client, initializing it on first call."""
    global _client
    if _client is not None:
        return _client

    settings = load_settings()
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. "
            "Populate it in your .env file before running."
        )
    _client = Anthropic(api_key=settings.anthropic_api_key)
    return _client
