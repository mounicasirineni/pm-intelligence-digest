from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict

from anthropic import Anthropic

from ..config import load_settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an intelligent research assistant for a Senior Product Manager "
    "preparing for interviews at top tech companies. Your job is to extract signal "
    "from content — not just summarize, but identify what actually matters for "
    "someone tracking industry trends, technology direction, company strategy, and "
    "market behavior shifts. "
    "For every insight, ask: would a reader get this from the headline or first paragraph alone? "
    "If yes, it is not an insight — it is a restatement. A genuine insight names a non-obvious implication, "
    "a second-order consequence, a strategic tradeoff, or a pattern that requires reading the full content to surface. "
    "Format each bullet as plain text only. Do not bold, italicize, or use any markdown formatting inside bullet text."
)


def _build_client() -> Anthropic:
    settings = load_settings()
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. "
            "Populate it in your .env file before running the summarizer."
        )
    return Anthropic(api_key=settings.anthropic_api_key)


def _extract_json(text: str) -> str:
    """
    Best-effort extraction of a JSON object from a Claude reply.

    Handles common cases where the model wraps JSON in ```json ... ``` or ``` ... ``` fences.
    """
    if not text:
        return text

    # First look for ```json ... ``` fenced blocks.
    json_fence = re.search(r"```json(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if json_fence:
        return json_fence.group(1).strip()

    # Then look for generic ``` ... ``` fenced blocks.
    generic_fence = re.search(r"```(.*?)```", text, flags=re.DOTALL)
    if generic_fence:
        return generic_fence.group(1).strip()

    return text.strip()


def summarize_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize a single content item using Claude.

    Args:
        item: dict with at least title, url, summary (or full content),
              source_name, and theme.

    Returns:
        {
          "insights": [ "...", ... ],        # 3–5 bullets
          "pm_interview_relevance": "...",   # text explanation
          "pm_relevance_score": "high" | "medium" | "low",  # categorical
          "confidence": "high" | "medium" | "low"
        }
    """
    client = _build_client()

    title = item.get("title") or ""
    url = item.get("url") or ""
    content = item.get("summary") or item.get("content") or ""
    source_name = item.get("source_name") or ""
    theme = item.get("theme") or ""

    user_prompt = f"""
You are analyzing a single content item for a Senior Product Manager.

Content metadata:
- Source: {source_name}
- Theme: {theme}
- Title: {title}
- URL: {url}

Content body:
\"\"\"{content}\"\"\"

Please respond in strict JSON with the following structure:
{{
  "insights": [
    "bullet 1",
    "bullet 2",
    "bullet 3"
  ],
  "pm_interview_relevance": "one line explaining why this matters (or doesn't) for a PM interview",
  "pm_relevance_score": "high" | "medium" | "low",
  "confidence": "high" | "medium" | "low"
}}

Guidance:
- Insights should be 3–5 bullets.
- Go beyond what happened; explain why it matters in terms of product strategy, AI/tech direction, company moves, and market behavior. Each bullet must pass this test: could a reader have written this bullet from the headline and first paragraph alone? If yes, rewrite it. Surface the implication, tradeoff, or pattern that only emerges from reading the full content.
- Avoid generic PM glosses like 'this has implications for product strategy' or 'PMs should pay attention to this trend.' Instead, name the specific implication: what decision does this change, what assumption does it challenge, or what risk does it reveal?
- INSIGHT PRIORITIZATION RULE: When you have more than 3 insight bullets and must choose which to include, rank them in this order:
    (1) Bullets naming a specific product design consequence, architectural decision, or measurable tradeoff — what should a PM build differently, test differently, or price differently as a result of this?
    (2) Bullets naming a strategic pattern or competitive dynamic with a named mechanism.
    (3) Bullets naming a market observation or trend without a concrete action attached.
  A bullet that tells a PM what to build or decide differently outranks a bullet that tells them what is happening. Both are valuable, but the synthesizer will select from your bullets — give it your most actionable ones first. When in doubt, ask: could a PM use this bullet to change a decision in a meeting tomorrow? If yes, it ranks above bullets where the answer is 'it depends' or 'it's a useful frame.'
- "pm_relevance_score" should be a categorical assessment of how useful this item is for PM interview preparation:
    high   = directly relevant to product strategy, company moves, or market shifts a PM interviewer would ask about
    medium = tangentially relevant; useful context but not a likely interview topic
    low    = not relevant to PM interview prep (e.g. sports tech, celebrity news, off-topic content)
- "pm_interview_relevance" should be a one-line text explanation supporting your pm_relevance_score judgment.
- "confidence" should reflect how much genuine signal (vs. noise) you believe this item contains for a Senior PM tracking the space — high means rich, substantive content; low means thin, paywalled, or off-topic.
- If this article is a newsletter containing multiple unrelated stories, focus your analysis exclusively on the lead story — the one reflected in the article title. Ignore secondary stories, roundups, and link digests further in the body.
""".strip()

    settings = load_settings()

    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=800,
        temperature=0.2,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
    )

    # anthropic-python returns content as a list of blocks
    try:
        content_block = response.content[0]
        text = getattr(content_block, "text", None) or content_block.get("text")  # type: ignore[union-attr]
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Unexpected Claude response format: %s", exc)
        raise

    logger.debug("Raw Claude response text: %s", text)
    print("Raw Claude response text from summarizer:")
    print(text)

    cleaned_text = _extract_json(text)

    try:
        parsed = json.loads(cleaned_text)
    except Exception:
        logger.warning("Claude response was not valid JSON, falling back to wrapper.")
        parsed = {
            "raw_text": text,
            "insights": [],
            "pm_interview_relevance": "",
            "pm_relevance_score": "medium",
            "confidence": "medium",
        }

    # Ensure required keys exist with safe defaults.
    insights = parsed.get("insights") or []
    if not isinstance(insights, list):
        insights = [str(insights)]

    pm_interview_relevance = parsed.get("pm_interview_relevance") or ""
    pm_relevance_score = str(parsed.get("pm_relevance_score") or "medium").lower()
    if pm_relevance_score not in {"high", "medium", "low"}:
        pm_relevance_score = "medium"

    confidence = str(parsed.get("confidence") or "medium").lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"

    return {
        "insights": insights,
        "pm_interview_relevance": str(pm_interview_relevance),
        "pm_relevance_score": pm_relevance_score,
        "confidence": confidence,
    }