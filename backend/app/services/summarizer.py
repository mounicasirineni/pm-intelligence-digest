from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from ..config import load_settings
from .client import get_client
from ..constants import OG_DESCRIPTION_PREFIX

logger = logging.getLogger(__name__)

# Hard skip — don't hit the API below this word count.
MINIMUM_CONTENT_WORDS = 200

# Articles below this word count get confidence capped at "medium" in code,
# regardless of what Claude returns.
CONFIDENCE_FLOOR_WORDS = 400

# ---------------------------------------------------------------------------
# pm_interview_relevance derived in code — no LLM call needed
# ---------------------------------------------------------------------------
_PM_RELEVANCE_DESCRIPTIONS = {
    "high":   "Directly relevant to product strategy, company moves, or market shifts a PM interviewer would ask about.",
    "medium": "Tangentially relevant — useful context but not a likely interview topic.",
    "low":    "Not relevant to PM interview preparation.",
}

# ---------------------------------------------------------------------------
# Call A — Extract (Sonnet)
# ---------------------------------------------------------------------------

_CALL_A_SYSTEM = (
    "You are an intelligent research assistant for a Senior Product Manager. "
    "Your job is to extract signal from content — not just summarize, but identify what actually matters. "
    "For every insight, ask: would a reader get this from the headline or first paragraph alone? "
    "If yes, it is not an insight — it is a restatement. A genuine insight names a non-obvious implication, "
    "a second-order consequence, a strategic tradeoff, or a pattern that requires reading the full content to surface. "
    "Format each bullet as plain text only. No bold, italics, or markdown inside bullet text.\n\n"

    "ORDERING: Order bullets from most specific (named mechanisms, products, concrete tradeoffs, "
    "verifiable numbers) to most abstract (patterns, strategic observations). "
    "Rank in this order: (1) specific product design consequence or measurable tradeoff, "
    "(2) strategic pattern with a named mechanism, (3) market observation without a concrete action.\n\n"

    "CONTRADICTION MANDATE: If the content contains a fact, data point, or claim that contradicts, "
    "qualifies, or significantly complicates the article's central claim, you MUST include it as a bullet. "
    "A qualifying fact that changes the conclusion a reader would draw is as important as an explicit contradiction.\n\n"

    "SOURCE FIDELITY: Every bullet must trace to a specific claim, data point, or quote in the content body. "
    "Do not introduce named entities not in the content. Do not assert specific numbers not in the content. "
    "Do not assign strategic motivations a source does not explicitly state. "
    "Match source hedge levels exactly — if source says 'suggests', write 'suggests', not 'demonstrates'. "
    "Scope claims to actual examples — a single company's move does not establish a universal pattern. "
    "Do not assert why a company did something unless the source explicitly states it.\n\n"

    "EDGE CASES: "
    "If content is cadence-driven (weekly roundup, event listing, award announcement) with no strategic "
    "mechanism revealed, return 1 bullet naming what it is and why it lacks signal. "
    "If content is a newsletter or roundup with multiple distinct stories, extract at least one bullet "
    "per PM-relevant story — do not limit to the lead story."
)



def _extract_json(text: str) -> str:
    """Best-effort extraction of a JSON object from a Claude reply."""
    if not text:
        return text
    # Strip complete blocks first
    text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL).strip()
    # Strip unclosed blocks (truncated responses — no </reasoning> present)
    text = re.sub(r"<reasoning>.*", "", text, flags=re.DOTALL).strip()
    # Try ```json ... ``` fence
    json_fence = re.search(r"```json(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if json_fence:
        return json_fence.group(1).strip()
    # Try generic ``` ... ``` fence
    generic_fence = re.search(r"```(.*?)```", text, flags=re.DOTALL)
    if generic_fence:
        return generic_fence.group(1).strip()
    # Find outermost { } or [ ] — whichever comes first
    brace_start = text.find("{")
    bracket_start = text.find("[")
    if brace_start == -1 and bracket_start == -1:
        return text.strip()
    if brace_start == -1:
        start, end_char = bracket_start, "]"
    elif bracket_start == -1:
        start, end_char = brace_start, "}"
    else:
        start, end_char = (brace_start, "}") if brace_start < bracket_start else (bracket_start, "]")
    end = text.rfind(end_char)
    if end != -1 and end > start:
        return text[start:end + 1]
    return text.strip()


def _build_call_a_prompt(
    source_name: str,
    theme: str,
    title: str,
    url: str,
    content: str,
) -> str:
    return f"""Source: {source_name}
Theme: {theme}
Title: {title}
URL: {url}

Content:
\"\"\"{content}\"\"\"

Extract 3-5 insight bullets from this content.

Return strict JSON: {{"insights": ["bullet 1", "bullet 2", ...]}}""".strip()


def _call_extract(
    client: Anthropic,
    settings: Any,
    content: str,
    content_word_count: int,
    title: str,
    source_name: str,
    theme: str,
    url: str,
) -> List[str]:
    """Call A: extract insight bullets from article content. Returns insights list."""
    max_tokens = 800

    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=max_tokens,
        temperature=0,
        system=_CALL_A_SYSTEM,
        messages=[{
            "role": "user",
            "content": _build_call_a_prompt(source_name, theme, title, url, content),
        }],
    )

    stop_reason = getattr(response, "stop_reason", None)

    if stop_reason == "max_tokens":
        logger.warning(
            "Call A truncated at max_tokens for '%s' (words=%d, max_tokens=%d)",
            title, content_word_count, max_tokens,
        )

    if stop_reason == "refusal":
        logger.warning("Call A refusal for '%s'", title)
        return []

    if not response.content:
        return []

    block = response.content[0]
    text = getattr(block, "text", None) or block.get("text")  # type: ignore[union-attr]
    cleaned = _extract_json(text or "")

    try:
        parsed = json.loads(cleaned)
        insights = parsed.get("insights") or []
        if not isinstance(insights, list):
            insights = [str(insights)]
        return [str(b) for b in insights if b]
    except Exception:
        logger.warning(
            "Call A JSON parse failed for '%s'. Raw (first 300): %s",
            title, (text or "")[:300],
        )
        return []


# ---------------------------------------------------------------------------
# Call B — Confidence (Haiku)
# ---------------------------------------------------------------------------

_CALL_B_SYSTEM = (
    "You are a content quality classifier for a PM intelligence digest. "
    "Assess how well a set of insight bullets is grounded in verifiable source content. "
    "Be skeptical — a source must earn high confidence by providing specific, traceable claims."
)


def _build_call_b_prompt(
    source_name: str,
    title: str,
    theme: str,
    content_word_count: int,
    is_og_fallback: bool,
    insights: List[str],
) -> str:
    insights_block = "\n".join(f"  - {b}" for b in insights) if insights else "  (none extracted)"
    og_note = "\nNote: content was sourced from og:description fallback — likely very thin." if is_og_fallback else ""

    return f"""
Source: {source_name}
Title: {title}
Theme: {theme}
Content word count: {content_word_count}{og_note}

Extracted bullets:
{insights_block}

Classify CONFIDENCE — how well does the source support producing accurate, grounded bullets?

high   = bullets name specific mechanisms, products, numbers, or quotes traceable to
         substantive source content (300+ words of original reporting or analysis).
         Bullets are specific and would require reading the full content to write.
medium = bullets are plausible but some inference required. Source was thin (200-300 words),
         partially paywalled, or a press release stub. 2-3 bullets grounded, rest inferred.
low    = bullets are primarily inference. Source was boilerplate, a stub, or a cadence post
         with no original analysis. Any bullet could have been written from the headline alone.

Return strict JSON: {{"confidence": "high|medium|low"}}
""".strip()


def _call_confidence(
    client: Anthropic,
    settings: Any,
    insights: List[str],
    title: str,
    source_name: str,
    theme: str,
    content_word_count: int,
    is_og_fallback: bool,
) -> str:
    """Call B (Haiku): classify confidence from bullets. Returns 'high'|'medium'|'low'."""

    if is_og_fallback:
        logger.info("Call B skipped for '%s' — og:description fallback → low", title)
        return "low"

    if content_word_count < CONFIDENCE_FLOOR_WORDS and not insights:
        return "low"

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=64,
        temperature=0,
        system=_CALL_B_SYSTEM,
        messages=[{
            "role": "user",
            "content": _build_call_b_prompt(
                source_name, title, theme,
                content_word_count, is_og_fallback, insights,
            ),
        }],
    )

    block = response.content[0]
    text = getattr(block, "text", None) or block.get("text")  # type: ignore[union-attr]
    try:
        parsed = json.loads(_extract_json(text or ""))
        confidence = str(parsed.get("confidence") or "medium").lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "medium"
        if content_word_count < CONFIDENCE_FLOOR_WORDS and confidence == "high":
            logger.info(
                "Call B: overriding 'high' → 'medium' for '%s' — only %d words",
                title, content_word_count,
            )
            confidence = "medium"
        return confidence
    except Exception:
        logger.warning("Call B parse failed for '%s' — defaulting to medium", title)
        return "medium"


# ---------------------------------------------------------------------------
# Call C — PM Relevance (Haiku)
# ---------------------------------------------------------------------------

_CALL_C_SYSTEM = (
    "You are a PM relevance classifier for a product management intelligence digest. "
    "Assess whether a set of insight bullets would be useful to a Senior PM "
    "doing their job — making product strategy decisions, tracking competitors, "
    "understanding how user behavior is evolving, monitoring regulatory constraints, "
    "and staying current on technology trends. "
    "The digest serves working PMs, not just interview candidates."
)


def _build_call_c_prompt(
    source_name: str,
    title: str,
    theme: str,
    insights: List[str],
) -> str:
    insights_block = "\n".join(f"  - {b}" for b in insights) if insights else "  (none)"

    return f"""
Source: {source_name}
Title: {title}
Theme: {theme}

Insight bullets:
{insights_block}

Classify PM_RELEVANCE_SCORE:

high   = bullets directly name product strategy decisions, competitive moves, architectural
         tradeoffs, market shifts, regulatory constraints, technology unlocks, or user
         behavior changes a working PM needs to track. A PM could read these and immediately
         update their product thinking, competitive awareness, or roadmap priorities.
medium = bullets provide useful industry context but are not immediately actionable.
         Good background signal — relevant to a PM's domain but not foreground decision input.
low    = bullets are not relevant to product management work.

DOMAIN FILTER RULE: Score low if bullets are primarily about:
  - Foreign policy, military operations, or armed conflict
  - Domestic party politics or electoral outcomes
  - Celebrity news or entertainment gossip
  - Sports outcomes or athlete news
  A PM analogy requiring translation through two or more conceptual layers does not qualify.
  Example: naval operations → negotiation strategy → compliance framing = low, not medium.

ROUTINE UPDATE RULE: Score low if bullets describe routine operational activity with no
strategic mechanism revealed — weekly content additions, cadence posts, minor feature
releases with no architectural significance, event listings, award announcements.
The company name alone does not determine the score — the strategic signal does.

GLOBAL SIGNAL RULE: Market shifts, regulatory changes, user behavior trends, and startup
activity in non-US markets (EU, Asia, Latin America, Africa) are valid high or medium signals
if they reveal patterns relevant to product strategy — do not score low purely due to geography.

Return strict JSON: {{"pm_relevance_score": "high|medium|low"}}
""".strip()


def _call_pm_relevance(
    client: Anthropic,
    settings: Any,
    insights: List[str],
    title: str,
    source_name: str,
    theme: str,
) -> str:
    """Call C (Haiku): classify PM relevance from bullets. Returns 'high'|'medium'|'low'."""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=64,
        temperature=0,
        system=_CALL_C_SYSTEM,
        messages=[{
            "role": "user",
            "content": _build_call_c_prompt(source_name, title, theme, insights),
        }],
    )

    block = response.content[0]
    text = getattr(block, "text", None) or block.get("text")  # type: ignore[union-attr]
    try:
        parsed = json.loads(_extract_json(text or ""))
        score = str(parsed.get("pm_relevance_score") or "medium").lower()
        if score not in {"high", "medium", "low"}:
            score = "medium"
        return score
    except Exception:
        logger.warning("Call C parse failed for '%s' — defaulting to medium", title)
        return "medium"


def _low_result(content_word_count: int, is_og_fallback: bool, reason: str = "") -> Dict[str, Any]:
    """Return a low-score result dict with consistent shape."""
    return {
        "insights": [],
        "pm_relevance_score": "low",
        "pm_interview_relevance": _PM_RELEVANCE_DESCRIPTIONS["low"],
        "confidence": "low",
        "content_word_count": content_word_count,
        "is_og_fallback": is_og_fallback,
        "_skip_reason": reason,
    }


def summarize_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize a single content item using a 3-call pipeline.

    Call A (Sonnet)  — extract insight bullets from article content
    Call B (Haiku)   — classify confidence (source quality)
    Call C (Haiku)   — classify PM relevance (only if confidence is high/medium)

    Short-circuit: low confidence → skip Call C, return immediately.
    """
    title = item.get("title") or ""
    url = item.get("url") or ""
    source_name = item.get("source_name") or ""
    theme = item.get("theme") or ""
    raw_content = item.get("summary") or item.get("content") or ""

    is_og_fallback = raw_content.startswith(OG_DESCRIPTION_PREFIX)
    content = raw_content[len(OG_DESCRIPTION_PREFIX):] if is_og_fallback else raw_content
    content_word_count = len(content.split())

    if is_og_fallback:
        logger.info(
            "Content for '%s' sourced from og:description fallback — "
            "confidence will be capped at low regardless of Claude output",
            title,
        )

    logger.info(
        "SUMMARIZER [%s]: %d words | og_fallback=%s | url=%s",
        title[:60], content_word_count, is_og_fallback, url,
    )

    if content_word_count < MINIMUM_CONTENT_WORDS:
        logger.info(
            "SUMMARIZER [%s]: hard skip — %d words < minimum %d",
            title[:60], content_word_count, MINIMUM_CONTENT_WORDS,
        )
        return _low_result(content_word_count, is_og_fallback, reason="content_too_short")

    settings = load_settings()
    client = get_client()

    insights = _call_extract(
        client, settings, content, content_word_count,
        title, source_name, theme, url,
    )

    if not insights:
        rss_fallback = item.get("rss_summary") or ""
        if (
            rss_fallback
            and len(rss_fallback.split()) >= MINIMUM_CONTENT_WORDS
            and not item.get("_refusal_retry")
        ):
            logger.warning("Call A returned no insights for '%s' — retrying with RSS summary", title)
            fallback_item = {**item, "summary": rss_fallback, "_refusal_retry": True}
            return summarize_item(fallback_item)

        logger.warning("Call A returned no insights for '%s' — skipping", title)
        return _low_result(content_word_count, is_og_fallback, reason="no_insights_extracted")

    confidence = _call_confidence(
        client, settings, insights, title, source_name,
        theme, content_word_count, is_og_fallback,
    )

    logger.info("SUMMARIZER [%s]: confidence=%s", title[:60], confidence)

    if confidence == "low":
        return {
            "insights": insights,
            "pm_relevance_score": "low",
            "pm_interview_relevance": _PM_RELEVANCE_DESCRIPTIONS["low"],
            "confidence": "low",
            "content_word_count": content_word_count,
            "is_og_fallback": is_og_fallback,
        }

    pm_relevance_score = _call_pm_relevance(
        client, settings, insights, title, source_name, theme,
    )

    logger.info(
        "SUMMARIZER OUTPUT [%s]: confidence=%s pm_relevance=%s words=%d",
        title[:60], confidence, pm_relevance_score, content_word_count,
    )

    return {
        "insights": insights,
        "pm_relevance_score": pm_relevance_score,
        "pm_interview_relevance": _PM_RELEVANCE_DESCRIPTIONS.get(pm_relevance_score, ""),
        "confidence": confidence,
        "content_word_count": content_word_count,
        "is_og_fallback": is_og_fallback,
    }
