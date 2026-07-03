from __future__ import annotations

import json
import logging
import re
import traceback
import uuid
from datetime import date
from typing import Any, Dict, List, Optional, Set, Tuple

from anthropic import Anthropic

from ..config import load_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Theme routing constants
# ---------------------------------------------------------------------------
WHATS_SHIFTING_THEMES = {
    "technology_trends",
    "market_signals",
    "user_behavior",
    "regulation_policy",
}

DEDICATED_SECTION_THEMES = {
    "company_strategy",
    "market_signals",
    "pm_craft",
}

HAIKU_MODEL = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# System prompt — shared across all calls
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a senior intelligence analyst briefing a Senior Product Manager. "
    "Your job is to reason across multiple sources and surface what is actually "
    "shifting in the industry — not what happened, but what it means and what "
    "patterns are emerging. "
    "For every insight, ask: what would a reader NOT get from reading any single source? "
    "A good insight names the underlying force driving multiple seemingly unrelated events, "
    "challenges a conventional assumption, or identifies a second-order consequence that "
    "practitioners haven't yet articulated. Avoid insights that merely restate a trend with a PM gloss. "
    "PM ACTIONABILITY STANDARD: Across all sections, when choosing between a strategic observation "
    "and a concrete product design consequence, always prefer the latter. "
    "A specific mechanical implication that tells a PM what decision to make, what assumption to test, "
    "or what design pattern to apply is always stronger than a generalizable pattern observation. "
    "Test every closing implication sentence on two dimensions: (1) Is it traceable to a specific "
    "bullet in a cited source? (2) Does it match the hedge level of that source? A closing sentence "
    "that asserts certainty where the source only suggests possibility is an inference boundary "
    "violation regardless of how actionable it sounds. Prefer a hedged specific consequence over a "
    "confident general one. "
    "The broad observation is usually derivable from the headline. "
    "The specific mechanical consequence requires reading the full content. Keep the latter."
)

# ---------------------------------------------------------------------------
# System prompt constants — registered at startup for version tracking.
# Call 1b and Call 1 fill share SYSTEM_PROMPT (above).
# ---------------------------------------------------------------------------

_CALL_1A_SYSTEM = (
    "You are a content classifier for a PM intelligence digest. "
    "Classify items as cross-market or company-specific based on their central claim. "
    "Be conservative — when in doubt, classify as company-specific."
)

_CALL_2_SYSTEM = (
    "You are writing a PM Craft insight for a Senior PM digest. "
    "Your job is to surface the single most actionable, non-obvious practitioner insight "
    "from product_craft and design_ux sources. Prefer specific reframes over general advice."
)

_CALL_3_SYSTEM = SYSTEM_PROMPT  # Company Watch uses the shared synthesizer system prompt

_CALL_4A_SYSTEM = (
    "You are a company maturity classifier for a PM intelligence digest. "
    "Classify companies as startup or established based on their market position. "
    "Be conservative — when in doubt about a well-known name, classify as established."
)

_CALL_4B_SYSTEM = SYSTEM_PROMPT  # Startup Radar uses the shared synthesizer system prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_client() -> Anthropic:
    settings = load_settings()
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. "
            "Populate it in your .env file before running the synthesizer."
        )
    return Anthropic(api_key=settings.anthropic_api_key)


def _extract_json(text: str) -> str:
    if not text:
        return text
    json_fence = re.search(r"```json(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if json_fence:
        return json_fence.group(1).strip()
    generic_fence = re.search(r"```(.*?)```", text, flags=re.DOTALL)
    if generic_fence:
        return generic_fence.group(1).strip()
    return text.strip()


def _extract_reasoning_block(text: str) -> Tuple[str, str]:
    if not text:
        return "", text
    match = re.search(r"<reasoning>(.*?)</reasoning>", text, flags=re.DOTALL)
    if match:
        reasoning = match.group(1).strip()
        remaining = text[:match.start()] + text[match.end():]
        return reasoning, remaining.strip()
    return "", text


def _strip_date_check_flags(text: str) -> str:
    if not text:
        return text
    return re.sub(r"\[DATE CHECK:[^\]]*\]", "", text).strip()


def _build_context_block(
    items: List[Dict[str, Any]],
    start_idx: int = 1,
) -> Tuple[str, List[Dict[str, Any]], int]:
    lines: List[str] = []
    indexed_items: List[Dict[str, Any]] = []
    idx = start_idx

    for item in items:
        insights = item["insights"]
        if not isinstance(insights, list):
            insights = [str(insights)]

        theme = item["theme"]
        if theme == "company_strategy":
            allowed_section = "company_watch ONLY"
        elif theme == "pm_craft":
            allowed_section = "pm_craft_today ONLY"
        else:
            allowed_section = "whats_shifting eligible"

        company_id = item.get("company_id")

        lines.append(f"Item [{idx}]:")
        lines.append(f"- Theme: {item['theme']}")
        lines.append(f"- Allowed section: {allowed_section}")
        if company_id:
            lines.append(f"- Company: {company_id}")
        lines.append(f"- Source: {item['source_name']}")
        lines.append(f"- Title: {item['title']}")
        lines.append("- Insights:")
        for bullet in insights:
            lines.append(f"  - {bullet}")
        lines.append("")

        indexed_items.append({
            "index": idx,
            "item_id": item.get("item_id"),
            "theme": item["theme"],
            "title": item["title"],
            "source_name": item["source_name"],
            "company_id": company_id,
        })
        idx += 1

    return "\n".join(lines), indexed_items, idx


def _normalize_whats_shifting(raw: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    if not isinstance(raw, list):
        return [{"headline": "", "paragraph": str(raw), "source_indices": []}]
    for entry in raw:
        if isinstance(entry, dict):
            headline = str(entry.get("headline") or "")
            paragraph = entry.get("paragraph") or entry.get("text") or ""
            indices = entry.get("source_indices") or entry.get("sources") or []
        else:
            headline = ""
            paragraph = str(entry)
            indices = []
        if not isinstance(indices, list):
            indices = [indices]
        cleaned: List[int] = []
        for i in indices:
            try:
                cleaned.append(int(i))
            except Exception:
                continue
        headline = _strip_date_check_flags(headline)
        paragraph = _strip_date_check_flags(paragraph)

        # Hard-cap headline to 20 words regardless of what Claude produced.
        headline_words = headline.split()
        if len(headline_words) > 20:
            logger.info(
                "HEADLINE TRUNCATED: %d words → 20 for paragraph starting '%s'",
                len(headline_words),
                paragraph[:60],
            )
            headline = " ".join(headline_words[:20]) + "…"

        normalized.append({
            "headline": headline,
            "paragraph": paragraph,
            "source_indices": cleaned,
            "declared_theme": _strip_date_check_flags(str(entry.get("theme") or "")) if isinstance(entry, dict) else "",
        })
    return normalized


def _normalize_company_watch(raw: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(raw, dict):
        return {}
    normalized: Dict[str, Dict[str, Any]] = {}
    for company, value in raw.items():
        if value in (None, "", {}):
            continue
        if isinstance(value, dict):
            paragraph = value.get("paragraph") or value.get("text") or ""
            indices = value.get("source_indices") or value.get("sources") or []
        else:
            paragraph = str(value)
            indices = []
        if not paragraph:
            continue
        if not isinstance(indices, list):
            indices = [indices]
        cleaned: List[int] = []
        for i in indices:
            try:
                cleaned.append(int(i))
            except Exception:
                continue
        paragraph = _strip_date_check_flags(paragraph)
        normalized[company] = {"paragraph": paragraph, "source_indices": cleaned}
    return normalized


def _normalize_startup_radar(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        raw = [str(raw)]
    normalized = []
    for entry in raw:
        if isinstance(entry, dict):
            bullet = entry.get("bullet") or entry.get("text") or str(entry)
            indices = entry.get("source_indices") or []
        else:
            bullet = str(entry)
            indices = []
        if not isinstance(indices, list):
            indices = [indices]
        cleaned: List[int] = []
        for i in indices:
            try:
                cleaned.append(int(i))
            except Exception:
                continue
        bullet = _strip_date_check_flags(bullet)
        normalized.append({"bullet": bullet, "source_indices": cleaned})
    return normalized


def _normalize_pm_craft(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        text = str(raw.get("text") or raw.get("pm_craft_today") or "")
        indices = raw.get("source_indices") or []
    else:
        text = str(raw)
        indices = []
    if not isinstance(indices, list):
        indices = [indices]
    cleaned: List[int] = []
    for i in indices:
        try:
            cleaned.append(int(i))
        except Exception:
            continue
    text = _strip_date_check_flags(text)
    return {"text": text, "source_indices": cleaned}


def _get_theme_for_ws(
    ws: Dict[str, Any],
    ws_indexed: List[Dict[str, Any]],
    source_index_lookup: Dict[str, Dict[str, Any]],
) -> str:
    """
    Identify the theme a WS paragraph belongs to.

    FIX: Check _fill_theme first — fill results carry this key explicitly.
    Previously, fill results with empty source_indices fell through to
    source_index_lookup, got theme="" and were classified as "unknown",
    causing false coverage-gap alarms even when the fill succeeded.
    """
    # 1. Model self-declared theme — highest fidelity signal.
    if ws.get("declared_theme"):
        return ws["declared_theme"]

    # 2. Fill results carry _fill_theme explicitly.
    if ws.get("_fill_theme"):
        return ws["_fill_theme"]

    indices = ws.get("source_indices", [])
    if indices:
        info = source_index_lookup.get(str(indices[0]), {})
        theme = info.get("theme", "")
        if theme:
            return theme
    return "unknown"


def _get_item_score(
    source_index: Optional[int],
    ws_items: List[Dict[str, Any]],
    ws_indexed: List[Dict[str, Any]],
) -> Tuple[int, int]:
    if source_index is None:
        return (1, 1)
    item_id = None
    for entry in ws_indexed:
        if entry["index"] == source_index:
            item_id = entry.get("item_id")
            break
    if not item_id:
        return (1, 1)
    for item in ws_items:
        if item.get("item_id") == item_id:
            rel = 0 if item.get("pm_relevance_score") == "high" else 1
            conf = 0 if item.get("confidence") == "high" else 1
            return (rel, conf)
    return (1, 1)


def _get_covered_themes(
    ws_paragraphs: List[Dict[str, Any]],
    ws_indexed: List[Dict[str, Any]],
    source_index_lookup: Dict[str, Dict[str, Any]],
) -> Set[str]:
    covered = set()
    for ws in ws_paragraphs:
        if not ws.get("paragraph", "").strip():
            continue
        theme = _get_theme_for_ws(ws, ws_indexed, source_index_lookup)
        if theme and theme != "unknown":
            covered.add(theme)
    return covered


def _deduplicate_by_theme(
    ws_paragraphs: List[Dict[str, Any]],
    ws_items: List[Dict[str, Any]],
    ws_indexed: List[Dict[str, Any]],
    source_index_lookup: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    seen_themes: Dict[str, Dict[str, Any]] = {}

    for ws in ws_paragraphs:
        if not ws.get("paragraph", "").strip():
            continue
        theme = _get_theme_for_ws(ws, ws_indexed, source_index_lookup)

        if theme not in seen_themes:
            seen_themes[theme] = ws
        else:
            existing_idx = seen_themes[theme]["source_indices"][0] if seen_themes[theme]["source_indices"] else None
            new_idx = ws["source_indices"][0] if ws["source_indices"] else None
            existing_score = _get_item_score(existing_idx, ws_items, ws_indexed)
            new_score = _get_item_score(new_idx, ws_items, ws_indexed)

            if new_score < existing_score:
                logger.info(
                    "THEME DEDUP: Replaced paragraph for theme '%s' — "
                    "new anchor scored %s vs existing %s",
                    theme, new_score, existing_score,
                )
                seen_themes[theme] = ws
            else:
                logger.info(
                    "THEME DEDUP: Kept existing paragraph for theme '%s' — "
                    "existing anchor scored %s vs new %s",
                    theme, existing_score, new_score,
                )

    return list(seen_themes.values())


# ---------------------------------------------------------------------------
# Call 1: What's Shifting + Interview Angle
# ---------------------------------------------------------------------------

def _call_whats_shifting(
    client: Anthropic,
    settings: Any,
    ws_items: List[Dict[str, Any]],
    today: str,
    ws_theme_distribution: Dict[str, int] | None = None,
    required_anchors: List[Dict[str, Any]] | None = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    context_block, ws_indexed, _ = _build_context_block(ws_items, start_idx=1)

    required_anchors_block = ""
    if required_anchors:
        anchor_lines = []
        for a in required_anchors:
            anchor_lines.append(f"  - {a['theme']}: \"{a['anchor_item']['title']}\"")
        required_anchors_block = (
            "REQUIRED PARAGRAPH ANCHORS:\n"
            "You must produce exactly one paragraph per theme listed below.\n"
            "Each paragraph must cover exactly one theme.\n"
            "Do not produce more paragraphs than anchors listed.\n"
            "Do not combine themes across paragraphs.\n"
            "For each anchor, build the strongest possible paragraph from ALL items "
            "in the pool that share that theme — the anchor item is your starting point, "
            "not your only source. Pull from multiple bullets across same-theme items "
            "to build a unified insight. Do not limit yourself to the anchor item alone.\n\n"
            "Anchors:\n"
            + "\n".join(anchor_lines)
            + "\n"
        )

    theme_availability_lines = []
    for theme, count in sorted((ws_theme_distribution or {}).items(), key=lambda x: -x[1]):
        theme_availability_lines.append(f"  - {theme}: {count} item{'s' if count != 1 else ''}")
    theme_availability_block = (
        "Available items by theme in today's pool (for reference):\n"
        + "\n".join(theme_availability_lines)
        + "\n"
    ) if theme_availability_lines else ""

    user_prompt = f"""
You are reasoning across multiple high/medium confidence items that a Senior PM is tracking.
Today's date is {today}.

{required_anchors_block}{theme_availability_block}
WHAT'S SHIFTING CONTENT BOUNDARY:
What's Shifting covers structural shifts in markets, technology landscapes, or regulatory
environments — forces that are changing the conditions under which products are built or
compete. It does NOT cover practitioner process advice, sprint methodology, documentation
practices, or product craft frameworks. The test: does this bullet describe a change in
the world, or a change in what a PM should do? The former belongs here. The latter does not.

Items:
{context_block}

First, write your anchor selection reasoning inside <reasoning>...</reasoning> tags.
For each required anchor theme:
(1) List ALL insight bullets across every source eligible for that theme, ranked by non-obviousness.
(2) Name the highest-ranked bullet and declare it the closing implication anchor. The closing sentence of the paragraph must be traceable to this bullet specifically — not to a synthesizer-constructed bridge across bullets, and not to a lower-ranked bullet. If you find yourself writing a closing sentence that does not trace to this bullet, return to the ranked list and select a different anchor.
(3) List every bullet from the cited sources that contradicts, qualifies, or limits the anchor's claim. For each one, decide: incorporate it into the paragraph, or explicitly acknowledge it as a scope limitation. Omission is not an option — if a qualifying bullet would change the conclusion a reader draws, it must appear in the paragraph or the closing implication must be scoped to exclude it.
(4) If you are combining items from more than one source into this paragraph, name the specific causal mechanism that justifies the combination. If you cannot name a mechanism beyond a shared category label, do not combine — anchor to the single strongest source instead.

Then produce a JSON object with this exact structure:
{{
  "whats_shifting": [
    {{
      "headline": "One declarative sentence, maximum 15 words, naming the underlying structural force or pattern. Not an event description. Renders as the visible collapsed card headline — scannable and self-contained.",
      "paragraph": "Open by restating the headline claim with one additional clause of context. Develop across 3-4 sentences. If drawing from a single source, build depth by incorporating its strongest and most complicating bullets. If drawing from multiple sources, only combine them if you can name a specific causal mechanism that connects them — one that neither source states alone. A shared category label ('both are about AI costs') is not a mechanism. A shared causal chain ('both reveal that X causes Y through mechanism Z') is. When in doubt, anchor to one source's strongest bullet and build depth rather than breadth. Close with one PM implication directly traceable to a cited source. Every sentence ends with inline [n] citations. HEDGE MATCH: match source hedge levels throughout — 'suggests' not 'demonstrates'. NO TIMELINE: omit any timeline not verbatim in a source. NO UNIVERSALITY: scope claims to actual examples. Draw from at least 2 distinct insight bullets — use the best available rather than omitting the paragraph. COMPLICATION RULE: Before finalizing the paragraph, check every cited source for bullets that contradict or qualify the central claim. If any exist, either incorporate them or scope the closing implication to reflect the limitation. A paragraph that ignores complicating evidence from its own cited sources will score lower than one that acknowledges the complication and commits to a narrower claim. INSIGHT SELECTION RULE: The closing implication must trace to the single highest-ranked bullet identified in your reasoning — the most non-obvious, specific, and source-grounded insight available. Do not close with a broad pattern observation when a more specific mechanical consequence is available in the source bullets. The broad observation is usually derivable from the headline. The specific mechanical consequence requires reading the full content. Keep the latter. SPLIT CHECK: Before writing the closing sentence, ask: does it contain 'and', 'while also', 'as well as', or 'in addition'? If yes, it contains two consequences. Split them, keep only the stronger one, and discard the other. A closing sentence that states two consequences is a split implication regardless of how tightly connected they seem. CLOSING SENTENCE: State exactly one consequence directly traceable to a specific bullet in the cited sources. Match the hedge level of the source — if the source says 'suggests', write 'suggests', not 'means' or 'demonstrates'. Do not convert a source observation into a prescription. If no source bullet explicitly states a PM action, use 'this suggests' or 'this may signal' framing. Never write 'this means PMs must/should' unless a cited source explicitly states that directive. A well-hedged implication that commits to one specific consequence scores higher than a confident assertion that goes beyond the source.",
      "source_indices": [1, 2],
      "theme": "The single theme this paragraph addresses. Must be exactly one of: technology_trends | market_signals | user_behavior | regulation_policy. Pick the theme whose core claim this paragraph develops — not a secondary mention."
    }}
  ],
  "interview_angle": "One specific tradeoff a PM should have a prepared opinion on this week, derived from a whats_shifting source. Empty string if no whats_shifting paragraph was produced. Frame as a debatable tradeoff at PM decision level — feature prioritization, architecture, safety design, retention, compliance, pricing, or go-to-market. Scope to the context the source describes. DOMAIN RULE: Do not frame in legal, financial, or policy terms even if the source is a legal or regulatory story — translate into a product decision a PM owns. The angle must be answerable by a PM from product judgment, not legal or financial expertise."
}}

IMPORTANT: Do not include anchor_reasoning inside the JSON. All reasoning goes in the <reasoning> block only.
OUTPUT RULE: Produce a paragraph for every required anchor theme. Imperfect paragraphs are better than empty arrays — missing themes will trigger individual retry calls that cost extra latency and tokens.
CITATION RULE: Only cite item [n] if a specific insight bullet from that item directly supports the exact claim. Every sentence must have at least one citation.
""".strip()

    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=5000,
        temperature=0.3,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    content_block = response.content[0]
    text = getattr(content_block, "text", None) or content_block.get("text")  # type: ignore[union-attr]
    # FIX: use logger.debug for raw responses (very long — not useful at info level)
    logger.debug("Raw Claude Call 1 (WS) response text: %s", text)

    reasoning_text, text_without_reasoning = _extract_reasoning_block(text or "")
    if reasoning_text:
        logger.info("Call 1 anchor reasoning extracted (%d chars)", len(reasoning_text))

    cleaned = _extract_json(text_without_reasoning)
    try:
        parsed = json.loads(cleaned)
    except Exception:
        logger.warning("Call 1 response was not valid JSON. Raw (first 500): %s", (text or "")[:500])
        parsed = {"whats_shifting": [], "interview_angle": ""}

    parsed["_call1_reasoning"] = reasoning_text
    return parsed, ws_indexed


# ---------------------------------------------------------------------------
# Per-theme fill call
# ---------------------------------------------------------------------------

def _call_whats_shifting_single_theme(
    client: Anthropic,
    settings: Any,
    theme_items: List[Dict[str, Any]],
    anchor: Dict[str, Any],
    today: str,
    ws_indexed: List[Dict[str, Any]],
    covered_themes: Set[str] | None = None,
) -> Optional[Dict[str, Any]]:
    if not theme_items:
        logger.warning("FILL [theme=%s]: No items available — skipping", anchor["theme"])
        return None

    lines: List[str] = []
    for item in theme_items:
        idx = None
        for entry in ws_indexed:
            if entry.get("item_id") == item.get("item_id"):
                idx = entry["index"]
                break
        if idx is None:
            continue

        insights = item["insights"]
        if not isinstance(insights, list):
            insights = [str(insights)]

        lines.append(f"Item [{idx}]:")
        lines.append(f"- Theme: {item['theme']}")
        lines.append(f"- Source: {item['source_name']}")
        lines.append(f"- Title: {item['title']}")
        lines.append("- Insights:")
        for bullet in insights:
            lines.append(f"  - {bullet}")
        lines.append("")

    context_block = "\n".join(lines)
    if not context_block.strip():
        logger.warning("FILL [theme=%s]: Empty context block — skipping", anchor["theme"])
        return None

    theme = anchor["theme"]
    anchor_title = anchor["anchor_item"]["title"]

    covered_themes_str = ", ".join(sorted(covered_themes)) if covered_themes else "none yet"
    fill_theme_context = f"""
THEME GAP CONTEXT:
The following themes are already covered by paragraphs produced in the main synthesis pass:
{covered_themes_str}

Your task is to fill the gap for theme: {theme}
The paragraph you produce must develop a claim whose central force is {theme} — not a theme
already covered above. If the available source bullets touch multiple themes, anchor to the
one that is genuinely about {theme} and treat others as supporting context only.
"""

    user_prompt = f"""
You are producing exactly one What's Shifting paragraph for a Senior PM digest.
Today's date is {today}.

Theme: {theme}
Anchor item: "{anchor_title}"

MANDATE: Produce exactly one paragraph for this theme. This is a targeted fill call — the main synthesis pass did not produce a paragraph for this theme. Output is required. A paragraph with 2 bullets and an imperfect closing is better than no paragraph.
{fill_theme_context}
Items for this theme:
{context_block}

First, write your reasoning inside <reasoning>...</reasoning> tags:
(1) List all insight bullets ranked by non-obviousness.
(2) Name the highest-ranked bullet and declare it the closing implication anchor.
(3) List every bullet that contradicts or qualifies the anchor's claim.
(4) Name the causal mechanism if combining multiple sources.

Then produce a JSON object:
{{
  "headline": "One declarative sentence, maximum 15 words, naming the structural force or pattern. Scannable, self-contained, not an event description.",
  "paragraph": "3-5 sentences. Open with the headline claim plus one clause of context. Develop with 2+ bullets from the items above. Close with one PM implication traceable to a cited source. Every sentence has inline [n] citations. HEDGE MATCH: match source hedge levels. NO TIMELINE unless verbatim in source. NO UNIVERSALITY beyond actual examples. COMPLICATION RULE: Before finalizing the paragraph, check every cited source for bullets that contradict or qualify the central claim. SPLIT CHECK: Before writing the closing sentence, ask: does it contain 'and', 'while also', 'as well as', or 'in addition'? If yes, split and keep only the stronger one. CLOSING SENTENCE: State exactly one consequence directly traceable to a specific bullet in the cited sources.",
  "source_indices": [],
  "theme": "{theme}"
}}

IMPORTANT: Do not include anchor_reasoning in the JSON. All reasoning goes in <reasoning> only.
CITATION RULE: Use the item index numbers shown above exactly as written.
""".strip()

    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=1500,
        temperature=0.3,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    content_block = response.content[0]
    text = getattr(content_block, "text", None) or content_block.get("text")  # type: ignore[union-attr]
    logger.debug("Raw Claude Fill Call (theme=%s) response text: %s", theme, text)

    reasoning_text, text_without_reasoning = _extract_reasoning_block(text or "")
    cleaned = _extract_json(text_without_reasoning)
    try:
        parsed = json.loads(cleaned)
    except Exception:
        logger.warning(
            "FILL [theme=%s]: Response was not valid JSON. Raw (first 300): %s",
            theme, (text or "")[:300],
        )
        return None

    headline = _strip_date_check_flags(str(parsed.get("headline") or ""))
    paragraph = _strip_date_check_flags(str(parsed.get("paragraph") or ""))
    indices = parsed.get("source_indices") or []
    if not isinstance(indices, list):
        indices = []
    cleaned_indices: List[int] = []
    for i in indices:
        try:
            cleaned_indices.append(int(i))
        except Exception:
            continue

    if not paragraph.strip():
        logger.warning("FILL [theme=%s]: Paragraph is empty after parsing — fill failed", theme)
        return None

    logger.info("FILL [theme=%s]: Paragraph produced (%d words)", theme, len(paragraph.split()))
    return {
        "headline": headline,
        "paragraph": paragraph,
        "source_indices": cleaned_indices,
        "_fill_theme": theme,
        "_fill_reasoning": reasoning_text,
    }


# ---------------------------------------------------------------------------
# Call 2a: Company Watch only (compressed prompt)
# ---------------------------------------------------------------------------

def _call_company_watch(
    client: Anthropic,
    settings: Any,
    cw_items: List[Dict[str, Any]],
    today: str,
    start_idx: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not cw_items:
        logger.info("Call 3 (CW): No company_strategy items — returning empty.")
        return {"company_watch": {}, "_call3_reasoning": ""}, []

    context_block, cw_indexed, _ = _build_context_block(cw_items, start_idx=start_idx)

    user_prompt = f"""
You are writing Company Watch entries for a Senior PM digest.
Today: {today}

COMPANY WATCH RULES (apply to every company entry below):
  STRUCTURE: 2-3 sentences.
    Sentence 1: what is strategically changing — a shift in positioning, priority,
                or competitive stance. Not news. A structural move.
    Sentence 2: evidence with inline [n] citations.
    Sentence 3 (optional): one implication, most specific and directly grounded.
  OMIT RULE: Write empty string if no item in the pool matches that company.
             Do not hallucinate coverage for companies with no items.
  SOURCE RULE: Only cite items whose Company field matches the entry company.
  HEDGE MATCH: Match source hedge levels — 'suggests' not 'demonstrates'.
  COMPLICATION RULE: If a cited source contains a bullet qualifying the central
                     claim, incorporate it or scope the implication accordingly.
  SPLIT CHECK: Before writing the closing sentence — does it contain 'and',
               'while also', 'as well as', or 'in addition'? If yes, split and
               keep only the stronger consequence.
  CLOSING SENTENCE: Exactly one consequence traceable to a specific cited bullet.
                    Never assert certainty beyond what the source states.

Items (company_strategy sources only):
{context_block}

First, write anchor reasoning inside <reasoning>...</reasoning> tags.
For each company that has items:
  (1) List all bullets ranked by non-obviousness.
  (2) Declare the closing implication anchor.
  (3) Note any qualifying bullets — incorporate or scope.

Then produce JSON:
{{
  "company_watch": {{
    "Google":    {{"paragraph": "...", "source_indices": []}},
    "Meta":      {{"paragraph": "...", "source_indices": []}},
    "Apple":     {{"paragraph": "...", "source_indices": []}},
    "Amazon":    {{"paragraph": "...", "source_indices": []}},
    "Netflix":   {{"paragraph": "...", "source_indices": []}},
    "Microsoft": {{"paragraph": "...", "source_indices": []}},
    "NVIDIA":    {{"paragraph": "...", "source_indices": []}},
    "OpenAI":    {{"paragraph": "...", "source_indices": []}},
    "Anthropic": {{"paragraph": "...", "source_indices": []}}
  }}
}}

IMPORTANT: Reasoning in <reasoning> only. JSON contains paragraphs only.
CITATION RULE: Only cite [n] if a specific bullet directly supports the exact claim.
""".strip()

    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=3000,
        temperature=0.3,
        system=_CALL_3_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )

    block = response.content[0]
    text = getattr(block, "text", None) or block.get("text")  # type: ignore[union-attr]
    logger.debug("Raw Call 3 (CW) response: %s", text)

    reasoning_text, text_without_reasoning = _extract_reasoning_block(text or "")
    if reasoning_text:
        logger.info("Call 3 (CW) reasoning extracted (%d chars)", len(reasoning_text))

    cleaned = _extract_json(text_without_reasoning)
    try:
        parsed = json.loads(cleaned)
    except Exception:
        logger.warning("Call 3 (CW) not valid JSON. Raw (first 500): %s", (text or "")[:500])
        parsed = {"company_watch": {}}

    parsed["_call3_reasoning"] = reasoning_text
    return parsed, cw_indexed


# ---------------------------------------------------------------------------
# Call 2b: Startup Radar + PM Craft
# ---------------------------------------------------------------------------

def _call_sr_pm_craft(
    client: Anthropic,
    settings: Any,
    sr_pm_items: List[Dict[str, Any]],
    today: str,
    start_idx: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Dedicated call for Startup Radar and PM Craft only.

    Separating these from Company Watch gives SR and PM Craft their own
    full attention budget. Previously these sections received whatever
    token headroom remained after 9 company evaluations — directly causing
    SR's 43% run rate and coherence volatility.
    """
    if not sr_pm_items:
        logger.info("Call 2b (SR+PC): No items — returning empty.")
        return {
            "startup_radar": [],
            "pm_craft_today": {"text": "", "source_indices": []},
            "_call2b_reasoning": "",
        }, []

    context_block, sr_pm_indexed, _ = _build_context_block(sr_pm_items, start_idx=start_idx)

    user_prompt = f"""
You are reasoning across startup_disruption and product_craft items for a Senior PM digest.
Today's date is {today}.

Items are tagged either:
  - startup_radar ONLY  (theme: market_signals)
  - pm_craft_today ONLY (theme: pm_craft)

Items:
{context_block}

First, write your anchor selection reasoning inside <reasoning>...</reasoning> tags.
For startup_radar and pm_craft_today:
(1) List ALL insight bullets ranked by non-obviousness.
(2) Name the highest-ranked bullet and declare it the closing implication anchor.
(3) List every bullet that contradicts or qualifies the anchor's claim.
(4) Name the causal mechanism if combining multiple sources.

Then produce a JSON object:

{{
  "startup_radar": [
    {{
      "bullet": "2-3 items on early-stage or emerging companies only. Structure: [what the company did] + [why it matters strategically] + [what pattern or shift it represents]. HEDGE MATCH. NO TIMELINE unless verbatim in source. METRICS: include funding amount or key metric if available. THEMATIC COMBINATION: only combine companies if you can name the specific causal mechanism both share. SPLIT CHECK: closing sentence must state exactly one consequence traceable to a cited bullet.",
      "source_indices": []
    }}
  ],
  "pm_craft_today": {{
    "text": "Single most actionable PM craft insight. Draw ONLY from items tagged pm_craft_today ONLY (product_craft) OR pm_craft_today eligible (design_ux). Empty string if no such item exists. INSIGHT QUALITY: non-obvious pattern, tradeoff, or reframe that changes how a PM approaches a real decision. SPLIT CHECK: closing sentence must state exactly one actionable consequence traceable to a cited bullet.",
    "source_indices": []
  }}
}}

IMPORTANT: Do not include reasoning inside the JSON. All reasoning goes in <reasoning> only.
SECTION ROUTING RULE: startup_radar ONLY items (market_signals) → startup_radar only. pm_craft_today ONLY items (pm_craft) → pm_craft_today only. Hard constraints, not suggestions.
CITATION RULE: Only cite item [n] if a specific insight bullet directly supports the exact claim.
STARTUP FILTER: Only include companies with company_maturity=startup in startup_radar. Established companies must not appear here.
EMPTY STRING RULE: If no eligible items exist for a section, return empty string or empty array — do not hallucinate content.
""".strip()

    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=2000,
        temperature=0.3,
        system=_CALL_4B_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )

    content_block = response.content[0]
    text = getattr(content_block, "text", None) or content_block.get("text")  # type: ignore[union-attr]
    logger.debug("Raw Claude Call 2b (SR+PC) response: %s", text)

    reasoning_text, text_without_reasoning = _extract_reasoning_block(text or "")
    if reasoning_text:
        logger.info("Call 2b (SR+PC) reasoning extracted (%d chars)", len(reasoning_text))

    cleaned = _extract_json(text_without_reasoning)
    try:
        parsed = json.loads(cleaned)
    except Exception:
        logger.warning("Call 2b (SR+PC) response not valid JSON. Raw (first 500): %s", (text or "")[:500])
        parsed = {
            "startup_radar": [],
            "pm_craft_today": {"text": "", "source_indices": []},
        }

    parsed["_call2b_reasoning"] = reasoning_text
    return parsed, sr_pm_indexed


# ---------------------------------------------------------------------------
# Call 1a — Cross-market classifier (Haiku)
# ---------------------------------------------------------------------------

def _call_cross_market_classifier(
    client: Anthropic,
    settings: Any,
    ws_items: List[Dict[str, Any]],
    today: str,
) -> Dict[str, bool]:
    """
    Classify each WS-theme item as cross-market or not.
    Returns {item_id: is_cross_market} dict.
    Runs on Haiku — pure classification, no synthesis.
    """
    if not ws_items:
        return {}

    items_block_lines = []
    for item in ws_items:
        bullets = item.get("insights") or []
        bullets_text = "\n".join(f"    - {b}" for b in bullets) if bullets else "    (none)"
        items_block_lines.append(
            f"[{item['item_id']}]\n"
            f"  Source: {item['source_name']}\n"
            f"  Theme: {item['theme']}\n"
            f"  Title: {item['title']}\n"
            f"  Bullets:\n{bullets_text}"
        )
    items_block = "\n\n".join(items_block_lines)

    user_prompt = f"""
Today: {today}

For each item below, classify whether it describes a CROSS-MARKET structural shift
or COMPANY-SPECIFIC news.

CROSS-MARKET = the central claim applies to an industry, product category, or regulatory
framework affecting multiple companies or builders. The insight would be relevant to a PM
at any company in that space, not just the company mentioned.

COMPANY-SPECIFIC = primarily about what one named company did, faces, or decided.
A regulatory action against one company is company-specific even if it has industry
implications — the central claim is still about that company's situation.

Items:
{items_block}

Return strict JSON — a dict mapping each item_id to true (cross-market) or false (company-specific):
{{
  "classifications": {{
    "<item_id>": true,
    "<item_id>": false
  }}
}}

Use the exact item_id strings shown in brackets above.
""".strip()

    try:
        response = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=512,
            temperature=0,
            system=_CALL_1A_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
        )
        block = response.content[0]
        text = getattr(block, "text", None) or block.get("text")  # type: ignore[union-attr]
        parsed = json.loads(_extract_json(text or ""))
        classifications = parsed.get("classifications") or {}
        result = {
            item["item_id"]: bool(classifications.get(item["item_id"], False))
            for item in ws_items
        }
        cross_market_count = sum(1 for v in result.values() if v)
        logger.info(
            "Call 1a (cross-market): %d/%d items classified as cross-market",
            cross_market_count, len(ws_items),
        )
        return result
    except Exception:
        logger.warning("Call 1a classification failed — defaulting all items to cross-market")
        return {item["item_id"]: True for item in ws_items}


# ---------------------------------------------------------------------------
# Call 4a — Startup classifier (Haiku)
# ---------------------------------------------------------------------------

def _call_startup_classifier(
    client: Anthropic,
    settings: Any,
    sr_items: List[Dict[str, Any]],
    today: str,
) -> Dict[str, bool]:
    """
    Classify each startup_disruption item as genuinely early-stage or established.
    Returns {item_id: is_startup} dict.
    Runs on Haiku — pure classification, no synthesis.
    """
    if not sr_items:
        return {}

    items_block_lines = []
    for item in sr_items:
        bullets = item.get("insights") or []
        bullets_text = "\n".join(f"    - {b}" for b in bullets) if bullets else "    (none)"
        items_block_lines.append(
            f"[{item['item_id']}]\n"
            f"  Company: {item.get('company_id') or 'unknown'}\n"
            f"  Source: {item['source_name']}\n"
            f"  Title: {item['title']}\n"
            f"  Bullets:\n{bullets_text}"
        )
    items_block = "\n\n".join(items_block_lines)

    user_prompt = f"""
Today: {today}

For each item below, classify whether the PRIMARY SUBJECT COMPANY is a startup
(early-stage, emerging) or established.

STARTUP = early-stage or emerging company, privately held, valued below $1B,
without dominant market position. Unknown companies are likely startups.

ESTABLISHED = publicly traded company, subsidiary of one, OR privately held with
$1B+ valuation AND significant market presence.

Named established companies (always established regardless of funding stage):
Anthropic, OpenAI, Stripe, SpaceX, Databricks, Canva, Intuit, Google, Meta,
Amazon, Salesforce, Microsoft, Apple, Netflix, NVIDIA, Uber, Airbnb, DoorDash,
Instacart, Reddit, Discord, Figma, Notion, Airtable, Hugging Face, Cohere,
Mistral, Scale AI, Weights & Biases.

When in doubt: if a PM at Google would consider this company a peer rather than
a startup, classify as established.

Items:
{items_block}

Return strict JSON:
{{
  "classifications": {{
    "<item_id>": true,
    "<item_id>": false
  }}
}}

true = startup, false = established.
Use the exact item_id strings shown in brackets above.
""".strip()

    try:
        response = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=256,
            temperature=0,
            system=_CALL_4A_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
        )
        block = response.content[0]
        text = getattr(block, "text", None) or block.get("text")  # type: ignore[union-attr]
        parsed = json.loads(_extract_json(text or ""))
        classifications = parsed.get("classifications") or {}
        result = {
            item["item_id"]: bool(classifications.get(item["item_id"], False))
            for item in sr_items
        }
        startup_count = sum(1 for v in result.values() if v)
        logger.info(
            "Call 4a (startup): %d/%d items classified as startup",
            startup_count, len(sr_items),
        )
        return result
    except Exception:
        logger.warning("Call 4a classification failed — defaulting all items to startup")
        return {item["item_id"]: True for item in sr_items}


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def _persist_synthesizer_input(
    grouped_summaries: Dict[str, List[Dict[str, Any]]],
    today: str,
) -> None:
    import sqlite3
    settings = load_settings()
    db_path = settings.database_path
    try:
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS synthesizer_inputs ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "run_date TEXT NOT NULL, "
            "created_at TEXT NOT NULL, "
            "grouped_summaries TEXT NOT NULL"
            ")"
        )
        conn.execute(
            "INSERT INTO synthesizer_inputs (run_date, created_at, grouped_summaries) VALUES (?, ?, ?)",
            (
                today,
                date.today().isoformat(),
                json.dumps(grouped_summaries),
            )
        )
        conn.commit()
        conn.close()
        logger.info("Synthesizer input persisted for %s", today)
    except Exception as exc:
        logger.warning("Failed to persist synthesizer input: %s", exc)


def synthesize_trends(grouped_summaries: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    client = _build_client()
    settings = load_settings()
    today = date.today().strftime("%B %d, %Y")

    _persist_synthesizer_input(grouped_summaries, today)

    # ---------------------------------------------------------------------------
    # Filter items
    # ---------------------------------------------------------------------------
    filtered_items: List[Dict[str, Any]] = []
    dropped_low_confidence = 0
    dropped_low_relevance = 0

    for theme, items in grouped_summaries.items():
        for item in items:
            conf_raw = str(item.get("confidence") or "low").lower()
            if conf_raw not in {"high", "medium"}:
                dropped_low_confidence += 1
                logger.info(
                    "FILTER [step=1 reason=low_confidence] skipping: %s — %s",
                    item.get("source_name"), item.get("title")
                )
                continue

            relevance_raw = str(item.get("pm_relevance_score") or "low").lower()
            if relevance_raw not in {"high", "medium"}:
                dropped_low_relevance += 1
                logger.info(
                    "FILTER [step=2 reason=low_relevance] dropped: %s — %s",
                    item.get("source_name"), item.get("title")
                )
                continue

            filtered_items.append({
                "item_id": str(uuid.uuid4()),
                "theme": theme,
                "title": item.get("title", ""),
                "source_name": item.get("source_name", ""),
                "company_id": item.get("company_id"),
                "insights": item.get("insights") or [],
                "confidence": conf_raw,
                "pm_relevance_score": relevance_raw,
            })

    if dropped_low_confidence:
        logger.info("Dropped %d items with low confidence before synthesis.", dropped_low_confidence)
    if dropped_low_relevance:
        logger.info("Dropped %d items with low PM relevance before synthesis.", dropped_low_relevance)

    theme_funnel_after_filter: Dict[str, int] = {}
    for item in filtered_items:
        t = item["theme"]
        theme_funnel_after_filter[t] = theme_funnel_after_filter.get(t, 0) + 1
    logger.info("THEME FUNNEL [stage=after_quality_filter]: %s", json.dumps(theme_funnel_after_filter))

    empty_result = {
        "whats_shifting": [],
        "company_watch": {},
        "startup_radar": [],
        "pm_craft_today": {"text": "", "source_indices": []},
        "interview_angle": "",
        "source_index_lookup": {},
        "editorial_warnings": {
            "multi_thread_violations": [],
            "date_warnings": [],
            "coherence_warnings": [],
            "routing_warnings": [],
            "omit_rule_violations": [],
            "split_implication_warnings": [],
            "theme_audit_warnings": [],
            "cw_source_integrity_violations": [],
            "pm_craft_source_violations": [],
            "source_concentration_warnings": [],
            "theme_diversity_warnings": [],
        },
    }

    if not filtered_items:
        logger.warning("No eligible items available for synthesis after filtering.")
        return empty_result

    # ---------------------------------------------------------------------------
    # Source concentration check
    # ---------------------------------------------------------------------------
    source_counts: Dict[str, List[str]] = {}
    for item in filtered_items:
        src = item.get("source_name", "unknown")
        if src not in source_counts:
            source_counts[src] = []
        source_counts[src].append(item.get("title", ""))

    source_concentration_warnings = []
    for src, titles in source_counts.items():
        if len(titles) >= 3:
            source_concentration_warnings.append({
                "source_name": src,
                "item_count": len(titles),
                "titles": titles,
                "warning": f"{src} contributes {len(titles)} items to today's filtered pool — review for source diversity"
            })

    if source_concentration_warnings:
        logger.warning("SOURCE CONCENTRATION WARNINGS: %s", json.dumps(source_concentration_warnings, indent=2))

    # ---------------------------------------------------------------------------
    # Source diversity cap
    # ---------------------------------------------------------------------------
    MAX_ITEMS_PER_SOURCE = 3

    source_item_counts: Dict[str, int] = {}
    diversity_capped_items: List[Dict[str, Any]] = []
    diversity_overflow: List[Dict[str, Any]] = []

    relevance_order = {"high": 0, "medium": 1}
    filtered_items.sort(key=lambda x: relevance_order.get(x.get("pm_relevance_score", "medium"), 1))

    for item in filtered_items:
        src = item.get("source_name", "unknown")
        current_count = source_item_counts.get(src, 0)
        if current_count < MAX_ITEMS_PER_SOURCE:
            diversity_capped_items.append(item)
            source_item_counts[src] = current_count + 1
        else:
            diversity_overflow.append(item)
            logger.info(
                "DIVERSITY CAP: '%s' from '%s' held in overflow (source already at %d items)",
                item.get("title"), src, MAX_ITEMS_PER_SOURCE
            )

    represented_themes = {item["theme"] for item in diversity_capped_items}
    for item in diversity_overflow:
        if item["theme"] not in represented_themes:
            diversity_capped_items.append(item)
            represented_themes.add(item["theme"])
            logger.info(
                "DIVERSITY CAP OVERRIDE: '%s' from '%s' restored — only item for theme '%s'",
                item.get("title"), item.get("source_name"), item["theme"]
            )

    filtered_items = diversity_capped_items

    theme_funnel_after_cap: Dict[str, int] = {}
    for item in filtered_items:
        t = item["theme"]
        theme_funnel_after_cap[t] = theme_funnel_after_cap.get(t, 0) + 1
    logger.info("THEME FUNNEL [stage=after_diversity_cap]: %s", json.dumps(theme_funnel_after_cap))

    # ---------------------------------------------------------------------------
    # Partition by routing eligibility
    # ---------------------------------------------------------------------------
    ws_items: List[Dict[str, Any]] = []
    sr_pm_items: List[Dict[str, Any]] = []
    cw_items: List[Dict[str, Any]] = []
    sr_items: List[Dict[str, Any]] = []

    items_for_1a = [
        item for item in filtered_items
        if item["theme"] in WHATS_SHIFTING_THEMES
    ]
    cross_market_map = _call_cross_market_classifier(
        client, settings, items_for_1a, today
    )

    for item in filtered_items:
        theme = item["theme"]
        if theme == "company_strategy":
            cw_items.append(item)
        elif theme == "pm_craft":
            sr_pm_items.append(item)
        elif theme in WHATS_SHIFTING_THEMES:
            ws_items.append(item)
            if theme == "market_signals":
                sr_items.append(item)

    design_ux_ws = sum(1 for i in ws_items if i["theme"] == "design_ux")
    design_ux_pm = sum(1 for i in sr_pm_items if i["theme"] == "design_ux")
    logger.info(
        "Call 1a design_ux routing: %d cross-market → WS, %d company-specific → PM Craft",
        design_ux_ws, design_ux_pm,
    )
    logger.info(
        "Call 1a: %d items in WS pool after cross-market filter",
        len(ws_items),
    )

    dedicated_items = cw_items + sr_items + sr_pm_items

    ws_theme_dist: Dict[str, int] = {}
    for item in ws_items:
        t = item["theme"]
        ws_theme_dist[t] = ws_theme_dist.get(t, 0) + 1
    dedicated_theme_dist: Dict[str, int] = {}
    for item in dedicated_items:
        t = item["theme"]
        dedicated_theme_dist[t] = dedicated_theme_dist.get(t, 0) + 1
    logger.info("THEME FUNNEL [stage=ws_items_post_partition]: %s", json.dumps(ws_theme_dist))
    logger.info("THEME FUNNEL [stage=dedicated_items_post_partition]: %s", json.dumps(dedicated_theme_dist))
    logger.info(
        "Routing: %d whats_shifting items, %d dedicated section items",
        len(ws_items), len(dedicated_items)
    )

    # ---------------------------------------------------------------------------
    # Build required anchors
    # ---------------------------------------------------------------------------
    WHATS_SHIFTING_THEMES_ORDERED = [
        "technology_trends",
        "market_signals",
        "regulation_policy",
        "user_behavior",
    ]

    required_anchors = []
    for theme in WHATS_SHIFTING_THEMES_ORDERED:
        candidates = [i for i in ws_items if i["theme"] == theme]
        if candidates:
            best = sorted(
                candidates,
                key=lambda x: (
                    0 if x.get("pm_relevance_score") == "high" else 1,
                    0 if x.get("confidence") == "high" else 1,
                )
            )[0]
            required_anchors.append({"theme": theme, "anchor_item": best})

    logger.info(
        "REQUIRED ANCHORS: %s",
        json.dumps([{"theme": a["theme"], "title": a["anchor_item"]["title"]} for a in required_anchors])
    )

    try:
        # ---------------------------------------------------------------------------
        # Call 1: all WS themes together
        # ---------------------------------------------------------------------------
        call1_parsed, ws_indexed = _call_whats_shifting(
            client, settings, ws_items, today,
            ws_theme_distribution=ws_theme_dist,
            required_anchors=required_anchors,
        )

        ws_paragraphs = _normalize_whats_shifting(call1_parsed.get("whats_shifting") or [])
        ws_paragraphs = [ws for ws in ws_paragraphs if ws.get("paragraph", "").strip()]

        source_index_lookup: Dict[str, Dict[str, Any]] = {}
        for entry in ws_indexed:
            source_index_lookup[str(entry["index"])] = {
                "title": entry["title"],
                "source_name": entry["source_name"],
                "theme": entry["theme"],
                "company_id": entry.get("company_id"),
                "item_id": entry.get("item_id"),  # propagate item_id for evaluator matching
            }

        # ---------------------------------------------------------------------------
        # Detect missing themes and fill
        # ---------------------------------------------------------------------------
        covered_themes = _get_covered_themes(ws_paragraphs, ws_indexed, source_index_lookup)
        missing_anchors = [a for a in required_anchors if a["theme"] not in covered_themes]

        logger.info("WS covered themes after Call 1: %s", sorted(covered_themes))
        logger.info("WS missing themes after Call 1: %s", [a["theme"] for a in missing_anchors])

        fill_reasonings: List[Dict[str, Any]] = []

        for anchor in missing_anchors:
            theme = anchor["theme"]
            theme_items = [i for i in ws_items if i["theme"] == theme]

            logger.info(
                "WS FILL [theme=%s]: Calling targeted fill with %d items",
                theme, len(theme_items)
            )

            fill_result = _call_whats_shifting_single_theme(
                client, settings, theme_items, anchor, today, ws_indexed,
                covered_themes=covered_themes,
            )

            if fill_result and fill_result.get("paragraph", "").strip():
                ws_paragraphs.append(fill_result)
                logger.info(
                    "WS FILL [theme=%s]: Merged into ws_paragraphs (%d total paragraphs now)",
                    theme, len(ws_paragraphs)
                )
                if fill_result.get("_fill_reasoning"):
                    fill_reasonings.append({
                        "theme": theme,
                        "reasoning": fill_result["_fill_reasoning"],
                    })
            else:
                logger.error(
                    "WS FILL [theme=%s]: Fill call produced no usable paragraph",
                    theme
                )

        # ---------------------------------------------------------------------------
        # Deduplicate — one paragraph per theme
        # ---------------------------------------------------------------------------
        ws_paragraphs = _deduplicate_by_theme(
            ws_paragraphs, ws_items, ws_indexed, source_index_lookup
        )

        # ---------------------------------------------------------------------------
        # Final coverage check
        # ---------------------------------------------------------------------------
        final_covered = _get_covered_themes(ws_paragraphs, ws_indexed, source_index_lookup)
        still_missing = [a["theme"] for a in required_anchors if a["theme"] not in final_covered]

        if still_missing:
            logger.error(
                "WS FINAL COVERAGE GAP: themes still missing after all fill calls: %s",
                still_missing
            )
        else:
            logger.info(
                "WS FINAL COVERAGE: all %d required themes covered: %s",
                len(required_anchors), sorted(final_covered)
            )

        # Remove WS-consumed market_signals items from Startup Radar pool
        ws_used_indices: Set[int] = set()
        for ws in ws_paragraphs:
            ws_used_indices.update(ws.get("source_indices", []))

        ws_used_item_ids: Set[str] = set()
        for entry in ws_indexed:
            if entry["index"] in ws_used_indices and entry.get("item_id"):
                ws_used_item_ids.add(entry["item_id"])

        if ws_used_item_ids:
            before_sr_count = len(sr_items)
            sr_items = [
                item for item in sr_items
                if item.get("item_id") not in ws_used_item_ids
            ]
            removed_sr = before_sr_count - len(sr_items)
            if removed_sr:
                logger.info(
                    "DEDUP [ws_consumed]: Removed %d market_signals item(s) from "
                    "Startup Radar pool already cited in WS",
                    removed_sr
                )

        # Call 4a — startup classification
        startup_map: Dict[str, bool] = {}
        if sr_items:
            startup_map = _call_startup_classifier(client, settings, sr_items, today)
            before_count = len(sr_items)
            sr_items = [
                item for item in sr_items
                if startup_map.get(item["item_id"], True)
            ]
            dropped = before_count - len(sr_items)
            logger.info(
                "Call 4a: %d established companies removed from startup_disruption pool",
                dropped,
            )

        # ---------------------------------------------------------------------------
        # Call 3: Company Watch
        # ---------------------------------------------------------------------------
        start_idx = len(ws_indexed) + 1

        call4_items = sr_items + sr_pm_items

        logger.info(
            "Call 3/4 split: %d company_strategy → Call 3, %d startup + %d PM Craft → Call 4",
            len(cw_items), len(sr_items), len(sr_pm_items),
        )

        call3_parsed, cw_indexed = _call_company_watch(
            client, settings, cw_items, today, start_idx=start_idx
        )

        # ---------------------------------------------------------------------------
        # Call 4: Startup Radar + PM Craft
        # ---------------------------------------------------------------------------
        call4_parsed, sr_pm_indexed = _call_sr_pm_craft(
            client, settings, call4_items, today,
            start_idx=start_idx + len(cw_indexed),
        )

        dedicated_indexed = cw_indexed + sr_pm_indexed

        # Merge into the shape the rest of the pipeline expects
        call2_parsed = {
            "company_watch":  call3_parsed.get("company_watch") or {},
            "startup_radar":  call4_parsed.get("startup_radar") or [],
            "pm_craft_today": call4_parsed.get("pm_craft_today") or {"text": "", "source_indices": []},
            "_call3_reasoning": call3_parsed.get("_call3_reasoning", ""),
            "_call2b_reasoning": call4_parsed.get("_call2b_reasoning", ""),
        }

        for entry in dedicated_indexed:
            source_index_lookup[str(entry["index"])] = {
                "title": entry["title"],
                "source_name": entry["source_name"],
                "theme": entry["theme"],
                "company_id": entry.get("company_id"),
                "item_id": entry.get("item_id"),
            }

        interview_angle = _strip_date_check_flags(str(call1_parsed.get("interview_angle") or ""))

        # FIX: documented poison strings with context about when each was added.
        # These catch specific prompt-bleed patterns observed in production.
        INTERVIEW_ANGLE_POISON_STRINGS = [
            "synthesis pipeline",  # Claude echoed orchestration context
            "surface this week",   # Claude echoed prompt phrasing verbatim
            "run the",             # Claude echoed "run the digest" from prompt
        ]
        if any(p in interview_angle.lower() for p in INTERVIEW_ANGLE_POISON_STRINGS):
            logger.warning(
                "INTERVIEW ANGLE: Prompt bleed detected — suppressing output. Raw: %s",
                interview_angle[:200],
            )
            interview_angle = ""

        normalized_company_watch = _normalize_company_watch(call2_parsed.get("company_watch") or {})
        normalized_startup_radar = _normalize_startup_radar(call2_parsed.get("startup_radar") or [])
        pm_craft_today = _normalize_pm_craft(call2_parsed.get("pm_craft_today") or {})

        # ---------------------------------------------------------------------------
        # Post-processing validators
        # ---------------------------------------------------------------------------

        ws_eligible_indices = {entry["index"] for entry in ws_indexed}
        dedicated_eligible_indices = {entry["index"] for entry in dedicated_indexed}

        # 1. Multi-thread check
        multi_thread_warnings = []
        for company, value in normalized_company_watch.items():
            indices = value.get("source_indices", [])
            if len(indices) > 2:
                multi_thread_warnings.append({
                    "company": company,
                    "source_count": len(indices),
                    "source_indices": indices,
                    "warning": "More than 2 sources cited — review for multiple threads"
                })
            paragraph = value.get("paragraph", "")
            and_count = paragraph.upper().count(" AND ")
            if and_count >= 3:
                multi_thread_warnings.append({
                    "company": company,
                    "and_count": and_count,
                    "warning": "High conjunction count — review for multiple threads"
                })

        if multi_thread_warnings:
            logger.warning("SINGLE THESIS WARNINGS: %s", json.dumps(multi_thread_warnings, indent=2))

        # 2. Date validation
        current_year = date.today().year
        current_month = date.today().month
        date_warnings = []

        all_paragraphs_for_check = []
        for ws in ws_paragraphs:
            all_paragraphs_for_check.append(("whats_shifting", ws.get("paragraph", "")))
        for company, value in normalized_company_watch.items():
            all_paragraphs_for_check.append((f"company_watch.{company}", value.get("paragraph", "")))
        for sr in normalized_startup_radar:
            all_paragraphs_for_check.append(("startup_radar", sr.get("bullet", "")))
        all_paragraphs_for_check.append(("pm_craft_today", pm_craft_today.get("text", "")))

        month_year_pattern = re.compile(
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)'
            r'\s+(20\d{2})\b'
        )
        months_map = {
            "January": 1, "February": 2, "March": 3, "April": 4,
            "May": 5, "June": 6, "July": 7, "August": 8,
            "September": 9, "October": 10, "November": 11, "December": 12
        }

        for section, paragraph in all_paragraphs_for_check:
            for match in month_year_pattern.finditer(paragraph):
                month_str, year_str = match.group(1), match.group(2)
                year = int(year_str)
                month = months_map.get(month_str, 0)
                if year < current_year or (year == current_year and month < current_month):
                    date_warnings.append({
                        "section": section,
                        "date_found": match.group(0),
                        "warning": "Date appears to be in the past — verify if stated as future milestone"
                    })

        if date_warnings:
            logger.warning("DATE WARNINGS: %s", json.dumps(date_warnings, indent=2))

        # 3. Cross-paragraph coherence
        source_to_companies: Dict[int, List[str]] = {}
        for company, value in normalized_company_watch.items():
            for i in value.get("source_indices", []):
                if i not in source_to_companies:
                    source_to_companies[i] = []
                source_to_companies[i].append(company)

        ws_all_indices: List[int] = []
        for ws in ws_paragraphs:
            ws_all_indices.extend(ws.get("source_indices", []))

        coherence_warnings = []
        for i, companies in source_to_companies.items():
            if len(companies) > 1:
                source_info = source_index_lookup.get(str(i), {})
                coherence_warnings.append({
                    "source_index": i,
                    "source_title": source_info.get("title", "unknown"),
                    "companies": companies,
                    "warning": "Same source cited in multiple company entries — review for contradictory framings"
                })

        cw_all_indices = set()
        for value in normalized_company_watch.values():
            cw_all_indices.update(value.get("source_indices", []))

        shared_cw_ws = set(ws_all_indices) & cw_all_indices
        for i in shared_cw_ws:
            source_info = source_index_lookup.get(str(i), {})
            companies_using = source_to_companies.get(i, [])
            coherence_warnings.append({
                "source_index": i,
                "source_title": source_info.get("title", "unknown"),
                "also_in_company_watch": companies_using,
                "warning": "SANITY CHECK: Source appears in both WS and CW — should not happen with two-call architecture"
            })

        if coherence_warnings:
            logger.warning("COHERENCE WARNINGS: %s", json.dumps(coherence_warnings, indent=2))

        # 4 + 5. Routing canary (merged — previously validators 4 and 5 checked the
        # same company_watch condition independently, producing duplicate warnings).
        routing_warnings = []
        omit_rule_warnings = []

        for i, ws in enumerate(ws_paragraphs):
            for idx_val in ws.get("source_indices", []):
                if idx_val in dedicated_eligible_indices:
                    source_info = source_index_lookup.get(str(idx_val), {})
                    source_theme = source_info.get("theme", "")
                    routing_warnings.append({
                        "section": f"whats_shifting[{i}]",
                        "source_index": idx_val,
                        "source_title": source_info.get("title", "unknown"),
                        "source_theme": source_theme,
                        "warning": "CANARY: DEDICATED_SECTION source cited in whats_shifting"
                    })

        for company, value in normalized_company_watch.items():
            for idx_val in value.get("source_indices", []):
                if idx_val in ws_eligible_indices:
                    source_info = source_index_lookup.get(str(idx_val), {})
                    routing_warnings.append({
                        "section": f"company_watch.{company}",
                        "source_index": idx_val,
                        "source_title": source_info.get("title", "unknown"),
                        "source_theme": source_info.get("theme", "unknown"),
                        "warning": "CANARY: WHATS_SHIFTING source cited in company_watch"
                    })
                    # Omit-rule canary for company_watch is now the same check —
                    # record under omit_rule_warnings too for backward compatibility.
                    omit_rule_warnings.append({
                        "section": f"company_watch.{company}",
                        "source_index": idx_val,
                        "source_title": source_info.get("title", "unknown"),
                        "source_theme": source_info.get("theme", "unknown"),
                        "warning": "CANARY: Company Watch cites WS-eligible source",
                        "action": "INVESTIGATE"
                    })

        # PM Craft omit-rule check (unique — not covered by routing canary)
        for idx_val in pm_craft_today.get("source_indices", []):
            if idx_val in ws_eligible_indices:
                source_info = source_index_lookup.get(str(idx_val), {})
                omit_rule_warnings.append({
                    "section": "pm_craft_today",
                    "source_index": idx_val,
                    "source_title": source_info.get("title", "unknown"),
                    "source_theme": source_info.get("theme", "unknown"),
                    "warning": "CANARY: PM Craft cites WS-eligible source",
                    "action": "INVESTIGATE"
                })

        if routing_warnings:
            logger.warning("ROUTING CANARY FIRED: %s", json.dumps(routing_warnings, indent=2))
        if omit_rule_warnings:
            logger.warning("OMIT RULE CANARY: %s", json.dumps(omit_rule_warnings, indent=2))

        # 5b. Company Watch source integrity check
        cw_source_integrity_violations = []
        companies_to_clear = []

        for company, value in normalized_company_watch.items():
            bad_indices = []
            for idx_val in value.get("source_indices", []):
                source_info = source_index_lookup.get(str(idx_val), {})
                source_theme = source_info.get("theme", "")
                source_company_id = source_info.get("company_id")

                if source_theme != "company_strategy":
                    bad_indices.append((idx_val, "non_company_strategy", source_info))
                elif source_company_id != company:
                    bad_indices.append((idx_val, "company_id_mismatch", source_info))

            if bad_indices:
                for idx_val, reason, source_info in bad_indices:
                    warning_msg = (
                        f"Company Watch entry for {company} cites a non-company_strategy source "
                        f"(theme: {source_info.get('theme', 'unknown')}). Entry cleared."
                        if reason == "non_company_strategy"
                        else
                        f"Company Watch entry for {company} cites source belonging to "
                        f"'{source_info.get('company_id', 'unknown')}'. Entry cleared."
                    )
                    cw_source_integrity_violations.append({
                        "company": company,
                        "source_index": idx_val,
                        "source_title": source_info.get("title", "unknown"),
                        "source_theme": source_info.get("theme", "unknown"),
                        "source_company_id": source_info.get("company_id"),
                        "reason": reason,
                        "action": "ENTRY_CLEARED",
                        "warning": warning_msg,
                    })
                companies_to_clear.append(company)

        for company in companies_to_clear:
            logger.warning("CW SOURCE INTEGRITY: Clearing %s entry", company)
            normalized_company_watch[company] = {"paragraph": "", "source_indices": []}

        if cw_source_integrity_violations:
            logger.warning("CW SOURCE INTEGRITY VIOLATIONS: %s", json.dumps(cw_source_integrity_violations, indent=2))

        # 5c. PM Craft source integrity check
        product_craft_indices = {
            entry["index"] for entry in dedicated_indexed
            if entry["theme"] in {"product_craft", "design_ux"}
        }

        pm_craft_source_violations = []
        pm_craft_indices = pm_craft_today.get("source_indices", [])
        bad_pm_craft_indices = [
            idx_val for idx_val in pm_craft_indices
            if idx_val not in product_craft_indices
        ]
        if bad_pm_craft_indices:
            for idx_val in bad_pm_craft_indices:
                source_info = source_index_lookup.get(str(idx_val), {})
                pm_craft_source_violations.append({
                    "source_index": idx_val,
                    "source_title": source_info.get("title", "unknown"),
                    "source_theme": source_info.get("theme", "unknown"),
                    "action": "ENTRY_CLEARED",
                    "warning": (
                        f"PM Craft cites a non-product_craft source "
                        f"(theme: {source_info.get('theme', 'unknown')}). Entry cleared."
                    )
                })
            logger.warning("PM CRAFT SOURCE VIOLATIONS: %s", json.dumps(pm_craft_source_violations, indent=2))
            pm_craft_today = {"text": "", "source_indices": []}

        # 6. Split implication detector
        SPLIT_SIGNALS = [
            " and ", " but also ", " as well as ", " while also ",
            " in addition ", " additionally "
        ]

        split_implication_warnings = []

        def check_split_implication(text: str, section: str) -> None:
            if not text:
                return
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
            if not sentences:
                return
            closing_sentence = sentences[-1].lower()
            closing_clean = re.sub(r'\[\d+\]', '', closing_sentence)

            for signal in SPLIT_SIGNALS:
                if signal in closing_clean:
                    parts = closing_clean.split(signal, 1)
                    if len(parts) == 2:
                        before_words = len(parts[0].split())
                        after_words = len(parts[1].split())
                        if before_words >= 10 and after_words >= 10:
                            split_implication_warnings.append({
                                "section": section,
                                "signal": signal.strip(),
                                "closing_sentence": sentences[-1][:200],
                                "warning": "Possible split implication — review for two separate consequences"
                            })
                            break

        for i, ws in enumerate(ws_paragraphs):
            check_split_implication(ws.get("paragraph", ""), f"whats_shifting[{i}]")
        for company, value in normalized_company_watch.items():
            check_split_implication(value.get("paragraph", ""), f"company_watch.{company}")
        for i, sr in enumerate(normalized_startup_radar):
            check_split_implication(sr.get("bullet", ""), f"startup_radar[{i}]")
        check_split_implication(pm_craft_today.get("text", ""), "pm_craft_today")

        if split_implication_warnings:
            logger.warning("SPLIT IMPLICATION WARNINGS: %s", json.dumps(split_implication_warnings, indent=2))

        # 7. Theme audit
        THEME_KEYWORDS: Dict[str, List[str]] = {
            "regulation_policy": [
                "regulat", "law", "legal", "court", "legislat", "policy", "government",
                "enforcement", "compliance", "antitrust", "jurisdiction", "decree",
                "shutdown", "ban", "ruling", "verdict", "ftc", "doj", "gdpr",
                "cybercrime", "arrest", "detained", "prosecution",
            ],
            "market_signals": [
                "market", "acqui", "merger", "ipo", "valuation", "invest", "fund",
                "compet", "price", "pricing", "revenue", "monetiz", "platform",
                "asset-light", "vertical integrat", "supply chain", "demand",
                "abandon", "pivot", "writedown", "infrastructure", "capital",
                "startup", "early-stage", "seed", "series",
            ],
            "user_behavior": [
                "consumer", "user", "worker", "employee", "customer", "adoption",
                "preference", "workforce", "talent", "credential", "overqualif",
                "retention", "engagement", "behavior", "choice", "lifestyle",
                "enterprise", "prosumer", "audience",
            ],
            "technology_trends": [
                "ai ", "artificial intelligence", "model", "llm", "machine learning",
                "gpt", "gemini", "claude", "neural", "inference", "token",
                "foundation model", "generative", "autonomous", "robotaxi",
                "compute", "gpu", "blackwell", "hopper", "technology", "tech",
                "platform", "infrastructure", "open source", "api",
            ],
        }

        def _classify_paragraph_theme(paragraph: str) -> str:
            if not paragraph:
                return "unknown"
            first_sentence = re.split(r"(?<=[.!?])\s+", paragraph.strip())[0].lower()
            first_sentence = re.sub(r"\[\d+\]", "", first_sentence)
            scores: Dict[str, int] = {}
            for t, keywords in THEME_KEYWORDS.items():
                scores[t] = sum(1 for kw in keywords if kw in first_sentence)
            best_theme = max(scores, key=lambda t: scores[t])
            if scores[best_theme] == 0:
                return "unknown"
            return best_theme

        theme_audit_warnings = []
        ws_theme_counts: Dict[str, List[int]] = {}
        for i, ws in enumerate(ws_paragraphs):
            paragraph = ws.get("paragraph", "")
            theme = _get_theme_for_ws(ws, ws_indexed, source_index_lookup)
            if theme == "unknown":
                theme = _classify_paragraph_theme(paragraph)
            if theme not in ws_theme_counts:
                ws_theme_counts[theme] = []
            ws_theme_counts[theme].append(i)

        for theme, paragraph_indices in ws_theme_counts.items():
            if len(paragraph_indices) > 1:
                theme_audit_warnings.append({
                    "theme": theme,
                    "paragraph_indices": paragraph_indices,
                    "warning": f"Theme '{theme}' anchors {len(paragraph_indices)} WS paragraphs after dedup — investigate",
                    "action": "INVESTIGATE_UPSTREAM"
                })

        if theme_audit_warnings:
            logger.error("THEME AUDIT WARNINGS: %s", json.dumps(theme_audit_warnings, indent=2))

        # 8. Theme diversity warnings
        theme_diversity_warnings = []
        ws_paragraph_theme_counts = {t: len(idxs) for t, idxs in ws_theme_counts.items()}
        for theme, item_count in ws_theme_dist.items():
            if item_count >= 2:
                anchored = ws_paragraph_theme_counts.get(theme, 0)
                if anchored == 0:
                    theme_diversity_warnings.append({
                        "theme": theme,
                        "pool_item_count": item_count,
                        "ws_paragraph_count": 0,
                        "warning": (
                            f"Theme '{theme}' has {item_count} items in the WS pool "
                            "but anchors 0 paragraphs after all fill calls — content issue"
                        ),
                        "action": "INVESTIGATE_UPSTREAM",
                    })

        if theme_diversity_warnings:
            logger.error("THEME DIVERSITY WARNINGS: %s", json.dumps(theme_diversity_warnings, indent=2))

        # ---------------------------------------------------------------------------
        # Build display payload
        # ---------------------------------------------------------------------------
        ws_display_payload = [
            {
                "headline": ws.get("headline", ""),
                "paragraph": ws.get("paragraph", ""),
                "source_indices": ws.get("source_indices", []),
                "theme": _get_theme_for_ws(ws, ws_indexed, source_index_lookup),
            }
            for ws in ws_paragraphs
            if ws.get("paragraph", "").strip()
        ]

        logger.info(
            "WS DISPLAY PAYLOAD: %d paragraphs, themes: %s",
            len(ws_display_payload),
            [ws.get("theme") for ws in ws_display_payload],
        )

        return {
            "whats_shifting": ws_display_payload,
            "company_watch": normalized_company_watch,
            "startup_radar": normalized_startup_radar,
            "pm_craft_today": pm_craft_today,
            "interview_angle": interview_angle,
            "source_index_lookup": source_index_lookup,
            "editorial_warnings": {
                "multi_thread_violations": multi_thread_warnings,
                "date_warnings": date_warnings,
                "coherence_warnings": coherence_warnings,
                "routing_warnings": routing_warnings,
                "omit_rule_violations": omit_rule_warnings,
                "cw_source_integrity_violations": cw_source_integrity_violations,
                "pm_craft_source_violations": pm_craft_source_violations,
                "source_concentration_warnings": source_concentration_warnings,
                "split_implication_warnings": split_implication_warnings,
                "theme_audit_warnings": theme_audit_warnings,
                "theme_diversity_warnings": theme_diversity_warnings,
                "call1_anchor_reasoning_debug": call1_parsed.get("_call1_reasoning", ""),
                "call1a_cross_market_debug": cross_market_map,
                "call3_anchor_reasoning_debug": call2_parsed.get("_call3_reasoning", ""),
                "call2b_anchor_reasoning_debug": call2_parsed.get("_call2b_reasoning", ""),
                "call4a_startup_debug": startup_map,
                "fill_anchor_reasoning_debug": fill_reasonings,
            },
        }

    except Exception as exc:
        logger.exception("Exception during synthesis: %s", exc)
        raise