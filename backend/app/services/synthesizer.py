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
    "ai_technology",
    "market_behavior",
    "consumer_behavior",
    "regulation_policy",
    "design_ux",
}

DEDICATED_SECTION_THEMES = {
    "company_strategy",
    "startup_disruption",
    "product_craft",
}


# ---------------------------------------------------------------------------
# System prompt — shared across all calls
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a senior intelligence analyst briefing a Product Manager who is "
    "actively interviewing at top tech companies including Google, Microsoft, "
    "Apple, Meta, Amazon, Netflix, NVIDIA, and OpenAI. Your job is to "
    "reason across multiple sources and surface what is actually shifting in "
    "the industry — not what happened, but what it means and what patterns are emerging. "
    "For every insight, ask: what would a reader NOT get from reading any single source? "
    "A good insight names the underlying force driving multiple seemingly unrelated events, "
    "challenges a conventional assumption, or identifies a second-order consequence that "
    "practitioners haven't yet articulated. Avoid insights that merely restate a trend with a PM gloss. "
    "PM ACTIONABILITY STANDARD: Across all sections, when choosing between a strategic observation "
    "and a concrete product design consequence, always prefer the latter. "
    "A specific mechanical implication that tells a PM what decision to make, what assumption to test, "
    "or what design pattern to apply is always stronger than a generalizable pattern observation. "
    "Test every closing implication sentence: could a PM walk into a meeting tomorrow and use this "
    "to change a decision? If the answer is 'it depends on context' or 'it is a useful frame,' "
    "the implication is too abstract — rewrite it. "
    "The broad observation is usually derivable from the headline. "
    "The specific mechanical consequence requires reading the full content. Keep the latter. "
    "A sharp PM should be able to walk into any interview and have a prepared opinion on the insights you surface."
)


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
    """
    Extract <reasoning>...</reasoning> block from Claude response.
    Returns (reasoning_text, remaining_text_with_json).
    """
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
        elif theme == "startup_disruption":
            allowed_section = "startup_radar ONLY"
        elif theme == "product_craft":
            allowed_section = "pm_craft_today ONLY"
        elif theme == "design_ux":
            allowed_section = "pm_craft_today eligible (design_ux)"
        else:
            allowed_section = "any dedicated section"

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
        # The prompt instructs 20 words max but Claude occasionally exceeds it
        # for complex topics. This is the code-level enforcement.
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
    Identify the theme a WS paragraph belongs to by looking up its
    primary source index. Falls back to keyword classification.
    """
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
    """
    Return (relevance_score, confidence_score) for a given source index.
    Lower numbers = better (0=high, 1=medium/low).
    Used for deduplication tie-breaking.
    """
    if source_index is None:
        return (1, 1)
    # Find item_id for this index
    item_id = None
    for entry in ws_indexed:
        if entry["index"] == source_index:
            item_id = entry.get("item_id")
            break
    if not item_id:
        return (1, 1)
    # Find the item in ws_items
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
    """
    Return the set of themes covered by non-empty WS paragraphs.
    """
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
    """
    Ensure at most one paragraph per theme.
    When duplicates exist, keep the paragraph anchored to the
    highest relevance/confidence source for that theme.
    """
    seen_themes: Dict[str, Dict[str, Any]] = {}

    for ws in ws_paragraphs:
        if not ws.get("paragraph", "").strip():
            continue
        theme = _get_theme_for_ws(ws, ws_indexed, source_index_lookup)

        if theme not in seen_themes:
            seen_themes[theme] = ws
        else:
            # Compare anchor quality — lower score tuple = better
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
# Call 1: What's Shifting + Interview Angle (all WS themes together)
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
(2) Name the highest-ranked bullet and explain why it is the anchor.
(3) List any bullets that contradict or qualify the anchor's claim and how you will address them.

Then produce a JSON object with this exact structure:
{{
  "whats_shifting": [
    {{
      "headline": "One declarative sentence, maximum 20 words, naming the underlying structural force or pattern. Not an event description. Renders as the visible collapsed card headline — scannable and self-contained.",
      "paragraph": "Open by restating the headline claim with one additional clause of context. Develop across 3-4 sentences connecting signals from different sources to reveal something non-obvious. Close with one PM implication directly traceable to a cited source. Every sentence ends with inline [n] citations. HEDGE MATCH: match source hedge levels throughout — 'suggests' not 'demonstrates'. NO TIMELINE: omit any timeline not verbatim in a source. NO UNIVERSALITY: scope claims to actual examples. Draw from at least 2 distinct insight bullets — use the best available rather than omitting the paragraph. CLOSING SENTENCE: single consequence, source-traceable, no constructed PM actions.",
      "source_indices": [1, 2]
    }}
  ],
  "interview_angle": "One specific tradeoff a PM should have a prepared opinion on this week, derived from a whats_shifting source. Empty string if no whats_shifting paragraph was produced. Frame as a debatable tradeoff at PM decision level — feature prioritization, architecture, safety design, retention, compliance, pricing, or go-to-market. Scope to the context the source describes."
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
    print("Raw Claude Call 1 (WS) response text:")
    print(text)

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
# Per-theme fill call: produces exactly one paragraph for one missing theme
# ---------------------------------------------------------------------------

def _call_whats_shifting_single_theme(
    client: Anthropic,
    settings: Any,
    theme_items: List[Dict[str, Any]],
    anchor: Dict[str, Any],
    today: str,
    ws_indexed: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Targeted call for a single WS theme that Call 1 failed to produce.
    Sends only that theme's items. Returns a single normalized WS entry
    or None if Claude still produces nothing usable.
    """
    if not theme_items:
        logger.warning("FILL [theme=%s]: No items available — skipping", anchor["theme"])
        return None

    # The fill call uses the same index numbers as ws_indexed so citations
    # remain consistent with the main source_index_lookup.
    # Build a context block using the existing indices from ws_indexed.
    lines: List[str] = []
    for item in theme_items:
        # Find this item's index in ws_indexed by item_id
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

    user_prompt = f"""
You are producing exactly one What's Shifting paragraph for a Senior PM digest.
Today's date is {today}.

Theme: {theme}
Anchor item: "{anchor_title}"

MANDATE: Produce exactly one paragraph for this theme. This is a targeted fill call — the main synthesis pass did not produce a paragraph for this theme. Output is required. A paragraph with 2 bullets and an imperfect closing is better than no paragraph.

Items for this theme:
{context_block}

First, write your reasoning inside <reasoning>...</reasoning> tags:
(1) List all insight bullets ranked by non-obviousness.
(2) Name your anchor bullet and why.
(3) Note any contradicting bullets and how you will address them.

Then produce a JSON object:
{{
  "headline": "One declarative sentence, maximum 20 words, naming the structural force or pattern. Scannable, self-contained, not an event description.",
  "paragraph": "3-5 sentences. Open with the headline claim plus one clause of context. Develop with 2+ bullets from the items above. Close with one PM implication traceable to a cited source. Every sentence has inline [n] citations. HEDGE MATCH: match source hedge levels. NO TIMELINE unless verbatim in source. NO UNIVERSALITY beyond actual examples.",
  "source_indices": []
}}

IMPORTANT: Do not include anchor_reasoning in the JSON. All reasoning goes in <reasoning> only.
CITATION RULE: Use the item index numbers shown above (e.g. [3], [7]) exactly as written.
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
    print(f"Raw Claude Fill Call (theme={theme}) response text:")
    print(text)

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
# Call 2: Company Watch + Startup Radar + PM Craft
# ---------------------------------------------------------------------------

def _call_dedicated_sections(
    client: Anthropic,
    settings: Any,
    dedicated_items: List[Dict[str, Any]],
    today: str,
    start_idx: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    context_block, dedicated_indexed, _ = _build_context_block(dedicated_items, start_idx=start_idx)

    user_prompt = f"""
You are reasoning across multiple high/medium confidence items that a Senior PM is tracking.
Today's date is {today}.

You are given items eligible for Company Watch, Startup Radar, and PM Craft.

Items:
{context_block}

First, write your anchor selection reasoning inside <reasoning>...</reasoning> tags.
For each company with available items, for each startup radar item, and for pm_craft_today:
(1) List ALL insight bullets ranked by non-obviousness.
(2) Name the highest-ranked bullet and explain why it is the anchor.
(3) Note any contradicting bullets and how you will address them.

Then produce a JSON object. Do not include anchor_reasoning fields inside the JSON.

{{
  "company_watch": {{
    "Google": {{
      "paragraph": "2-3 sentences of strategic signal. Sentence 1: what is strategically changing — not news, but a shift in positioning, priority, or competitive stance. Sentence 2: evidence with inline [n] citations. Sentence 3 (optional): one implication, most specific and directly grounded. OMIT RULE: empty string if no company_watch ONLY item matches this company. SOURCE RULE: only cite items tagged company_watch ONLY whose Company field matches. HEDGE MATCH throughout. CLOSING: single consequence, source-traceable, no constructed framings.",
      "source_indices": []
    }},
    "Meta": {{"paragraph": "Same rules as Google. Empty string if no matching item.", "source_indices": []}},
    "Apple": {{"paragraph": "Same rules as Google. Empty string if no matching item.", "source_indices": []}},
    "Amazon": {{"paragraph": "Same rules as Google. Empty string if no matching item.", "source_indices": []}},
    "Netflix": {{"paragraph": "Same rules as Google. Empty string if no matching item.", "source_indices": []}},
    "Microsoft": {{"paragraph": "Same rules as Google. Empty string if no matching item.", "source_indices": []}},
    "NVIDIA": {{"paragraph": "Same rules as Google. Empty string if no matching item.", "source_indices": []}},
    "OpenAI": {{"paragraph": "Same rules as Google. Empty string if no matching item.", "source_indices": []}},
    "Anthropic": {{"paragraph": "Same rules as Google. Empty string if no matching item.", "source_indices": []}}
  }},
  "startup_radar": [
    {{
      "bullet": "2-3 items on early-stage or emerging companies only. Structure: [what the company did] + [why it matters strategically] + [what pattern or shift it represents]. HEDGE MATCH. NO TIMELINE unless verbatim in source. METRICS: include funding amount or key metric. THEMATIC COMBINATION: only combine companies if you can name the specific causal mechanism both share — a shared category label is not a mechanism. CLOSING: single consequence, source-traceable, no constructed assertions.",
      "source_indices": []
    }}
  ],
  "pm_craft_today": {{
    "text": "Single most actionable PM craft insight. Draw ONLY from items tagged pm_craft_today ONLY (product_craft) OR pm_craft_today eligible (design_ux). Empty string if no such item exists. INSIGHT QUALITY: non-obvious pattern, tradeoff, or reframe that changes how a PM approaches a real decision. CLOSING: single consequence, source-traceable, no constructed PM actions.",
    "source_indices": []
  }}
}}

IMPORTANT: Do not include anchor_reasoning anywhere in the JSON. All reasoning goes in <reasoning> only.
SECTION ROUTING RULE: Items tagged company_watch ONLY → that company's entry only. startup_radar ONLY → startup_radar only. pm_craft_today ONLY → pm_craft_today only. pm_craft_today eligible (design_ux) → pm_craft_today only. Hard constraints, not suggestions.
CITATION RULE: Only cite item [n] if a specific insight bullet directly supports the exact claim.
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
    print("Raw Claude Call 2 (Dedicated) response text:")
    print(text)

    reasoning_text, text_without_reasoning = _extract_reasoning_block(text or "")
    if reasoning_text:
        logger.info("Call 2 anchor reasoning extracted (%d chars)", len(reasoning_text))

    cleaned = _extract_json(text_without_reasoning)
    try:
        parsed = json.loads(cleaned)
    except Exception:
        logger.warning("Call 2 response was not valid JSON. Raw (first 500): %s", (text or "")[:500])
        parsed = {
            "company_watch": {},
            "startup_radar": [],
            "pm_craft_today": {"text": "", "source_indices": []},
        }

    parsed["_call2_reasoning"] = reasoning_text
    return parsed, dedicated_indexed


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

            company_maturity = str(item.get("company_maturity") or "not_applicable").lower()

            if theme == "startup_disruption" and company_maturity != "startup":
                logger.info(
                    "FILTER [step=3 reason=non_startup_in_startup_radar] dropped: %s — %s",
                    item.get("source_name"), item.get("title")
                )
                continue

            filtered_items.append({
                "item_id": str(uuid.uuid4()),
                "theme": theme,
                "title": item.get("title", ""),
                "source_name": item.get("source_name", ""),
                "company_id": item.get("company_id"),
                "scope": str(item.get("scope") or "cross_market").lower(),
                "insights": item.get("insights") or [],
                "confidence": conf_raw,
                "pm_relevance_score": relevance_raw,
                "company_maturity": company_maturity,
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
    print(f"THEME FUNNEL [stage=after_quality_filter]: {theme_funnel_after_filter}")

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
        print("SOURCE CONCENTRATION WARNINGS:")
        for w in source_concentration_warnings:
            print(json.dumps(w, indent=2))

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
    print(f"THEME FUNNEL [stage=after_diversity_cap]: {theme_funnel_after_cap}")

    # ---------------------------------------------------------------------------
    # Partition by routing eligibility
    # ---------------------------------------------------------------------------
    ws_items = []
    dedicated_items = []

    for item in filtered_items:
        theme = item["theme"]
        scope = item.get("scope", "cross_market")

        if theme in DEDICATED_SECTION_THEMES:
            dedicated_items.append(item)
        elif theme in WHATS_SHIFTING_THEMES:
            if theme == "regulation_policy" and scope == "company_specific":
                logger.info(
                    "FILTER [step=4 reason=regulation_policy_company_specific_excluded_from_ws] "
                    "routed to dedicated: %s — %s",
                    item.get("source_name"), item.get("title")
                )
                dedicated_items.append(item)
            elif theme == "design_ux":
                if scope == "cross_market":
                    ws_items.append(item)
                    logger.info(
                        "ROUTING [design_ux cross_market → ws_items + dedicated_items]: %s — %s",
                        item.get("source_name"), item.get("title")
                    )
                else:
                    logger.info(
                        "ROUTING [design_ux company_specific → dedicated_items only]: %s — %s",
                        item.get("source_name"), item.get("title")
                    )
                dedicated_items.append(item)
            else:
                ws_items.append(item)

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
    print(f"THEME FUNNEL [stage=ws_items_post_partition]: {ws_theme_dist}")
    print(f"THEME FUNNEL [stage=dedicated_items_post_partition]: {dedicated_theme_dist}")

    logger.info(
        "Routing: %d whats_shifting items, %d dedicated section items",
        len(ws_items), len(dedicated_items)
    )

    # ---------------------------------------------------------------------------
    # Build required anchors — one per WS theme with items available
    # ---------------------------------------------------------------------------
    WHATS_SHIFTING_THEMES_ORDERED = [
        "ai_technology",
        "market_behavior",
        "regulation_policy",
        "consumer_behavior",
        "design_ux",
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
    print(f"REQUIRED ANCHORS: {[{'theme': a['theme'], 'title': a['anchor_item']['title']} for a in required_anchors]}")

    try:
        # ---------------------------------------------------------------------------
        # Call 1: all WS themes together
        # ---------------------------------------------------------------------------
        call1_parsed, ws_indexed = _call_whats_shifting(
            client, settings, ws_items, today,
            ws_theme_distribution=ws_theme_dist,
            required_anchors=required_anchors,
        )

        # Normalize Call 1 output into the live ws_paragraphs list
        ws_paragraphs = _normalize_whats_shifting(call1_parsed.get("whats_shifting") or [])
        ws_paragraphs = [ws for ws in ws_paragraphs if ws.get("paragraph", "").strip()]

        # Build source_index_lookup from ws_indexed now so fill calls can use it
        source_index_lookup: Dict[str, Dict[str, Any]] = {}
        for entry in ws_indexed:
            source_index_lookup[str(entry["index"])] = {
                "title": entry["title"],
                "source_name": entry["source_name"],
                "theme": entry["theme"],
                "company_id": entry.get("company_id"),
            }

        # ---------------------------------------------------------------------------
        # Step 2: detect which themes Call 1 covered
        # ---------------------------------------------------------------------------
        covered_themes = _get_covered_themes(ws_paragraphs, ws_indexed, source_index_lookup)
        missing_anchors = [a for a in required_anchors if a["theme"] not in covered_themes]

        logger.info("WS covered themes after Call 1: %s", sorted(covered_themes))
        logger.info("WS missing themes after Call 1: %s", [a["theme"] for a in missing_anchors])
        print(f"WS covered themes after Call 1: {sorted(covered_themes)}")
        print(f"WS missing themes after Call 1: {[a['theme'] for a in missing_anchors]}")

        # ---------------------------------------------------------------------------
        # Step 3: per-theme fill calls for each missing theme
        # Results are merged into ws_paragraphs immediately after each call
        # so they cannot be silently dropped downstream.
        # ---------------------------------------------------------------------------
        fill_reasonings: List[Dict[str, Any]] = []

        for anchor in missing_anchors:
            theme = anchor["theme"]
            theme_items = [i for i in ws_items if i["theme"] == theme]

            logger.info(
                "WS FILL [theme=%s]: Calling targeted fill with %d items",
                theme, len(theme_items)
            )
            print(f"WS FILL [theme={theme}]: Targeted fill call with {len(theme_items)} items")

            fill_result = _call_whats_shifting_single_theme(
                client, settings, theme_items, anchor, today, ws_indexed
            )

            if fill_result and fill_result.get("paragraph", "").strip():
                # Merge immediately — this is the guarantee that fill results
                # cannot be bypassed by subsequent processing steps.
                ws_paragraphs.append(fill_result)
                logger.info(
                    "WS FILL [theme=%s]: Merged into ws_paragraphs (%d total paragraphs now)",
                    theme, len(ws_paragraphs)
                )
                print(f"WS FILL [theme={theme}]: SUCCESS — merged. Total paragraphs: {len(ws_paragraphs)}")
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
                print(f"WS FILL [theme={theme}]: FAILED — no paragraph produced")

        # ---------------------------------------------------------------------------
        # Step 4: deduplicate — one paragraph per theme, keep best anchor
        # ---------------------------------------------------------------------------
        ws_paragraphs = _deduplicate_by_theme(
            ws_paragraphs, ws_items, ws_indexed, source_index_lookup
        )

        # ---------------------------------------------------------------------------
        # Step 5: verify final coverage before building display payload
        # ---------------------------------------------------------------------------
        final_covered = _get_covered_themes(ws_paragraphs, ws_indexed, source_index_lookup)
        still_missing = [a["theme"] for a in required_anchors if a["theme"] not in final_covered]

        if still_missing:
            logger.error(
                "WS FINAL COVERAGE GAP: themes still missing after all fill calls: %s",
                still_missing
            )
            print(f"WS FINAL COVERAGE GAP: {still_missing} — content issue, not a pipeline drop")
        else:
            logger.info(
                "WS FINAL COVERAGE: all %d required themes covered: %s",
                len(required_anchors), sorted(final_covered)
            )
            print(f"WS FINAL COVERAGE: all themes covered: {sorted(final_covered)}")

        # ---------------------------------------------------------------------------
        # Dedup: remove design_ux items consumed by WS from dedicated_items
        # ---------------------------------------------------------------------------
        ws_used_indices: Set[int] = set()
        for ws in ws_paragraphs:
            ws_used_indices.update(ws.get("source_indices", []))

        ws_used_item_ids: Set[str] = set()
        for entry in ws_indexed:
            if entry["index"] in ws_used_indices and entry.get("item_id"):
                ws_used_item_ids.add(entry["item_id"])

        if ws_used_item_ids:
            before_count = len(dedicated_items)
            dedicated_items = [
                item for item in dedicated_items
                if item.get("item_id") not in ws_used_item_ids
            ]
            removed = before_count - len(dedicated_items)
            if removed:
                logger.info(
                    "DEDUP [ws_consumed]: Removed %d item(s) from dedicated_items already consumed by WS",
                    removed
                )
                print(f"DEDUP [ws_consumed]: Removed {removed} item(s) from dedicated_items")

        # ---------------------------------------------------------------------------
        # Call 2: dedicated sections
        # ---------------------------------------------------------------------------
        start_idx = len(ws_indexed) + 1
        call2_parsed, dedicated_indexed = _call_dedicated_sections(
            client, settings, dedicated_items, today, start_idx=start_idx
        )

        # Complete source_index_lookup with dedicated items
        for entry in dedicated_indexed:
            source_index_lookup[str(entry["index"])] = {
                "title": entry["title"],
                "source_name": entry["source_name"],
                "theme": entry["theme"],
                "company_id": entry.get("company_id"),
            }

        interview_angle = _strip_date_check_flags(str(call1_parsed.get("interview_angle") or ""))
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
            print("SINGLE THESIS WARNINGS:")
            for w in multi_thread_warnings:
                print(json.dumps(w, indent=2))

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
            print("DATE WARNINGS:")
            for w in date_warnings:
                print(json.dumps(w, indent=2))

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
            print("COHERENCE WARNINGS:")
            for w in coherence_warnings:
                print(json.dumps(w, indent=2))

        # 4. Routing canary
        routing_warnings = []

        for i, ws in enumerate(ws_paragraphs):
            for idx_val in ws.get("source_indices", []):
                if idx_val in dedicated_eligible_indices:
                    source_info = source_index_lookup.get(str(idx_val), {})
                    routing_warnings.append({
                        "section": f"whats_shifting[{i}]",
                        "source_index": idx_val,
                        "source_title": source_info.get("title", "unknown"),
                        "source_theme": source_info.get("theme", "unknown"),
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

        if routing_warnings:
            logger.warning("ROUTING CANARY FIRED: %s", json.dumps(routing_warnings, indent=2))
            print("ROUTING CANARY FIRED:")
            for w in routing_warnings:
                print(json.dumps(w, indent=2))

        # 5. Omit rule canary
        omit_rule_warnings = []

        for company, value in normalized_company_watch.items():
            for idx_val in value.get("source_indices", []):
                if idx_val in ws_eligible_indices:
                    source_info = source_index_lookup.get(str(idx_val), {})
                    omit_rule_warnings.append({
                        "section": f"company_watch.{company}",
                        "source_index": idx_val,
                        "source_title": source_info.get("title", "unknown"),
                        "source_theme": source_info.get("theme", "unknown"),
                        "warning": "CANARY: Company Watch cites WS-eligible source",
                        "action": "INVESTIGATE"
                    })

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

        if omit_rule_warnings:
            logger.warning("OMIT RULE CANARY: %s", json.dumps(omit_rule_warnings, indent=2))
            print("OMIT RULE CANARY FIRED:")
            for w in omit_rule_warnings:
                print(json.dumps(w, indent=2))

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
            print("CW SOURCE INTEGRITY VIOLATIONS:")
            for v in cw_source_integrity_violations:
                print(json.dumps(v, indent=2))

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
            print("PM CRAFT SOURCE VIOLATIONS:")
            for v in pm_craft_source_violations:
                print(json.dumps(v, indent=2))
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
            print("SPLIT IMPLICATION WARNINGS:")
            for w in split_implication_warnings:
                print(json.dumps(w, indent=2))

        # 7. Theme audit
        THEME_KEYWORDS: Dict[str, List[str]] = {
            "regulation_policy": [
                "regulat", "law", "legal", "court", "legislat", "policy", "government",
                "enforcement", "compliance", "antitrust", "jurisdiction", "decree",
                "shutdown", "ban", "ruling", "verdict", "ftc", "doj", "gdpr",
                "cybercrime", "arrest", "detained", "prosecution",
            ],
            "market_behavior": [
                "market", "acqui", "merger", "ipo", "valuation", "invest", "fund",
                "compet", "price", "pricing", "revenue", "monetiz", "platform",
                "asset-light", "vertical integrat", "supply chain", "demand",
                "abandon", "pivot", "writedown", "infrastructure", "capital",
            ],
            "consumer_behavior": [
                "consumer", "user", "worker", "employee", "customer", "adoption",
                "preference", "workforce", "talent", "credential", "overqualif",
                "retention", "engagement", "behavior", "choice", "lifestyle",
            ],
            "design_ux": [
                "design", "ux", "interface", "user experience", "interaction",
                "consent", "onboarding", "friction", "accessibility", "pattern",
                "navigation", "layout", "visual",
            ],
            "ai_technology": [
                "ai ", "artificial intelligence", "model", "llm", "machine learning",
                "gpt", "gemini", "claude", "neural", "inference", "token",
                "foundation model", "generative", "autonomous", "robotaxi",
                "compute", "gpu", "blackwell", "hopper",
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
            print("THEME AUDIT WARNINGS:")
            for w in theme_audit_warnings:
                print(json.dumps(w, indent=2))

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
            print("THEME DIVERSITY WARNINGS:")
            for w in theme_diversity_warnings:
                print(json.dumps(w, indent=2))

        # ---------------------------------------------------------------------------
        # Step 6: Build display payload from ws_paragraphs — the live merged list.
        # This is the single source of truth. Call 1 parsed output is never
        # referenced again after step 1 normalization. Fill results that were
        # appended in step 3 are guaranteed to appear here.
        # ---------------------------------------------------------------------------
        ws_display_payload = [
            {
                "headline": ws.get("headline", ""),
                "paragraph": ws.get("paragraph", ""),
                "source_indices": ws.get("source_indices", []),
            }
            for ws in ws_paragraphs
            if ws.get("paragraph", "").strip()
        ]

        logger.info(
            "WS DISPLAY PAYLOAD: %d paragraphs, themes: %s",
            len(ws_display_payload),
            [_get_theme_for_ws(ws, ws_indexed, source_index_lookup) for ws in ws_display_payload],
        )
        print(
            f"WS DISPLAY PAYLOAD: {len(ws_display_payload)} paragraphs, "
            f"themes: {[_get_theme_for_ws(ws, ws_indexed, source_index_lookup) for ws in ws_display_payload]}"
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
                "call2_anchor_reasoning_debug": call2_parsed.get("_call2_reasoning", ""),
                "fill_anchor_reasoning_debug": fill_reasonings,
            },
        }

    except Exception as exc:
        print("Exception during synthesis:", exc)
        traceback.print_exc()
        raise