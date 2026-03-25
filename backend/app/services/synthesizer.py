from __future__ import annotations

import json
import logging
import re
import traceback
from datetime import date
from typing import Any, Dict, List, Tuple

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
# System prompt — shared across both calls
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


def _build_client() -> Anthropic:
    settings = load_settings()
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. "
            "Populate it in your .env file before running the synthesizer."
        )
    return Anthropic(api_key=settings.anthropic_api_key)


def _extract_json(text: str) -> str:
    """
    Best-effort extraction of a JSON object from a Claude reply, handling
    ```json ... ``` and ``` ... ``` fenced code blocks.
    """
    if not text:
        return text

    json_fence = re.search(r"```json(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if json_fence:
        return json_fence.group(1).strip()

    generic_fence = re.search(r"```(.*?)```", text, flags=re.DOTALL)
    if generic_fence:
        return generic_fence.group(1).strip()

    return text.strip()


def _build_context_block(
    items: List[Dict[str, Any]],
    start_idx: int = 1,
) -> Tuple[str, List[Dict[str, Any]], int]:
    """
    Build a numbered context block from a list of items.
    Returns (context_block, indexed_items, next_idx).
    """
    lines: List[str] = []
    indexed_items: List[Dict[str, Any]] = []
    idx = start_idx

    for item in items:
        insights = item["insights"]
        if not isinstance(insights, list):
            insights = [str(insights)]

        # Derive allowed section from theme
        theme = item["theme"]
        if theme == "company_strategy":
            allowed_section = "company_watch ONLY"
        elif theme == "startup_disruption":
            allowed_section = "startup_radar ONLY"
        elif theme == "product_craft":
            allowed_section = "pm_craft_today ONLY"
        else:
            allowed_section = "any dedicated section"

        lines.append(f"Item [{idx}]:")
        lines.append(f"- Theme: {item['theme']}")
        lines.append(f"- Allowed section: {allowed_section}")
        lines.append(f"- Source: {item['source_name']}")
        lines.append(f"- Title: {item['title']}")
        lines.append("- Insights:")
        for bullet in insights:
            lines.append(f"  - {bullet}")
        lines.append("")

        indexed_items.append({
            "index": idx,
            "theme": item["theme"],
            "title": item["title"],
            "source_name": item["source_name"],
        })
        idx += 1

    return "\n".join(lines), indexed_items, idx


def _normalize_whats_shifting(raw: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    if not isinstance(raw, list):
        return [{"paragraph": str(raw), "source_indices": []}]
    for entry in raw:
        if isinstance(entry, dict):
            paragraph = entry.get("paragraph") or entry.get("text") or ""
            indices = entry.get("source_indices") or entry.get("sources") or []
        else:
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
        normalized.append({"paragraph": paragraph, "source_indices": cleaned})
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
    return {"text": text, "source_indices": cleaned}


# ---------------------------------------------------------------------------
# Call 1: What's Shifting + Interview Angle
# Uses ONLY whats_shifting_eligible items — structural routing enforcement
# ---------------------------------------------------------------------------

def _call_whats_shifting(
    client: Anthropic,
    settings: Any,
    ws_items: List[Dict[str, Any]],
    today: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Generate whats_shifting paragraphs and interview_angle.
    Only sees WHATS_SHIFTING_ELIGIBLE items — cannot cite dedicated section sources.
    """
    context_block, ws_indexed, _ = _build_context_block(ws_items, start_idx=1)

    user_prompt = f"""
You are reasoning across multiple high/medium confidence items that a Senior PM is tracking.
Today's date is {today}.

You are given items eligible for What's Shifting analysis. Use these to produce whats_shifting paragraphs and an interview_angle.

Items:
{context_block}

Produce a structured JSON object:
{{
  "whats_shifting": [
    {{
      "paragraph": "4-5 paragraph-length insights. Each paragraph must: "
                   "(1) open with a single declarative sentence naming the underlying force or pattern — not an event description; "
                   "(2) develop the insight across 3-4 sentences by connecting signals from different sources or themes to reveal something non-obvious; "
                   "(3) close with the strategic implication for a PM — what decision, risk, or opportunity does this pattern create? "
                   "The implication must be directly derivable from the cited sources. "
                   "Each sentence ends with inline [n] citations. Only cite [n] if a specific bullet from item [n] directly supports that sentence. "
                   "READER CONTEXT RULE: Write for a reader who has NOT seen the source articles. Provide plain-language context for any unfamiliar term or company. "
                   "LEDE PRECISION RULE: Opening sentence makes a claim the paragraph must fully deliver. Avoid absolute framing unless a source explicitly uses it. "
                   "IMPLICATION FOCUS RULE: Closing PM implication makes exactly one claim. If writing 'and' connecting two consequences, cut one. "
                   "SPLIT IMPLICATION SELF-CHECK: Count distinct actionable consequences in closing sentence. If more than one, cut the weaker. "
                   "EXAMPLE DISCIPLINE RULE: No more than three distinct examples per paragraph. "
                   "MINIMUM VIABLE PARAGRAPH RULE: If fewer than two examples pass the connective tissue test, do not publish the paragraph. "
                   "THEME AUDIT SELF-CHECK: Before finalizing, list the central theme of each paragraph opening sentence. "
                   "Eligible themes: AI & technology, market behavior, consumer behavior, regulation & policy, design & UX. "
                   "No theme should appear as the central claim of more than one paragraph. "
                   "If any theme appears twice, rewrite the weaker paragraph around a different theme. "
                   "A four-paragraph brief with four distinct themes is better than five paragraphs with a duplicate theme. "
                   "DATE VALIDATION RULE: Today's date is {today}. If any milestone date is earlier than today, flag it with [DATE CHECK: this date may already have passed].",
      "source_indices": [1, 2]
    }}
  ],
  "interview_angle": "One specific thing a PM should have a prepared opinion on before interviews this week. "
                     "Anchor to a specific named company, case, or development from today's sources. "
                     "Frame as a debatable claim or tradeoff, not a fact to recite. "
                     "Rotate focus across product strategy, consumer insight, regulatory navigation, and AI. "
                     "PM DECISION LEVEL RULE: The angle must be grounded in a decision a PM actually owns — "
                     "feature prioritization, product architecture, safety design, retention mechanics, "
                     "compliance strategy, pricing tradeoffs, or go-to-market sequencing. "
                     "Do not anchor to decisions owned by executives, infrastructure teams, or investors "
                     "(e.g. compute allocation, M&A timing, fundraising strategy, CEO org changes). "
                     "If the most interesting story today is an exec-level decision, reframe it as: "
                     "what should a PM building on that platform or in that market decide differently as a result? "
                     "VERIFIED MOTIVATION RULE: Only assert a company's strategic motivation if it is explicitly "
                     "stated in the source. Do not infer why a company made a decision and present it as fact. "
                     "If the motivation is unclear, frame the angle around the observable outcome and the "
                     "PM-level tradeoff it reveals, not the company's presumed intent. "
                     "STRONG ANCHOR PREFERENCE: Prefer stories where the source explicitly names a product "
                     "decision, design tradeoff, or architectural choice over stories where the PM implication "
                     "must be inferred from a business event. "
                     "Good anchors from today's typical sources: permission model design tradeoffs, "
                     "content moderation architecture, safety-as-product-surface decisions, "
                     "platform protocol design, compliance-as-feature tradeoffs. "
                     "Weak anchors: company shutdowns where motivation is unconfirmed, exec org changes, "
                     "fundraising rounds without product detail."
}}

Guidance:
- INSIGHT DEPTH RULE: Every whats_shifting paragraph must reveal something a reader could NOT get from any single source. If your paragraph could have been written from a single source, rewrite it.
- PM ACTIONABILITY RULE: When finalizing closing implications, ask: does this tell a PM what to do differently, or does it tell them something interesting? Prefer concrete architectural decisions, pricing tradeoffs, measurement approaches, or design patterns over market trend observations without concrete action.
- GROUNDING RULE: Do not introduce external statistics, historical references, or general knowledge in implication sentences. Use 'this suggests...' or 'this implies...' rather than asserting as established fact.
- CITATION RULE: Only cite item [n] if a specific insight bullet from that item directly supports the exact claim. Every sentence in whats_shifting must have at least one citation.
- REFRAMING RULE: Do not reproduce a named framework from a source as your insight. Ask what it reveals when placed alongside other signals.
- MULTI-SOURCE DEPTH RULE: When a paragraph draws from 3+ sources, verify that at least one source contributes content from beyond its first insight bullet. If all evidence comes from bullet 1 of each source, the paragraph is missing the most specific and verifiable content.
- REGULATORY CLUSTER RULE: When multiple regulatory sources (municipal, national, international) are grouped into a single paragraph, they must share a mechanistic connection beyond surface theme similarity. Different enforcement mechanisms, different legal theories, and different compliance surfaces are not unified by the label 'regulation.' If three regulatory sources do not share a single mechanistic implication for product teams, split into two paragraphs or drop the weakest story.
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

    cleaned = _extract_json(text)
    try:
        parsed = json.loads(cleaned)
    except Exception:
        logger.warning("Call 1 response was not valid JSON. Raw (first 500): %s", text[:500] if text else "")
        parsed = {"whats_shifting": [], "interview_angle": ""}

    return parsed, ws_indexed


# ---------------------------------------------------------------------------
# Call 2: Company Watch + Startup Radar + PM Craft
# Uses ONLY dedicated_section_eligible items — structural routing enforcement
# ---------------------------------------------------------------------------

def _call_dedicated_sections(
    client: Anthropic,
    settings: Any,
    dedicated_items: List[Dict[str, Any]],
    today: str,
    start_idx: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Generate company_watch, startup_radar, and pm_craft_today.
    Only sees DEDICATED_SECTION_ELIGIBLE items — cannot cite whats_shifting sources.
    """
    context_block, dedicated_indexed, _ = _build_context_block(dedicated_items, start_idx=start_idx)

    user_prompt = f"""
You are reasoning across multiple high/medium confidence items that a Senior PM is tracking.
Today's date is {today}.

You are given items eligible for Company Watch, Startup Radar, and PM Craft. Use these to produce company_watch entries, startup_radar bullets, and pm_craft_today.

Items:
{context_block}

Produce a structured JSON object:
{{
  "company_watch": {{
    "Google": {{
      "paragraph": "2-3 sentences of strategic signal. "
                   "Sentence 1: name what is strategically changing for this company — not news, but a shift in positioning, priority, or competitive stance. "
                   "Sentence 2: provide the evidence from cited sources with inline [n] citations. "
                   "Sentence 3 (optional): name the implication — one claim only, the most specific and directly grounded. "
                   "Only include this company if there is genuine signal today from the items provided. "
                   "COMPANY WATCH OMIT RULE: If no item directly covers this company's strategy or product moves, set paragraph to empty string. Do not substitute a tangentially related item. "
                   "COMPANY WATCH CONVERGENCE RULE: Multiple threads allowed only if all threads converge on a single closing implication. "
                   "If the closing sentence does not follow from all threads, cut to the strongest single thread. "
                   "LEDE PRECISION RULE: Avoid absolute framing unless a source explicitly uses it. "
                   "IMPLICATION FOCUS RULE: Sentence 3 must make exactly one claim. Cut any 'and' connecting two consequences. "
                   "BULLET DEPTH RULE: Do not build this entry from the first insight bullet alone. Read all insight bullets for this company's sources before writing. "
                   "The most specific and verifiable content (named products, specific numbers, architectural details) is often in bullets 2-4, not bullet 1. "
                   "METRICS PRESERVATION RULE: If a source contains a specific number (dollar amount, percentage, named product, date), include it if it supports the entry. Named companies, products, and dollar figures ground the entry. "
                   "SCOPE FIDELITY RULE: Reflect the actual scope stated in the source. If a source explicitly limits scope (e.g. 'non-safety parts only'), the entry must reflect that limit, not expand it. "
                   "INFERENCE BOUNDARY RULE: Do not assert competitive framings, strategic motivations, or market positions not explicitly stated in the source. "
                   "If the source says 'open infrastructure for non-safety functions,' do not write 'competing with safety-critical vendors.' "
                   "If the source says 'minority investment participant,' do not write 'vertically integrating' or 'infrastructure ownership.' "
                   "If a source shows a 1.75x fund size increase, do not assert '3-5x capital requirements' — that multiplier is not in the source.",
      "source_indices": []
    }},
    "Meta": {{"paragraph": "2-3 sentences of strategic signal. Same rules as Google.", "source_indices": []}},
    "Apple": {{"paragraph": "2-3 sentences of strategic signal. Same rules as Google.", "source_indices": []}},
    "Amazon": {{"paragraph": "2-3 sentences of strategic signal. Same rules as Google.", "source_indices": []}},
    "Netflix": {{"paragraph": "2-3 sentences of strategic signal. Same rules as Google.", "source_indices": []}},
    "Microsoft": {{"paragraph": "2-3 sentences of strategic signal. Same rules as Google.", "source_indices": []}},
    "NVIDIA": {{"paragraph": "2-3 sentences of strategic signal. Same rules as Google.", "source_indices": []}},
    "OpenAI": {{"paragraph": "2-3 sentences of strategic signal. Same rules as Google.", "source_indices": []}},
    "Anthropic": {{"paragraph": "2-3 sentences of strategic signal. Same rules as Google.", "source_indices": []}}
  }},
  "startup_radar": [
    {{
      "bullet": "2-3 items on early-stage or emerging companies making unexpected moves. "
                "Structure each bullet as: [what the company did] + [why it matters strategically] + [what pattern or shift it represents]. "
                "Only include early-stage or emerging companies — do not include established large-cap companies. "
                "IMPLICATION FOCUS RULE: Each bullet must close with exactly one strategic consequence. Cut any 'and' connecting two separate consequences. "
                "METRICS PRESERVATION RULE: Include the funding amount, round size, or key metric from the source. Do not omit specific numbers that ground the strategic claim. "
                "INFERENCE BOUNDARY RULE: Do not assert a specific multiplier, ratio, or benchmark unless it appears verbatim in the source. Inferred benchmarks must use 'suggests' or 'implies' framing, never assertion.",
      "source_indices": []
    }}
  ],
  "pm_craft_today": {{
    "text": "Single most actionable PM craft insight from today's content. "
            "Draw from product_craft or design_ux items first. If none available, use startup_disruption or company_strategy items. "
            "Must be non-obvious — a specific pattern, tradeoff, or reframe that changes how a PM would approach a real decision. "
            "Avoid generic advice. Name the specific insight: what assumption does it challenge, what decision does it change, or what pattern does it reveal? "
            "Write for a reader who has NOT read the source. "
            "If no craft-relevant insight exists, set text to empty string.",
    "source_indices": []
  }}
}}

Guidance:
- SECTION ROUTING RULE: Each item is tagged with an "Allowed section" field. This is a hard constraint, not a suggestion.
    Items tagged "company_watch ONLY" (theme: company_strategy) may ONLY appear in company_watch entries.
    Items tagged "startup_radar ONLY" (theme: startup_disruption) may ONLY appear in startup_radar bullets.
    Items tagged "pm_craft_today ONLY" (theme: product_craft) may ONLY appear in pm_craft_today.
  A TechCrunch article about Amazon is tagged startup_disruption → startup_radar ONLY. Do not use it in company_watch even if it describes a major company's strategy. Company Watch entries must be built exclusively from company_strategy sources (official company blogs, newsrooms, first-party announcements). If no company_strategy item covers a given company today, set that company's paragraph to empty string — do not substitute a startup_disruption item.
- COMPANY WATCH GROUNDING RULE: Do not connect two separate signals for the same company into a causal narrative unless that connection is explicitly made in the sources. Each sentence must be traceable to a specific cited source. Do not infer competitive intent, strategic motivation, or market position that the source does not explicitly state. Scope fidelity is required: if a source limits a product to 'non-safety functions,' do not frame it as competing with safety-critical vendors. If a source describes a minority VC investment, do not frame it as vertical integration or infrastructure ownership. If a source shows a fund grew 1.75x, do not assert industry-wide capital requirements are 3-5x higher.
- COMPANY WATCH INSIGHT RULE: Each company paragraph must answer 'what is strategically shifting for this company today' — not just 'what did they do.'
- STARTUP RADAR RULE: Each bullet must contain a genuine 'so what' — the strategic implication, competitive threat, or market pattern revealed, not just a description of the event.
- PM CRAFT OMIT RULE: If no item contains a craft-relevant insight, set pm_craft_today text to empty string rather than forcing a weak insight.
- PM ACTIONABILITY RULE: Prefer concrete product design consequences over strategic observations. Does this tell a PM what to build or decide differently?
- IMPLICATION FOCUS RULE: Every closing implication must make exactly one claim.
- DATE VALIDATION RULE: Today's date is {today}. If any milestone date is earlier than today, flag it with [DATE CHECK: this date may already have passed].
- CITATION RULE: Only cite item [n] if a specific insight bullet directly supports the exact claim.
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

    cleaned = _extract_json(text)
    try:
        parsed = json.loads(cleaned)
    except Exception:
        logger.warning("Call 2 response was not valid JSON. Raw (first 500): %s", text[:500] if text else "")
        parsed = {
            "company_watch": {},
            "startup_radar": [],
            "pm_craft_today": {"text": "", "source_indices": []},
        }

    return parsed, dedicated_indexed


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def synthesize_trends(grouped_summaries: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Run a two-call synthesis across all summarized items.

    Call 1: What's Shifting + Interview Angle (WHATS_SHIFTING_ELIGIBLE items only)
    Call 2: Company Watch + Startup Radar + PM Craft (DEDICATED_SECTION_ELIGIBLE items only)

    Structural routing enforcement: each call only sees items it is allowed to cite.
    The model cannot violate routing because the wrong items are never in its context.

    Filters to items that are:
      - confidence: high or medium
      - pm_relevance_score: high or medium
    """
    client = _build_client()
    settings = load_settings()
    today = date.today().strftime("%B %d, %Y")

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
                "theme": theme,
                "title": item.get("title", ""),
                "source_name": item.get("source_name", ""),
                "insights": item.get("insights") or [],
                "confidence": conf_raw,
                "pm_relevance_score": relevance_raw,
            })

    if dropped_low_confidence:
        logger.info("Dropped %d items with low confidence before synthesis.", dropped_low_confidence)
    if dropped_low_relevance:
        logger.info("Dropped %d items with low PM relevance before synthesis.", dropped_low_relevance)

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
        },
    }

    if not filtered_items:
        logger.warning("No eligible items available for synthesis after filtering.")
        return empty_result

    # ---------------------------------------------------------------------------
    # Partition by routing eligibility
    # ---------------------------------------------------------------------------
    ws_items = [i for i in filtered_items if i["theme"] in WHATS_SHIFTING_THEMES]
    dedicated_items = [i for i in filtered_items if i["theme"] in DEDICATED_SECTION_THEMES]

    logger.info(
        "Routing: %d whats_shifting items, %d dedicated section items",
        len(ws_items), len(dedicated_items)
    )

    try:
        # ---------------------------------------------------------------------------
        # Call 1: What's Shifting + Interview Angle
        # ---------------------------------------------------------------------------
        call1_parsed, ws_indexed = _call_whats_shifting(client, settings, ws_items, today)

        # ---------------------------------------------------------------------------
        # Call 2: Company Watch + Startup Radar + PM Craft
        # Sequential numbering continues from where Call 1 left off
        # ---------------------------------------------------------------------------
        start_idx = len(ws_indexed) + 1
        call2_parsed, dedicated_indexed = _call_dedicated_sections(
            client, settings, dedicated_items, today, start_idx=start_idx
        )

        # ---------------------------------------------------------------------------
        # Merge indexed items and build source_index_lookup
        # ---------------------------------------------------------------------------
        indexed_items = ws_indexed + dedicated_indexed
        source_index_lookup: Dict[str, Dict[str, Any]] = {}
        for entry in indexed_items:
            source_index_lookup[str(entry["index"])] = {
                "title": entry["title"],
                "source_name": entry["source_name"],
                "theme": entry["theme"],
            }

        # ---------------------------------------------------------------------------
        # Normalize outputs
        # ---------------------------------------------------------------------------
        normalized_whats_shifting = _normalize_whats_shifting(call1_parsed.get("whats_shifting") or [])
        interview_angle = str(call1_parsed.get("interview_angle") or "")

        normalized_company_watch = _normalize_company_watch(call2_parsed.get("company_watch") or {})
        normalized_startup_radar = _normalize_startup_radar(call2_parsed.get("startup_radar") or [])
        pm_craft_today = _normalize_pm_craft(call2_parsed.get("pm_craft_today") or {})

        # ---------------------------------------------------------------------------
        # Post-processing validators
        # ---------------------------------------------------------------------------

        ws_eligible_indices = {entry["index"] for entry in ws_indexed}
        dedicated_eligible_indices = {entry["index"] for entry in dedicated_indexed}

        # 1. Multi-thread / single thesis check for company_watch
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

        all_paragraphs = []
        for ws in normalized_whats_shifting:
            all_paragraphs.append(("whats_shifting", ws.get("paragraph", "")))
        for company, value in normalized_company_watch.items():
            all_paragraphs.append((f"company_watch.{company}", value.get("paragraph", "")))
        for sr in normalized_startup_radar:
            all_paragraphs.append(("startup_radar", sr.get("bullet", "")))
        all_paragraphs.append(("pm_craft_today", pm_craft_today.get("text", "")))

        month_year_pattern = re.compile(
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)'
            r'\s+(20\d{2})\b'
        )
        months_map = {
            "January": 1, "February": 2, "March": 3, "April": 4,
            "May": 5, "June": 6, "July": 7, "August": 8,
            "September": 9, "October": 10, "November": 11, "December": 12
        }

        for section, paragraph in all_paragraphs:
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

        # 3. Cross-paragraph coherence: flag sources shared across multiple company entries
        source_to_companies: Dict[int, List[str]] = {}
        for company, value in normalized_company_watch.items():
            for i in value.get("source_indices", []):
                if i not in source_to_companies:
                    source_to_companies[i] = []
                source_to_companies[i].append(company)

        ws_all_indices: List[int] = []
        for ws in normalized_whats_shifting:
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

        # Sanity check: with two-call architecture WS and CW cannot share sources by design
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

        # 4. Routing canary — should always be empty with two-call architecture
        routing_warnings = []

        for i, ws in enumerate(normalized_whats_shifting):
            for idx_val in ws.get("source_indices", []):
                if idx_val in dedicated_eligible_indices:
                    source_info = source_index_lookup.get(str(idx_val), {})
                    routing_warnings.append({
                        "section": f"whats_shifting[{i}]",
                        "source_index": idx_val,
                        "source_title": source_info.get("title", "unknown"),
                        "source_theme": source_info.get("theme", "unknown"),
                        "warning": "CANARY: DEDICATED_SECTION source cited in whats_shifting — partitioning may have failed"
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
                        "warning": "CANARY: WHATS_SHIFTING source cited in company_watch — partitioning may have failed"
                    })

        if routing_warnings:
            logger.warning("ROUTING CANARY FIRED: %s", json.dumps(routing_warnings, indent=2))
            print("ROUTING CANARY FIRED — investigate partitioning logic:")
            for w in routing_warnings:
                print(json.dumps(w, indent=2))

        # 5. Omit rule canary — should always be empty with two-call architecture
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
                        "warning": "CANARY: Company Watch cites WS-eligible source — partitioning may have failed",
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
                    "warning": "CANARY: PM Craft cites WS-eligible source — partitioning may have failed",
                    "action": "INVESTIGATE"
                })

        if omit_rule_warnings:
            logger.warning("OMIT RULE CANARY: %s", json.dumps(omit_rule_warnings, indent=2))
            print("OMIT RULE CANARY FIRED:")
            for w in omit_rule_warnings:
                print(json.dumps(w, indent=2))

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
                        if before_words >= 6 and after_words >= 6:
                            split_implication_warnings.append({
                                "section": section,
                                "signal": signal.strip(),
                                "closing_sentence": sentences[-1][:200],
                                "warning": "Possible split implication — review for two separate consequences"
                            })
                            break

        for i, ws in enumerate(normalized_whats_shifting):
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

        # 7. Theme audit for What's Shifting
        theme_audit_warnings = []
        ws_theme_counts: Dict[str, List[int]] = {}
        for i, ws in enumerate(normalized_whats_shifting):
            indices = ws.get("source_indices", [])
            if not indices:
                continue
            primary_idx = indices[0]
            source_info = source_index_lookup.get(str(primary_idx), {})
            theme = source_info.get("theme", "unknown")
            if theme not in ws_theme_counts:
                ws_theme_counts[theme] = []
            ws_theme_counts[theme].append(i)

        for theme, paragraph_indices in ws_theme_counts.items():
            if len(paragraph_indices) > 1:
                theme_audit_warnings.append({
                    "theme": theme,
                    "paragraph_indices": paragraph_indices,
                    "warning": f"Theme '{theme}' anchors {len(paragraph_indices)} What's Shifting paragraphs — should appear at most once",
                    "action": "REWRITE_DUPLICATE_RECOMMENDED"
                })

        if theme_audit_warnings:
            logger.warning("THEME AUDIT WARNINGS: %s", json.dumps(theme_audit_warnings, indent=2))
            print("THEME AUDIT WARNINGS:")
            for w in theme_audit_warnings:
                print(json.dumps(w, indent=2))

        # ---------------------------------------------------------------------------
        # Return
        # ---------------------------------------------------------------------------
        return {
            "whats_shifting": normalized_whats_shifting,
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
                "split_implication_warnings": split_implication_warnings,
                "theme_audit_warnings": theme_audit_warnings,
            },
        }

    except Exception as exc:
        print("Exception during synthesis:", exc)
        traceback.print_exc()
        raise