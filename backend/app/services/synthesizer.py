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
    if not text:
        return text
    json_fence = re.search(r"```json(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if json_fence:
        return json_fence.group(1).strip()
    generic_fence = re.search(r"```(.*?)```", text, flags=re.DOTALL)
    if generic_fence:
        return generic_fence.group(1).strip()
    return text.strip()


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
        paragraph = _strip_date_check_flags(paragraph)
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

    # Build required anchors block
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

    # Build theme availability block (kept for evaluator transparency, no longer drives selection)
    theme_availability_lines = []
    for theme, count in sorted((ws_theme_distribution or {}).items(), key=lambda x: -x[1]):
        theme_availability_lines.append(f"  - {theme}: {count} item{'s' if count != 1 else ''}")
    if theme_availability_lines:
        theme_availability_block = (
            "Available items by theme in today's pool (for reference):\n"
            + "\n".join(theme_availability_lines)
            + "\n"
        )
    else:
        theme_availability_block = ""

    user_prompt = f"""
You are reasoning across multiple high/medium confidence items that a Senior PM is tracking.
Today's date is {today}.

{required_anchors_block}{theme_availability_block}You are given items eligible for What's Shifting analysis. Use these to produce whats_shifting paragraphs and an interview_angle.

Items:
{context_block}

Produce a structured JSON object:
{{
  "whats_shifting": [
    {{
      "paragraph": "Each paragraph must: "
                   "(1) open with a single declarative sentence naming the underlying force or pattern — not an event description; "
                   "(2) develop the insight across 3-4 sentences by connecting signals from different sources or themes to reveal something non-obvious; "
                   "(3) close with the strategic implication for a PM — what decision, risk, or opportunity does this pattern create? "
                   "The implication must be directly derivable from the cited sources. "
                   "Each sentence ends with inline [n] citations. Only cite [n] if a specific bullet from item [n] directly supports that sentence. "
                   "CITATION CLAIM-LEVEL RULE: Citations are claim-level, not paragraph-level. "
                   "When the topic shifts to content from a different source, drop the prior source citation unless it independently supports the new claim. "
                   "Do not carry a citation forward from one sentence into the next if the new sentence draws from a different source. "
                   "Every citation must answer: does this specific source contain a bullet that directly supports this specific sentence? If no, remove the citation. "
                   "READER CONTEXT RULE: Assume the reader has not heard of any company or product mentioned — provide one clause of context on first mention. "
                   "LEDE PRECISION RULE: Opening sentence makes a claim the paragraph must fully deliver. "
                   "PARAGRAPH INTEGRITY RULE: These constraints apply to every sentence in the paragraph, not just the closing: "
                   "(1) HEDGE MATCH: Match the hedge level of your sources throughout. If a source says 'suggests,' 'implies,' 'may,' or 'could,' use equivalent hedged language at every claim that traces to that source. Do not convert a source observation into an assertion anywhere in the paragraph. 'This suggests...' not 'This demonstrates...' "
                   "(2) NO TIMELINE: Do not assert a specific timeline ('within weeks,' 'before the window closes,' 'within a year') unless that timeline appears verbatim in a source bullet. If no source names it, remove it. "
                   "(3) NO UNIVERSALITY: Do not assert a pattern applies broadly ('all platforms,' 'every PM,' 'any company') when sources show 1-3 examples. Scope it: 'in categories where X applies...' or 'among companies that...' "
                   "ANCHOR SELECTION RULE: Before drafting each paragraph, complete these steps in order: "
                   "(1) Rank ALL insight bullets across every source eligible for that theme by non-obviousness. The most non-obvious bullet is the one that: "
                   "(a) names a structural constraint, counter-intuitive tradeoff, or unintended consequence, "
                   "(b) contradicts or qualifies the headline's apparent conclusion, or "
                   "(c) reveals a mechanism the headline actively obscures. "
                   "(2) Identify the highest-ranked bullet. This is your anchor — build the paragraph's opening claim around it. "
                   "(3) Before drafting, scan ALL remaining bullets for any that contradict, qualify, or limit the anchor's claim. "
                   "For each one found, the paragraph must contain a sentence that directly addresses it — either incorporating it as a qualification or steelmanning your thesis against it. "
                   "If no such sentence exists in your draft, the paragraph is not ready to publish. "
                   "(4) Use remaining bullets as supporting evidence or complication. Do not start from bullet 1 of the first source unless it is genuinely the most non-obvious — "
                   "it rarely is. Bullets 2-4 contain the most specific mechanisms, named products, concrete tradeoffs, and verifiable numbers. "
                   "COMBINATION AND CONSTRUCTION RULE: "
                   "Before combining two sources into one paragraph, ask: can I complete this sentence from a specific source bullet — "
                   "'These sources both demonstrate that [specific causal chain / failure mode / design implication]'? "
                   "If you cannot complete that sentence using words or clear implications from at least one source bullet, do not combine. "
                   "A shared category label ('AI', 'regulation') is not a mechanism. "
                   "If one source contributes 4+ strong bullets, write a single-source paragraph instead of combining. "
                   "Regardless of source count, the paragraph must draw from at least 3 distinct insight bullets. If you cannot find 3, do not publish the paragraph. "
                   "DROPPED BULLET REVIEW: After writing, review every bullet you did NOT use from every cited source and apply these two tests: "
                   "(1) STRONGER INSIGHT TEST: Does this bullet contain a more specific, actionable, or non-obvious insight than the bullets you used? If yes, replace the weakest used bullet with this one. "
                   "(2) SCOPE TEST: Does this bullet limit the geographic, demographic, or use-case scope of the closing implication in a way that materially changes its applicability? If yes, add the qualifier or revise the implication. "
                   "If yes to either, revise before publishing. "
                   "CLOSING SENTENCE RULE: The closing implication must pass all three tests before publishing: "
                   "(1) SINGLE CONSEQUENCE: Contains exactly one actionable consequence. Do not use 'and,' 'but also,' 'as well as,' or 'while also.' If you find yourself writing a conjunction, stop — delete everything after it. "
                   "(2) SOURCE TRACEABILITY: Traces directly to a specific bullet in a cited source. If you cannot identify that bullet, reframe as 'these cases suggest...' not 'this demonstrates that PMs should...' "
                   "(3) NO CONSTRUCTED ACTION: Does not prescribe a specific PM action ('build X,' 'architect around Y') that no source bullet recommends. If constructed, reframe as 'these cases suggest considering...' "
                   "If the closing sentence fails any of (1)-(3), rewrite before publishing. ",
      "source_indices": [1, 2]
    }}
  ],
  "interview_angle": "One specific thing a PM should have a prepared opinion on before interviews this week. "
                     "SOURCE RESTRICTION: Must derive from a source already cited in one of the whats_shifting paragraphs above. "
                     "Do not introduce a new source that did not appear in whats_shifting. "
                     "If no whats_shifting paragraph was produced, set interview_angle to empty string. "
                     "ANCHOR RULE: Anchor to a specific named company, case, product decision, design tradeoff, or architectural choice from today's sources. "
                     "Prefer stories where the source explicitly names a product decision or design tradeoff over stories where the PM implication must be inferred from a business event. "
                     "Weak anchors — do not use: company shutdowns where motivation is unconfirmed, exec org changes, fundraising rounds without product detail. "
                     "PM DECISION LEVEL RULE: The angle must be grounded in a decision a PM actually owns — "
                     "feature prioritization, product architecture, safety design, retention mechanics, "
                     "compliance strategy, pricing tradeoffs, or go-to-market sequencing. "
                     "Do not anchor to decisions owned by executives, infrastructure teams, or investors. "
                     "If the most interesting story today is an exec-level decision, reframe it as: "
                     "what should a PM building on that platform or in that market decide differently as a result? "
                     "FRAMING RULE: Frame as a debatable tradeoff a PM must take a position on — not a fact to recite, not a trend to acknowledge. "
                     "Only assert a company's strategic motivation if it is explicitly stated in the source. "
                     "If motivation is unclear, frame around the observable outcome and the PM-level tradeoff it reveals. "
                     "PARAGRAPH INTEGRITY RULE: "
                     "(1) HEDGE MATCH: Match the hedge level of the source. Do not assert motivations, outcomes, or patterns not explicitly stated. "
                     "(2) NO UNIVERSALITY: Do not assert the tradeoff applies to all PMs or all companies. Scope it to the context the source describes."
}}

Guidance:
- INSIGHT DEPTH RULE: Every whats_shifting paragraph must reveal something a reader could NOT get from any single source. If your paragraph could have been written from a single source, rewrite it.
- PM ACTIONABILITY RULE: When finalizing closing implications, ask: does this tell a PM what to do differently, or does it tell them something interesting? Prefer concrete architectural decisions, pricing tradeoffs, measurement approaches, or design patterns over market trend observations without concrete action.
- GROUNDING RULE: Do not introduce external statistics, historical references, or general knowledge in implication sentences. Use 'this suggests...' or 'this implies...' rather than asserting as established fact.
- CITATION RULE: Only cite item [n] if a specific insight bullet from that item directly supports the exact claim. Every sentence in whats_shifting must have at least one citation.
- REFRAMING RULE: Do not reproduce a named framework from a source as your insight. Ask what it reveals when placed alongside other signals.
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
                   "COMPANY WATCH OMIT RULE: If no item directly covers this company's strategy or product moves, set paragraph to empty string. Do not substitute a tangentially related item. "
                   "If no item has Company field matching this company, set paragraph to empty string. Do not cite any other company's source. "
                   "COMPANY WATCH SOURCE RULE: Company Watch entries may ONLY cite sources tagged 'company_watch ONLY' in their Allowed section field AND whose Company field matches this company. "
                   "An item tagged 'company_watch ONLY' whose Company field is 'NVIDIA' must NOT be used in the Google entry. "
                   "An item tagged 'company_watch ONLY' whose Company field is 'Google' may ONLY be used in the Google entry. "
                   "PARAGRAPH INTEGRITY RULE: These constraints apply to every sentence in the entry, not just the closing: "
                   "(1) HEDGE MATCH: Match the hedge level of your sources throughout. If a source says 'suggests,' 'implies,' 'may,' or 'could,' use equivalent hedged language at every claim that traces to that source. Do not convert a source observation into an assertion anywhere. 'This suggests...' not 'This demonstrates...' "
                   "(2) NO TIMELINE: Do not assert a specific timeline unless it appears verbatim in a source bullet. If no source names it, remove it. "
                   "(3) NO UNIVERSALITY: Do not assert a pattern applies broadly when sources show 1-3 examples. Scope it: 'in categories where X applies...' "
                   "ANCHOR SELECTION RULE: Before drafting, complete these steps in order: "
                   "(1) Rank ALL insight bullets for this company's sources by non-obviousness. The most non-obvious bullet is the one that: "
                   "(a) names a structural constraint, counter-intuitive tradeoff, or unintended consequence, "
                   "(b) contradicts or qualifies the headline's apparent conclusion, or "
                   "(c) reveals a mechanism the headline actively obscures. "
                   "Do not build this entry from bullet 1 alone — the most specific and verifiable content is often in bullets 2-4. "
                   "(2) Identify the highest-ranked bullet. Build the entry's opening claim around it. "
                   "(3) Before drafting, scan ALL remaining bullets for any that contradict, qualify, or limit the anchor's claim. "
                   "For each one found, the entry must contain a sentence that directly addresses it — either incorporating it as a qualification or steelmanning your thesis against it. "
                   "If no such sentence exists in your draft, the entry is not ready to publish. "
                   "(4) Use remaining bullets as supporting evidence or complication. "
                   "THREAD SELECTION RULE: Before writing, identify all available threads for this company. "
                   "Ask: can I identify a single thread where the sources contain the most specific bullets and at least one contradicting bullet? Write that thread only. "
                   "Do not combine threads unless they share a specific mechanism — a specific causal chain, failure mode, or design implication. A shared category label ('AI', 'cloud', 'regulation') is not a mechanism. "
                   "If you cannot name a mechanism that meets this condition from at least one source bullet, cut to the strongest single thread. "
                   "A tight 2-sentence entry built on one deep thread is stronger than a 4-sentence entry that skims three stories. "
                   "DROPPED BULLET REVIEW: After writing, review every bullet you did NOT use and apply these two tests: "
                   "(1) STRONGER INSIGHT TEST: Does this bullet contain a more specific, actionable, or non-obvious insight than the bullets you used? If yes, replace the weakest used bullet with this one. "
                   "(2) SCOPE TEST: Does this bullet limit the scope of the closing implication in a way that materially changes its applicability? If yes, add the qualifier or revise. "
                   "If yes to either, revise before publishing. "
                   "METRICS PRESERVATION RULE: If a source contains a specific number (dollar amount, percentage, named product, date), include it if it supports the entry. Named companies, products, and dollar figures ground the entry. "
                   "SCOPE FIDELITY RULE: Reflect the actual scope stated in the source. If a source explicitly limits scope (e.g. 'non-safety parts only'), the entry must reflect that limit, not expand it. "
                   "CLOSING SENTENCE RULE: Sentence 3 must pass all three tests before publishing: "
                   "(1) SINGLE CONSEQUENCE: Contains exactly one actionable consequence. Do not use 'and,' 'but also,' 'as well as,' or 'while also.' If you find yourself writing a conjunction, stop — delete everything after it. "
                   "(2) SOURCE TRACEABILITY: Traces directly to a specific bullet in a cited source. If you cannot identify that bullet, reframe as 'this suggests...' not 'this demonstrates...' "
                   "(3) NO CONSTRUCTED ACTION: Does not assert competitive framings, strategic motivations, or market positions not explicitly stated in the source. If inferred, frame as 'this suggests...' not 'this demonstrates...' "
                   "If sentence 3 fails any of (1)-(3), rewrite before publishing.",
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
                "PARAGRAPH INTEGRITY RULE: These constraints apply to the entire bullet, not just the closing: "
                "(1) HEDGE MATCH: Match the hedge level of your sources throughout. If a source says 'suggests,' 'implies,' 'may,' or 'could,' use equivalent hedged language at every claim that traces to that source. Do not convert a source observation into an assertion anywhere. 'This suggests...' not 'This demonstrates...' "
                "(2) NO TIMELINE: Do not assert a specific timeline unless it appears verbatim in a source bullet. If no source names it, remove it. "
                "(3) NO UNIVERSALITY: Do not assert a pattern applies broadly when sources show 1-3 examples. Scope it: 'in categories where X applies...' "
                "ANCHOR SELECTION RULE: Before writing each bullet, complete these steps in order: "
                "(1) Rank ALL insight bullets for that source by non-obviousness. The most non-obvious bullet is the one that: "
                "(a) names a structural constraint, counter-intuitive tradeoff, or unintended consequence, "
                "(b) contradicts or qualifies the headline's apparent conclusion, or "
                "(c) reveals a mechanism the headline actively obscures. "
                "(2) Identify the highest-ranked bullet. Build your radar bullet around it. "
                "(3) Before drafting, scan ALL remaining bullets for any that contradict, qualify, or limit the anchor's claim. "
                "For each one found, the bullet must contain a sentence that directly addresses it — either incorporating it as a qualification or steelmanning your thesis against it. "
                "If no such sentence exists in your draft, the bullet is not ready to publish. "
                "(4) Use remaining bullets as supporting evidence only. Do not start from bullet 1 unless it is genuinely the most non-obvious — it rarely is. Bullets 2-4 contain the most specific mechanisms, named products, concrete tradeoffs, and verifiable numbers. "
                "VERBATIM COPY CHECK: After writing each bullet, read the source bullet text and your output side by side. "
                "If more than 4 consecutive words appear in the same order in both, you have copied rather than synthesized — rewrite. "
                "The structural test: your bullet must name the PATTERN or MECHANISM the source example reveals, not describe the example itself. "
                "Apply this two-step rewrite test before finalizing: "
                "(1) EXAMPLE TEST: Identify the named company or event in your bullet. Ask: is my bullet primarily describing what this company did? "
                "If yes, rewrite to lead with what this example reveals about a broader pattern, constraint, or market dynamic — then use the company as evidence. "
                "(2) SUBSTITUTION TEST: Ask: if I replaced this company with a different company doing the same thing, would my bullet still be true? "
                "If no, your bullet is too tied to the specific event. Rewrite to name the underlying mechanism that makes the example significant. "
                "The source example is evidence. Your bullet is the insight the evidence supports. These are different sentences. "
                "DROPPED BULLET REVIEW: After writing, review every bullet you did NOT use and apply these two tests: "
                "(1) STRONGER INSIGHT TEST: Does this bullet contain a more specific, actionable, or non-obvious insight than the bullets you used? If yes, replace the weakest used bullet with this one. "
                "(2) SCOPE TEST: Does this bullet limit the scope of the closing consequence in a way that materially changes its applicability? If yes, add the qualifier or revise. "
                "If yes to either, revise before publishing. "
                "METRICS PRESERVATION RULE: Include the funding amount, round size, or key metric from the source. Do not omit specific numbers that ground the strategic claim. "
                "THEMATIC COMBINATION RULE: Each bullet must cover a single company or a single strategic pattern. "
                "Before combining two companies or stories into one bullet, ask: can I complete this sentence from a specific source bullet — "
                "'These cases both demonstrate that [specific causal chain / failure mode / design implication]'? "
                "If you cannot complete that sentence using words or clear implications from at least one source bullet, do not combine. "
                "A shared category label ('AI', 'fintech', 'regulation') is not a mechanism. "
                "If two stories share only a category but not a specific causal mechanism, they belong in separate bullets. "
                "CLOSING SENTENCE RULE: The closing consequence must pass all three tests before publishing: "
                "(1) SINGLE CONSEQUENCE: Contains exactly one strategic consequence. Do not use 'and,' 'but also,' 'as well as,' or 'while also.' If you find yourself writing a conjunction, stop — delete everything after it. "
                "(2) SOURCE TRACEABILITY: Traces directly to a specific bullet in the cited source. If you cannot identify that bullet, reframe as 'this suggests...' not 'this demonstrates...' "
                "(3) NO CONSTRUCTED ACTION: Does not assert a specific outcome, ratio, or benchmark not present in any source bullet. If inferred, use 'suggests' or 'implies' framing, never assertion. "
                "If the closing consequence fails any of (1)-(3), rewrite before publishing.",
      "source_indices": []
    }}
  ],
  "pm_craft_today": {{
    "text": "Single most actionable PM craft insight from today's content. "
            "Draw ONLY from items tagged 'pm_craft_today ONLY' (theme: product_craft) OR items tagged 'pm_craft_today eligible (design_ux)'. "
            "Do NOT use startup_disruption or company_strategy items — even if no product_craft or design_ux item is available. "
            "If no product_craft or design_ux item is available today, set text to empty string. "
            "INSIGHT QUALITY RULE: The insight must be non-obvious — a specific pattern, tradeoff, or reframe that changes how a PM would approach a real decision. "
            "Avoid generic advice. Name the specific insight: what assumption does it challenge, what decision does it change, or what pattern does it reveal? "
            "Write for a reader who has NOT read the source. "
            "PARAGRAPH INTEGRITY RULE: These constraints apply to the entire entry, not just the closing: "
            "(1) HEDGE MATCH: Match the hedge level of your sources throughout. If a source says 'suggests,' 'implies,' 'may,' or 'could,' use equivalent hedged language at every claim that traces to that source. Do not convert a source observation into an assertion anywhere. "
            "(2) NO TIMELINE: Do not assert a specific timeline unless it appears verbatim in a source bullet. If no source names it, remove it. "
            "(3) NO UNIVERSALITY: Do not assert a pattern applies broadly when sources show 1-3 examples. Scope it: 'in contexts where X applies...' "
            "ANCHOR SELECTION RULE: Before drafting, complete these steps in order: "
            "(1) Rank ALL insight bullets for every eligible source by non-obviousness. The most non-obvious bullet is the one that: "
            "(a) names a structural constraint, counter-intuitive tradeoff, or unintended consequence, "
            "(b) contradicts or qualifies the headline's apparent conclusion, or "
            "(c) reveals a mechanism the headline actively obscures. "
            "(2) Identify the highest-ranked bullet. Build the insight around it. "
            "(3) Before drafting, scan ALL remaining bullets for any that contradict, qualify, or limit the anchor's claim. "
            "For each one found, the entry must contain a sentence that directly addresses it — either incorporating it as a qualification or steelmanning your thesis against it. "
            "If no such sentence exists in your draft, the entry is not ready to publish. "
            "(4) Use remaining bullets as supporting evidence. Do not start from bullet 1 unless it is genuinely the most non-obvious — bullets 2-4 contain the most specific mechanisms and concrete tradeoffs. "
            "DROPPED BULLET REVIEW: After writing, review every bullet you did NOT use and apply these two tests: "
            "(1) STRONGER INSIGHT TEST: Does this bullet contain a more specific, actionable, or non-obvious insight than the bullets you used? If yes, replace the weakest used bullet with this one. "
            "(2) SCOPE TEST: Does this bullet limit the scope of the closing insight in a way that materially changes its applicability? If yes, add the qualifier or revise. "
            "If yes to either, revise before publishing. "
            "CLOSING SENTENCE RULE: The closing sentence must pass all three tests before publishing: "
            "(1) SINGLE CONSEQUENCE: Contains exactly one actionable consequence. Do not use 'and,' 'but also,' 'as well as,' or 'while also.' If you find yourself writing a conjunction, stop — delete everything after it. "
            "(2) SOURCE TRACEABILITY: Traces directly to a specific bullet in a cited source. If you cannot identify that bullet, reframe as 'this suggests...' not 'this demonstrates...' "
            "(3) NO CONSTRUCTED ACTION: Does not prescribe a specific PM action not present in any source bullet. If constructed, reframe as 'this suggests considering...' "
            "If the closing sentence fails any of (1)-(3), rewrite before publishing. "
            "If no craft-relevant insight exists after applying all rules above, set text to empty string.",
    "source_indices": []
  }}
}}

Guidance:
- SECTION ROUTING RULE: Each item is tagged with an "Allowed section" field and a "Company" field. Both are hard constraints, not suggestions.
    Items tagged "company_watch ONLY" may ONLY appear in the company_watch entry whose company name matches the item's Company field.
    Items tagged "startup_radar ONLY" (theme: startup_disruption) may ONLY appear in startup_radar bullets.
    Items tagged "pm_craft_today ONLY" (theme: product_craft) may ONLY appear in pm_craft_today.
    Items tagged "pm_craft_today eligible (design_ux)" may ONLY appear in pm_craft_today.
  A TechCrunch article about Amazon is tagged startup_disruption → startup_radar ONLY. Do not use it in company_watch even if it describes a major company's strategy.
  A YourStory article about Anthropic is tagged startup_disruption → startup_radar ONLY. Do not use it in company_watch.
  An NVIDIA Blog item has Company field 'NVIDIA' → it may only appear in the NVIDIA company_watch entry. Do not use it for Google or any other company.
  Company Watch entries must be built exclusively from items tagged "company_watch ONLY" whose Company field matches that company.
  If no such item exists for a given company today, set that company's paragraph to empty string.
- COMPANY WATCH INSIGHT RULE: Each company paragraph must answer 'what is strategically shifting for this company today' — not just 'what did they do.'
- PM ACTIONABILITY RULE: Prefer concrete product design consequences over strategic observations. Does this tell a PM what to build or decide differently?
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

    # Persist input for controlled re-runs
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

            if theme == "startup_disruption" and company_maturity == "established":
                logger.info(
                    "FILTER [step=3 reason=established_company_in_startup_radar] dropped: %s — %s",
                    item.get("source_name"), item.get("title")
                )
                continue

            filtered_items.append({
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

    # Theme funnel stage 1: after quality/relevance filter
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
    # Source concentration check — observational, runs on pre-cap pool
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
    # Source diversity cap — applied before synthesis
    # High relevance items from each source are retained first.
    # Once a source hits MAX_ITEMS_PER_SOURCE, additional items are held in
    # overflow and only restored if their theme has zero other representation.
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

    # Theme funnel stage 2: after diversity cap
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
            else:
                ws_items.append(item)
                if theme == "design_ux":
                    # design_ux items are eligible for both WS and PM Craft
                    dedicated_items.append(item)
                    logger.info(
                        "ROUTING [design_ux dual-routed to ws_items + dedicated_items]: %s — %s",
                        item.get("source_name"), item.get("title")
                    )

    # Theme funnel stage 3: after partition
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
    # Build required anchors — one per WS theme with at least one filtered item
    # Guarantees every eligible theme gets a paragraph, not just the ones
    # the synthesizer would naturally gravitate toward.
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
        call1_parsed, ws_indexed = _call_whats_shifting(
            client, settings, ws_items, today,
            ws_theme_distribution=ws_theme_dist,
            required_anchors=required_anchors,
        )

        start_idx = len(ws_indexed) + 1
        call2_parsed, dedicated_indexed = _call_dedicated_sections(
            client, settings, dedicated_items, today, start_idx=start_idx
        )

        indexed_items = ws_indexed + dedicated_indexed
        source_index_lookup: Dict[str, Dict[str, Any]] = {}
        for entry in indexed_items:
            source_index_lookup[str(entry["index"])] = {
                "title": entry["title"],
                "source_name": entry["source_name"],
                "theme": entry["theme"],
                "company_id": entry.get("company_id"),
            }

        normalized_whats_shifting = _normalize_whats_shifting(call1_parsed.get("whats_shifting") or [])
        interview_angle = _strip_date_check_flags(str(call1_parsed.get("interview_angle") or ""))

        normalized_company_watch = _normalize_company_watch(call2_parsed.get("company_watch") or {})
        normalized_startup_radar = _normalize_startup_radar(call2_parsed.get("startup_radar") or [])
        pm_craft_today = _normalize_pm_craft(call2_parsed.get("pm_craft_today") or {})

        # ---------------------------------------------------------------------------
        # Post-processing validators
        # ---------------------------------------------------------------------------

        ws_eligible_indices = {entry["index"] for entry in ws_indexed}
        dedicated_eligible_indices = {entry["index"] for entry in dedicated_indexed}

        company_strategy_by_company: Dict[str, set] = {}
        for entry in dedicated_indexed:
            if entry["theme"] == "company_strategy":
                cid = entry.get("company_id")
                if cid:
                    if cid not in company_strategy_by_company:
                        company_strategy_by_company[cid] = set()
                    company_strategy_by_company[cid].add(entry["index"])

        # 1. Multi-thread / single thesis check
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

        # 3. Cross-paragraph coherence
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
                    if reason == "non_company_strategy":
                        warning_msg = (
                            f"Company Watch entry for {company} cites a non-company_strategy source "
                            f"(theme: {source_info.get('theme', 'unknown')}). "
                            "Entry cleared — only first-party company_strategy sources are permitted."
                        )
                    else:
                        warning_msg = (
                            f"Company Watch entry for {company} cites a company_strategy source "
                            f"belonging to '{source_info.get('company_id', 'unknown')}', not '{company}'. "
                            "Entry cleared — each company entry must only cite its own first-party sources."
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
            logger.warning(
                "CW SOURCE INTEGRITY: Clearing %s entry — violation(s): %s",
                company,
                [v["warning"] for v in cw_source_integrity_violations if v["company"] == company]
            )
            normalized_company_watch[company] = {"paragraph": "", "source_indices": []}

        if cw_source_integrity_violations:
            logger.warning(
                "CW SOURCE INTEGRITY VIOLATIONS: %s",
                json.dumps(cw_source_integrity_violations, indent=2)
            )
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
                        f"(theme: {source_info.get('theme', 'unknown')}). "
                        "Entry cleared — only product_craft or design_ux sources are permitted."
                    )
                })
            logger.warning(
                "PM CRAFT SOURCE VIOLATIONS: %s",
                json.dumps(pm_craft_source_violations, indent=2)
            )
            print("PM CRAFT SOURCE VIOLATIONS:")
            for v in pm_craft_source_violations:
                print(json.dumps(v, indent=2))
            # Clear the entry — same enforcement as CW source integrity
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
                        # Raised threshold from 6 to 10 words on each side
                        # to reduce false positives on list constructions
                        # and single-implication sentences with conjunctions
                        if before_words >= 10 and after_words >= 10:
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
        # Classification uses the paragraph's opening sentence content rather than
        # the source's feed-level theme tag, which caused systematic miscounting
        # when MIT Tech Review (tagged ai_technology) was cited in market_behavior
        # or regulation_policy paragraphs.
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
            """
            Classify a paragraph's central theme from its opening sentence.
            Uses keyword matching with priority ordering so more specific themes
            win over ai_technology when both match.
            """
            if not paragraph:
                return "unknown"
            # Use first sentence only — the opening claim determines the theme
            first_sentence = re.split(r"(?<=[.!?])\s+", paragraph.strip())[0].lower()
            # Strip inline citations so [1] doesn't interfere
            first_sentence = re.sub(r"\[\d+\]", "", first_sentence)

            # Score each theme by keyword hit count
            scores: Dict[str, int] = {}
            for theme, keywords in THEME_KEYWORDS.items():
                scores[theme] = sum(1 for kw in keywords if kw in first_sentence)

            best_theme = max(scores, key=lambda t: scores[t])
            # If no keywords matched at all, fall back to source tag
            if scores[best_theme] == 0:
                return "unknown"
            return best_theme

        theme_audit_warnings = []
        ws_theme_counts: Dict[str, List[int]] = {}
        for i, ws in enumerate(normalized_whats_shifting):
            paragraph = ws.get("paragraph", "")
            theme = _classify_paragraph_theme(paragraph)
            # Fall back to source tag only if keyword classifier returns unknown
            if theme == "unknown":
                indices = ws.get("source_indices", [])
                if indices:
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
                    "warning": f"Theme '{theme}' anchors {len(paragraph_indices)} What's Shifting paragraphs — required_anchors constraint may have failed upstream",
                    "action": "INVESTIGATE_UPSTREAM"
                })

        if theme_audit_warnings:
            logger.error("THEME AUDIT WARNINGS: %s", json.dumps(theme_audit_warnings, indent=2))
            print("THEME AUDIT WARNINGS — INVESTIGATE UPSTREAM:")
            for w in theme_audit_warnings:
                print(json.dumps(w, indent=2))

        # 8. Theme diversity warnings
        # Check if any theme with 2+ items in the WS pool anchors zero paragraphs.
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
                            "but anchors 0 What's Shifting paragraphs — possible selection bias"
                        ),
                        "action": "INVESTIGATE_UPSTREAM",
                    })

        if theme_diversity_warnings:
            logger.error("THEME DIVERSITY WARNINGS: %s", json.dumps(theme_diversity_warnings, indent=2))
            print("THEME DIVERSITY WARNINGS — INVESTIGATE UPSTREAM (required_anchors may have failed):")
            for w in theme_diversity_warnings:
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
                "cw_source_integrity_violations": cw_source_integrity_violations,
                "pm_craft_source_violations": pm_craft_source_violations,
                "source_concentration_warnings": source_concentration_warnings,
                "split_implication_warnings": split_implication_warnings,
                "theme_audit_warnings": theme_audit_warnings,
                "theme_diversity_warnings": theme_diversity_warnings,
            },
        }

    except Exception as exc:
        print("Exception during synthesis:", exc)
        traceback.print_exc()
        raise