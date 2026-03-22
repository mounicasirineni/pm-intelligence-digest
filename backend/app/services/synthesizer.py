from __future__ import annotations

import json
import logging
import re
import traceback
from datetime import date
from typing import Any, Dict, List

from anthropic import Anthropic

from ..config import load_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Theme routing constants
# Controls which themes are eligible for each brief section.
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
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a senior intelligence analyst briefing a Product Manager who is "
    "actively interviewing at top tech companies including Google, Microsoft, "
    "Apple, Meta, Amazon, Netflix, NVIDIA, and OpenAI. Your job is to "
    "reason across multiple sources covering 8 themes: AI & technology, company "
    "strategy, product craft, startup disruption, market behavior, consumer "
    "behavior, regulation & policy, and design & UX. Surface what is actually "
    "shifting in the industry — not what happened, but what it means and what "
    "patterns are emerging. "
    "For every insight, ask: what would a reader NOT get from reading any single source? What is the non-obvious implication "
    "that only emerges when you reason across multiple signals? A good insight names the underlying force driving multiple "
    "seemingly unrelated events, challenges a conventional assumption, or identifies a second-order consequence that "
    "practitioners haven't yet articulated. Avoid insights that merely restate a trend with a PM gloss — "
    "'companies are investing in AI' is not an insight. 'AI investment is accelerating consolidation around "
    "infrastructure players while commoditizing application-layer differentiation' is an insight. "
    "CRITICAL: In your whats_shifting paragraphs, distribute central claims across the five eligible themes: "
    "AI & technology, market behavior, consumer behavior, regulation & policy, and design & UX. "
    "No single theme should be the central claim of more than one paragraph in a five-paragraph brief. "
    "A paragraph that mentions a theme as supporting context does not count against that theme's allocation — "
    "only the central claim of the opening sentence determines the theme. "
    "Company strategy, product craft, and startup disruption belong in their dedicated sections and should not anchor a whats_shifting paragraph. "
    "SOURCE ROUTING RULE: The items list is partitioned into two labeled sections: "
    "WHATS_SHIFTING_ELIGIBLE and DEDICATED_SECTION_ELIGIBLE. "
    "whats_shifting paragraphs must ONLY cite items from the WHATS_SHIFTING_ELIGIBLE section. "
    "company_watch, startup_radar, and pm_craft_today must ONLY cite items from the DEDICATED_SECTION_ELIGIBLE section. "
    "Do not cite a WHATS_SHIFTING_ELIGIBLE item in company_watch or startup_radar. "
    "Do not cite a DEDICATED_SECTION_ELIGIBLE item in whats_shifting. "
    "This is a hard constraint — not a preference. Violating it will cause the brief to recycle "
    "the same source across multiple sections, degrading breadth and introducing repetition. "
    "PM ACTIONABILITY STANDARD: Across all sections — whats_shifting, company_watch, startup_radar, and pm_craft_today — "
    "when choosing between a strategic observation and a concrete product design consequence, always prefer the latter. "
    "A specific mechanical implication that tells a PM what decision to make, what assumption to test, or what design pattern to apply "
    "is always stronger than a generalizable pattern observation. "
    "Test every closing implication sentence: could a PM walk into a meeting tomorrow and use this to change a decision? "
    "If the answer is 'it depends on context' or 'it is a useful frame,' the implication is too abstract — rewrite it. "
    "Not 'product teams must balance accuracy and convenience' but "
    "'the revenue model only works if conversion rates justify the subsidy — validate this before committing to the pricing architecture.' "
    "The broad observation is usually derivable from the headline. The specific mechanical consequence requires reading the full content. Keep the latter. "
    "A sharp PM should be able to walk into any interview and have a prepared "
    "opinion on the insights you surface."
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


def synthesize_trends(grouped_summaries: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Run a second-pass synthesis across all summarized items.

    Filters to items that are:
      - confidence: high or medium (summary is reliable)
      - pm_relevance_score: high or medium (topic is PM-relevant)

    Args:
        grouped_summaries: dict[theme, list[items]] where each item contains at least:
            - title
            - source_name
            - insights (list[str])
            - confidence: "high" | "medium" | "low"
            - pm_relevance_score: "high" | "medium" | "low"

    Returns:
        Structured JSON with source attribution and editorial warnings.
    """
    client = _build_client()

    # Two-step filter before synthesis:
    # Step 1 — confidence: drop items the model couldn't summarize reliably
    # Step 2 — pm_relevance_score: drop items that aren't relevant to PM interviews
    filtered_items: List[Dict[str, Any]] = []
    dropped_low_confidence = 0
    dropped_low_relevance = 0

    for theme, items in grouped_summaries.items():
        for item in items:
            conf_raw = str(item.get("confidence") or "medium").lower()
            if conf_raw not in {"high", "medium"}:
                dropped_low_confidence += 1
                logger.info(
                    "FILTER [step=1 reason=low_confidence] skipping relevance check: %s — %s",
                    item.get("source_name"), item.get("title")
                )
                continue

            relevance_raw = str(item.get("pm_relevance_score") or "medium").lower()
            if relevance_raw not in {"high", "medium"}:
                dropped_low_relevance += 1
                logger.info(
                    "FILTER [step=2 reason=low_relevance] dropped after confidence passed: %s — %s",
                    item.get("source_name"), item.get("title")
                )
                continue

            filtered_items.append(
                {
                    "theme": theme,
                    "title": item.get("title", ""),
                    "source_name": item.get("source_name", ""),
                    "insights": item.get("insights") or [],
                    "confidence": conf_raw,
                    "pm_relevance_score": relevance_raw,
                }
            )

    if dropped_low_confidence:
        logger.info("Dropped %d items with low confidence before synthesis.", dropped_low_confidence)
    if dropped_low_relevance:
        logger.info("Dropped %d items with low PM relevance before synthesis.", dropped_low_relevance)

    if not filtered_items:
        logger.warning("No eligible items available for synthesis after filtering.")
        return {
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

    # ---------------------------------------------------------------------------
    # Partition filtered items by routing eligibility
    # ---------------------------------------------------------------------------
    whats_shifting_items: List[Dict[str, Any]] = []
    dedicated_section_items: List[Dict[str, Any]] = []

    for item in filtered_items:
        theme = item.get("theme", "")
        if theme in WHATS_SHIFTING_THEMES:
            whats_shifting_items.append(item)
        else:
            dedicated_section_items.append(item)

    # ---------------------------------------------------------------------------
    # Build context block with routing-aware numbered sections
    # ---------------------------------------------------------------------------
    lines: List[str] = []
    lines.append("You are given a set of analyzed content items partitioned by routing eligibility.")
    lines.append("")

    lines.append("=== WHATS_SHIFTING_ELIGIBLE ITEMS ===")
    lines.append("Use ONLY these items for whats_shifting paragraphs.")
    lines.append("")

    ws_indexed: List[Dict[str, Any]] = []
    idx = 1
    for item in whats_shifting_items:
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

        ws_indexed.append({
            "index": idx,
            "theme": item["theme"],
            "title": item["title"],
            "source_name": item["source_name"],
        })
        idx += 1

    lines.append("=== DEDICATED_SECTION_ELIGIBLE ITEMS ===")
    lines.append("Use ONLY these items for company_watch, startup_radar, and pm_craft_today.")
    lines.append("")

    dedicated_indexed: List[Dict[str, Any]] = []
    for item in dedicated_section_items:
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

        dedicated_indexed.append({
            "index": idx,
            "theme": item["theme"],
            "title": item["title"],
            "source_name": item["source_name"],
        })
        idx += 1

    indexed_items = ws_indexed + dedicated_indexed
    context_block = "\n".join(lines)

    # ---------------------------------------------------------------------------
    # User prompt
    # ---------------------------------------------------------------------------
    today = date.today().strftime("%B %d, %Y")
    user_prompt = f"""
You are reasoning across multiple high/medium confidence items that a Senior PM is tracking.
Today's date is {today}.

Use only the information in the items list below.

{context_block}

Now produce a structured JSON object with the following shape:
{{
  "whats_shifting": [
    {{
      "paragraph": "One of 4-5 paragraph-length insights. Each paragraph must: "
                   "(1) open with a single declarative sentence naming the underlying force or pattern — not an event description; "
                   "(2) develop the insight across 3-4 sentences by connecting signals from different sources or themes to reveal something non-obvious; "
                   "(3) close with the strategic implication for a PM — what decision, risk, or opportunity does this pattern create? "
                   "The implication must be directly derivable from the cited sources — do not introduce external facts, statistics, historical claims, or general knowledge not present in the items list. "
                   "If you cannot ground the implication in a specific cited item, state it as a logical inference from the evidence rather than as a fact. "
                   "Balance AI/tech signals WITH business model shifts, consumer behavior changes, regulatory moves, and design/UX trends. "
                   "Each sentence ends with inline [n] citations. Only cite [n] if a specific bullet from item [n] directly supports that sentence. "
                   "READER CONTEXT RULE: Write every paragraph for a reader who has NOT seen any of the source articles. "
                   "Before using any company name, product name, technical term, or domain-specific concept that would not be familiar to a general PM audience, provide one clause of plain-language context inline. "
                   "EXAMPLE DISCIPLINE RULE: Each paragraph should use no more than three distinct examples. "
                   "MINIMUM VIABLE PARAGRAPH RULE: If fewer than two examples pass the connective tissue test, do not publish the paragraph. "
                   "LEDE PRECISION RULE: The opening sentence makes a claim the paragraph must fully deliver. Avoid absolute framing unless a source explicitly uses it. "
                   "IMPLICATION FOCUS RULE: The closing PM implication must make exactly one claim — the sharpest consequence that follows directly from the paragraph's examples. If you find yourself writing 'and' or 'but also' connecting two separate consequences, cut one. Keep the claim that is more specific and more directly grounded in the cited sources. "
                   "SPLIT IMPLICATION SELF-CHECK: Before finalizing the closing sentence, count distinct actionable consequences; signal words like 'and', 'but also', 'as well as', 'while also', 'in addition', 'both...and' connecting two consequences mean a split — keep the more specific, source-grounded claim only; see Guidance for full test. "
                   "ATTRIBUTION PRECISION RULE: Do not attribute intentionality, motivation, or incentive to systems, algorithms, or automated processes.",
      "source_indices": [1, 3]
    }}
  ],
  "company_watch": {{
    "Google": {{
      "paragraph": "2-3 sentences of strategic signal. "
                   "Sentence 1: name what is strategically changing for this company. "
                   "Sentence 2: provide the evidence from cited sources with inline [n] citations. "
                   "Sentence 3 (optional): name the implication — one claim only, the most specific and directly grounded. "
                   "Only include this company if there is genuine signal today. "
                   "LEDE PRECISION RULE: Avoid absolute framing unless a source explicitly uses it. "
                   "IMPLICATION FOCUS RULE: Sentence 3 must make exactly one claim. Cut any 'and' connecting two consequences. "
                   "COMPANY WATCH CONVERGENCE RULE: Multiple threads only if all converge on one closing implication; max two threads. Write closing first, then convergence test; three+ distinct claims = rewrite.",
      "source_indices": []
    }},
    "Meta": {{
      "paragraph": "2-3 sentences of strategic signal. Same rules as Google.",
      "source_indices": []
    }},
    "Apple": {{
      "paragraph": "2-3 sentences of strategic signal. Same rules as Google.",
      "source_indices": []
    }},
    "Amazon": {{
      "paragraph": "2-3 sentences of strategic signal. Same rules as Google.",
      "source_indices": []
    }},
    "Netflix": {{
      "paragraph": "2-3 sentences of strategic signal. Same rules as Google.",
      "source_indices": []
    }},
    "Microsoft": {{
      "paragraph": "2-3 sentences of strategic signal. Same rules as Google.",
      "source_indices": []
    }},
    "NVIDIA": {{
      "paragraph": "2-3 sentences of strategic signal. Same rules as Google.",
      "source_indices": []
    }},
    "OpenAI": {{
      "paragraph": "2-3 sentences of strategic signal. Same rules as Google.",
      "source_indices": []
    }}
  }},
  "startup_radar": [
    {{
      "bullet": "2-3 items on early-stage or emerging companies making unexpected moves. "
                "Structure: [what the company did] + [why it matters strategically] + [what pattern or shift it represents]. "
                "IMPLICATION FOCUS RULE: Each bullet must close with exactly one strategic consequence. Cut any 'and' connecting two separate consequences.",
      "source_indices": []
    }}
  ],
  "pm_craft_today": {{
    "text": "Single most actionable PM craft insight from today's content. "
            "Must be non-obvious — a specific pattern, tradeoff, or reframe that changes how a PM would approach a real decision. "
            "Avoid generic advice. Name the specific insight: what assumption does it challenge, what decision does it change, or what pattern does it reveal? "
            "Write for a reader who has NOT read the source.",
    "source_indices": []
  }},
  "interview_angle": "One specific thing a PM should have a prepared opinion on before interviews this week. "
                     "Anchored to a specific named company, case, or development from today's sources. "
                     "Frame as a debatable claim or tradeoff, not a fact to recite. "
                     "Rotate focus across product strategy, consumer insight, regulatory navigation, and AI."
}}

Guidance:
- INSIGHT DEPTH RULE: Every whats_shifting paragraph must reveal something a reader could NOT get from any single source. If your paragraph could have been written from a single source, rewrite it.
- PM ACTIONABILITY RULE: When finalizing closing implication sentences in any section, apply this test: does this tell a PM what to do differently, or does it tell them something interesting? Prefer bullets that name a specific architectural decision, pricing tradeoff, measurement approach, or design pattern over bullets that name a market trend without a concrete action attached. This applies to whats_shifting implications, company_watch sentence 3, startup_radar 'so what' clauses, and pm_craft_today. When choosing between a strategic observation and a concrete product design consequence, always prefer the latter.
- REFRAMING RULE: Do not reproduce a named framework from a source as your insight. Ask: what does this framework reveal when placed alongside signals from other sources?
- GROUNDING RULE FOR IMPLICATIONS: Do not introduce external statistics, historical references, or general knowledge in implication sentences. Rewrite as logical inference: 'this suggests...' or 'this implies...' rather than asserting as established fact.
- SPLIT IMPLICATION SELF-CHECK: Before finalizing any closing implication sentence across all sections, run this test: Read the closing sentence and count how many distinct actionable consequences it contains. Signal words that indicate a split: 'and', 'but also', 'as well as', 'while also', 'in addition', 'both...and'. If your closing sentence contains any of these connecting two separate consequences, you have a split implication. Cut the weaker consequence using this test: which claim is more specific and more directly grounded in the cited sources? Keep that one. Discard the other entirely — do not move it to a different sentence in the same paragraph. A split implication that is split across two sentences is still a split implication. The entire paragraph must commit to one consequence. If you cannot choose, the paragraph needs a stronger thesis.
- DATE VALIDATION RULE: Today's date is {today}. Before finalizing any paragraph, check every date, milestone, or timeline claim against today's date. If a cited milestone date is earlier than today, flag it inline with [DATE CHECK: this date may already have passed] rather than stating it as a future event. Do not silently reproduce a past date as if it were upcoming.
- CITATION RULE: Only cite item [n] if a specific insight bullet from that item directly supports the exact claim you are making. Do not cite thematically related items.
- COMPANY WATCH CONVERGENCE RULE: Each company entry may contain multiple threads only if all threads converge on a single closing implication. The convergence test: write your closing implication sentence first. Then check — does every preceding sentence in the paragraph directly support that closing sentence? If any sentence requires its own separate closing to be meaningful, it belongs in a different paragraph or should be cut. A paragraph passes the convergence test if you can remove any one sentence and the closing implication still follows logically from the remaining sentences. A paragraph fails the convergence test if the closing sentence only follows from one thread and ignores the other. Examples: PASS — 'Google has technical debt in its ad stack AND is shifting to agentic ops without human approval' both support 'Google is accepting higher platform risk in exchange for speed.' One closing covers both threads. FAIL — 'Apple failed to acquire Halide AND Apple's Camera app is competitively vulnerable in the enthusiast segment' — the closing 'indie developers are vulnerable to talent poaching' only follows from thread 1, not thread 2. Cut thread 2 or find a closing that covers both. HARD LIMIT: No company entry may contain more than two threads regardless of convergence. Three or more threads always require splitting or cutting — a focused two-thread paragraph is always better than a sprawling three-thread paragraph where one thread doesn't fully belong. THREAD COUNT TEST: Read your draft and count distinct strategic claims. One claim = always valid. Two claims = apply convergence test. Three or more claims = rewrite required, no exceptions.
- COMPANY WATCH GROUNDING RULE: Do not connect two separate signals for the same company into a causal narrative unless that connection is explicitly made in the sources.
- CROSS-COMPANY CONSISTENCY RULE: Before finalizing company_watch, check whether any source index appears in more than one company entry. If the same source feeds two company entries, those entries must not present contradictory interpretations of the same event. When a source contains multiple interpretations, pick one and apply it consistently. If two sources genuinely support different interpretations, acknowledge the difference explicitly rather than presenting both as established facts in different paragraphs.
- THEME DIVERSITY RULE: Audit the central claim of each whats_shifting paragraph against five themes: AI & technology, market behavior, consumer behavior, regulation & policy, design & UX. No theme should appear as the central claim more than once.
- THEME AUDIT SELF-CHECK: Before finalizing whats_shifting, list the central theme of each paragraph's opening sentence explicitly: paragraph 1 theme: [theme], paragraph 2 theme: [theme], etc. If any theme appears more than once in this list, you must rewrite the duplicate paragraph around a different theme before proceeding. Do not submit a whats_shifting section where the same theme anchors two paragraphs. If you cannot find a strong enough insight for an underrepresented theme, reduce to four paragraphs rather than duplicate a theme in five. A four-paragraph brief with four distinct themes is always better than a five-paragraph brief with a theme duplication.
- SOURCE ROUTING ENFORCEMENT: whats_shifting must only cite WHATS_SHIFTING_ELIGIBLE item numbers. company_watch, startup_radar, pm_craft_today must only cite DEDICATED_SECTION_ELIGIBLE item numbers. Violation = same source recycled across sections.
- COMPANY WATCH OMIT RULE: Only include a company entry if at least one DEDICATED_SECTION_ELIGIBLE item directly covers that company's strategy, product moves, or competitive positioning today. If no DEDICATED_SECTION_ELIGIBLE item covers a company, omit that company entirely — do not substitute a WHATS_SHIFTING_ELIGIBLE source to fill the slot. An absent company entry is always better than a company entry sourced from a third-party market behavior or regulation article. Test before writing each entry: is the source index I am citing in the DEDICATED_SECTION_ELIGIBLE list? If not, do not write the entry.
- PM CRAFT OMIT RULE: pm_craft_today must only cite DEDICATED_SECTION_ELIGIBLE items. If no product_craft or design_ux source has a strong PM craft insight today, pull from the strongest available startup_disruption or company_strategy item rather than reaching into WHATS_SHIFTING_ELIGIBLE sources. If no DEDICATED_SECTION_ELIGIBLE source has a craft-relevant insight, set pm_craft_today text to empty string rather than sourcing from a WHATS_SHIFTING_ELIGIBLE item.
- For startup_radar, each bullet must contain a genuine 'so what' — not just a description of what happened.
- For interview_angle, rotate focus across different PM skill areas over time instead of defaulting to AI every time.
""".strip()

    settings = load_settings()

    try:
        response = client.messages.create(
            model=settings.claude_model,
            max_tokens=8000,
            temperature=0.3,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
        )

        print("Raw Claude synthesis response content:")
        print(response.content)

        try:
            content_block = response.content[0]
            text = getattr(content_block, "text", None) or content_block.get("text")  # type: ignore[union-attr]
        except Exception as exc:
            logger.exception("Unexpected Claude response format in synthesizer: %s", exc)
            raise

        logger.debug("Raw Claude synthesis response text: %s", text)
        print("Raw Claude synthesis response text:")
        print(text)

        cleaned = _extract_json(text)
        try:
            parsed = json.loads(cleaned)
        except Exception:
            logger.warning(f"Claude synthesis response was not valid JSON, returning wrapper. Raw response (first 500 chars): {text[:500]}")
            parsed = {
                "raw_text": text,
                "whats_shifting": [],
                "company_watch": {},
                "startup_radar": [],
                "pm_craft_today": {"text": "", "source_indices": []},
                "interview_angle": "",
            }

        # ---------------------------------------------------------------------------
        # Normalize whats_shifting
        # ---------------------------------------------------------------------------
        raw_whats_shifting = parsed.get("whats_shifting") or []
        normalized_whats_shifting: List[Dict[str, Any]] = []
        if isinstance(raw_whats_shifting, list):
            for entry in raw_whats_shifting:
                if isinstance(entry, dict):
                    paragraph = entry.get("paragraph") or entry.get("text") or ""
                    indices = entry.get("source_indices") or entry.get("sources") or []
                else:
                    paragraph = str(entry)
                    indices = []

                if not isinstance(indices, list):
                    indices = [indices]
                cleaned_indices: List[int] = []
                for i in indices:
                    try:
                        cleaned_indices.append(int(i))
                    except Exception:
                        continue

                normalized_whats_shifting.append(
                    {
                        "paragraph": paragraph,
                        "source_indices": cleaned_indices,
                    }
                )
        else:
            normalized_whats_shifting.append(
                {"paragraph": str(raw_whats_shifting), "source_indices": []}
            )

        # ---------------------------------------------------------------------------
        # Normalize company_watch
        # ---------------------------------------------------------------------------
        raw_company_watch = parsed.get("company_watch") or {}
        if not isinstance(raw_company_watch, dict):
            raw_company_watch = {"_raw": str(raw_company_watch)}

        normalized_company_watch: Dict[str, Dict[str, Any]] = {}
        for company, value in raw_company_watch.items():
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
            cleaned_indices_cw: List[int] = []
            for i in indices:
                try:
                    cleaned_indices_cw.append(int(i))
                except Exception:
                    continue

            normalized_company_watch[company] = {
                "paragraph": paragraph,
                "source_indices": cleaned_indices_cw,
            }

        # ---------------------------------------------------------------------------
        # Normalize startup_radar
        # ---------------------------------------------------------------------------
        raw_startup_radar = parsed.get("startup_radar") or []
        if not isinstance(raw_startup_radar, list):
            raw_startup_radar = [str(raw_startup_radar)]

        normalized_startup_radar = []
        for entry in raw_startup_radar:
            if isinstance(entry, dict):
                bullet = entry.get("bullet") or entry.get("text") or str(entry)
                indices = entry.get("source_indices") or []
            else:
                bullet = str(entry)
                indices = []
            if not isinstance(indices, list):
                indices = [indices]
            cleaned_indices_sr = []
            for i in indices:
                try:
                    cleaned_indices_sr.append(int(i))
                except Exception:
                    continue
            normalized_startup_radar.append({
                "bullet": bullet,
                "source_indices": cleaned_indices_sr,
            })

        # ---------------------------------------------------------------------------
        # Normalize pm_craft_today
        # ---------------------------------------------------------------------------
        raw_pm_craft = parsed.get("pm_craft_today") or {}
        if isinstance(raw_pm_craft, dict):
            pm_craft_text = str(raw_pm_craft.get("text") or raw_pm_craft.get("pm_craft_today") or "")
            pm_craft_indices = raw_pm_craft.get("source_indices") or []
        else:
            pm_craft_text = str(raw_pm_craft)
            pm_craft_indices = []
        if not isinstance(pm_craft_indices, list):
            pm_craft_indices = [pm_craft_indices]
        cleaned_pm_craft_indices = []
        for i in pm_craft_indices:
            try:
                cleaned_pm_craft_indices.append(int(i))
            except Exception:
                continue
        pm_craft_today = {
            "text": pm_craft_text,
            "source_indices": cleaned_pm_craft_indices,
        }

        interview_angle = parsed.get("interview_angle") or ""

        # ---------------------------------------------------------------------------
        # Build source index lookup
        # ---------------------------------------------------------------------------
        source_index_lookup: Dict[str, Dict[str, Any]] = {}
        for entry in indexed_items:
            source_index_lookup[str(entry["index"])] = {
                "title": entry["title"],
                "source_name": entry["source_name"],
                "theme": entry["theme"],
            }

        # ---------------------------------------------------------------------------
        # Post-processing validators
        # ---------------------------------------------------------------------------

        # 1. Single thesis rule: flag company entries citing more than 2 source indices
        # or with high conjunction counts suggesting multiple threads
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
            logger.warning("SINGLE THESIS VIOLATIONS: %s", json.dumps(multi_thread_warnings, indent=2))
            print("SINGLE THESIS WARNINGS:")
            for w in multi_thread_warnings:
                print(json.dumps(w, indent=2))

        # 2. Date validation: scan all paragraphs for past month+year references
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
        # and sources shared between whats_shifting and company_watch
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
                "warning": "Source used in both whats_shifting and company_watch — review for routing violation or contradictory framing"
            })

        if coherence_warnings:
            logger.warning("COHERENCE WARNINGS: %s", json.dumps(coherence_warnings, indent=2))
            print("COHERENCE WARNINGS:")
            for w in coherence_warnings:
                print(json.dumps(w, indent=2))

        # 4. Routing violation check: whats_shifting should not cite dedicated section indices
        ws_eligible_indices = {entry["index"] for entry in ws_indexed}
        dedicated_eligible_indices = {entry["index"] for entry in dedicated_indexed}

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
                        "warning": "DEDICATED_SECTION source cited in whats_shifting — routing violation"
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
                        "warning": "WHATS_SHIFTING source cited in company_watch — routing violation"
                    })

        if routing_warnings:
            logger.warning("ROUTING VIOLATIONS: %s", json.dumps(routing_warnings, indent=2))
            print("ROUTING VIOLATIONS:")
            for w in routing_warnings:
                print(json.dumps(w, indent=2))

        # Omit rule enforcement: flag company entries and pm_craft_today
        # that cite whats_shifting eligible sources
        omit_rule_warnings: List[Dict[str, Any]] = []

        for company, value in normalized_company_watch.items():
            for idx_val in value.get("source_indices", []):
                if idx_val in ws_eligible_indices:
                    source_info = source_index_lookup.get(str(idx_val), {})
                    omit_rule_warnings.append({
                        "section": f"company_watch.{company}",
                        "source_index": idx_val,
                        "source_title": source_info.get("title", "unknown"),
                        "source_theme": source_info.get("theme", "unknown"),
                        "warning": "Company Watch entry cites WHATS_SHIFTING_ELIGIBLE source — entry should be omitted, not substituted",
                        "action": "OMIT_RECOMMENDED"
                    })

        for idx_val in pm_craft_today.get("source_indices", []):
            if idx_val in ws_eligible_indices:
                source_info = source_index_lookup.get(str(idx_val), {})
                omit_rule_warnings.append({
                    "section": "pm_craft_today",
                    "source_index": idx_val,
                    "source_title": source_info.get("title", "unknown"),
                    "source_theme": source_info.get("theme", "unknown"),
                    "warning": "PM Craft Today cites WHATS_SHIFTING_ELIGIBLE source — should use DEDICATED_SECTION_ELIGIBLE source or omit",
                    "action": "OMIT_RECOMMENDED"
                })

        if omit_rule_warnings:
            logger.warning("OMIT RULE VIOLATIONS: %s", json.dumps(omit_rule_warnings, indent=2))
            print("OMIT RULE VIOLATIONS:")
            for w in omit_rule_warnings:
                print(json.dumps(w, indent=2))

        # Split implication detector: scan closing sentences of all paragraphs
        # for conjunction patterns suggesting two separate consequences
        SPLIT_SIGNALS = [
            " and ", " but also ", " as well as ", " while also ",
            " in addition ", " both ", " additionally ",
        ]

        split_implication_warnings: List[Dict[str, Any]] = []

        def check_split_implication(text: str, section: str) -> None:
            if not text:
                return
            # Get the last sentence of the paragraph
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
            if not sentences:
                return
            closing_sentence = sentences[-1].lower()
            # Remove citation markers like [1], [2] before checking
            closing_clean = re.sub(r"\[\d+\]", "", closing_sentence)

            signals_found = [s for s in SPLIT_SIGNALS if s in closing_clean]
            if len(signals_found) >= 1:
                # Additional check: conjunction must connect two verb phrases
                # (rough proxy: word count after conjunction is substantial)
                for signal in signals_found:
                    parts = closing_clean.split(signal, 1)
                    if len(parts) == 2:
                        before_words = len(parts[0].split())
                        after_words = len(parts[1].split())
                        # Only flag if both sides have enough content to be separate claims
                        if before_words >= 5 and after_words >= 5:
                            split_implication_warnings.append({
                                "section": section,
                                "signal": signal.strip(),
                                "closing_sentence": sentences[-1][:200],
                                "warning": "Possible split implication detected in closing sentence — review for two separate consequences",
                            })
                            break

        # Check all sections
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

        # Theme audit: check for duplicate theme anchors in whats_shifting paragraphs
        # Uses source_index_lookup to determine theme of each paragraph's primary source
        theme_audit_warnings: List[Dict[str, Any]] = []

        ws_theme_counts: Dict[str, List[int]] = {}
        for i, ws in enumerate(normalized_whats_shifting):
            indices = ws.get("source_indices", [])
            if not indices:
                continue
            # Use the first cited source as the theme anchor proxy
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
                    "action": "REWRITE_DUPLICATE_RECOMMENDED",
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
            "interview_angle": str(interview_angle),
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
        print("Exception during Claude synthesis call/parse:", exc)
        traceback.print_exc()
        raise