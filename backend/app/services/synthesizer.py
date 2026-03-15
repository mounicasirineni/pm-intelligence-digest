from __future__ import annotations

import json
import logging
import re
import traceback
from typing import Any, Dict, List

from anthropic import Anthropic

from ..config import load_settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a senior intelligence analyst briefing a Product Manager who is "
    "actively interviewing at top tech companies including Google, Microsoft, "
    "Apple, Meta, Amazon, OpenAI, Anthropic, NVIDIA, and Uber. Your job is to "
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
    "CRITICAL: In your whats_shifting paragraphs, maintain a 60/40 balance — approximately 60% grounded primarily in non-AI themes "
    "(business model shifts, consumer behavior changes, regulatory moves, market dynamics, or design/UX trends) and approximately 40% "
    "covering AI/tech developments. Both directions matter: do not let AI dominate, but do not drop AI coverage entirely either. "
    "An insight that mentions AI as a secondary factor is acceptable as non-AI. A paragraph whose central claim is an AI development counts as AI coverage. "
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
        Structured JSON with source attribution.
    """
    client = _build_client()

    # Two-step filter before synthesis:
    # Step 1 — confidence: drop items the model couldn't summarize reliably
    # Step 2 — pm_relevance_score: drop items that aren't relevant to PM interviews
    # Order matters: no point evaluating relevance on an unreliable summary.
    filtered_items: List[Dict[str, Any]] = []
    dropped_low_confidence = 0
    dropped_low_relevance = 0

    for theme, items in grouped_summaries.items():
        for item in items:
            # Step 1: confidence check
            conf_raw = str(item.get("confidence") or "medium").lower()
            if conf_raw not in {"high", "medium"}:
                dropped_low_confidence += 1
                logger.info(
                    "FILTER [step=1 reason=low_confidence] skipping relevance check: %s — %s",
                    item.get("source_name"), item.get("title")
                )
                continue

            # Step 2: PM relevance check (only reached if confidence passes)
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
            "pm_craft_today": "",
            "interview_angle": "",
            "source_index_lookup": {},
        }

    # Build compact prompt content: title + insights bullets only, with explicit indices.
    lines: List[str] = []
    lines.append("You are given a set of analyzed content items with high/medium confidence.")
    lines.append("Each item includes a title, source, theme, and 3–5 insight bullets.")
    lines.append(
        "Themes span AI & technology, company strategy, product craft, startup disruption, "
        "market behavior, consumer behavior, regulation & policy, and design & UX."
    )
    lines.append("Use these to reason across sources and produce a single weekly brief.")
    lines.append("")
    lines.append("Items:")

    indexed_items: List[Dict[str, Any]] = []
    for idx, item in enumerate(filtered_items, start=1):
        insights = item["insights"]
        if not isinstance(insights, list):
            insights = [str(insights)]

        lines.append(f"\nItem [{idx}]:")
        lines.append(f"- Theme: {item['theme']}")
        lines.append(f"- Source: {item['source_name']}")
        lines.append(f"- Title: {item['title']}")
        lines.append("- Insights:")
        for bullet in insights:
            lines.append(f"  - {bullet}")

        indexed_items.append(
            {
                "index": idx,
                "theme": item["theme"],
                "title": item["title"],
                "source_name": item["source_name"],
            }
        )

    context_block = "\n".join(lines)

    user_prompt = f"""
You are reasoning across multiple high/medium confidence items that a Senior PM is tracking.

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
                   "Balance AI/tech signals WITH business model shifts, consumer behavior changes, regulatory moves, and design/UX trends. "
                   "Each sentence ends with inline [n] citations. Only cite [n] if a specific bullet from item [n] directly supports that sentence.",
      "source_indices": [1, 3]
    }}
  ],
  "company_watch": {{
    "Google": {{
      "paragraph": "2-3 sentences of strategic signal. "
                   "Sentence 1: name what is strategically changing for this company — not news, but a shift in positioning, priority, or competitive stance. "
                   "Sentence 2: provide the evidence from cited sources with inline [n] citations. "
                   "Sentence 3 (optional): name the implication — what does this mean for competitors, partners, or PMs building on or against this platform? "
                   "Only include this company if there is genuine signal today.",
      "source_indices": [2, 4]
    }},
    "Microsoft": {{
      "paragraph": "2-3 sentences of strategic signal. "
                   "Sentence 1: name what is strategically changing for this company — not news, but a shift in positioning, priority, or competitive stance. "
                   "Sentence 2: provide the evidence from cited sources with inline [n] citations. "
                   "Sentence 3 (optional): name the implication — what does this mean for competitors, partners, or PMs building on or against this platform? "
                   "Only include this company if there is genuine signal today.",
      "source_indices": [2, 4]
    }},
    "Apple": {{
      "paragraph": "2-3 sentences of strategic signal. "
                   "Sentence 1: name what is strategically changing for this company — not news, but a shift in positioning, priority, or competitive stance. "
                   "Sentence 2: provide the evidence from cited sources with inline [n] citations. "
                   "Sentence 3 (optional): name the implication — what does this mean for competitors, partners, or PMs building on or against this platform? "
                   "Only include this company if there is genuine signal today.",
      "source_indices": [2, 4]
    }},
    "Meta": {{
      "paragraph": "2-3 sentences of strategic signal. "
                   "Sentence 1: name what is strategically changing for this company — not news, but a shift in positioning, priority, or competitive stance. "
                   "Sentence 2: provide the evidence from cited sources with inline [n] citations. "
                   "Sentence 3 (optional): name the implication — what does this mean for competitors, partners, or PMs building on or against this platform? "
                   "Only include this company if there is genuine signal today.",
      "source_indices": [2, 4]
    }},
    "Amazon": {{
      "paragraph": "2-3 sentences of strategic signal. "
                   "Sentence 1: name what is strategically changing for this company — not news, but a shift in positioning, priority, or competitive stance. "
                   "Sentence 2: provide the evidence from cited sources with inline [n] citations. "
                   "Sentence 3 (optional): name the implication — what does this mean for competitors, partners, or PMs building on or against this platform? "
                   "Only include this company if there is genuine signal today.",
      "source_indices": [2, 4]
    }},
    "OpenAI": {{
      "paragraph": "2-3 sentences of strategic signal. "
                   "Sentence 1: name what is strategically changing for this company — not news, but a shift in positioning, priority, or competitive stance. "
                   "Sentence 2: provide the evidence from cited sources with inline [n] citations. "
                   "Sentence 3 (optional): name the implication — what does this mean for competitors, partners, or PMs building on or against this platform? "
                   "Only include this company if there is genuine signal today.",
      "source_indices": [2, 4]
    }},
    "Anthropic": {{
      "paragraph": "2-3 sentences of strategic signal. "
                   "Sentence 1: name what is strategically changing for this company — not news, but a shift in positioning, priority, or competitive stance. "
                   "Sentence 2: provide the evidence from cited sources with inline [n] citations. "
                   "Sentence 3 (optional): name the implication — what does this mean for competitors, partners, or PMs building on or against this platform? "
                   "Only include this company if there is genuine signal today.",
      "source_indices": [2, 4]
    }},
    "NVIDIA": {{
      "paragraph": "2-3 sentences of strategic signal. "
                   "Sentence 1: name what is strategically changing for this company — not news, but a shift in positioning, priority, or competitive stance. "
                   "Sentence 2: provide the evidence from cited sources with inline [n] citations. "
                   "Sentence 3 (optional): name the implication — what does this mean for competitors, partners, or PMs building on or against this platform? "
                   "Only include this company if there is genuine signal today.",
      "source_indices": [2, 4]
    }},
    "Uber": {{
      "paragraph": "2-3 sentences of strategic signal. "
                   "Sentence 1: name what is strategically changing for this company — not news, but a shift in positioning, priority, or competitive stance. "
                   "Sentence 2: provide the evidence from cited sources with inline [n] citations. "
                   "Sentence 3 (optional): name the implication — what does this mean for competitors, partners, or PMs building on or against this platform? "
                   "Only include this company if there is genuine signal today.",
      "source_indices": [2, 4]
    }}
  }},
  "startup_radar": [
    {{
      "bullet": "2-3 items on early-stage or emerging companies making unexpected moves. Each bullet MUST go beyond describing what happened — it must explain the strategic pattern it reveals, the incumbent it threatens, or the market shift it signals. Structure each bullet as: [what the company did] + [why it matters strategically] + [what pattern or shift it represents]. Avoid restating facts without synthesis. Exclude established research labs, geopolitical incidents, and large-cap company moves — those belong in company_watch or whats_shifting.",
      "source_indices": [1, 2]
    }}
  ],
  "pm_craft_today": {{
    "text": "single most actionable PM craft insight from today's content, drawing especially from product_craft, design_ux, and consumer_behavior themes (not just AI sources). "
            "Must be a non-obvious takeaway that a reader would NOT get from the headline alone — a specific pattern, tradeoff, or reframe that changes how a PM would approach a real decision. "
            "Avoid generic advice like 'PMs should focus on user needs' or 'test before building.' "
            "Instead name the specific insight: what assumption does it challenge, what decision does it change, or what pattern does it reveal?",
    "source_indices": [3]
  }},
  "interview_angle": "one specific thing a PM should have a prepared opinion on before interviews this week, "
                     "sometimes focusing on product strategy, sometimes consumer insight, sometimes regulatory "
                     "navigation, sometimes AI — not always AI"
}}

Guidance:
- INSIGHT DEPTH RULE: Every whats_shifting paragraph must reveal something a reader could NOT get from any single source. Ask yourself: am I naming an underlying force that connects multiple signals, or am I just describing what happened with a PM gloss? 'Platforms are degrading quality' is a description. 'Market dominance removes the competitive pressure that originally forced quality — creating a predictable degradation lifecycle that PMs can use to time competitive entry' is an insight. If your paragraph could have been written from a single source, rewrite it.
- When making a claim in whats_shifting, you MUST cite which item numbers support it using [n] notation at the end of each sentence. Every sentence in whats_shifting must have at least one citation.
- CRITICAL CITATION RULE: Only cite item [n] if a specific insight bullet from that item directly supports the exact claim you are making in that sentence. Do not cite an item merely because it is thematically related or appeared in the same section. If you cannot point to a specific bullet from item [n] that supports the claim, do not cite it.
- The source_indices array for each whats_shifting entry must list all item numbers that meaningfully support that paragraph.
- For company_watch entries, also include inline [n] citations and a matching source_indices array for each company you populate.
- Apply the same citation rule to company_watch: only cite an item if its insight bullets directly support the specific claim made about that company.
- COMPANY WATCH INSIGHT RULE: Each company paragraph must answer 'what is strategically shifting for this company today' — not just 'what did they do.' A paragraph that only describes a product launch or announcement without explaining the strategic positioning, competitive implication, or market signal it represents is insufficient. Ask: does this paragraph tell a PM something they could use to form an opinion about this company's direction in an interview? If not, either deepen it or omit the company.
- For company_watch, only include companies (from Google, Microsoft, Apple, Meta, Amazon, OpenAI, Anthropic, NVIDIA, Uber) that have clear signal today; omit or set null for companies without signal.
- Do not restate per-source summaries; always combine signals across sources and themes. The test: if you removed all but one citation from a paragraph and it still made sense, you have summarized, not synthesized. A synthesized paragraph requires at least two sources because the insight only emerges from their combination.
- Ensure at least 60% of whats_shifting paragraphs have a non-AI theme as their central claim (business model shifts, consumer behavior, regulatory moves, market dynamics, design/UX). With 4 paragraphs that means 3 non-AI; with 5 paragraphs that means 3 non-AI. A paragraph that mentions AI as context but leads with a non-AI insight counts. A paragraph whose main point is an AI development does not count.
- For pm_craft_today, favor insights grounded in product_craft, design_ux, and consumer_behavior themes, even when they intersect with AI.
- For startup_radar, each bullet must contain a genuine 'so what' — the strategic implication, competitive threat, or market pattern revealed, not just a description of the event. A bullet that only describes what a company did without explaining what it means strategically is insufficient.
- For interview_angle, rotate focus across different PM skill areas (product strategy, consumer insight, regulatory navigation, AI, etc.) over time instead of defaulting to AI every time.
""".strip()

    settings = load_settings()

    try:
        response = client.messages.create(
            model=settings.claude_model,
            max_tokens=4000,
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
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Unexpected Claude response format in synthesizer: %s", exc)
            raise

        logger.debug("Raw Claude synthesis response text: %s", text)
        print("Raw Claude synthesis response text:")
        print(text)

        cleaned = _extract_json(text)
        try:
            parsed = json.loads(cleaned)
        except Exception:
            logger.warning("Claude synthesis response was not valid JSON, returning wrapper.")
            parsed = {
                "raw_text": text,
                "whats_shifting": [],
                "company_watch": {},
                "startup_radar": [],
                "pm_craft_today": {"text": "", "source_indices": []},
                "interview_angle": "",
            }

        # Normalize whats_shifting
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
                for idx in indices:
                    try:
                        cleaned_indices.append(int(idx))
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

        # Normalize company_watch
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
            for idx in indices:
                try:
                    cleaned_indices_cw.append(int(idx))
                except Exception:
                    continue

            normalized_company_watch[company] = {
                "paragraph": paragraph,
                "source_indices": cleaned_indices_cw,
            }

        # Normalize startup_radar
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
            for idx in indices:
                try:
                    cleaned_indices_sr.append(int(idx))
                except Exception:
                    continue
            normalized_startup_radar.append({
                "bullet": bullet,
                "source_indices": cleaned_indices_sr,
            })

        # Normalize pm_craft_today
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
        for idx in pm_craft_indices:
            try:
                cleaned_pm_craft_indices.append(int(idx))
            except Exception:
                continue
        pm_craft_today = {
            "text": pm_craft_text,
            "source_indices": cleaned_pm_craft_indices,
        }

        interview_angle = parsed.get("interview_angle") or ""

        # Build lookup table for UI citation resolution
        source_index_lookup: Dict[int, Dict[str, Any]] = {}
        for entry in indexed_items:
            idx = entry["index"]
            source_index_lookup[str(idx)] = {
                "title": entry["title"],
                "source_name": entry["source_name"],
                "theme": entry["theme"],
            }

        return {
            "whats_shifting": normalized_whats_shifting,
            "company_watch": normalized_company_watch,
            "startup_radar": normalized_startup_radar,
            "pm_craft_today": pm_craft_today,
            "interview_angle": str(interview_angle),
            "source_index_lookup": source_index_lookup,
        }

    except Exception as exc:
        print("Exception during Claude synthesis call/parse:", exc)
        traceback.print_exc()
        raise