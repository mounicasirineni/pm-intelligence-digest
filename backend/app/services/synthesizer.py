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
    "CRITICAL: In your whats_shifting paragraphs, distribute central claims across the five eligible themes: "
    "AI & technology, market behavior, consumer behavior, regulation & policy, and design & UX. "
    "No single theme should be the central claim of more than one paragraph in a five-paragraph brief. "
    "A paragraph that mentions a theme as supporting context does not count against that theme's allocation — "
    "only the central claim of the opening sentence determines the theme. "
    "Company strategy, product craft, and startup disruption belong in their dedicated sections and should not anchor a whats_shifting paragraph."
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
                   "The implication must be directly derivable from the cited sources — do not introduce external facts, statistics, historical claims, or general knowledge not present in the items list. "
                   "If you cannot ground the implication in a specific cited item, state it as a logical inference from the evidence rather than as a fact. "
                   "Balance AI/tech signals WITH business model shifts, consumer behavior changes, regulatory moves, and design/UX trends. "
                   "Each sentence ends with inline [n] citations. Only cite [n] if a specific bullet from item [n] directly supports that sentence. "
                   "READER CONTEXT RULE: Write every paragraph for a reader who has NOT seen any of the source articles. "
                   "Before using any company name, product name, technical term, or domain-specific concept that would not be familiar to a general PM audience, provide one clause of plain-language context inline — for example, 'Kalshi, a prediction markets platform,' or 'microdramas, short serialized video episodes of 60-90 seconds popular in mobile-first markets.' "
                   "Do not assume the reader knows what a specific company does, what a product category is, or what a cited metric represents without that context. "
                   "If an example requires more than one clause to explain before it supports the paragraph's thesis, it is likely the wrong example — either find a cleaner one or cut it. "
                   "EXAMPLE DISCIPLINE RULE: Each paragraph should use no more than three distinct examples. If you have four or more examples supporting the same thesis, pick the three that are most directly grounded in cited sources and most familiar to a PM audience. Do not stretch a fourth example into the paragraph to make a pattern appear more universal than the evidence supports. "
                   "Before including any example, test it against the paragraph's opening sentence: does this example directly illustrate the named force or pattern, or does it require its own sub-argument to connect? "
                   "If it requires a sub-argument, it belongs in a different paragraph or should be cut entirely. "
                   "Examples drawn from wildly different domains — for instance, US financial regulation, broadcast licensing, and Indian digital identity all in one paragraph — may each be valid individually but collectively signal that the thesis is too broad. When this happens, either narrow the thesis to fit the strongest two examples, or split into two paragraphs each with a tighter claim. "
                   "MINIMUM VIABLE PARAGRAPH RULE: Before publishing a paragraph, count how many examples pass the connective tissue test — directly illustrating the opening sentence without requiring a sub-argument. If fewer than two examples pass, do not publish the paragraph. Instead, either reframe the thesis to fit the examples you have, fold the strongest example into a different paragraph where it fits cleanly, or hold it for a future brief when stronger supporting evidence exists. A tight two-example paragraph is always better than a sprawling three-example paragraph where one example doesn't belong. "
                   "LEDE PRECISION RULE: The opening sentence makes a claim the paragraph must fully deliver. Before finalizing, check: does the evidence show what the lede claims, or something weaker? "
                   "Avoid absolute framing — words like 'primary', 'fundamental', 'definitive', 'repositioning', 'institutional' — unless a source explicitly uses that framing. "
                   "Two examples in two markets do not establish a company-wide repositioning. One incident and one monitoring system do not establish that a risk category has become the primary risk. "
                   "Replace absolutes with relative claims: 'an emerging risk' instead of 'the primary risk', 'prioritizing X over Y in specific markets' instead of 'repositioning as X'. "
                   "Choose the strongest claim the evidence actually supports — not the strongest claim you wish it supported. "
                   "IMPLICATION FOCUS RULE: The closing PM implication must make exactly one claim — the sharpest consequence that follows directly from the paragraph's examples. "
                   "If you find yourself writing a closing sentence with 'and' or 'but also' connecting two separate consequences, you have two claims — cut one. "
                   "When choosing which claim to keep, apply this test: which claim is more specific and more directly grounded in the cited sources? "
                   "A specific mechanical consequence ('the revenue model only works if conversion rates justify the subsidy') is always stronger than a broad generalizable observation ('product teams must balance accuracy and convenience'). "
                   "The broad observation is usually derivable from the headline — the specific mechanical consequence requires reading the full content. Keep the latter, cut the former. "
                   "Do not close by covering both a generalizable insight AND a company-specific observation — pick the one that is more non-obvious and let it stand alone. "
                   "ATTRIBUTION PRECISION RULE: Do not attribute intentionality, motivation, or incentive to systems, algorithms, or automated processes. 'The model lacks an independent validation layer' is precise. 'The model has an incentive to overstate' implies agency it does not have. Use mechanistic language for technical systems.",
      "source_indices": [1, 3]
    }}
  ],
  "company_watch": {{
    "Google": {{
      "paragraph": "2-3 sentences of strategic signal. "
                   "Sentence 1: name what is strategically changing for this company — not news, but a shift in positioning, priority, or competitive stance. "
                   "Sentence 2: provide the evidence from cited sources with inline [n] citations. "
                   "Sentence 3 (optional): name the implication — what does this mean for competitors, partners, or PMs building on or against this platform? "
                   "Only include this company if there is genuine signal today. "
                   "LEDE PRECISION RULE: The opening sentence makes a claim the evidence must fully support. Avoid absolute framing — words like 'repositioning', 'fundamental shift', 'primary', 'definitive' — unless a source explicitly uses that framing. One product move or one market does not establish a company-wide strategic shift. Replace absolutes with relative claims: 'prioritizing X in specific markets' instead of 'repositioning as X'. Choose the strongest claim the evidence actually supports. "
                   "IMPLICATION FOCUS RULE: If sentence 3 is present, it must make exactly one claim. If you find yourself writing 'and' or 'but also' connecting two separate consequences, cut one. Keep the claim that is more specific and more directly grounded in the cited sources.",
      "source_indices": [2, 4]
    }},
    "Microsoft": {{
      "paragraph": "2-3 sentences of strategic signal. "
                   "Sentence 1: name what is strategically changing for this company — not news, but a shift in positioning, priority, or competitive stance. "
                   "Sentence 2: provide the evidence from cited sources with inline [n] citations. "
                   "Sentence 3 (optional): name the implication — what does this mean for competitors, partners, or PMs building on or against this platform? "
                   "Only include this company if there is genuine signal today. "
                   "LEDE PRECISION RULE: The opening sentence makes a claim the evidence must fully support. Avoid absolute framing — words like 'repositioning', 'fundamental shift', 'primary', 'definitive' — unless a source explicitly uses that framing. One product move or one market does not establish a company-wide strategic shift. Replace absolutes with relative claims: 'prioritizing X in specific markets' instead of 'repositioning as X'. Choose the strongest claim the evidence actually supports. "
                   "IMPLICATION FOCUS RULE: If sentence 3 is present, it must make exactly one claim. If you find yourself writing 'and' or 'but also' connecting two separate consequences, cut one. Keep the claim that is more specific and more directly grounded in the cited sources.",
      "source_indices": [2, 4]
    }},
    "Apple": {{
      "paragraph": "2-3 sentences of strategic signal. "
                   "Sentence 1: name what is strategically changing for this company — not news, but a shift in positioning, priority, or competitive stance. "
                   "Sentence 2: provide the evidence from cited sources with inline [n] citations. "
                   "Sentence 3 (optional): name the implication — what does this mean for competitors, partners, or PMs building on or against this platform? "
                   "Only include this company if there is genuine signal today. "
                   "LEDE PRECISION RULE: The opening sentence makes a claim the evidence must fully support. Avoid absolute framing — words like 'repositioning', 'fundamental shift', 'primary', 'definitive' — unless a source explicitly uses that framing. One product move or one market does not establish a company-wide strategic shift. Replace absolutes with relative claims: 'prioritizing X in specific markets' instead of 'repositioning as X'. Choose the strongest claim the evidence actually supports. "
                   "IMPLICATION FOCUS RULE: If sentence 3 is present, it must make exactly one claim. If you find yourself writing 'and' or 'but also' connecting two separate consequences, cut one. Keep the claim that is more specific and more directly grounded in the cited sources.",
      "source_indices": [2, 4]
    }},
    "Meta": {{
      "paragraph": "2-3 sentences of strategic signal. "
                   "Sentence 1: name what is strategically changing for this company — not news, but a shift in positioning, priority, or competitive stance. "
                   "Sentence 2: provide the evidence from cited sources with inline [n] citations. "
                   "Sentence 3 (optional): name the implication — what does this mean for competitors, partners, or PMs building on or against this platform? "
                   "Only include this company if there is genuine signal today. "
                   "LEDE PRECISION RULE: The opening sentence makes a claim the evidence must fully support. Avoid absolute framing — words like 'repositioning', 'fundamental shift', 'primary', 'definitive' — unless a source explicitly uses that framing. One product move or one market does not establish a company-wide strategic shift. Replace absolutes with relative claims: 'prioritizing X in specific markets' instead of 'repositioning as X'. Choose the strongest claim the evidence actually supports. "
                   "IMPLICATION FOCUS RULE: If sentence 3 is present, it must make exactly one claim. If you find yourself writing 'and' or 'but also' connecting two separate consequences, cut one. Keep the claim that is more specific and more directly grounded in the cited sources.",
      "source_indices": [2, 4]
    }},
    "Amazon": {{
      "paragraph": "2-3 sentences of strategic signal. "
                   "Sentence 1: name what is strategically changing for this company — not news, but a shift in positioning, priority, or competitive stance. "
                   "Sentence 2: provide the evidence from cited sources with inline [n] citations. "
                   "Sentence 3 (optional): name the implication — what does this mean for competitors, partners, or PMs building on or against this platform? "
                   "Only include this company if there is genuine signal today. "
                   "LEDE PRECISION RULE: The opening sentence makes a claim the evidence must fully support. Avoid absolute framing — words like 'repositioning', 'fundamental shift', 'primary', 'definitive' — unless a source explicitly uses that framing. One product move or one market does not establish a company-wide strategic shift. Replace absolutes with relative claims: 'prioritizing X in specific markets' instead of 'repositioning as X'. Choose the strongest claim the evidence actually supports. "
                   "IMPLICATION FOCUS RULE: If sentence 3 is present, it must make exactly one claim. If you find yourself writing 'and' or 'but also' connecting two separate consequences, cut one. Keep the claim that is more specific and more directly grounded in the cited sources.",
      "source_indices": [2, 4]
    }},
    "OpenAI": {{
      "paragraph": "2-3 sentences of strategic signal. "
                   "Sentence 1: name what is strategically changing for this company — not news, but a shift in positioning, priority, or competitive stance. "
                   "Sentence 2: provide the evidence from cited sources with inline [n] citations. "
                   "Sentence 3 (optional): name the implication — what does this mean for competitors, partners, or PMs building on or against this platform? "
                   "Only include this company if there is genuine signal today. "
                   "LEDE PRECISION RULE: The opening sentence makes a claim the evidence must fully support. Avoid absolute framing — words like 'repositioning', 'fundamental shift', 'primary', 'definitive' — unless a source explicitly uses that framing. One product move or one market does not establish a company-wide strategic shift. Replace absolutes with relative claims: 'prioritizing X in specific markets' instead of 'repositioning as X'. Choose the strongest claim the evidence actually supports. "
                   "IMPLICATION FOCUS RULE: If sentence 3 is present, it must make exactly one claim. If you find yourself writing 'and' or 'but also' connecting two separate consequences, cut one. Keep the claim that is more specific and more directly grounded in the cited sources.",
      "source_indices": [2, 4]
    }},
    "Anthropic": {{
      "paragraph": "2-3 sentences of strategic signal. "
                   "Sentence 1: name what is strategically changing for this company — not news, but a shift in positioning, priority, or competitive stance. "
                   "Sentence 2: provide the evidence from cited sources with inline [n] citations. "
                   "Sentence 3 (optional): name the implication — what does this mean for competitors, partners, or PMs building on or against this platform? "
                   "Only include this company if there is genuine signal today. "
                   "LEDE PRECISION RULE: The opening sentence makes a claim the evidence must fully support. Avoid absolute framing — words like 'repositioning', 'fundamental shift', 'primary', 'definitive' — unless a source explicitly uses that framing. One product move or one market does not establish a company-wide strategic shift. Replace absolutes with relative claims: 'prioritizing X in specific markets' instead of 'repositioning as X'. Choose the strongest claim the evidence actually supports. "
                   "IMPLICATION FOCUS RULE: If sentence 3 is present, it must make exactly one claim. If you find yourself writing 'and' or 'but also' connecting two separate consequences, cut one. Keep the claim that is more specific and more directly grounded in the cited sources.",
      "source_indices": [2, 4]
    }},
    "NVIDIA": {{
      "paragraph": "2-3 sentences of strategic signal. "
                   "Sentence 1: name what is strategically changing for this company — not news, but a shift in positioning, priority, or competitive stance. "
                   "Sentence 2: provide the evidence from cited sources with inline [n] citations. "
                   "Sentence 3 (optional): name the implication — what does this mean for competitors, partners, or PMs building on or against this platform? "
                   "Only include this company if there is genuine signal today. "
                   "LEDE PRECISION RULE: The opening sentence makes a claim the evidence must fully support. Avoid absolute framing — words like 'repositioning', 'fundamental shift', 'primary', 'definitive' — unless a source explicitly uses that framing. One product move or one market does not establish a company-wide strategic shift. Replace absolutes with relative claims: 'prioritizing X in specific markets' instead of 'repositioning as X'. Choose the strongest claim the evidence actually supports. "
                   "IMPLICATION FOCUS RULE: If sentence 3 is present, it must make exactly one claim. If you find yourself writing 'and' or 'but also' connecting two separate consequences, cut one. Keep the claim that is more specific and more directly grounded in the cited sources.",
      "source_indices": [2, 4]
    }},
    "Uber": {{
      "paragraph": "2-3 sentences of strategic signal. "
                   "Sentence 1: name what is strategically changing for this company — not news, but a shift in positioning, priority, or competitive stance. "
                   "Sentence 2: provide the evidence from cited sources with inline [n] citations. "
                   "Sentence 3 (optional): name the implication — what does this mean for competitors, partners, or PMs building on or against this platform? "
                   "Only include this company if there is genuine signal today. "
                   "LEDE PRECISION RULE: The opening sentence makes a claim the evidence must fully support. Avoid absolute framing — words like 'repositioning', 'fundamental shift', 'primary', 'definitive' — unless a source explicitly uses that framing. One product move or one market does not establish a company-wide strategic shift. Replace absolutes with relative claims: 'prioritizing X in specific markets' instead of 'repositioning as X'. Choose the strongest claim the evidence actually supports. "
                   "IMPLICATION FOCUS RULE: If sentence 3 is present, it must make exactly one claim. If you find yourself writing 'and' or 'but also' connecting two separate consequences, cut one. Keep the claim that is more specific and more directly grounded in the cited sources.",
      "source_indices": [2, 4]
    }}
  }},
  "startup_radar": [
    {{
      "bullet": "2-3 items on early-stage or emerging companies making unexpected moves. Each bullet MUST go beyond describing what happened — it must explain the strategic pattern it reveals, the incumbent it threatens, or the market shift it signals. Structure each bullet as: [what the company did] + [why it matters strategically] + [what pattern or shift it represents]. Avoid restating facts without synthesis. Exclude established research labs, geopolitical incidents, and large-cap company moves — those belong in company_watch or whats_shifting. "
                "IMPLICATION FOCUS RULE: Each bullet must close with exactly one strategic consequence — the sharpest 'so what' that follows from the company's move. If your closing clause contains 'and' connecting two separate consequences, cut one. Keep the claim that is more specific and more directly grounded in what the company actually did. A specific mechanical consequence ('the revenue model depends on conversion rates justifying the testing subsidy') is stronger than a broad pattern observation ('product teams must balance accuracy and convenience').",
      "source_indices": [1, 2]
    }}
  ],
  "pm_craft_today": {{
    "text": "single most actionable PM craft insight from today's content, drawing especially from product_craft, design_ux, and consumer_behavior themes (not just AI sources). "
            "Must be a non-obvious takeaway that a reader would NOT get from the headline alone — a specific pattern, tradeoff, or reframe that changes how a PM would approach a real decision. "
            "Avoid generic advice like 'PMs should focus on user needs' or 'test before building.' "
            "Instead name the specific insight: what assumption does it challenge, what decision does it change, or what pattern does it reveal? "
            "Write for a reader who has NOT read the source — do not reference source-specific names, characters, or proprietary frameworks without briefly explaining them in plain language first. "
            "The insight must stand alone without requiring the reader to know the source material.",
    "source_indices": [3]
  }},
  "interview_angle": "one specific thing a PM should have a prepared opinion on before interviews this week. "
                     "Must be anchored to a specific named company, case, or development from today's sources — not a general theme. "
                     "Frame it as a debatable claim or tradeoff a PM would be asked to reason through, not a fact to recite. "
                     "Rotate focus across product strategy, consumer insight, regulatory navigation, and AI — not always AI.",
}}

Guidance:
- INSIGHT DEPTH RULE: Every whats_shifting paragraph must reveal something a reader could NOT get from any single source. Ask yourself: am I naming an underlying force that connects multiple signals, or am I just describing what happened with a PM gloss? 'Platforms are degrading quality' is a description. 'Market dominance removes the competitive pressure that originally forced quality — creating a predictable degradation lifecycle that PMs can use to time competitive entry' is an insight. If your paragraph could have been written from a single source, rewrite it.
- REFRAMING RULE: If a source already contains sharp analysis or a named framework (e.g. 'binary compliance cliff', 'enshittification', 'agency paradox'), do NOT reproduce that framework as your insight — the reader can get that from the source directly. Instead, ask: what does this framework reveal when placed alongside signals from other sources? What assumption does it challenge that the source author didn't explicitly address? What is the second-order consequence that follows from combining this framework with a different domain's signal? Your insight should be one step of reasoning beyond the sharpest thing in your sources, not a restatement of it.
- GROUNDING RULE FOR IMPLICATIONS: The closing PM implication sentence in each paragraph is the most common source of ungrounded claims. Do not introduce external statistics, historical references, or general knowledge claims in implication sentences — these cannot be cited and will fail grounding checks. If your implication relies on external knowledge (e.g. 'decades of research show...', 'historically...', 'studies suggest...'), rewrite it as a logical inference from the sources you have cited: 'this suggests...' or 'this implies...' rather than asserting it as established fact.
- When making a claim in whats_shifting, you MUST cite which item numbers support it using [n] notation at the end of each sentence. Every sentence in whats_shifting must have at least one citation.
- CRITICAL CITATION RULE: Only cite item [n] if a specific insight bullet from that item directly supports the exact claim you are making in that sentence. Do not cite an item merely because it is thematically related or appeared in the same section. If you cannot point to a specific bullet from item [n] that supports the claim, do not cite it.
- The source_indices array for each whats_shifting entry must list all item numbers that meaningfully support that paragraph.
- For company_watch entries, also include inline [n] citations and a matching source_indices array for each company you populate.
- Apply the same citation rule to company_watch: only cite an item if its insight bullets directly support the specific claim made about that company.
- COMPANY WATCH INSIGHT RULE: Each company paragraph must answer 'what is strategically shifting for this company today' — not just 'what did they do.' A paragraph that only describes a product launch or announcement without explaining the strategic positioning, competitive implication, or market signal it represents is insufficient. Ask: does this paragraph tell a PM something they could use to form an opinion about this company's direction in an interview? If not, either deepen it or omit the company.
- COMPANY WATCH SINGLE THESIS RULE: Each company entry must have one connecting thesis that runs through all sentences. If you have signal from two unrelated sources for the same company, pick the stronger one rather than combining them into a paragraph with two disconnected threads. A focused paragraph covering one strategic shift cleanly is better than a sprawling paragraph that covers two unrelated moves — the latter will read as incoherent and fail the coherence check. Test: can you state the paragraph's central claim in one sentence? If not, it has too many threads.
- COMPANY WATCH REFRAMING RULE: If a source already names a strategic shift for a company, do not reproduce that framing as your insight. Ask: what does this move reveal about the company's broader competitive positioning that the source didn't explicitly connect? What does it mean for PMs building on or against this platform that isn't stated in the article?
- COMPANY WATCH GROUNDING RULE: Do not connect two separate signals for the same company into a causal narrative unless that connection is explicitly made in the sources. If OpenAI is mentioned in one source for military contracts and another for enterprise strategy, do not imply these are causally related unless a source makes that link. Each sentence in a company entry must be traceable to a specific cited source — do not use one source to reinterpret another.
- For company_watch, only include companies (from Google, Microsoft, Apple, Meta, Amazon, OpenAI, Anthropic, NVIDIA, Uber) that have clear signal today; omit or set null for companies without signal.
- Do not restate per-source summaries; always combine signals across sources and themes. The test: if you removed all but one citation from a paragraph and it still made sense, you have summarized, not synthesized. A synthesized paragraph requires at least two sources because the insight only emerges from their combination.
- THEME DIVERSITY RULE: Before finalizing whats_shifting, audit the central claim of each paragraph's opening sentence against these five themes: AI & technology, market behavior, consumer behavior, regulation & policy, and design & UX. No theme should appear as the central claim more than once. If two paragraphs share the same central theme, either reframe the weaker one around a different theme or cut it and replace it with a paragraph from an underrepresented theme. A five-paragraph brief should ideally touch all five themes; a four-paragraph brief should cover at least four. Themes appearing only as supporting examples do not count toward a theme's allocation.
- For pm_craft_today, favor insights grounded in product_craft, design_ux, and consumer_behavior themes, even when they intersect with AI.
- For startup_radar, each bullet must contain a genuine 'so what' — the strategic implication, competitive threat, or market pattern revealed, not just a description of the event. A bullet that only describes what a company did without explaining what it means strategically is insufficient.
- For interview_angle, rotate focus across different PM skill areas (product strategy, consumer insight, regulatory navigation, AI, etc.) over time instead of defaulting to AI every time.
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
            logger.warning(f"Claude synthesis response was not valid JSON, returning wrapper. Raw response (first 500 chars): {text[:500]}")
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