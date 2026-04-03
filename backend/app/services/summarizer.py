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
          "confidence": "high" | "medium" | "low",
          "company_maturity": "startup" | "established" | "not_applicable",
          "scope": "cross_market" | "company_specific",
          "content_word_count": int  # words in content body sent to the model
        }
    """
    client = _build_client()

    title = item.get("title") or ""
    url = item.get("url") or ""
    source_name = item.get("source_name") or ""
    theme = item.get("theme") or ""
    content = item.get("summary") or item.get("content") or ""
    content_word_count = len(content.split())
    logger.info("Content word count for '%s': %d words", title, content_word_count)
    print(f"Content word count for '{title}': {content_word_count} words")
    logger.info(
        "Fetch quality for '%s': %d words | url=%s",
        source_name,
        content_word_count,
        url,
    )
    print(f"Fetch quality for '{source_name}': {content_word_count} words | url={url}")

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
  "confidence": "high" | "medium" | "low",
  "company_maturity": "startup" | "established" | "not_applicable",
  "scope": "cross_market" | "company_specific"
}}

Guidance:
- Insights should be 3–5 bullets.
- Go beyond what happened; explain why it matters in terms of product strategy, AI/tech direction, company moves, and market behavior. Each bullet must pass this test: could a reader have written this bullet from the headline and first paragraph alone? If yes, rewrite it. Surface the implication, tradeoff, or pattern that only emerges from reading the full content.
- SPECIFICITY-FIRST RULE: Structure your bullets from most specific to most abstract. Put bullets naming specific mechanisms, named products, concrete tradeoffs, or verifiable numbers first. Put bullets naming broader patterns or strategic observations last. This ordering helps the synthesizer find your most grounded content quickly rather than defaulting to the first bullet which is often the most abstract framing.
- Avoid generic PM glosses like 'this has implications for product strategy' or 'PMs should pay attention to this trend.' Instead, name the specific implication: what decision does this change, what assumption does it challenge, or what risk does it reveal?
- INSIGHT PRIORITIZATION RULE: When you have more than 3 insight bullets and must choose which to include, rank them in this order:
    (1) Bullets naming a specific product design consequence, architectural decision, or measurable tradeoff — what should a PM build differently, test differently, or price differently as a result of this?
    (2) Bullets naming a strategic pattern or competitive dynamic with a named mechanism.
    (3) Bullets naming a market observation or trend without a concrete action attached.
  A bullet that tells a PM what to build or decide differently outranks a bullet that tells them what is happening. Both are valuable, but the synthesizer will select from your bullets — give it your most actionable ones first. When in doubt, ask: could a PM use this bullet to change a decision in a meeting tomorrow? If yes, it ranks above bullets where the answer is 'it depends' or 'it's a useful frame.'
- COMPLICATION RULE: If the content body contains a fact, data point, mechanism, or claim that contradicts, qualifies, or significantly complicates the article's central claim, you MUST include it as a bullet regardless of where it falls in the prioritization order. This is not optional. A bullet that reverses or limits the central implication is more valuable than a bullet that restates it from a different angle. Ask: is there anything in this content that would make a reader second-guess the main takeaway? If yes, that belongs in the bullets. CONTRADICTION MANDATE: If the content body contains a named data point, statistic, expert claim, or explicit mechanism that directly contradicts, qualifies, or limits the article's central thesis, it must appear as a bullet. You may not omit it because it complicates the narrative. A qualifying fact (e.g. a number that limits the scope of a claim, a geographic or demographic restriction, a named alternative that undermines the central argument) is as important as an explicit contradiction. The trigger is not 'does this directly oppose the thesis' but 'does this change the conclusion a reader would draw.' If yes, include it.
- SOURCE FIDELITY RULE: Every bullet must be traceable to a specific claim, data point, or quote in the content body above. You may reason one logical step beyond the source (e.g. identifying an implication), but you may not:
    (a) Introduce named entities (companies, products, people, technologies) that do not appear in the content body
    (b) Assert specific numbers (multipliers, percentages, dollar figures, timelines) that do not appear in the content body
    (c) Assign strategic motivations to a company that the source does not state
  If you find yourself writing "this is similar to how X did Y" or "this follows the pattern of Z" using a company or event not mentioned in the content — stop. That bullet is from your training knowledge, not from the source. Either ground it in the source or cut it.
  HALLUCINATION TEST: Before finalizing each bullet, ask: "Is the specific company name, product name, number, or causal claim I am asserting actually in the content body above?" If no, rewrite without it.
  QUALIFIER PRESERVATION RULE: If the source uses hedged language to describe a finding ('suggests,' 'implies,' 'may,' 'could,' 'changes,' 'shifts'), your bullet must match that hedge level. Do not convert a source observation into a prescription. If the source says 'this changes how PMs prioritize,' do not write 'PMs must prioritize X from day one.' If the source says 'this suggests a tradeoff,' do not write 'PMs should always choose Y.' Preserving the source's epistemic confidence level is part of source fidelity. Converting hedged observations into prescriptive mandates is an inference boundary violation at the summarizer stage.
- "pm_relevance_score" should be a categorical assessment of how useful this item is for PM interview preparation:
    high   = directly relevant to product strategy, company moves, or market shifts a PM interviewer would ask about
    medium = tangentially relevant; useful context but not a likely interview topic
    low    = not relevant to PM interview prep (e.g. sports tech, celebrity news, off-topic content)
  ROUTINE UPDATE RULE: Content that reports routine operational activity without revealing strategic intent scores low, even if published by a major company. This includes: weekly content additions (new games, new titles, new episodes), cadence-driven posts (GFN Thursday, weekly roundups), minor feature releases with no architectural significance, and event listings or award announcements. A post scores medium or high only if it explains WHY the company made a move, WHAT it reveals about competitive positioning or product direction, or WHAT new capability it demonstrates. The company name alone does not determine the score — the strategic signal does.
- COMPANY MATURITY RULE: If this article's primary subject is a named company, assess whether that company is early-stage or established. Set "company_maturity" to:
    startup     = primary subject is an early-stage, emerging, or privately held company without dominant market position
    established = primary subject is a publicly traded company, a subsidiary of one, or a company with $1B+ valuation and significant market presence (e.g. Intuit, Google, Meta, Amazon, Salesforce)
    not_applicable = article is not primarily about a named company, or covers multiple companies without a single primary subject
  This field is used downstream to filter Startup Radar — established companies will not appear in Startup Radar regardless of feed tag.
- SCOPE RULE: Assess whether this article describes a pattern affecting multiple companies or an entire product category ('cross_market'), or whether it is primarily about one named company's specific regulatory situation, legal case, government contract, or product decision ('company_specific').
    cross_market    = the article's central claim applies to an industry, a product category, or a regulatory framework that affects multiple companies or builders. Example: an EFF article about how 3D printer regulations create repurposable censorship infrastructure is cross_market. An MIT Technology Review article about plastics supply chain vulnerabilities is cross_market.
    company_specific = the article is primarily about what one named company did, faces, or decided. Example: a TechCrunch article about Google's new API is company_specific. An RTI filing about OpenAI's military contract is company_specific.
  This field is used downstream to route regulation_policy articles — only cross_market articles are eligible for What's Shifting. Company_specific articles route to Company Watch or are dropped.
- "pm_interview_relevance" should be a one-line text explanation supporting your pm_relevance_score judgment.
- "confidence" should reflect how well the content body above supports producing accurate, grounded insight bullets — it is a self-assessment of source quality, NOT a judgment of topic interest. Ask: how much of what I would write comes from the content body vs. from my own training knowledge?
    high   = content body is substantive (300+ words of original reporting, analysis, or primary source material) and provides enough specific detail to write 3-5 grounded bullets without drawing on outside knowledge
    medium = content body is adequate but thin (100-300 words, or partially paywalled) — can produce 2-3 grounded bullets but some inference required
    low    = content body is too thin to support grounded bullets (under 100 words, pure press release boilerplate, "In Brief" stub, or heavily paywalled with only a lede visible) — any bullets would be primarily inference from training knowledge, not from the source
  IMPORTANT: A thin source on an interesting topic still gets low confidence. Topic relevance is measured by pm_relevance_score, not confidence. These are independent judgments.
- ROUNDUP HANDLING RULE: If this article is a newsletter or roundup containing multiple distinct stories, extract insights from each distinct story separately — do not limit analysis to the lead story. Each distinct story should contribute at least one insight bullet if PM-relevant. Bullets from different stories within the same roundup should each be self-contained and traceable to their specific story. If the roundup contains 4 stories, aim for 4-5 bullets total covering all stories, not 3-5 bullets covering only the first. Exception: if secondary stories are clearly minor (brief mentions, event listings, link digests with no analysis), skip them. Apply the same PM-relevance test to each story independently.
""".strip()

    settings = load_settings()

    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=1200,
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
            "company_maturity": "not_applicable",
            "scope": "cross_market",
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

    company_maturity = str(parsed.get("company_maturity") or "not_applicable").lower()
    if company_maturity not in {"startup", "established", "not_applicable"}:
        company_maturity = "not_applicable"

    scope = str(parsed.get("scope") or "cross_market").lower()
    if scope not in {"cross_market", "company_specific"}:
        scope = "cross_market"

    return {
        "insights": insights,
        "pm_interview_relevance": str(pm_interview_relevance),
        "pm_relevance_score": pm_relevance_score,
        "confidence": confidence,
        "company_maturity": company_maturity,
        "scope": scope,
        "content_word_count": content_word_count,
    }