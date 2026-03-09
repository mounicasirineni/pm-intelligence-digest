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
    "patterns are emerging. Weight AI and non-AI signals equally. A sharp PM "
    "should be able to walk into any interview and have a prepared opinion on "
    "the insights you surface."
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

    Args:
        grouped_summaries: dict[theme, list[items]] where each item contains at least:
            - title
            - source_name
            - insights (list[str])
            - confidence: "high" | "medium" | "low"

    Returns:
        Structured JSON with source attribution, e.g.:
        {
          "whats_shifting": [
            {
              "paragraph": "Paragraph text with [1][3] inline citations.",
              "source_indices": [1, 3]
            }
          ],
          "company_watch": {
            "Google": {
              "paragraph": "Signal for Google with [2] citation.",
              "source_indices": [2]
            },
            ...
          },
          "startup_radar": [ "...", ... ],
          "pm_craft_today": "...",
          "interview_angle": "...",
          "source_index_lookup": {
            1: {"title": "...", "source_name": "...", "theme": "..."},
            ...
          }
        }
    """
    client = _build_client()

    # Flatten and filter to high/medium confidence items.
    filtered_items: List[Dict[str, Any]] = []
    for theme, items in grouped_summaries.items():
        for item in items:
            conf_raw = (item.get("confidence") or "").lower()
            if conf_raw not in {"high", "medium"}:
                continue
            filtered_items.append(
                {
                    "theme": theme,
                    "title": item.get("title", ""),
                    "source_name": item.get("source_name", ""),
                    "insights": item.get("insights") or [],
                    "confidence": conf_raw,
                }
            )

    if not filtered_items:
        logger.warning("No high/medium confidence items available for synthesis.")
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
      "paragraph": "One of 3-4 paragraph-length insights that synthesize across sources and themes, "
                   "balancing AI/tech signals WITH business model shifts, consumer behavior changes, "
                   "regulatory moves, and design/UX trends. Each sentence ends with inline [n] style "
                   "citations referencing item numbers, and each paragraph draws on at least two distinct themes.",
      "source_indices": [1, 3]
    }}
  ],
  "company_watch": {{
    "Google": {{
      "paragraph": "2-3 sentences of signal for Google, with inline [2][4] citations.",
      "source_indices": [2, 4]
    }},
    "Microsoft": null,
    "Apple": null,
    "Meta": null,
    "Amazon": null,
    "OpenAI": null,
    "Anthropic": null,
    "NVIDIA": null,
    "Uber": null
  }},
  "startup_radar": [
    "2-3 startup moves worth knowing about, including both AI and non-AI startups and disruption patterns, "
    "each as a sentence or short paragraph"
  ],
  "pm_craft_today": "single most actionable PM craft insight from today's content, drawing especially from "
                    "product_craft, design_ux, and consumer_behavior themes (not just AI sources)",
  "interview_angle": "one specific thing a PM should have a prepared opinion on before interviews this week, "
                     "sometimes focusing on product strategy, sometimes consumer insight, sometimes regulatory "
                     "navigation, sometimes AI — not always AI"
}}

Guidance:
- When making a claim in whats_shifting, you MUST cite which item numbers support it using [n] notation at the end of each sentence. Every sentence in whats_shifting must have at least one citation.
- The source_indices array for each whats_shifting entry must list all item numbers that meaningfully support that paragraph.
- For company_watch entries, also include inline [n] citations and a matching source_indices array for each company you populate.
- For company_watch, only include companies (from Google, Microsoft, Apple, Meta, Amazon, OpenAI, Anthropic, NVIDIA, Uber) that have clear signal today; omit or set null for companies without signal.
- Do not restate per-source summaries; always combine signals across sources and themes.
- Ensure whats_shifting paragraphs balance AI/tech stories WITH business model shifts, consumer behavior changes, regulatory moves, and design/UX trends; explicitly avoid an AI-only framing.
- For pm_craft_today, favor insights grounded in product_craft, design_ux, and consumer_behavior themes, even when they intersect with AI.
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

        # Print the raw response content object before any processing.
        print("Raw Claude synthesis response content:")
        print(response.content)

        # Extract the primary text block.
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
                "pm_craft_today": "",
                "interview_angle": "",
            }

        # Ensure required keys exist with reasonable defaults, and normalize structure.
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

        startup_radar = parsed.get("startup_radar") or []
        if not isinstance(startup_radar, list):
            startup_radar = [str(startup_radar)]

        pm_craft_today = parsed.get("pm_craft_today") or ""
        interview_angle = parsed.get("interview_angle") or ""

        # Build a lookup table so the UI can resolve item indices to titles/sources.
        source_index_lookup: Dict[int, Dict[str, Any]] = {}
        for entry in indexed_items:
            idx = entry["index"]
            source_index_lookup[idx] = {
                "title": entry["title"],
                "source_name": entry["source_name"],
                "theme": entry["theme"],
            }

        return {
            "whats_shifting": normalized_whats_shifting,
            "company_watch": normalized_company_watch,
            "startup_radar": startup_radar,
            "pm_craft_today": str(pm_craft_today),
            "interview_angle": str(interview_angle),
            "source_index_lookup": source_index_lookup,
        }

    except Exception as exc:
        print("Exception during Claude synthesis call/parse:", exc)
        traceback.print_exc()
        # Re-raise so the calling code sees the failure during debugging.
        raise

