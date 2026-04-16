from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Tuple

from anthropic import Anthropic

from ..config import load_settings
from .summarizer import _extract_json


logger = logging.getLogger(__name__)

EVAL_MODEL = "claude-haiku-4-5-20251001"

_CITATION_RE = re.compile(r"\[\d+\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class EvalResult:
    pipeline_funnel: Dict[str, Any]
    pm_relevance: Dict[str, Any]
    llm_judge: Dict[str, Any]
    pm_craft: Dict[str, Any]
    interview_angle: Dict[str, Any]
    overall_score: float
    flags: Dict[str, Any]


def _get_db_path() -> Path:
    settings = load_settings()
    db_path = settings.database_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def _get_connection() -> sqlite3.Connection:
    return sqlite3.connect(str(_get_db_path()))


def _ensure_evals_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS evals (
          date TEXT PRIMARY KEY,
          pipeline_funnel_json TEXT,
          pm_relevance_json TEXT,
          llm_judge_json TEXT,
          pm_craft_json TEXT,
          interview_angle_json TEXT,
          overall_score REAL,
          flags_json TEXT,
          evaluated_at TEXT
        )
        """
    )
    for col, col_type in [
        ("pipeline_funnel_json", "TEXT"),
        ("pm_relevance_json", "TEXT"),
        ("pm_craft_json", "TEXT"),
        ("interview_angle_json", "TEXT"),
    ]:
        try:
            conn.execute(f"ALTER TABLE evals ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass
    conn.commit()


def _build_llm_client() -> Anthropic:
    settings = load_settings()
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. "
            "Populate it in your .env file before running LLM-based evals."
        )
    return Anthropic(api_key=settings.anthropic_api_key)


# --------------------------------
# Guardrail 1 — pipeline_funnel
# --------------------------------


def pipeline_funnel(
    items_by_theme: Dict[str, List[Dict[str, Any]]],
    synthesis: Dict[str, Any],
    fetch_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    meta = fetch_metadata or {}

    sources_configured = int(meta.get("sources_configured") or 0)
    sources_active = int(meta.get("sources_active") or 0)
    empty_source_names = meta.get("empty_source_names") or []

    all_items: List[Dict[str, Any]] = [
        item
        for items in (items_by_theme or {}).values()
        for item in (items or [])
        if isinstance(item, dict)
    ]

    fetched = len(all_items)

    confident = sum(
        1 for item in all_items
        if str(item.get("confidence") or "medium").lower() in {"high", "medium"}
    )

    relevant = sum(
        1 for item in all_items
        if str(item.get("confidence") or "medium").lower() in {"high", "medium"}
        and str(item.get("pm_relevance_score") or "medium").lower() in {"high", "medium"}
    )

    source_index_lookup = synthesis.get("source_index_lookup") or {}

    used_indices: set[str] = set()
    for insight in (synthesis.get("whats_shifting") or []):
        if isinstance(insight, dict):
            used_indices.update(str(i) for i in (insight.get("source_indices") or []))
    for company in (synthesis.get("company_watch") or {}).values():
        if isinstance(company, dict):
            used_indices.update(str(i) for i in (company.get("source_indices") or []))
    for item in (synthesis.get("startup_radar") or []):
        if isinstance(item, dict):
            used_indices.update(str(i) for i in (item.get("source_indices") or []))
    pm_craft = synthesis.get("pm_craft_today")
    if isinstance(pm_craft, dict):
        used_indices.update(str(i) for i in (pm_craft.get("source_indices") or []))

    output_cited_titles: set[str] = {
        str(source_index_lookup[k]["title"])
        for k in used_indices
        if k in source_index_lookup
        and isinstance(source_index_lookup[k], dict)
        and source_index_lookup[k].get("title")
    }

    relevant_titles: set[str] = {
        str(item.get("title") or "")
        for item in all_items
        if str(item.get("confidence") or "medium").lower() in {"high", "medium"}
        and str(item.get("pm_relevance_score") or "medium").lower() in {"high", "medium"}
        and item.get("title")
    }

    utilized = len(output_cited_titles & relevant_titles)

    return {
        "sources_configured": sources_configured,
        "sources_active": sources_active,
        "sources_active_pct": (sources_active / sources_configured * 100.0) if sources_configured else 0.0,
        "empty_source_names": empty_source_names,
        "fetched": fetched,
        "confident": confident,
        "confident_pct": (confident / fetched * 100.0) if fetched else 0.0,
        "relevant": relevant,
        "relevant_pct": (relevant / confident * 100.0) if confident else 0.0,
        "utilized": utilized,
        "utilized_pct": (utilized / relevant * 100.0) if relevant else 0.0,
    }


# --------------------------------
# Guardrail 2 — pm_relevance
# --------------------------------


def pm_relevance(
    items_by_theme: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    counts: Dict[str, int] = {"high": 0, "medium": 0, "low": 0, "unknown": 0}

    for items in (items_by_theme or {}).values():
        for item in (items or []):
            if not isinstance(item, dict):
                continue
            score = str(item.get("pm_relevance_score") or "").strip().lower()
            if score in {"high", "medium", "low"}:
                counts[score] += 1
            else:
                counts["unknown"] += 1

    total = counts["high"] + counts["medium"] + counts["low"]
    total_with_unknown = sum(counts.values())

    return {
        "total_items": total_with_unknown,
        "high_count": counts["high"],
        "medium_count": counts["medium"],
        "low_count": counts["low"],
        "unknown_count": counts["unknown"],
        "high_pct": (counts["high"] / total * 100.0) if total else 0.0,
        "medium_pct": (counts["medium"] / total * 100.0) if total else 0.0,
        "low_pct": (counts["low"] / total * 100.0) if total else 0.0,
    }


# --------------------------------
# Eval — llm_judge (async)
# --------------------------------


async def llm_judge(
    synthesis: Dict[str, Any],
    items_by_theme: Dict[str, Any],
    ws_available_themes: Dict[str, int] | None = None,
) -> Dict[str, Any]:
    client = _build_llm_client()
    whats_shifting = synthesis.get("whats_shifting") or []
    company_watch = synthesis.get("company_watch") or {}
    startup_radar = synthesis.get("startup_radar") or []
    source_index_lookup = synthesis.get("source_index_lookup") or {}

    def _build_source_summaries(indices: List[Any]) -> Dict[str, List[str]]:
        indexed_items: Dict[Tuple[str, str], List[str]] = {}
        for items in (items_by_theme or {}).values():
            for item in (items or []):
                if not isinstance(item, dict):
                    continue
                src = str(item.get("source_name") or "").strip()
                ttl = str(item.get("title") or "").strip()
                if not src or not ttl:
                    continue
                insights = item.get("insights") or []
                indexed_items[(src, ttl)] = [str(b) for b in (insights if isinstance(insights, list) else [str(insights)])]

        summaries: Dict[str, List[str]] = {}
        for idx in indices:
            meta = source_index_lookup.get(str(idx)) or {}
            if not isinstance(meta, dict):
                continue
            src = str(meta.get("source_name") or meta.get("source") or "").strip()
            ttl = str(meta.get("title") or "").strip()
            bullets = indexed_items.get((src, ttl)) or []
            if bullets:
                existing = summaries.get(src, [])
                seen: set[str] = set(existing)
                summaries[src] = existing + [b for b in bullets if b not in seen]
        return summaries

    def _extract_indices_from_text(text: str) -> List[int]:
        return [int(m) for m in re.findall(r"\[(\d+)\]", text)]

    def _score_paragraph(paragraph: str, indices: List[Any], section_context: str) -> Dict[str, Any] | None:
        if not paragraph:
            return None

        source_summaries = _build_source_summaries(indices)
        evidence_block = (
            "Source evidence (ALL bullets for each cited source — not just the ones used in synthesis):\n"
            "NOTE: The bullet numbers [1], [2], [3] below are internal reference numbers for this evaluation only. "
            "They do NOT correspond to the citation indices [n] used in the synthesis paragraph. "
            "Match claims to sources by source name and claim content, not by index number.\n"
            + "\n".join(
                f"{s}:\n" + "\n".join(f"  [{i+1}] {b}" for i, b in enumerate(bullets))
                for s, bullets in source_summaries.items()
            )
            if source_summaries else
            "Source evidence:\n(none found in underlying items)"
        )

        user_prompt = (
            f"Rate this {section_context} paragraph on three dimensions. "
            "All source bullets for each cited source are provided — not just the ones the synthesis used.\n\n"
            "1. COHERENCE (1-5): Does the paragraph deliver a unified insight, AND is that unity emergent from the sources or constructed by the synthesizer?\n\n"
            "Check two things:\n"
            "  (a) Internal consistency — do all sentences build toward one claim without disconnected threads?\n"
            "  (b) Source-emergent unity — does the unifying claim actually appear in any source, or did the synthesizer impose it?\n"
            "      A paragraph that selects only narrative-supporting bullets and ignores complicating evidence from the same sources "
            "is not coherent — it is selectively constructed. If the synthesis ignores bullets that challenge or complicate its central claim, "
            "that is a coherence failure even if the included sentences are internally consistent.\n"
            "  (c) Lede fidelity — does the evidence support the strength of the opening claim?\n\n"
            "Scoring:\n"
            "1 = completely disconnected OR lede substantially overclaims OR unity is fully synthesizer-imposed with no source support\n"
            "2 = sentences are consistent but unifying claim contradicts or ignores major source evidence\n"
            "3 = sentences are consistent but lede is slightly stronger than evidence, or unity is synthesizer-constructed from loosely related sources\n"
            "4 = tight thread, lede matches evidence, unity is grounded but one source has a complicating bullet the synthesis omitted\n"
            "5 = tight single thread throughout, lede matches exactly what evidence delivers, AND the unifying claim is present in at least one source (not purely synthesizer construction)\n\n"
            "2. INSIGHT_DEPTH (1-5): Is this a genuine synthesis revealing something non-obvious, does the closing implication commit to one sharp claim, AND is that implication grounded in the sources?\n\n"
            "Check three things:\n"
            "  (a) Synthesis quality — would a reader get this from any single source?\n"
            "  (b) Implication focus — does the closing PM implication make exactly one specific claim?\n"
            "  (c) Inference discipline — is the closing implication traceable to a specific bullet in the cited sources, "
            "or is it the synthesizer's own reasoning presented as a source-supported conclusion? "
            "A well-written implication that goes beyond what any source states must be scored down. "
            "Framing like 'these cases suggest...' is acceptable inference. "
            "Framing like 'this demonstrates that PMs should...' when no source makes that claim is an inference boundary violation.\n"
            "Also check: are there bullets in the source evidence that contain a sharper or more actionable PM insight "
            "than what the synthesis leads with? If the synthesis chose a weaker insight while a stronger one was available in the source bullets, "
            "that is an insight selection failure and should reduce this score.\n\n"
            "Scoring:\n"
            "1 = pure summary, or closing implication is generic and unsourced, or a sharper available insight was ignored\n"
            "2 = some synthesis but closing implication is an unsourced assertion, or meaningfully stronger insight was available and dropped\n"
            "3 = genuine synthesis but closing implication is split across two claims, or slightly overstates what sources support\n"
            "4 = genuine insight, single sharp implication, tightly grounded — minor inference acceptable\n"
            "5 = genuine insight a reader wouldn't get from any single source, closing implication commits to exactly one sharp specific consequence traceable to a source bullet\n\n"
            "3. CITATION_SUPPORT (1-5): Can every specific claim trace to a source passage AND did the synthesis fairly represent what the sources contain?\n\n"
            "This dimension has TWO components:\n\n"
            "  FORWARD TRACEABILITY: Can every specific claim in the paragraph be traced to a specific passage or data point?\n"
            "    Apply to each claim: (a) identify the claim, (b) find the exact source sentence, (c) if none exists, claim fails.\n\n"
            "  BACKWARD COMPLETENESS: Did the synthesis fairly represent the sources, or did it selectively omit evidence?\n"
            "    For each cited source, review ALL bullets provided above. Ask:\n"
            "    - Did the synthesis omit a bullet containing a named contradiction or named expert challenge to the paragraph's central claim?\n"
            "    - Did the synthesis omit a bullet containing a sharper, more specific, or more actionable insight than what was included?\n"
            "    - Did the synthesis build its conclusion primarily from 1-2 bullets while ignoring 3+ equally relevant bullets from the same source?\n"
            "    Selective omission that distorts the paragraph's conclusion is a citation integrity failure, "
            "even if every included claim is correctly sourced.\n\n"
            "Scoring:\n"
            "5 = every claim traces to a specific source passage AND synthesis fairly represents the full source evidence with no distorting omissions\n"
            "4 = one minor inference tightly constrained by source, no significant omissions\n"
            "3 = one moderate inference presented as fact, OR one high-value bullet omitted that would complicate but not reverse the conclusion\n"
            "2 = multiple unsourced inferences OR a named contradiction/expert challenge was omitted OR conclusion built from 1-2 bullets while ignoring stronger ones\n"
            "1 = paragraph's central claim has no source basis, directly contradicts the source, OR a named source bullet explicitly refutes the central claim and was suppressed\n\n"
            "CRITICAL RULES:\n"
            "  - Plausibility is NOT evidence. A claim consistent with the source topic but not stated is an unsourced inference.\n"
            "  - Any specific number not appearing verbatim in source evidence must be treated as unsourced.\n"
            "  - A claim contradicting an explicit source statement scores 1 regardless of writing quality.\n"
            "  - Omitting a bullet that explicitly challenges the synthesis conclusion scores 1 or 2 depending on severity.\n"
            "  - MULTI-SOURCE CITATIONS: For paragraphs citing multiple sources, check each specific claim against ALL cited sources before concluding it is unsourced. "
            "A claim may be sourced in one source even if it does not appear in another. "
            "Do not conclude a claim is unsourced until you have checked every cited source's bullet list individually.\n"
            "  - INDEX COLLISION RULE: The synthesis paragraph contains citation markers like [1], [4], [17], [23]. "
            "The evidence block above contains internal evaluation reference numbers also written as [1], [2], [3]. "
            "THESE ARE TWO COMPLETELY DIFFERENT NUMBERING SYSTEMS. They share the same bracket notation but have no relationship to each other. "
            "The synthesis [n] refers to a source by its pool index. The evidence block [n] is just a bullet counter within each source's list. "
            "NEVER use either set of numbers in your justification text. "
            "When writing your coherence_reason, insight_depth_reason, and citation_support_reason: "
            "refer to sources ONLY by their publication name (e.g. 'MIT Technology Review', 'Simon Willison Blog', 'TechCrunch'). "
            "BAD examples — never write these: 'source [3]', 'bullet [2]', 'Amazon News [8]', 'TechCrunch [14]', 'the [5] source'. "
            "GOOD examples — always write these: 'MIT Technology Review', 'the TechCrunch article on Monarch Tractor', 'the Amazon News source'. "
            "If you reference an index number of any kind in your reason text, your evaluation is invalid.\n\n"
            f"Paragraph:\n{paragraph}\n\n"
            f"{evidence_block}\n\n"
            'Return only valid JSON:\n'
            '{\n'
            '  "coherence": N,\n'
            '  "coherence_reason": "one sentence — identify whether unity is source-emergent or synthesizer-constructed, and note any ignored complicating evidence. Use source names only, never index numbers.",\n'
            '  "insight_depth": N,\n'
            '  "insight_depth_reason": "one sentence — identify whether closing implication is sourced or inferred, and whether a stronger available insight was dropped. Use source names only, never index numbers.",\n'
            '  "citation_support": N,\n'
            '  "citation_support_reason": "one sentence — identify the weakest forward traceability claim AND whether any significant source bullet was omitted that distorts the conclusion. Use source names only, never index numbers."\n'
            '}'
        )

        response = client.messages.create(
            model=EVAL_MODEL,
            max_tokens=512,
            temperature=0.0,
            system=(
                "You are an expert evaluator of product management intelligence briefs. "
                "You assess synthesis quality with precision and consistency. "
                "You are skeptical by default — a paragraph must earn high scores by meeting explicit criteria, "
                "not by sounding confident or well-written. "
                "You have access to ALL source bullets for each cited source, not just the ones the synthesis used. "
                "Your job is to check both what the synthesis included AND what it omitted. "
                "A well-constructed argument built on selectively chosen evidence does not earn a high score. "
                "CRITICAL OUTPUT RULE: In your reason text fields, refer to sources by publication name only. "
                "Never include any index number, bracket number, or numeric reference in reason text. "
                "Write 'MIT Technology Review' not 'source [1]'. Write 'the TechCrunch article' not 'TechCrunch [14]'. "
                "Any index number in your output text invalidates the evaluation."
            ),
            messages=[{"role": "user", "content": user_prompt}],
        )
        block = response.content[0]
        text = getattr(block, "text", None) or block.get("text")  # type: ignore[union-attr]
        parsed = json.loads(_extract_json(text))

        return {
            "paragraph_preview": paragraph[:160],
            "coherence": int(parsed.get("coherence") or 0),
            "coherence_reason": str(parsed.get("coherence_reason") or ""),
            "insight_depth": int(parsed.get("insight_depth") or 0),
            "insight_depth_reason": str(parsed.get("insight_depth_reason") or ""),
            "citation_support": int(parsed.get("citation_support") or 0),
            "citation_support_reason": str(parsed.get("citation_support_reason") or ""),
        }

    def _score_topical_breadth(
        ws_paragraphs: List[str],
        available_themes: Dict[str, int] | None = None,
    ) -> Dict[str, Any]:
        if not ws_paragraphs:
            return {"topical_breadth": 0, "topical_breadth_reason": "No paragraphs to evaluate."}
        client = _build_llm_client()
        combined = "\n\n".join(ws_paragraphs)

        available_themes_block = ""
        if available_themes:
            theme_lines = "\n".join(
                f"  - {theme}: {count} item(s) available"
                for theme, count in sorted(available_themes.items(), key=lambda x: -x[1])
            )
            available_themes_block = (
                f"\nAvailable source material by theme today:\n{theme_lines}\n"
                "If a theme had 2+ available items and does not appear as the central claim of any paragraph, "
                "that is a breadth failure — the synthesizer had material to work with and chose not to use it. "
                "Penalize accordingly.\n"
            )

        response = client.messages.create(
            model=EVAL_MODEL,
            max_tokens=256,
            temperature=0.0,
            system=(
                "You are an expert evaluator of product management intelligence briefs. "
                "You assess whether synthesis achieves genuine thematic diversity across industry topics."
            ),
            messages=[{"role": "user", "content": (
                "Evaluate the TOPICAL BREADTH of this What's Shifting section.\n\n"
                "The five eligible themes for What's Shifting are:\n"
                "1. AI & technology — central claim depends on an AI capability, product, adoption pattern, or AI safety/policy\n"
                "2. Market behavior — central claim is about market dynamics, competitive shifts, pricing, supply/demand, or financial markets\n"
                "3. Consumer behavior — central claim is about how end users are changing what they want, do, or expect\n"
                "4. Regulation & policy — central claim is about regulatory moves, compliance requirements, or government policy\n"
                "5. Design & UX — central claim is about product design patterns, user experience shifts, or interface paradigms\n\n"
                "For each paragraph, identify its central theme based solely on the opening sentence's primary claim. "
                "A theme appearing only as a supporting example does not count — only the central claim determines the theme.\n\n"
                + available_themes_block +
                "Score based on theme diversity across paragraphs:\n"
                "1 = One theme dominates all or nearly all paragraphs (e.g. 4-5 regulation paragraphs)\n"
                "2 = Only two distinct themes represented across all paragraphs\n"
                "3 = Three distinct themes represented, but one theme appears in 2+ paragraphs\n"
                "4 = Four distinct themes represented with no theme appearing more than once\n"
                "5 = Five distinct themes each represented exactly once — ideal spread\n\n"
                "A five-paragraph brief scoring 5 must have each paragraph covering a different theme. "
                "A four-paragraph brief scoring 5 must cover at least four distinct themes.\n\n"
                f"What's Shifting section:\n{combined}\n\n"
                'Return only valid JSON: '
                '{"topical_breadth": N, "topical_breadth_reason": "one sentence listing the central theme of each paragraph, how many distinct themes are represented, and whether any theme with available source material was omitted"}'
            )}],
        )
        try:
            content_block = response.content[0]
            text = getattr(content_block, "text", None) or content_block.get("text")
            parsed = json.loads(_extract_json(text))
        except Exception as e:
            logger.warning("Could not parse breadth evaluation response: %s | raw text: %s", e, text[:200] if text else "None")
            return {"topical_breadth": 3, "topical_breadth_reason": "Could not parse breadth evaluation."}

        score = parsed.get("topical_breadth", 3)
        reason = parsed.get("topical_breadth_reason", "")
        return {"topical_breadth": score, "topical_breadth_reason": reason}

    # ── Score each section ────────────────────────────────────────────────────

    async def score_ws_one(insight: Any) -> Dict[str, Any] | None:
        paragraph = str(insight.get("paragraph") or "").strip() if isinstance(insight, dict) else str(insight).strip()
        indices = insight.get("source_indices") or [] if isinstance(insight, dict) else []
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, lambda: _score_paragraph(paragraph, indices, "whats_shifting synthesis"))
        except Exception:
            return None

    async def score_cw_one(company: str, value: Any) -> Dict[str, Any] | None:
        paragraph = str(value.get("paragraph") or "").strip() if isinstance(value, dict) else str(value).strip()
        indices = value.get("source_indices") or [] if isinstance(value, dict) else []
        if not paragraph:
            return None
        loop = asyncio.get_running_loop()
        try:
            scored = await loop.run_in_executor(None, lambda: _score_paragraph(paragraph, indices, f"company_watch ({company})"))
            if scored:
                scored["company"] = company
            return scored
        except Exception:
            return None

    async def score_sr_one(bullet_entry: Any) -> Dict[str, Any] | None:
        if isinstance(bullet_entry, dict):
            text = str(bullet_entry.get("bullet") or "").strip()
            indices = bullet_entry.get("source_indices") or []
        else:
            text = str(bullet_entry).strip()
            indices = _extract_indices_from_text(text)

        if not text:
            return None

        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, lambda: _score_paragraph(text, indices, "startup_radar"))
        except Exception:
            return None

    async def score_topical_breadth_async() -> Dict[str, Any]:
        ws_paragraphs = [
            str(i.get("paragraph") or "").strip() if isinstance(i, dict) else str(i).strip()
            for i in whats_shifting
        ]
        ws_paragraphs = [p for p in ws_paragraphs if p]
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, lambda: _score_topical_breadth(ws_paragraphs, ws_available_themes))
        except Exception:
            return {"topical_breadth": 0, "topical_breadth_reason": "Eval failed."}

    ws_scores: List[Dict[str, Any]] = []
    for insight in whats_shifting:
        scored = await score_ws_one(insight)
        if scored:
            ws_scores.append(scored)

    cw_scores: List[Dict[str, Any]] = []
    if isinstance(company_watch, dict):
        for company, value in company_watch.items():
            scored = await score_cw_one(company, value)
            if scored:
                cw_scores.append(scored)

    sr_scores: List[Dict[str, Any]] = []
    for bullet_entry in startup_radar:
        scored = await score_sr_one(bullet_entry)
        if scored:
            sr_scores.append(scored)

    topical_breadth_result = await score_topical_breadth_async()

    def _averages(scores: List[Dict[str, Any]]) -> Tuple[float, float, float]:
        if not scores:
            return 0.0, 0.0, 0.0
        return (
            sum(p["coherence"] for p in scores) / len(scores),
            sum(p["insight_depth"] for p in scores) / len(scores),
            sum(p["citation_support"] for p in scores) / len(scores),
        )

    def _representative_reason(scores: List[Dict[str, Any]], dimension: str) -> str:
        if not scores:
            return ""

        key = dimension
        reason_key = f"{dimension}_reason"

        all_scores = [float(p.get(key) or 0) for p in scores]
        min_score = min(all_scores)
        max_score = max(all_scores)

        lowest = min(scores, key=lambda p: p.get(key) or 5)
        reason = str(lowest.get(reason_key) or "")

        if min_score == max_score:
            return f"All paragraphs scored {min_score:.1f}: {reason}"

        return f"Lowest-scoring paragraph ({min_score:.1f}): {reason}"

    ws_avg_c, ws_avg_i, ws_avg_g = _averages(ws_scores)
    cw_avg_c, cw_avg_i, cw_avg_g = _averages(cw_scores)
    sr_avg_c, sr_avg_i, sr_avg_g = _averages(sr_scores)
    ws_topical_breadth = float(topical_breadth_result.get("topical_breadth") or 0.0)

    all_scored = ws_scores + cw_scores + sr_scores
    flagged = [
        p for p in all_scored
        if (p["coherence"] <= 2 or p["insight_depth"] <= 2 or p["citation_support"] <= 2)
    ]
    total_scored = len(all_scored)
    weak_pct = (len(flagged) / total_scored * 100.0) if total_scored else 0.0

    return {
        "ws_paragraph_scores": ws_scores,
        "ws_avg_coherence": ws_avg_c,
        "ws_avg_insight_depth": ws_avg_i,
        "ws_avg_citation_support": ws_avg_g,
        "ws_topical_breadth": ws_topical_breadth,
        "ws_topical_breadth_reason": topical_breadth_result.get("topical_breadth_reason", ""),
        "ws_coherence_reason": _representative_reason(ws_scores, "coherence"),
        "ws_insight_reason": _representative_reason(ws_scores, "insight_depth"),
        "ws_grounding_reason": _representative_reason(ws_scores, "citation_support"),
        "cw_paragraph_scores": cw_scores,
        "cw_avg_coherence": cw_avg_c,
        "cw_avg_insight_depth": cw_avg_i,
        "cw_avg_citation_support": cw_avg_g,
        "cw_coherence_reason": _representative_reason(cw_scores, "coherence"),
        "cw_insight_reason": _representative_reason(cw_scores, "insight_depth"),
        "cw_grounding_reason": _representative_reason(cw_scores, "citation_support"),
        "sr_paragraph_scores": sr_scores,
        "sr_avg_coherence": sr_avg_c,
        "sr_avg_insight_depth": sr_avg_i,
        "sr_avg_citation_support": sr_avg_g,
        "sr_coherence_reason": _representative_reason(sr_scores, "coherence"),
        "sr_insight_reason": _representative_reason(sr_scores, "insight_depth"),
        "sr_grounding_reason": _representative_reason(sr_scores, "citation_support"),
        "avg_coherence": ws_avg_c,
        "avg_insight_depth": ws_avg_i,
        "avg_citation_support": ws_avg_g,
        "flagged_paragraphs": flagged,
        "total_scored": total_scored,
        "weak_pct": weak_pct,
    }


# --------------------------------
# Eval — interview_angle_quality
# --------------------------------


async def interview_angle_quality(
    synthesis: Dict[str, Any],
) -> Dict[str, Any]:
    client = _build_llm_client()
    interview_angle = str(synthesis.get("interview_angle") or "").strip()

    if not interview_angle:
        return {"paragraph_preview": "", "relevance": 0, "relevance_reason": "No interview_angle found."}

    def _call_claude() -> Dict[str, Any]:
        response = client.messages.create(
            model=EVAL_MODEL,
            max_tokens=128,
            temperature=0.0,
            system=(
                "You are an expert evaluator of product management interview preparation material. "
                "You assess how useful synthesized intelligence is for PM interview contexts."
            ),
            messages=[{"role": "user", "content": (
                "Rate this interview angle on PM RELEVANCE (1-5):\n\n"
                "PM RELEVANCE: Would a strong PM candidate use this insight to demonstrate strategic thinking "
                "in a product sense, system design, or product strategy interview? "
                "Is it grounded in a real, recent development rather than generic PM advice? "
                "Does it test a transferable architectural or strategic principle, not just surface-domain knowledge? "
                "A fintech example that tests a general product architecture tradeoff should score as highly as a broadly applicable example — "
                "domain specificity is not a penalty if the underlying principle is transferable across PM roles. "
                "Penalize only when the insight is useful only for a single narrow domain and the principle does not transfer. "
                "1=generic advice any PM book would give, 5=sharply tied to a current development "
                "that reveals genuine product strategy thinking applicable across PM contexts\n\n"
                f"Interview Angle: {interview_angle}\n\n"
                'Return only valid JSON: {"relevance": N, "relevance_reason": "one sentence"}'
            )}],
        )
        block = response.content[0]
        text = getattr(block, "text", None) or block.get("text")  # type: ignore[union-attr]
        return json.loads(_extract_json(text))

    loop = asyncio.get_running_loop()
    try:
        parsed = await loop.run_in_executor(None, _call_claude)
    except Exception:
        return {"paragraph_preview": interview_angle[:160], "relevance": 0, "relevance_reason": "Eval failed."}

    return {
        "paragraph_preview": interview_angle[:160],
        "relevance": int(parsed.get("relevance") or 0),
        "relevance_reason": str(parsed.get("relevance_reason") or ""),
    }


# --------------------------------
# Eval — pm_craft_quality
# --------------------------------


async def pm_craft_quality(
    synthesis: Dict[str, Any],
) -> Dict[str, Any]:
    client = _build_llm_client()

    pm_craft_raw = synthesis.get("pm_craft_today") or {}
    pm_craft = (
        str(pm_craft_raw.get("text") or "").strip()
        if isinstance(pm_craft_raw, dict)
        else str(pm_craft_raw).strip()
    )

    if not pm_craft:
        return {"paragraph_preview": "", "insight_depth": 0, "insight_depth_reason": "No pm_craft_today found."}

    def _call_claude() -> Dict[str, Any]:
        response = client.messages.create(
            model=EVAL_MODEL,
            max_tokens=128,
            temperature=0.0,
            system=(
                "You are an expert evaluator of product management craft content. "
                "You assess how actionable and non-obvious PM craft insights are."
            ),
            messages=[{"role": "user", "content": (
                "Rate this PM craft insight on INSIGHT DEPTH (1-5):\n\n"
                "INSIGHT DEPTH: Is this a genuinely actionable PM craft insight that a practitioner "
                "would find non-obvious? Does it go beyond generic advice found in any PM book? "
                "Is it grounded in a specific, real observation from the sources? "
                "1=generic advice any PM blog would give, 5=sharp, specific, immediately applicable "
                "insight that reflects genuine synthesis from the source material\n\n"
                f"PM Craft insight: {pm_craft}\n\n"
                'Return only valid JSON: {"insight_depth": N, "insight_depth_reason": "one sentence"}'
            )}],
        )
        block = response.content[0]
        text = getattr(block, "text", None) or block.get("text")
        return json.loads(_extract_json(text))

    loop = asyncio.get_running_loop()
    try:
        parsed = await loop.run_in_executor(None, _call_claude)
    except Exception:
        return {"paragraph_preview": pm_craft[:160], "insight_depth": 0, "insight_depth_reason": "Eval failed."}

    return {
        "paragraph_preview": pm_craft[:160],
        "insight_depth": int(parsed.get("insight_depth") or 0),
        "insight_depth_reason": str(parsed.get("insight_depth_reason") or ""),
    }


# --------------------------------
# Orchestrator
# --------------------------------


def run(
    date_str: str | None,
    synthesis: Dict[str, Any],
    items_by_theme: Dict[str, Any],
    fetch_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if not date_str:
        date_str = date.today().isoformat()

    pipeline_funnel_result = pipeline_funnel(items_by_theme, synthesis, fetch_metadata)
    pm_relevance_result = pm_relevance(items_by_theme)

    WHATS_SHIFTING_THEMES = {
        "ai_technology", "market_behavior", "consumer_behavior", "regulation_policy", "design_ux"
    }
    ws_available_themes: Dict[str, int] = {}
    for theme, items in (items_by_theme or {}).items():
        if theme in WHATS_SHIFTING_THEMES:
            relevant_count = sum(
                1 for item in (items or [])
                if isinstance(item, dict)
                and str(item.get("confidence") or "low").lower() in {"high", "medium"}
                and str(item.get("pm_relevance_score") or "low").lower() in {"high", "medium"}
            )
            if relevant_count > 0:
                ws_available_themes[theme] = relevant_count

    try:
        llm_judge_result = asyncio.run(llm_judge(synthesis, items_by_theme, ws_available_themes))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            llm_judge_result = loop.run_until_complete(llm_judge(synthesis, items_by_theme, ws_available_themes))
        finally:
            loop.close()

    try:
        pm_craft_result = asyncio.run(pm_craft_quality(synthesis))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            pm_craft_result = loop.run_until_complete(pm_craft_quality(synthesis))
        finally:
            loop.close()

    try:
        interview_angle_result = asyncio.run(interview_angle_quality(synthesis))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            interview_angle_result = loop.run_until_complete(interview_angle_quality(synthesis))
        finally:
            loop.close()

    ws_coherence = float(llm_judge_result.get("ws_avg_coherence") or 0.0)
    ws_insight   = float(llm_judge_result.get("ws_avg_insight_depth") or 0.0)
    ws_grounding = float(llm_judge_result.get("ws_avg_citation_support") or 0.0)
    ws_breadth   = float(llm_judge_result.get("ws_topical_breadth") or 0.0)

    cw_coherence = float(llm_judge_result.get("cw_avg_coherence") or 0.0)
    cw_insight   = float(llm_judge_result.get("cw_avg_insight_depth") or 0.0)
    cw_grounding = float(llm_judge_result.get("cw_avg_citation_support") or 0.0)

    sr_coherence = float(llm_judge_result.get("sr_avg_coherence") or 0.0)
    sr_insight   = float(llm_judge_result.get("sr_avg_insight_depth") or 0.0)
    sr_grounding = float(llm_judge_result.get("sr_avg_citation_support") or 0.0)

    pc_insight   = float(pm_craft_result.get("insight_depth") or 0.0)
    ia_relevance = float(interview_angle_result.get("relevance") or 0.0)

    score_components: List[float] = []
    score_weights: List[float] = []

    ws_scores = llm_judge_result.get("ws_paragraph_scores") or []
    cw_scores = llm_judge_result.get("cw_paragraph_scores") or []
    sr_scores = llm_judge_result.get("sr_paragraph_scores") or []

    if ws_scores:
        score_components.extend([
            ws_coherence / 5.0 * 10.0,
            ws_insight / 5.0 * 10.0,
            ws_grounding / 5.0 * 10.0,
            ws_breadth / 5.0 * 10.0,
        ])
        score_weights.extend([10.0, 10.0, 10.0, 10.0])

    if cw_scores:
        score_components.extend([
            cw_coherence / 5.0 * 8.3,
            cw_insight / 5.0 * 8.3,
            cw_grounding / 5.0 * 8.4,
        ])
        score_weights.extend([8.3, 8.3, 8.4])

    if sr_scores:
        score_components.extend([
            sr_coherence / 5.0 * 6.7,
            sr_insight / 5.0 * 6.7,
            sr_grounding / 5.0 * 6.6,
        ])
        score_weights.extend([6.7, 6.7, 6.6])

    if pc_insight > 0:
        score_components.append(pc_insight / 5.0 * 10.0)
        score_weights.append(10.0)

    if ia_relevance > 0:
        score_components.append(ia_relevance / 5.0 * 5.0)
        score_weights.append(5.0)

    total_weight = sum(score_weights)
    if total_weight > 0:
        raw_score = sum(score_components)
        overall_score = raw_score / total_weight * 100.0
    else:
        overall_score = 0.0

    sections_scored = []
    if ws_scores:
        sections_scored.append("whats_shifting")
    if cw_scores:
        sections_scored.append("company_watch")
    if sr_scores:
        sections_scored.append("startup_radar")
    if pc_insight > 0:
        sections_scored.append("pm_craft")
    if ia_relevance > 0:
        sections_scored.append("interview_angle")

    flags = {
        "flagged_paragraphs": llm_judge_result.get("flagged_paragraphs") or [],
        "total_scored": int(llm_judge_result.get("total_scored") or 0),
        "weak_pct": float(llm_judge_result.get("weak_pct") or 0.0),
        "sections_scored": sections_scored,
        "sections_scored_count": len(sections_scored),
    }

    eval_result = EvalResult(
        pipeline_funnel=pipeline_funnel_result,
        pm_relevance=pm_relevance_result,
        llm_judge=llm_judge_result,
        pm_craft=pm_craft_result,
        interview_angle=interview_angle_result,
        overall_score=overall_score,
        flags=flags,
    )

    conn = _get_connection()
    try:
        _ensure_evals_table(conn)
        conn.execute(
            """
            INSERT OR REPLACE INTO evals (
              date, pipeline_funnel_json,
              pm_relevance_json, llm_judge_json, pm_craft_json, interview_angle_json,
              overall_score, flags_json, evaluated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                date_str,
                json.dumps(eval_result.pipeline_funnel),
                json.dumps(eval_result.pm_relevance),
                json.dumps(eval_result.llm_judge),
                json.dumps(eval_result.pm_craft),
                json.dumps(eval_result.interview_angle),
                eval_result.overall_score,
                json.dumps(eval_result.flags),
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "date": date_str,
        "pipeline_funnel": eval_result.pipeline_funnel,
        "pm_relevance": eval_result.pm_relevance,
        "llm_judge": eval_result.llm_judge,
        "pm_craft": eval_result.pm_craft,
        "interview_angle": eval_result.interview_angle,
        "overall_score": eval_result.overall_score,
        "flags": eval_result.flags,
    }