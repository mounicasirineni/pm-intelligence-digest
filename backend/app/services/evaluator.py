from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Tuple

from anthropic import Anthropic

from ..config import load_settings
from .summarizer import _extract_json


EVAL_MODEL = "claude-haiku-4-5-20251001"


@dataclass(frozen=True)
class EvalResult:
    theme_balance: Dict[str, Any]
    citation_coverage: Dict[str, Any]
    source_utilization: Dict[str, Any]
    llm_judge: Dict[str, Any]
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
          theme_balance_json TEXT,
          citation_coverage_json TEXT,
          source_utilization_json TEXT,
          llm_judge_json TEXT,
          overall_score REAL,
          flags_json TEXT,
          evaluated_at TEXT
        )
        """
    )
    conn.commit()


def _build_llm_client() -> Anthropic:
    settings = load_settings()
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. "
            "Populate it in your .env file before running LLM-based evals."
        )
    return Anthropic(api_key=settings.anthropic_api_key)


# -------------------------
# Eval 1 — theme_balance
# -------------------------


def theme_balance(
    items_by_theme: Dict[str, List[Dict[str, Any]]],
    synthesis: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare how many 'what's shifting' sources come from AI vs non-AI themes.

    Assumes AI-related items are under the 'ai_technology' theme key
    (case-insensitive), falling back to a simple substring check if needed.
    """
    source_index_lookup = synthesis.get("source_index_lookup") or {}
    whats_shifting = synthesis.get("whats_shifting") or []

    # Build a mapping from source_name → theme using today's items.
    source_to_theme: Dict[str, str] = {}
    for theme, items in (items_by_theme or {}).items():
        for item in items or []:
            name = (item or {}).get("source_name")
            if not name:
                continue
            source_to_theme[str(name)] = str(theme)

    # Determine which theme keys we treat as "AI".
    ai_theme_keys = {k for k in items_by_theme.keys() if "ai" in str(k).lower()}
    # Always treat a canonical 'ai_technology' key as AI if present.
    if "ai_technology" in items_by_theme:
        ai_theme_keys.add("ai_technology")

    ai_source_count = 0
    non_ai_source_count = 0
    themes_represented: set[str] = set()

    for insight in whats_shifting:
        if isinstance(insight, dict):
            indices = insight.get("source_indices") or []
        else:
            indices = []

        for idx in indices:
            meta = source_index_lookup.get(str(idx)) or {}
            source_name = meta.get("source_name") or meta.get("source") or meta.get("publisher")
            theme = None
            if source_name:
                theme = source_to_theme.get(str(source_name))

            if theme:
                themes_represented.add(theme)

            if theme and theme in ai_theme_keys:
                ai_source_count += 1
            else:
                non_ai_source_count += 1

    total = ai_source_count + non_ai_source_count
    if total == 0:
        ai_pct = 0.0
        non_ai_pct = 0.0
    else:
        ai_pct = (ai_source_count / total) * 100.0
        non_ai_pct = (non_ai_source_count / total) * 100.0

    return {
        "ai_source_pct": ai_pct,
        "non_ai_source_pct": non_ai_pct,
        "themes_represented": sorted(themes_represented),
        # Defined as non-AI percentage to encourage breadth.
        "theme_balance_score": non_ai_pct,
    }


# -----------------------------
# Eval 2 — citation_coverage
# -----------------------------


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_CITATION_RE = re.compile(r"\[\d+\]")


def _iter_paragraphs_for_citation_eval(synthesis: Dict[str, Any]):
    # What's shifting paragraphs
    for insight in synthesis.get("whats_shifting") or []:
        if isinstance(insight, dict):
            para = insight.get("paragraph") or ""
        else:
            para = str(insight)
        para = para.strip()
        if para:
            yield para

    # Company watch signals
    company_watch = synthesis.get("company_watch") or {}
    if isinstance(company_watch, dict):
        for value in company_watch.values():
            if isinstance(value, dict):
                para = value.get("paragraph") or ""
            else:
                para = str(value)
            para = para.strip()
            if para:
                yield para


def citation_coverage(synthesis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Measure how many sentences across key sections are backed by [n]-style citations.
    """
    total_sentences = 0
    uncited_sentences = 0
    uncited_examples: List[str] = []

    for paragraph in _iter_paragraphs_for_citation_eval(synthesis):
        sentences = _SENTENCE_SPLIT_RE.split(paragraph)
        for raw in sentences:
            sent = raw.strip()
            if not sent:
                continue
            total_sentences += 1
            if _CITATION_RE.search(sent):
                continue
            uncited_sentences += 1
            if len(uncited_examples) < 3:
                uncited_examples.append(sent)

    if total_sentences == 0:
        coverage_pct = 0.0
    else:
        coverage_pct = ((total_sentences - uncited_sentences) / total_sentences) * 100.0

    return {
        "total_claims": total_sentences,
        "uncited_claims": uncited_sentences,
        "citation_coverage_pct": coverage_pct,
        "uncited_examples": uncited_examples,
    }


# -----------------------------
# Eval 3 — source_utilization
# -----------------------------


def source_utilization(
    items_by_theme: Dict[str, List[Dict[str, Any]]],
    synthesis: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare how many fetched sources make it into the synthesis.
    """
    fetched_sources: set[Tuple[str | None, str | None]] = set()
    for theme, items in (items_by_theme or {}).items():
        for item in items or []:
            name = (item or {}).get("source_name")
            title = (item or {}).get("title")
            fetched_sources.add((str(name) if name else None, str(title) if title else None))

    source_index_lookup = synthesis.get("source_index_lookup") or {}
    cited_sources: set[Tuple[str | None, str | None]] = set()
    for meta in source_index_lookup.values():
        if not isinstance(meta, dict):
            continue
        name = meta.get("source_name") or meta.get("source")
        title = meta.get("title")
        cited_sources.add((str(name) if name else None, str(title) if title else None))

    fetched_sources.discard((None, None))
    cited_sources.discard((None, None))

    sources_fetched = len(fetched_sources)
    sources_cited = len(cited_sources)

    if sources_fetched == 0:
        util_pct = 0.0
    else:
        util_pct = (sources_cited / sources_fetched) * 100.0

    return {
        "sources_fetched": sources_fetched,
        "sources_cited": sources_cited,
        "utilization_pct": util_pct,
    }


# -----------------------------
# Eval 4 — llm_judge (async)
# -----------------------------


async def llm_judge(
    synthesis: Dict[str, Any],
    items_by_theme: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Ask Claude to score each 'what's shifting' paragraph on coherence, insight depth,
    and citation support.
    """
    client = _build_llm_client()
    whats_shifting = synthesis.get("whats_shifting") or []
    source_index_lookup = synthesis.get("source_index_lookup") or {}

    paragraph_scores: List[Dict[str, Any]] = []

    def _build_source_summaries(indices: List[Any]) -> Dict[str, List[str]]:
        """
        For each cited index, find the matching item in items_by_theme by source_name and title,
        and collect its insight bullets.
        """
        summaries: Dict[str, List[str]] = {}

        indexed_items: Dict[Tuple[str, str], List[str]] = {}
        for _theme, items in (items_by_theme or {}).items():
            for item in items or []:
                if not isinstance(item, dict):
                    continue
                src_name = str(item.get("source_name") or "").strip()
                title = str(item.get("title") or "").strip()
                if not src_name or not title:
                    continue
                insights = item.get("insights") or []
                if not isinstance(insights, list):
                    insights = [str(insights)]
                insights_strs = [str(b) for b in insights]
                indexed_items[(src_name, title)] = insights_strs

        for idx in indices:
            meta = source_index_lookup.get(str(idx)) or {}
            if not isinstance(meta, dict):
                continue
            src_name = str(meta.get("source_name") or meta.get("source") or "").strip()
            title = str(meta.get("title") or "").strip()
            if not src_name or not title:
                continue

            bullets = indexed_items.get((src_name, title)) or []
            if not bullets:
                continue

            prev = summaries.get(src_name, [])
            summaries[src_name] = prev + bullets

        for k, v in list(summaries.items()):
            seen: set[str] = set()
            deduped: List[str] = []
            for b in v:
                if b in seen:
                    continue
                seen.add(b)
                deduped.append(b)
            summaries[k] = deduped

        return summaries

    async def score_one(insight: Any) -> Dict[str, Any] | None:
        if isinstance(insight, dict):
            paragraph = str(insight.get("paragraph") or "").strip()
            indices = insight.get("source_indices") or []
        else:
            paragraph = str(insight).strip()
            indices = []

        if not paragraph:
            return None

        source_summaries = _build_source_summaries(indices)
        if source_summaries:
            evidence_lines = []
            for src_name, bullets in source_summaries.items():
                joined = " | ".join(bullets)
                evidence_lines.append(f"{src_name}: {joined}")
            evidence_block = "Source evidence:\n" + "\n".join(evidence_lines)
        else:
            evidence_block = "Source evidence:\n(none found in underlying items)"

        system_prompt = (
            "You are an expert evaluator of product management intelligence briefs. "
            "You assess synthesis quality with precision and consistency."
        )
        user_prompt = (
            "Rate this synthesis paragraph on three dimensions:\n\n"
            "1. COHERENCE (1-5): Do all sentences support a single unified insight, or are they loosely connected observations? "
            "1=completely disconnected, 5=tight single thread throughout\n\n"
            "2. INSIGHT_DEPTH (1-5): Is this a genuine cross-source synthesis revealing something non-obvious, or a restatement of what happened? "
            "1=pure summary, 5=genuine insight a reader wouldn't get from any single source\n\n"
            "3. CITATION_SUPPORT (1-5): For each sentence in the paragraph, does the cited source actually contain evidence for that specific claim, "
            "or are some sentences unsupported inferences that go beyond what the sources say? "
            "1=multiple sentences are unsupported inferences, 5=every claim is directly evidenced by the cited sources\n\n"
            f"Paragraph: {paragraph}\n\n"
            f"{evidence_block}\n\n"
            'Return only valid JSON: '
            '{"coherence": N, "coherence_reason": "one sentence", '
            '"insight_depth": N, "insight_depth_reason": "one sentence", '
            '"citation_support": N, "citation_support_reason": "one sentence"}'
        )

        def _call_claude() -> Dict[str, Any]:
            response = client.messages.create(
                model=EVAL_MODEL,
                max_tokens=256,
                temperature=0.0,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            block = response.content[0]
            text = getattr(block, "text", None) or block.get("text")  # type: ignore[union-attr]
            cleaned = _extract_json(text)
            return json.loads(cleaned)

        loop = asyncio.get_running_loop()
        try:
            parsed = await loop.run_in_executor(None, _call_claude)
        except Exception:
            return None

        coherence = int(parsed.get("coherence") or 0)
        insight_depth = int(parsed.get("insight_depth") or 0)
        citation_support = int(parsed.get("citation_support") or 0)

        return {
            "paragraph_preview": paragraph[:160],
            "coherence": coherence,
            "coherence_reason": str(parsed.get("coherence_reason") or ""),
            "insight_depth": insight_depth,
            "insight_depth_reason": str(parsed.get("insight_depth_reason") or ""),
            "citation_support": citation_support,
            "citation_support_reason": str(parsed.get("citation_support_reason") or ""),
        }

    for insight in whats_shifting:
        scored = await score_one(insight)
        if scored:
            paragraph_scores.append(scored)

    if not paragraph_scores:
        avg_coherence = 0.0
        avg_insight_depth = 0.0
        avg_citation_support = 0.0
    else:
        avg_coherence = sum(p["coherence"] for p in paragraph_scores) / len(paragraph_scores)
        avg_insight_depth = sum(p["insight_depth"] for p in paragraph_scores) / len(paragraph_scores)
        avg_citation_support = sum(p["citation_support"] for p in paragraph_scores) / len(paragraph_scores)

    flagged_paragraphs = [
        p
        for p in paragraph_scores
        if (p["coherence"] <= 2 or p["insight_depth"] <= 2 or p["citation_support"] <= 2)
    ]

    return {
        "paragraph_scores": paragraph_scores,
        "avg_coherence": avg_coherence,
        "avg_insight_depth": avg_insight_depth,
        "avg_citation_support": avg_citation_support,
        "flagged_paragraphs": flagged_paragraphs,
    }


# -----------------------------
# Orchestrator
# -----------------------------


def run(
    date_str: str | None,
    synthesis: Dict[str, Any],
    items_by_theme: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run all evals, persist to SQLite, and return a structured result dict.
    """
    if not date_str:
        date_str = date.today().isoformat()

    theme_balance_result = theme_balance(items_by_theme, synthesis)
    citation_coverage_result = citation_coverage(synthesis)
    source_utilization_result = source_utilization(items_by_theme, synthesis)

    try:
        llm_judge_result = asyncio.run(llm_judge(synthesis, items_by_theme))
    except RuntimeError:
        llm_judge_result = {
            "paragraph_scores": [],
            "avg_coherence": 0.0,
            "avg_insight_depth": 0.0,
            "avg_citation_support": 0.0,
            "flagged_paragraphs": [],
        }

    avg_coherence = float(llm_judge_result.get("avg_coherence") or 0.0)
    avg_insight_depth = float(llm_judge_result.get("avg_insight_depth") or 0.0)
    avg_citation_support = float(llm_judge_result.get("avg_citation_support") or 0.0)
    theme_balance_score = float(theme_balance_result.get("theme_balance_score") or 0.0)

    overall_score = (
        (avg_coherence / 5.0 * 30.0)
        + (avg_insight_depth / 5.0 * 30.0)
        + (avg_citation_support / 5.0 * 30.0)
        + (theme_balance_score / 100.0 * 10.0)
    )

    flags = {
        "flagged_paragraphs": llm_judge_result.get("flagged_paragraphs") or [],
    }

    eval_result = EvalResult(
        theme_balance=theme_balance_result,
        citation_coverage=citation_coverage_result,
        source_utilization=source_utilization_result,
        llm_judge=llm_judge_result,
        overall_score=overall_score,
        flags=flags,
    )

    conn = _get_connection()
    try:
        _ensure_evals_table(conn)
        conn.execute(
            """
            INSERT OR REPLACE INTO evals (
              date,
              theme_balance_json,
              citation_coverage_json,
              source_utilization_json,
              llm_judge_json,
              overall_score,
              flags_json,
              evaluated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                date_str,
                json.dumps(eval_result.theme_balance),
                json.dumps(eval_result.citation_coverage),
                json.dumps(eval_result.source_utilization),
                json.dumps(eval_result.llm_judge),
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
        "theme_balance": eval_result.theme_balance,
        "citation_coverage": eval_result.citation_coverage,
        "source_utilization": eval_result.source_utilization,
        "llm_judge": eval_result.llm_judge,
        "overall_score": eval_result.overall_score,
        "flags": eval_result.flags,
    }

