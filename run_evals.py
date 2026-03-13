from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import date
from typing import Any, Dict, List, Tuple

from backend.app.config import load_settings
from backend.app.services import evaluator


def _get_connection() -> sqlite3.Connection:
    settings = load_settings()
    db_path = settings.database_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(db_path))


def _load_digest_for_date(
    conn: sqlite3.Connection, date_str: str
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]] | None:
    cur = conn.execute(
        "SELECT synthesis_json, items_by_theme_json, fetch_metadata_json FROM digests WHERE date = ?",
        (date_str,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    synthesis = json.loads(row[0])
    items_by_theme = json.loads(row[1])
    fetch_metadata = json.loads(row[2]) if row[2] else {}
    return synthesis, items_by_theme, fetch_metadata


def _iter_all_digest_dates(conn: sqlite3.Connection) -> List[str]:
    cur = conn.execute("SELECT date FROM digests ORDER BY date")
    return [row[0] for row in cur.fetchall()]


def _run_for_date(conn: sqlite3.Connection, date_str: str) -> None:
    loaded = _load_digest_for_date(conn, date_str)
    if loaded is None:
        print(f"[skip] No digest found for {date_str}")
        return

    synthesis, items_by_theme, fetch_metadata = loaded
    print(f"[run] Evaluating {date_str}...", end=" ", flush=True)
    result = evaluator.run(date_str, synthesis, items_by_theme, fetch_metadata=fetch_metadata)
    print(f"done (overall_score={result.get('overall_score'):.1f})")


def _print_report(conn: sqlite3.Connection) -> None:
    # Ensure table + columns exist
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS evals (
          date TEXT PRIMARY KEY,
          pipeline_funnel_json TEXT,
          pm_relevance_json TEXT,
          llm_judge_json TEXT,
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
        ("interview_angle_json", "TEXT"),
    ]:
        try:
            conn.execute(f"ALTER TABLE evals ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass
    conn.commit()

    cur = conn.execute(
        "SELECT date, pipeline_funnel_json, pm_relevance_json, "
        "llm_judge_json, pm_craft_json, interview_angle_json, overall_score "
        "FROM evals ORDER BY date"
    )
    rows = cur.fetchall()
    if not rows:
        print("No evals found. Run evals first.")
        return

    # ── QUALITY TABLE ──────────────────────────────────────────────────────────
    print()
    print("QUALITY SCORES (out of 100)")
    print(
        f"{'Date':<12} {'Overall':>8} "
        f"{'WS-Coh':>7} {'WS-Ins':>7} {'WS-Grd':>7} {'WS-Brd':>7} "
        f"{'CW-Coh':>7} {'CW-Ins':>7} {'CW-Grd':>7} "
        f"{'SR-Coh':>7} {'SR-Ins':>7} {'SR-Grd':>7} "
        f"{'PC-Ins':>7} {'IA-Rel':>7}"
    )
    print("-" * 120)

    for row in rows:
        date_str, pf_json, pm_json, llm_json, pc_json, ia_json, overall = row
        llm = json.loads(llm_json) if llm_json else {}
        pc  = json.loads(pc_json)  if pc_json  else {}
        ia  = json.loads(ia_json)  if ia_json  else {}

        ws_c = float(llm.get("ws_avg_coherence")        or llm.get("avg_coherence")        or 0.0)
        ws_i = float(llm.get("ws_avg_insight_depth")    or llm.get("avg_insight_depth")    or 0.0)
        ws_g = float(llm.get("ws_avg_citation_support") or llm.get("avg_citation_support") or 0.0)
        ws_b = float(llm.get("ws_topical_breadth")      or 0.0)
        cw_c = float(llm.get("cw_avg_coherence")        or 0.0)
        cw_i = float(llm.get("cw_avg_insight_depth")    or 0.0)
        cw_g = float(llm.get("cw_avg_citation_support") or 0.0)
        sr_c = float(llm.get("sr_avg_coherence")        or 0.0)
        sr_i = float(llm.get("sr_avg_insight_depth")    or 0.0)
        sr_g = float(llm.get("sr_avg_citation_support") or 0.0)
        pc_i = float(pc.get("insight_depth")            or 0.0)
        ia_r = float(ia.get("relevance")                or 0.0)
        ov   = float(overall or 0.0)

        print(
            f"{date_str:<12} {ov:>8.1f} "
            f"{ws_c:>7.2f} {ws_i:>7.2f} {ws_g:>7.2f} {ws_b:>7.2f} "
            f"{cw_c:>7.2f} {cw_i:>7.2f} {cw_g:>7.2f} "
            f"{sr_c:>7.2f} {sr_i:>7.2f} {sr_g:>7.2f} "
            f"{pc_i:>7.2f} {ia_r:>7.2f}"
        )

    print()
    print("  WS=What's Shifting(40pts) CW=Company Watch(25pts) SR=Startup Radar(20pts) PC=PM Craft(10pts) IA=Interview Angle(5pts)")
    print("  Coh=Coherence  Ins=Insight Depth  Grd=Grounding  Brd=Topical Breadth  Rel=PM Relevance")

    # ── GUARDRAILS TABLE ───────────────────────────────────────────────────────
    print()
    print("GUARDRAILS (diagnostic — not in score)")
    print(
        f"{'Date':<12} {'Silent%':>8} {'Fetched':>8} "
        f"{'Conf%':>7} {'Rel%':>7} {'Util%':>7}"
    )
    print("-" * 56)

    for row in rows:
        date_str, pf_json, pm_json, llm_json, pc_json, ia_json, overall = row
        pf = json.loads(pf_json) if pf_json else {}

        src_cfg    = int(pf.get("sources_configured") or 0)
        src_act    = int(pf.get("sources_active")     or 0)
        silent_pct = ((src_cfg - src_act) / src_cfg * 100.0) if src_cfg else 0.0
        fetched    = int(pf.get("fetched")            or 0)
        conf_pct   = float(pf.get("confident_pct")   or 0.0)
        rel_pct    = float(pf.get("relevant_pct")     or 0.0)
        util_pct   = float(pf.get("utilized_pct")     or 0.0)

        print(
            f"{date_str:<12} {silent_pct:>8.1f} {fetched:>8} "
            f"{conf_pct:>7.1f} {rel_pct:>7.1f} {util_pct:>7.1f}"
        )

    print()
    print("  Silent% — sources with no new articles in the lookback window / configured sources")
    print("  Fetched — total articles collected across all active sources;")
    print("            exceeds source count when a source publishes multiple articles per day")
    print("  Conf%   — articles summarized with high/med confidence / fetched;")
    print("            low confidence means the model could not reliably extract signal")
    print("  Rel%    — high/med PM relevance / confident;")
    print("            low relevance means the content is not useful for PM interview prep")
    print("  Util%   — articles cited in the final synthesis / relevant;")
    print("            measures how much of the eligible pool the synthesizer used")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quality evals over stored digests.")
    parser.add_argument("date", nargs="?", help="Specific date (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--all", action="store_true", help="Run evals for all archived dates.")
    parser.add_argument("--report", action="store_true", help="Print trend summary table.")
    args = parser.parse_args()

    conn = _get_connection()
    try:
        if args.all:
            dates = _iter_all_digest_dates(conn)
            if not dates:
                print("No digests found in archive.")
                return
            for d in dates:
                _run_for_date(conn, d)
        else:
            _run_for_date(conn, args.date or date.today().isoformat())

        if args.report:
            _print_report(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()