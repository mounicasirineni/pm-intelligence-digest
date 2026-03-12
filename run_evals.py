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


def _load_digest_for_date(conn: sqlite3.Connection, date_str: str) -> Tuple[Dict[str, Any], Dict[str, Any]] | None:
    cur = conn.execute(
        "SELECT synthesis_json, items_by_theme_json FROM digests WHERE date = ?",
        (date_str,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    synthesis_json, items_json = row
    return json.loads(synthesis_json), json.loads(items_json)


def _iter_all_digest_dates(conn: sqlite3.Connection) -> List[str]:
    cur = conn.execute("SELECT date FROM digests ORDER BY date")
    return [row[0] for row in cur.fetchall()]


def _run_for_date(conn: sqlite3.Connection, date_str: str) -> None:
    loaded = _load_digest_for_date(conn, date_str)
    if loaded is None:
        print(f"[skip] No digest found for {date_str}")
        return

    synthesis, items_by_theme = loaded
    print(f"[run] Evaluating {date_str}...", end=" ", flush=True)
    result = evaluator.run(date_str, synthesis, items_by_theme)
    print(f"done (overall_score={result.get('overall_score'):.1f})")


def _print_report(conn: sqlite3.Connection) -> None:
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

    cur = conn.execute(
        "SELECT date, theme_balance_json, citation_coverage_json, llm_judge_json, overall_score "
        "FROM evals ORDER BY date"
    )
    rows = cur.fetchall()
    if not rows:
        print("No evals found. Run evals first.")
        return

    print(
        f"{'Date':<12} {'Overall':>8} {'Coherence':>10} {'Insight':>8} {'Grounding':>10} {'ThemeBal%':>10} {'Citation%':>10}"
    )
    print("-" * 86)

    for date_str, tb_json, cc_json, llm_json, overall in rows:
        theme_balance = json.loads(tb_json) if tb_json else {}
        citation_cov = json.loads(cc_json) if cc_json else {}
        llm = json.loads(llm_json) if llm_json else {}

        citation_pct = float(citation_cov.get("citation_coverage_pct") or 0.0)
        avg_coh = float(llm.get("avg_coherence") or 0.0)
        avg_insight = float(llm.get("avg_insight_depth") or 0.0)
        avg_cit_sup = float(llm.get("avg_citation_support") or 0.0)
        theme_balance_score = float(theme_balance.get("theme_balance_score") or 0.0)
        overall_score = float(overall or 0.0)

        print(
            f"{date_str:<12} "
            f"{overall_score:>8.1f} "
            f"{avg_coh:>10.2f} "
            f"{avg_insight:>8.2f} "
            f"{avg_cit_sup:>10.2f} "
            f"{theme_balance_score:>10.1f} "
            f"{citation_pct:>10.1f}"
        )

    print()
    print("* Citation% is diagnostic only — not included in overall score")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quality evals over stored digests.")
    parser.add_argument(
        "date",
        nargs="?",
        help="Specific date to run (YYYY-MM-DD). Defaults to today if omitted and --all is not set.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run evals for all dates in the digests archive.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print a trend summary table from the evals table.",
    )
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
            if args.date:
                date_str = args.date
            else:
                date_str = date.today().isoformat()
            _run_for_date(conn, date_str)

        if args.report:
            print()
            _print_report(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()

