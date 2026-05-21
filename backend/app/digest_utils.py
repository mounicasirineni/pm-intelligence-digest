from __future__ import annotations

from typing import Any, Dict, Set


def get_used_indices(synthesis: Dict[str, Any]) -> Set[str]:
    """
    Return the set of source index strings that Claude actually cited in the
    synthesis output (whats_shifting, company_watch, startup_radar,
    pm_craft_today).

    This is the single authoritative implementation.  Previously duplicated
    in main.py (_build_utilized_keys) and evaluator.py (pipeline_funnel).
    Both now import this function so that adding a new digest section only
    requires updating one place.
    """
    used: Set[str] = set()

    for insight in (synthesis.get("whats_shifting") or []):
        if isinstance(insight, dict):
            used.update(str(i) for i in (insight.get("source_indices") or []))

    for company in (synthesis.get("company_watch") or {}).values():
        if isinstance(company, dict):
            used.update(str(i) for i in (company.get("source_indices") or []))

    for item in (synthesis.get("startup_radar") or []):
        if isinstance(item, dict):
            used.update(str(i) for i in (item.get("source_indices") or []))

    pm_craft = synthesis.get("pm_craft_today")
    if isinstance(pm_craft, dict):
        used.update(str(i) for i in (pm_craft.get("source_indices") or []))

    return used
