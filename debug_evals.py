import json
import sqlite3
import sys

sys.path.insert(0, ".")

from backend.app.services.evaluator import llm_judge  # noqa: F401
import asyncio  # noqa: F401

conn = sqlite3.connect("data/digest.sqlite3")
syn_row = conn.execute(
    "SELECT synthesis_json, items_by_theme_json FROM digests WHERE date = '2026-03-09'"
).fetchone()

synthesis = json.loads(syn_row[0])
items_by_theme = json.loads(syn_row[1])

# Test the lookup manually
lookup = synthesis.get("source_index_lookup", {})
whats_shifting = synthesis.get("whats_shifting", [])

# Check first paragraph
first = whats_shifting[0]
indices = first.get("source_indices", [])
print("indices:", indices)
for idx in indices:
    meta = lookup.get(str(idx))
    print(f"  str({idx}) -> {meta}")




