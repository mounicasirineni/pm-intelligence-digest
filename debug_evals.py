import json
import sqlite3
import traceback

from backend.app.config import load_settings
from backend.app.services.evaluator import _build_llm_client, EVAL_MODEL
from backend.app.services.summarizer import _extract_json


settings = load_settings()
conn = sqlite3.connect(str(settings.database_path))
cur = conn.execute(
    "SELECT synthesis_json, items_by_theme_json FROM digests WHERE date = ?",
    ("2026-03-12",),
)
row = cur.fetchone()
conn.close()

synthesis = json.loads(row[0])
items_by_theme = json.loads(row[1])

insight = synthesis["whats_shifting"][0]
paragraph = insight["paragraph"]
print("Paragraph:", paragraph[:80])

# Test a basic API call first
print("\n--- Testing basic API call ---")
try:
    client = _build_llm_client()
    resp = client.messages.create(
        model=EVAL_MODEL,
        max_tokens=64,
        temperature=0.0,
        system="You are an evaluator.",
        messages=[{"role": "user", "content": 'Return only valid JSON: {"coherence": 4}'}],
    )
    print("API OK:", resp.content[0].text)
except Exception:
    traceback.print_exc()

# Test the full scoring call
print("\n--- Testing _score_paragraph equivalent ---")
try:
    client = _build_llm_client()
    user_prompt = (
        "Rate this whats_shifting synthesis paragraph on three dimensions:\n\n"
        "1. COHERENCE (1-5): 1=disconnected, 5=tight single thread\n"
        "2. INSIGHT_DEPTH (1-5): 1=pure summary, 5=genuine insight\n"
        "3. CITATION_SUPPORT (1-5): 1=unsupported, 5=fully evidenced\n\n"
        f"Paragraph: {paragraph[:500]}\n\n"
        "Source evidence:\n(none)\n\n"
        'Return only valid JSON: {"coherence": N, "coherence_reason": "one sentence", '
        '"insight_depth": N, "insight_depth_reason": "one sentence", '
        '"citation_support": N, "citation_support_reason": "one sentence"}'
    )
    resp = client.messages.create(
        model=EVAL_MODEL,
        max_tokens=256,
        temperature=0.0,
        system="You are an expert evaluator of product management intelligence briefs.",
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = resp.content[0].text
    print("Raw response:", text)
    cleaned = _extract_json(text)
    parsed = json.loads(cleaned)
    print("Parsed:", parsed)
except Exception:
    traceback.print_exc()