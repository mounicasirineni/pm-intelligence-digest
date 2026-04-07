# PM Intelligence Digest

A daily intelligence brief for a product manager who is actively interviewing. It reads from 40 curated sources, reasons across them with Claude, grades its own output, and ships a one-page brief every morning.

**[→ Live demo](https://pm-intelligence-digest-production.up.railway.app/)** · **[→ Evals dashboard](https://pm-intelligence-digest-production.up.railway.app/evals)** · **[→ Archive](https://pm-intelligence-digest-production.up.railway.app/history)**

---

## What this is

A production Flask app that fetches 40 RSS/podcast feeds every morning, runs a two-pass Claude pipeline (per-item signal extraction, then cross-source synthesis), evaluates its own output with a separate Haiku-based judge, and serves the result as a five-section editorial brief with inline citations and a visible quality score. I built it as a working demonstration of how I think about product decisions, LLM correctness guarantees, and evaluation rigor — each of which is visible in the code rather than claimed in a doc. It runs daily in production on Railway.

## Why I built it

I was preparing for PM interviews at top tech companies and couldn't find a source that did the thing I actually needed: reason *across* the day's news and tell me what was shifting, not restate what happened. Every newsletter I tried was either a link dump or a summary, and both collapse under the same problem — a PM in an interview gets rewarded for having a *prepared opinion*, not for having read the headlines. So I built the tool I wanted: a daily brief written for one specific reader (a PM interviewing this week), where every paragraph has to pass a "would you get this from any single source alone" test, and every closing implication has to be concrete enough that a PM could walk into a meeting tomorrow and use it to change a decision. The product decisions in the code — actionability standards, theme diversity enforcement, the Interview Angle section, the filtered-out panel that shows what the system rejected and why — all follow from that one user.

## How it works

The pipeline is three stages plus evaluation, all orchestrated from `backend/app/main.py:_run_pipeline`.

**1. Fetch** — `services/rss.py:fetch_items_grouped_by_theme` reads every feed in `config/sources.json`, applies a UTC-aware 24-hour lookback, isolates per-source failures so one dead feed can't take the run down, and returns items grouped by theme along with a `fetch_metadata` blob that later feeds the pipeline funnel.

**2. Summarize** — `services/summarizer.py:summarize_item` sends each article to Claude Sonnet with a strict prompt: extract 3–5 bullets ordered most-specific-to-most-abstract, and label the article on `confidence` (how grounded the bullets can be in the content body), `pm_relevance_score`, `company_maturity`, and `scope`. The prompt enforces source fidelity, qualifier preservation, and a contradiction mandate so complicating bullets can't be silently dropped.

**3. Synthesize** — `services/synthesizer.py:synthesize_trends` is the architecturally interesting step. Summarized items are filtered and then **partitioned** into two pools by routing eligibility. One Claude call sees only the *What's Shifting* pool and produces the cross-source trend paragraphs plus the Interview Angle. A second call sees only the *dedicated-section* pool and produces Company Watch, Startup Radar, and PM Craft. Because each call's context is partitioned, the model literally cannot cite across sections — routing is enforced structurally, not by prompt instruction. A battery of deterministic post-processors then runs: routing canaries that should stay silent forever, a Company Watch source-integrity check that clears any entry citing the wrong company, a theme audit, a date-in-the-past detector, and a split-implication heuristic.

**4. Persist** — `services/cache.py:save_digest` writes the synthesis, per-theme item map, and fetch metadata to SQLite, one row per date. An in-memory cache in `main.py` fronts it so page loads don't hit disk.

**5. Evaluate** — `services/evaluator.py:run` computes two things. The deterministic **guardrails** (`pipeline_funnel`, `pm_relevance`) walk the 5-stage funnel from sources active → fetched → confident → relevant → utilized. The **quality scores** use a separate Claude Haiku model as a judge, scoring every paragraph on coherence, insight depth, and citation support, plus a topical-breadth score for What's Shifting as a whole. Sections that produced no output are excluded from both the score and the weight denominator, so the 0–100 overall score stays comparable across days. Eval rows are stored alongside the digest in the same SQLite file.

**6. Serve** — `main.py` exposes `/`, `/<YYYY-MM-DD>`, `/history`, and `/evals`. `templates/index.html` renders the brief with inline citations, an expandable Source Details panel split into *utilized* vs *filtered out with reason*, and a visible quality-score bar that links to the evals dashboard. An APScheduler cron triggers the full pipeline daily at the configured time.

### What the brief produces

- **What's Shifting** — 4–5 cross-source insight paragraphs with inline `[n]` citations, distributed across AI & technology, market behavior, consumer behavior, regulation & policy, and design & UX. No single theme is allowed to anchor more than one paragraph.
- **Interview Angle** — one specific debatable claim a PM should walk in prepared to defend, anchored to a source already cited in What's Shifting.
- **PM Craft Today** — the single most actionable craft insight, drawn exclusively from product_craft or design_ux sources.
- **Company Watch** — strategic signal for nine named companies, sourced exclusively from first-party feeds with a deterministic integrity check.
- **Startup Radar** — 2–3 early-stage moves with a named "so what"; established companies are filtered out regardless of feed tag.
- **Source Details** — every underlying article with its insight bullets, split into what the synthesizer actually cited vs. what it saw and rejected.

## What's technically interesting

**Structural routing beats prompt instructions.** Early versions of the synthesizer asked one Claude call to produce all five sections and route items itself via prompt rules. It cheated — recycling Company Watch sources into What's Shifting, combining mechanistically unrelated stories under broad category labels, and violating routing rules given 3,000 tokens earlier in the prompt. The fix was to split synthesis into two calls where each call's context is partitioned at the input level. The model can't violate routing because the wrong items are never in its context. The routing canaries in `synthesizer.py` were written to fire if my structural assumption ever breaks — they've been silent since the rewrite.

**LLM-as-judge needs backward completeness, not just forward traceability.** The first version of the grounding evaluator only checked whether each claim traced to a source — and gave 5.0s to paragraphs that suppressed named expert contradictions. Manual QA caught three failure modes the automated eval missed: contradiction suppression, selective bullet use (1–2 bullets padded while 3–4 stronger ones were dropped), and mechanistically forced thematic combination. I rewrote the judge in `evaluator.py:_score_paragraph` to see *all* source bullets for every cited source — not just the ones the synthesis used — and to score selective omission as a grounding failure even when every included claim is correctly cited. The judge and the synthesizer prompts have been co-evolving ever since; each revealed a failure mode the other couldn't catch alone.

**The "utilized" metric has to be honest.** An earlier version of the pipeline funnel counted any article that passed filtering as utilized. That made the funnel look healthy even when the synthesizer only cited 4 of 20 eligible articles. The current version (`evaluator.py:pipeline_funnel`) resolves utilized against the actual `source_indices` referenced in the synthesis output, not against the filtered pool. Together with the visible Filtered Out panel on the frontend — which labels each rejected article with its reason (Low Confidence, Low Relevance, or Not Selected) — this makes the system honest about what it's throwing away. Trust in an editorial product comes from showing your work, not from looking infallible.

**Feed tags are insufficient for section routing.** A YourStory article about Intuit (a large public company) was tagged `startup_disruption` and reached Startup Radar because the routing logic only read the feed tag. The fix is two layers: the summarizer now assesses `company_maturity` per article from the content, and the synthesizer filters out `startup_disruption` items whose primary subject is an established company *before* they reach the prompt. Structural filters on content, not tags.

**Dynamic score normalization keeps day-over-day comparisons meaningful.** Some days have no first-party company news. Some days have no early-stage signal. A fixed 100-point scale would punish a day for being quiet. `evaluator.py:run` only includes sections that produced output in the weight denominator, so the 0–100 overall score stays comparable across days regardless of which sections appeared — and the evals dashboard lists which sections were scored so the comparison stays interpretable.

## Stack

- Python 3.12, Flask
- Anthropic Claude Sonnet (synthesis), Claude Haiku (evaluation)
- feedparser for RSS/podcast ingestion
- APScheduler for the daily cron
- SQLite for digest + eval persistence (one row per day, keyed by date)
- Jinja2 templates for the editorial frontend
- Railway for deployment, with a mounted volume for SQLite persistence
- Nixpacks pinned to Python 3.12.8 (3.13 free-threaded builds break `mise` on Railway)

## Live demo

**[→ LIVE DEMO](https://pm-intelligence-digest-production.up.railway.app/)** — today's brief, the archive, and the evals dashboard are all linked from the header.

## What I learned / what I'd do differently

**Synthesizers optimize for narrative coherence over source fidelity, and prompt rules alone won't fix it.** The single most consistent failure mode across every version of the synthesizer has been selective evidence use — picking 1–2 supporting bullets, dropping the rest, and writing a clean paragraph. Every round of prompt-tightening made the symptom rarer but never eliminated it. The real fix was architectural (see the two-call partitioning) plus evaluative (backward completeness in the judge). If I were starting over, I'd build the judge's backward completeness check *first*, before touching the synthesizer prompt at all, because the evals are what actually reveal which prompt rules are load-bearing and which are decorative.

**I over-invested in prompt engineering before I had evals.** The synthesizer prompt is enormous (hundreds of lines of rules) and most of it was written before I had any way to measure which rules mattered. Once I added the LLM judge and saw scores per dimension per day, I could have deleted half the prompt and the quality would have held. Next time: evals first, prompts second, and treat every prompt rule as a hypothesis the eval has to validate before it earns a permanent place in the prompt.

**RSS is more broken than I expected.** About 30% of feeds I curated were dead, blocked by Cloudflare, or redirected to marketing pages. Even with 40 configured sources, roughly 46% are silent on any given day. That's why the Silent % guardrail exists on the evals page — it's a real operating metric, not a diagnostic nicety. If I were designing source intake from scratch, I'd budget for a scraper tier alongside the RSS tier rather than treating RSS as a complete solution.

**Railway's ephemeral filesystem bit me in production.** The SQLite database was being wiped on every deploy until I mounted a persistent volume. I knew this in principle; I still shipped without it because local testing hid the problem. Lesson: stateful services on PaaS deploys need a volume from day one, not "we'll add it when it matters."

**A one-character bug made the grounding score 1.00 for an entire day.** `source_index_lookup` used integer keys in the synthesizer and string keys in the evaluator. Every evidence lookup in the judge failed silently, so it scored every paragraph as unsourced. The fix was `str(idx)`. The lesson isn't the bug — it's that the eval was designed well enough to make the failure loud (a floor score, visible on the dashboard) instead of degrading quietly. An eval that fails silently is worse than no eval.

**The hardest part of building an LLM product isn't the LLM, it's deciding what "good" means.** Most of my time on this project went into defining — and then redefining — the quality rubric: what counts as a genuine insight, what separates a prepared opinion from a restatement, when a multi-source combination is real synthesis vs. forced category labeling. The model was capable of producing good output from day one. What took months was building the editorial taste into the system so it could tell the difference between good output and plausible output, and surface that judgment to the user as a visible score they could trust.

**What I'd change in the architecture.** I'd make the pipeline async end-to-end so `/refresh` returns immediately and progress is streamed to the user, rather than the current synchronous-with-scheduler compromise. I'd also move the evaluator's paragraph scoring into a proper task queue so reruns don't block a web worker. And I'd separate the "editorial prompt" from the "format prompt" — right now both live inside the same giant user message, which makes it hard to iterate on editorial voice without risking JSON schema drift.
