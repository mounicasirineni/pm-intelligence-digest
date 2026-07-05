# PM Intelligence Digest

A daily industry intelligence brief for product managers who want to stay genuinely current without manually hunting down what matters. It reads from 44 curated sources, reasons across them with Claude, grades its own output, and ships a one-page brief every morning.

**[→ Live demo](https://pm-intelligence-digest-production.up.railway.app/)** · **[→ Digest Health dashboard](https://pm-intelligence-digest-production.up.railway.app/digest-health)** · **[→ Archive](https://pm-intelligence-digest-production.up.railway.app/history)**

---

## What this is

A production Flask app that fetches 44 RSS/podcast feeds every morning, runs a multi-call Claude pipeline (per-item fetch + extraction, then cross-source synthesis with partitioned context), evaluates its own output with a separate Claude-based judge, and serves the result as a five-section editorial brief with inline citations and a visible quality score. I built it as a working demonstration of how I think about product decisions, LLM correctness guarantees, and evaluation rigor — each of which is visible in the code rather than claimed in a doc. It runs daily in production on Railway.

## Why I built it

The problem I kept running into as a PM is that staying genuinely connected to the industry requires either too much time or too much tolerance for noise. Every newsletter I tried was either a link dump or a summary — and both collapse under the same limitation: they restate what happened, not what it means. A PM who has read the headlines but hasn't formed an opinion about them is no better prepared to make decisions or lead a conversation than one who hasn't. What I actually needed was something that reasoned *across* the day's news and surfaced what was shifting — not ten links, but a prepared point of view.

So I built the tool I wanted: a daily brief for one specific reader (a PM who wants to stay sharp and stay current), where every paragraph has to pass a "would you get this from any single source alone" test, and every closing implication has to be concrete enough that someone could walk into a meeting and use it to change a decision. The product decisions in the code — actionability standards, theme diversity enforcement, the Interview Angle section, the filtered-out panel that shows what the system rejected and why — all follow from that one user.

## How it works

The pipeline is seven stages orchestrated from `backend/app/main.py:_run_pipeline`.

**1. Fetch** — `services/rss.py:fetch_items_grouped_by_theme` reads every feed in `config/sources.json`, applies a UTC-aware 24-hour lookback, isolates per-source failures so one dead feed can't take the run down, and returns items grouped by theme along with a `fetch_metadata` blob that later feeds the pipeline funnel. Sources support per-source flags: `thin_feed` (always attempt full article fetch even when RSS body is present), `fetch_blocked` (skip full fetch entirely, use RSS summary only), `max_items`, and `lookback_hours` overrides.

**2. Article Fetch** — `services/fetcher.py:fetch_article_text` runs a three-tier fallback per article URL: (1) direct httpx + BeautifulSoup fetch, skipped for domains in `JINA_PREFERRED_DOMAINS`; (2) Jina Reader (`r.jina.ai`) for JS-rendered or paywalled pages; (3) `og:description` meta tag as last resort, prefixed with a sentinel constant so downstream code can cap confidence to "low". Paywall detection runs on tier-1 and tier-2 output.

**3. Summarize** — `services/summarizer.py:summarize_item` sends each article through three Claude calls, run in parallel across articles (max 10 workers):
- **Call A (Sonnet):** Extract 3–5 insight bullets ordered most-specific-to-most-abstract. Contradiction mandate: complicating bullets must not be silently dropped.
- **Call B (Haiku):** Classify `confidence` (high/medium/low). Short content or og:description fallback caps at "low" or "medium".
- **Call C (Haiku, skipped if confidence == low):** Classify `pm_relevance_score` (high/medium/low). Domain filter drops politics/sports/military. Routine update rule drops cadence posts with no strategic signal.

If the full article fetch returns no usable content, the summarizer falls back to the RSS summary.

**4. Synthesize** — `services/synthesizer.py:synthesize_trends` is the architecturally interesting step. Summarized items are filtered (dropping low-confidence and low-relevance items), capped at three items per source (highest-relevance kept), and then **partitioned** by routing eligibility before any synthesis call sees them:

- **Call 1a (Haiku):** Cross-market classifier — classifies each What's Shifting candidate as cross-market or company-specific. Company-specific items from `technology_trends`, `regulation_policy`, or `user_behavior` are dropped; company-specific `market_signals` items are routed to Startup Radar.
- **Call 1 (Sonnet):** Produces What's Shifting paragraphs (one per theme) with inline `[n]` citations. Required anchors enforce coverage of every available theme.
- **Call 1b (Sonnet, fill calls):** For any WS theme that Call 1 missed, a targeted single-theme fill call is made.
- **Call 1c (Sonnet):** Interview Angle — one specific debatable claim derived from the completed WS section.
- **Call 4a (Haiku):** Startup classifier — removes established companies from the Startup Radar pool before synthesis.
- **Call 2 (Sonnet):** PM Craft Today — single-source extraction from `pm_craft` items only.
- **Call 3 (Sonnet):** Company Watch for nine named companies (Google, Microsoft, Apple, Meta, Amazon, Netflix, NVIDIA, OpenAI, Anthropic), sourced exclusively from that company's own `company_strategy` feed.
- **Call 4b (Sonnet):** Startup Radar — cross-source synthesis on startup-classified items.

Because each synthesis call's context is partitioned at the input level, the model literally cannot cite across sections — routing is enforced structurally, not by prompt instruction.

A battery of deterministic post-processors then runs: a source-concentration warning, a multi-thread / single-thesis check on Company Watch entries, a date-in-the-past validator that scans every paragraph, a cross-paragraph coherence check (same source cited in multiple company entries), two routing canaries that should stay silent forever given the partitioning, a Company Watch source-integrity check that clears any entry citing a non-`company_strategy` source or the wrong company, a PM Craft source-integrity check, a split-implication detector on closing sentences, and a theme audit that flags when one theme anchors more than one What's Shifting paragraph.

**5. Persist** — `services/cache.py:save_digest` writes the synthesis, per-theme item map, fetch metadata, and raw pre-synthesis summaries to SQLite. An in-memory cache in `main.py` fronts it so page loads don't hit disk.

**6. Evaluate** — `services/evaluator.py:run` runs immediately after persistence on a force-refresh. It computes two things. The deterministic **guardrails** (`pipeline_funnel`, `pm_relevance`) walk the 5-stage funnel from sources configured → sources active → fetched → confident → relevant → utilized, where "utilized" is resolved against the actual `source_indices` cited in the synthesis output, not against the filtered pool. The **quality scores** use Claude as a judge, scoring every What's Shifting paragraph, Company Watch entry, and Startup Radar bullet on coherence, insight depth, and citation support (using all source bullets for every cited source — backward completeness, not just forward traceability), plus a whole-section topical-breadth score for What's Shifting, an insight-depth score for PM Craft, and a relevance score for the Interview Angle. Paragraph-level judge calls run concurrently via `asyncio.gather`. Sections that produced no output are excluded from both the score and the weight denominator, so the 0–100 overall score stays comparable across days. Eval rows are written to an `evals` table; warning frequency is tracked in a `warning_counts` table, and prompt changes are logged in `prompt_versions` / `prompt_patches` tables via `services/prompt_registry.py`.

**7. Serve** — `main.py` exposes the following routes:
- `GET /` — today's brief (runs pipeline if no cached result)
- `GET /<YYYY-MM-DD>` — historical brief
- `GET /history` — archive list
- `GET /evals` — evals dashboard
- `GET /digest-health` — current health signals (warnings, score trend)
- `GET /digest-health/pipeline` — 14-day pipeline funnel table
- `GET /digest-health/deviations` — 30-day warning history with prompt change dates overlaid
- `GET /digest-health/quality` — 30-day quality scores
- `GET /refresh?token=<secret>` — force pipeline re-run
- `GET /debug-eval/<date>` — raw synthesis JSON (no DB write)

`templates/index.html` renders the brief with inline citations, an expandable Source Details panel split into *utilized* vs *filtered out with reason* (Low Confidence / Low Relevance / Not Selected), and a visible quality-score bar that links to the evals dashboard. An APScheduler cron in `_start_scheduler_if_needed` triggers the full pipeline daily at the configured timezone-aware hour.

### What the brief produces

- **What's Shifting** — 4–5 cross-source insight paragraphs with inline `[n]` citations, distributed across AI & technology, market behavior, consumer behavior, regulation & policy, and design & UX. No single theme is allowed to anchor more than one paragraph.
- **Interview Angle** — one specific debatable claim worth having a prepared opinion on, anchored to a source already cited in What's Shifting.
- **PM Craft Today** — the single most actionable craft insight, drawn exclusively from `pm_craft` or `design_ux` sources.
- **Company Watch** — strategic signal for nine named companies (Google, Microsoft, Apple, Meta, Amazon, Netflix, NVIDIA, OpenAI, Anthropic), sourced exclusively from that company's own first-party feed with a deterministic integrity check that clears any entry citing the wrong company or a non-company source.
- **Startup Radar** — 2–3 early-stage moves with a named "so what"; established companies are filtered out regardless of feed tag, via a Haiku classifier that runs before synthesis.
- **Source Details** — every underlying article with its insight bullets, split into what the synthesizer actually cited vs. what it saw and rejected.

## What's technically interesting

**Structural routing beats prompt instructions.** Early versions of the synthesizer asked one Claude call to produce all five sections and route items itself via prompt rules. It cheated — recycling Company Watch sources into What's Shifting, combining mechanistically unrelated stories under broad category labels, and violating routing rules given 3,000 tokens earlier in the prompt. The fix was to split synthesis into multiple calls where each call's context is partitioned at the input level. The model can't violate routing because the wrong items are never in its context. The routing canaries in `synthesizer.py` were written to fire if this structural assumption ever breaks — they've been silent since the rewrite.

**LLM-as-judge needs backward completeness, not just forward traceability.** The first version of the grounding evaluator only checked whether each claim traced to a source — and gave high scores to paragraphs that suppressed named expert contradictions. Manual QA caught three failure modes the automated eval missed: contradiction suppression, selective bullet use (1–2 bullets padded while 3–4 stronger ones were dropped), and mechanistically forced thematic combination. I rewrote the judge in `evaluator.py:_score_paragraph` to see *all* source bullets for every cited source — not just the ones the synthesis used — and to score selective omission as a grounding failure even when every included claim is correctly cited. The judge and the synthesizer prompts have been co-evolving ever since; each revealed a failure mode the other couldn't catch alone.

**The "utilized" metric has to be honest.** An earlier version of the pipeline funnel counted any article that passed filtering as utilized. That made the funnel look healthy even when the synthesizer only cited 4 of 20 eligible articles. The current version (`evaluator.py:pipeline_funnel`) resolves utilized against the actual `source_indices` referenced in the synthesis output, not against the filtered pool. Together with the visible Filtered Out panel on the frontend — which labels each rejected article with its reason (Low Confidence, Low Relevance, or Not Selected) — this makes the system honest about what it's throwing away. Trust in an editorial product comes from showing your work, not from looking infallible.

**Feed tags are insufficient for section routing.** A YourStory article about Intuit (a large public company) was tagged `startup_disruption` and reached Startup Radar because the routing logic only read the feed tag. The fix is two layers: the summarizer now assesses `company_maturity` per article from the content, and a Haiku classifier (`Call 4a`) removes established companies from the Startup Radar pool before synthesis. Structural filters on content, not tags.

**Dynamic score normalization keeps day-over-day comparisons meaningful.** Some days have no first-party company news. Some days have no early-stage signal. A fixed 100-point scale would punish a day for being quiet. `evaluator.py:run` only includes sections that produced output in the weight denominator, so the 0–100 overall score stays comparable across days regardless of which sections appeared — and the evals dashboard lists which sections were scored so the comparison stays interpretable.

**Prompt version tracking makes regressions attributable.** `services/prompt_registry.py` registers every system prompt by SHA256 hash at startup. The `prompt_versions` and `prompt_patches` tables log when prompts changed and why. The `/digest-health/deviations` route overlays prompt change dates on the 30-day warning history so it's possible to see whether a quality regression followed a prompt edit.

## Stack

- Python 3.12.8, Flask, Jinja2
- Anthropic Claude Sonnet 4.5 (summarizer + synthesizer), Claude Haiku 4.5 (classifiers / confidence / relevance), Claude Sonnet 4.6 (evaluator / LLM-as-judge)
- `feedparser` for RSS/podcast ingestion; `httpx` + BeautifulSoup for full article fetch; Jina Reader for JS-rendered and paywalled pages
- APScheduler for the daily cron, `pytz` for timezone-aware scheduling
- SQLite for digest + eval persistence (tables: `digests`, `synthesizer_inputs`, `evals`, `warning_counts`, `prompt_versions`, `prompt_patches`)
- Railway for deployment, with a mounted volume so SQLite survives deploys
- Nixpacks pinned to Python 3.12.8 (3.13 free-threaded builds break `mise` on Railway)

## Live demo

**[→ LIVE DEMO](https://pm-intelligence-digest-production.up.railway.app/)** — today's brief, the archive, and the evals dashboard are all linked from the header.

## What I learned / what I'd do differently

**Synthesizers optimize for narrative coherence over source fidelity, and prompt rules alone won't fix it.** The single most consistent failure mode across every version of the synthesizer has been selective evidence use — picking 1–2 supporting bullets, dropping the rest, and writing a clean paragraph. Every round of prompt-tightening made the symptom rarer but never eliminated it. The real fix was architectural (two-call partitioning with a cross-market classifier pre-pass) plus evaluative (backward completeness in the judge). If I were starting over, I'd build the judge's backward completeness check *first*, before touching the synthesizer prompt at all, because the evals are what actually reveal which prompt rules are load-bearing and which are decorative.

**I over-invested in prompt engineering before I had evals.** The synthesizer prompt is enormous (hundreds of lines of rules) and most of it was written before I had any way to measure which rules mattered. Once I added the LLM judge and saw scores per dimension per day, I could have deleted half the prompt and the quality would have held. Next time: evals first, prompts second, and treat every prompt rule as a hypothesis the eval has to validate before it earns a permanent place in the prompt.

**RSS is more broken than I expected.** About 30% of feeds I curated were dead, blocked by Cloudflare, or redirected to marketing pages. Even with 44 configured sources, roughly 46% are silent on any given day. That's why the Silent % guardrail exists on the evals page — it's a real operating metric, not a diagnostic nicety. The three-tier article fetcher (httpx → Jina → og:description) exists because tier-1 alone failed too often on JS-rendered and paywalled pages. If I were designing source intake from scratch, I'd budget for a scraper tier alongside the RSS tier rather than treating RSS as a complete solution.

**Railway's ephemeral filesystem bit me in production.** The SQLite database was being wiped on every deploy until I mounted a persistent volume. I knew this in principle; I still shipped without it because local testing hid the problem. Lesson: stateful services on PaaS deploys need a volume from day one, not "we'll add it when it matters."

**A one-character bug made the grounding score 1.00 for an entire day.** `source_index_lookup` used integer keys in the synthesizer and string keys in the evaluator. Every evidence lookup in the judge failed silently, so it scored every paragraph as unsourced. The fix was `str(idx)`. The lesson isn't the bug — it's that the eval was designed well enough to make the failure loud (a floor score, visible on the dashboard) instead of degrading quietly. An eval that fails silently is worse than no eval.

**The hardest part of building an LLM product isn't the LLM, it's deciding what "good" means.** Most of my time on this project went into defining — and then redefining — the quality rubric: what counts as a genuine insight, what separates a formed opinion from a restatement, when a multi-source combination is real synthesis vs. forced category labeling. The model was capable of producing good output from day one. What took months was building the editorial taste into the system so it could tell the difference between good output and plausible output, and surface that judgment to the user as a visible score they could trust.

**What I'd change in the architecture.** I'd make the pipeline async end-to-end so `/refresh` returns immediately and progress is streamed to the user, rather than the current synchronous-with-scheduler compromise. I'd also move the evaluator's paragraph scoring into a proper task queue so reruns don't block a web worker. And I'd separate the "editorial prompt" from the "format prompt" — right now both live inside the same giant user message, which makes it hard to iterate on editorial voice without risking JSON schema drift.