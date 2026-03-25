# PM Intelligence Digest

> A daily AI-powered intelligence digest that surfaces what's actually shifting in the industry — not just what happened, but what it means.

**[→ Live Demo](https://pm-intelligence-digest-production.up.railway.app)** · **[→ Evals Dashboard](https://pm-intelligence-digest-production.up.railway.app/evals)** · **[→ Archive](https://pm-intelligence-digest-production.up.railway.app/history)**

---

## What This Is

Most news digests summarize. This one synthesizes.

The brief runs every morning at 7am, pulls from 40 curated sources across 8 themes, and uses Claude to do two things most digest tools don't:

1. **Extract signal, not summaries** — each article is analyzed for what it means, not just what happened. Every insight bullet must pass a non-obvious test: would a reader get this from the headline alone? If yes, it's not an insight.
2. **Reason across sources** — a second AI pass connects signals across multiple sources to surface patterns that don't appear in any single article.

Built for product managers who want a prepared opinion on what's shifting — across AI, business strategy, consumer behavior, regulation, and design.

## What It Produces

Every day the brief generates:

* **What's Shifting** — 4-5 cross-source insights with inline citations, distributed across five themes: AI & technology, market behavior, consumer behavior, regulation & policy, and design & UX — no single theme dominates more than one paragraph per brief
* **Interview Angle** — one specific thing to have a prepared opinion on, rotated across product strategy, consumer insight, regulatory navigation, and AI
* **PM Craft Today** — the most actionable PM craft insight from the day's content, grounded in a specific source
* **Company Watch** — strategic signal for Google, Microsoft, Apple, Meta, Amazon, Netflix, NVIDIA, OpenAI, and Anthropic — what's strategically shifting, not just what they announced, sourced exclusively from first-party official feeds
* **Startup Radar** — 2-3 disruption moves worth knowing about, each with a "so what" — the competitive threat or market pattern it reveals
* **Source Details** — every underlying article with confidence score, relevance score, and insight bullets

## How It Works

```
RSS/Podcast Sources (40)
        ↓
    Fetcher
    (24hr lookback window)
        ↓
  Summarizer (Pass 1)
  Claude extracts signal per item
  scores confidence + PM relevance
  non-obvious insight test applied
  PM actionability ranking applied
        ↓
  Synthesizer (Pass 2)
  Sources partitioned by routing eligibility
  Claude reasons across all items
  adds inline citations [n]
  builds source_index_lookup
        ↓
  Evaluator (Pass 3)
  LLM-as-judge scores quality
  5 dimensions, 100pt weighted scale
        ↓
  Post-processing Validators
  multi-thread violations
  date warnings
  coherence warnings
  routing violations
        ↓
  SQLite Cache
  (date-keyed, Railway Volume)
        ↓
  Flask Web App
  (daily auto-refresh at 7am IST)
```

## Evals Framework

Every digest run is automatically evaluated by an LLM judge across 5 quality dimensions, producing a weighted score out of 100:

| Section | Dimensions | Weight |
|---|---|---|
| What's Shifting | Coherence, Insight Depth, Grounding, Topical Breadth | 40pts |
| Company Watch | Coherence, Insight Depth, Grounding | 25pts |
| Startup Radar | Coherence, Insight Depth, Grounding | 20pts |
| PM Craft Today | Insight Depth | 10pts |
| Interview Angle | PM Relevance | 5pts |

**The 5 dimensions:**

| Dimension | What it measures |
|---|---|
| **Coherence** | Do all sentences in a paragraph support a single unified insight? |
| **Insight Depth** | Is this a genuine synthesis revealing something non-obvious, or just a summary? |
| **Grounding** | Does the cited source actually contain evidence for each claim? |
| **Topical Breadth (What's Shifting)** | Does this section distribute central claims across the five eligible themes (AI & technology, market behavior, consumer behavior, regulation & policy, design & UX)? |
| **Relevance (Interview Angle)** | Can a PM use this insight to demonstrate strategic thinking in an interview? |

**Pipeline guardrails** (diagnostic, not scored):

| Metric | What it measures |
|---|---|
| Silent % | Sources with no new articles in the lookback window |
| Confident % | Articles summarized with high/medium confidence |
| Relevant % | Articles with high/medium PM relevance |
| Utilized % | Relevant articles actually cited in the synthesis |
| Weak % | Paragraphs scoring ≤2 on any quality dimension |

**Post-processing editorial warnings** (logged per run, not scored):

| Warning | What it catches |
|---|---|
| Multi-thread violations | Company Watch entries citing more than 2 sources or with high conjunction counts suggesting multiple disconnected claims |
| Date warnings | Milestone or timeline claims where the cited date is earlier than today — catches past dates stated as future events |
| Coherence warnings | Same source cited in multiple company entries (risk of contradictory framings) or shared between What's Shifting and Company Watch (routing violation) |
| Routing violations | What's Shifting paragraphs citing dedicated-section sources, or Company Watch citing What's Shifting sources |

Every brief on the [Evals page](https://pm-intelligence-digest-production.up.railway.app/evals) includes an inline reasons row in the table showing the judge's one-sentence reasoning per dimension.

## Source Design

40 sources across 8 themes, curated for balance:

| Theme | Sources |
|---|---|
| AI & Technology | Import AI, Simon Willison, Benedict Evans, AI Snake Oil, a16z, Dwarkesh |
| Company Strategy | Google, Meta, Apple, Amazon, OpenAI, Microsoft Research, NVIDIA, Netflix Tech Blog, Anthropic News |
| Product Craft | Shreyas Doshi, SVPG, Lenny's Podcast, Lenny's Newsletter, How I AI |
| Startup Disruption | TechCrunch, Y Combinator, Hacker News, YourStory |
| Market Behavior | Platformer, Stratechery, Rest of World, Hard Fork, Sifted, The Verge, Vulcan Post |
| Consumer Behavior | Quartz, Axios, Pivot |
| Regulation & Policy | Politico Tech, EFF, Medianama |
| Design & UX | Nielsen Norman Group, UX Collective |

**Source routing:** Company Watch is sourced exclusively from first-party official feeds (company blogs and newsrooms). What's Shifting, Startup Radar, and PM Craft draw from third-party sources only — ensuring Company Watch and What's Shifting never recycle the same source across sections.

Sources were chosen to balance AI-optimist vs. skeptic voices, US vs. global perspective, and primary sources vs. independent analysis.

## Hallucination Mitigation

Every claim in the synthesis is grounded in cited source items. The `source_index_lookup` maps each `[n]` citation back to the original article title and source name, making it easy to verify any claim in seconds.

The evaluator's **Grounding** dimension specifically checks whether cited sources actually contain evidence for each claim. The grounding rubric requires passage-level traceability — for each synthesis claim, the evaluator must identify the specific source sentence or data point that supports it. Plausibility and topic consistency are explicitly excluded as proxies for evidence. Claims that contradict an explicit source statement (e.g. scope inversions) or assert specific numbers not present in the source are scored as grounding failures regardless of how well-written they appear.

Only high and medium confidence items (scored by Claude in Pass 1) are fed into the synthesis pass.

## Setup

### Prerequisites

* Python 3.9+
* pip
* Git
* [Cursor](https://cursor.sh) — AI-assisted development environment used to build this project
* Anthropic API key — get one at [console.anthropic.com](https://console.anthropic.com)
* Private RSS URLs (optional) — `LENNYS_PODCAST_RSS`, `HOW_I_AI_RSS`, `LENNYS_NEWSLETTER_RSS` for private feed sources. Without them those sources are skipped — the pipeline runs fine with the remaining public sources.

### Installation

```bash
# Clone the repo
git clone https://github.com/mounicasirineni/pm-intelligence-digest.git
cd pm-intelligence-digest

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# Run
python run.py
```

Open `http://127.0.0.1:8000` — the first load triggers the full pipeline. Runtime depends on how many sources are active that day (typically 2-5 minutes locally). Results are cached to SQLite after completion, so subsequent loads are instant.

> **Note for production deployments:** The pipeline runs too long for a synchronous HTTP request. Use the built-in scheduler instead — it runs the pipeline in the background at the configured time. Avoid triggering `/refresh` manually in production.

### Railway / Nixpacks (Python version)

The repo pins **Python 3.12.8** via `nixpacks.toml`, `mise.toml`, and `.python-version` so Railway’s build does not pull **Python 3.13.x** builds that can fail under `mise` with *“Python installation is missing a `lib` directory”* (often tied to experimental freethreaded installers). If you deploy elsewhere, use Python **3.12.x** or **3.11.x** for the same reason.

### Configuration

All settings are in `.env`:

```
ANTHROPIC_API_KEY=        # Required
CLAUDE_MODEL=claude-sonnet-4-5
LOOKBACK_HOURS=24         # Content window
DIGEST_SCHEDULE_HOUR=7    # Daily refresh time
DIGEST_SCHEDULE_MINUTE=0
DIGEST_TIMEZONE=Asia/Kolkata
```

### Validating Sources

```bash
python validate_sources.py
```

Checks all RSS/podcast feeds and reports pass/fail with entry counts.

## Project Structure

```
pm-intelligence-digest/
├── backend/app/
│   ├── main.py             # Flask app, routes, scheduler
│   ├── config.py           # Settings from .env
│   ├── services/
│   │   ├── rss.py          # RSS + podcast fetcher
│   │   ├── summarizer.py   # Pass 1: per-item signal extraction + PM actionability ranking
│   │   ├── synthesizer.py  # Pass 2: cross-source reasoning + citations + routing + validators
│   │   ├── evaluator.py    # Pass 3: LLM-as-judge quality scoring
│   │   └── cache.py        # SQLite date-based caching
│   └── templates/
│       ├── index.html      # Editorial frontend
│       ├── evals.html      # Evals dashboard
│       └── history.html    # Archive
├── config/
│   └── sources.json        # 40 curated sources with theme-based routing
├── data/                   # SQLite digest + evals storage
├── validate_sources.py     # Source health checker
├── test_fetcher.py         # Pipeline test script
└── run.py                  # Entry point
```

## What I Learned Building This

**RSS feeds are unreliable at scale.** 30%+ of URLs I tried were dead, blocked, or redirected. Built a validation script to catch this systematically. Even with 40 configured sources, ~46% are silent on any given day — the digest runs on 20-25 active sources.

**LLM output needs token budgets.** The synthesizer was silently truncating JSON until I diagnosed it and increased `max_tokens` to 4000.

**Type mismatches cause silent eval failures.** The grounding score was 1.00 (floor) on day one because `source_index_lookup` used integer keys in the synthesizer but the evaluator looked up string keys. Every lookup failed silently, so the judge had no evidence to verify claims against. The fix was one character: `source_index_lookup[str(idx)]`.

**Caching strategy matters for cost.** Without date-based SQLite caching, every page reload would re-run ~$0.10 in API calls. The evaluator alone makes 10-15 LLM calls per digest run.

**Production deployment reveals bugs local testing misses.** Railway's ephemeral filesystem wiped the SQLite DB on every deploy until a persistent volume was mounted. The `digest_by_date` route crashed with `citation_index_map is undefined` because only the index route passed that variable to the template.

**Prompt and eval design improve together.** The synthesizer prompt and the eval scorers have been in continuous co-evolution since deployment. Topical breadth went through three rewrites — from "reward non-AI" to "penalize both extremes" to "score distance from 60/40 ideal" to "a five-theme diversity model" — each time because the eval revealed a pattern the current rule couldn't catch. Coherence and insight depth scorers were extended to explicitly check lede fidelity and implication focus after paragraph-level review identified overclaiming and multi-part implications the original rubric scored as fine. The eval reasons UI — judge reasoning displayed inline in the evals table per dimension — made all of this debuggable in production rather than opaque.

**Source routing prevents cross-section recycling.** Early versions of What's Shifting consistently recycled the strongest Company Watch insights at a higher abstraction level — the same Nvidia GTC or Apple India source appearing in both sections. The fix was architectural: partition the 40 sources into routing buckets at the synthesizer level, so Company Watch only sees first-party company feeds and What's Shifting only sees third-party market, consumer, regulation, and design sources. The breadth eval now checks theme diversity; a future version will check source independence across sections.

**Post-processing validators catch what prompts miss.** Prompt rules alone don't reliably enforce structural constraints — a model under a long context will occasionally violate a rule it was given 3,000 tokens earlier. Adding deterministic post-processing validators (multi-thread detection, date validation, cross-source coherence checks, routing violation flags) creates a second enforcement layer that runs after every synthesis pass and logs warnings before anything reaches the frontend.

**QA and prompt co-evolution requires source verification.** Automated eval scores can mask factual inversions — the grounding evaluator gave 5.0 to a paragraph that directly contradicted its source (scope limited to non-safety functions, synthesis claimed safety-critical competition). Manual source verification caught three categories the evaluator missed: claims that contradicted explicit source statements, specific numbers generated by the summarizer with no source basis, and strategic framings inferred from thin sources (4-sentence press releases). The fix was architectural: rewrite the grounding rubric to require passage-level citation traceability, rewrite the confidence definition to measure source depth rather than topic interest, and add section routing tags so the model cannot misroute third-party articles into first-party-only sections.

## Roadmap

* [ ] Email delivery option
* [ ] "Save to prep notes" — tag insights directly into interview prep
* [ ] Source health monitoring — auto-flag dead feeds
* [ ] Mobile-optimized layout
* [ ] Async pipeline — background refresh so `/refresh` returns immediately
* [ ] Timestamp localization — show IST instead of UTC
* [ ] Scraper support for Uber newsrooms (no native RSS feeds available)
* [ ] Editorial warnings surfaced in the Evals UI alongside judge scores

## Built With

| Tool | Role |
|---|---|
| Python / Flask | Web framework and request routing |
| Claude Sonnet (`claude-sonnet-4-5`) | Signal extraction (Pass 1) and cross-source synthesis (Pass 2) |
| Claude Haiku (`claude-haiku-4-5-20251001`) | LLM-as-judge quality evaluation (Pass 3) |
| feedparser | RSS and podcast feed parsing |
| APScheduler | Daily background pipeline execution at 7am IST |
| SQLite | Date-keyed digest and eval storage |
| Railway | Production deployment with persistent volume for SQLite |
| Cursor | AI-assisted development environment |