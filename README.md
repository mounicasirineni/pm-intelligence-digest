# PM Intelligence Brief

> A daily AI-powered intelligence digest that surfaces what's actually shifting in the industry — not just what happened, but what it means.

**[→ Live Demo](https://pm-intelligence-digest-production.up.railway.app)** · **[→ Evals Dashboard](https://pm-intelligence-digest-production.up.railway.app/evals)** · **[→ Archive](https://pm-intelligence-digest-production.up.railway.app/history)**

---

## What This Is

Most news digests summarize. This one synthesizes.

The brief runs every morning at 7am, pulls from 39 curated sources across 8 themes, and uses Claude to do two things most digest tools don't:

1. **Extract signal, not summaries** — each article is analyzed for what it means, not just what happened. Every insight bullet must pass a non-obvious test: would a reader get this from the headline alone? If yes, it's not an insight.
2. **Reason across sources** — a second AI pass connects signals across multiple sources to surface patterns that don't appear in any single article.

Built for product managers who want a prepared opinion on what's shifting — across AI, business strategy, consumer behavior, regulation, and design.

## What It Produces

Every day the brief generates:

* **What's Shifting** — 4-5 cross-source insights with inline citations, balanced 60/40 across non-AI topics (business strategy, consumer behavior, regulation, design) and AI/tech developments
* **Interview Angle** — one specific thing to have a prepared opinion on, rotated across product strategy, consumer insight, regulatory navigation, and AI
* **PM Craft Today** — the most actionable PM craft insight from the day's content, grounded in a specific source
* **Company Watch** — strategic signal for Google, Microsoft, Apple, Meta, Amazon, OpenAI, Anthropic, NVIDIA, and Uber — what's strategically shifting, not just what they announced
* **Startup Radar** — 2-3 disruption moves worth knowing about, each with a "so what" — the competitive threat or market pattern it reveals
* **Source Details** — every underlying article with confidence score, relevance score, and insight bullets

## How It Works

```
RSS/Podcast Sources (39)
        ↓
    Fetcher
    (24hr lookback window)
        ↓
  Summarizer (Pass 1)
  Claude extracts signal per item
  scores confidence + PM relevance
  non-obvious insight test applied
        ↓
  Synthesizer (Pass 2)
  Claude reasons across all items
  adds inline citations [n]
  builds source_index_lookup
        ↓
  Evaluator (Pass 3)
  LLM-as-judge scores quality
  5 dimensions, 100pt weighted scale
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
| **Coherence** | Do all sentences support a single unified insight? |
| **Insight Depth** | Is this a genuine synthesis revealing something non-obvious, or just a summary? |
| **Grounding** | Does the cited source actually contain evidence for each claim? |
| **Topical Breadth** | Does What's Shifting maintain the 60/40 non-AI/AI balance? Penalizes both AI-dominated and non-AI-dominated digests. |
| **PM Relevance** | Can a PM use this insight to demonstrate strategic thinking in an interview? |

**Pipeline guardrails** (diagnostic, not scored):

| Metric | What it measures |
|---|---|
| Silent % | Sources with no new articles in the lookback window |
| Confident % | Articles summarized with high/medium confidence |
| Relevant % | Articles with high/medium PM relevance |
| Utilized % | Relevant articles actually cited in the synthesis |
| Weak % | Paragraphs scoring ≤2 on any quality dimension |

Every score cell on the [Evals page](https://pm-intelligence-digest-production.up.railway.app/evals) has a hover tooltip showing the judge's reasoning for that dimension.

**Score progression since deployment:**

| Date | Overall | Grounding | Weak % |
|---|---|---|---|
| March 13 | 52.4 | 1.00 | 100% |
| March 14 | 75.6 | 4.60 | 14.3% |
| March 15 | 80.3 | 4.00 | 0% |

## Source Design

39 sources across 8 themes, curated for balance:

| Theme | Sources |
|---|---|
| AI & Technology | Import AI, Simon Willison, Benedict Evans, AI Snake Oil, NVIDIA, a16z, Dwarkesh |
| Company Strategy | Google, Meta, Apple, Amazon, OpenAI, Microsoft Research, Verge Transportation |
| Product Craft | Shreyas Doshi, Gibson Biddle, Lenny's Podcast, Stratechery, Acquired |
| Startup Disruption | TechCrunch, Y Combinator, Hacker News |
| Market Behavior | Platformer, MIT Technology Review, Rest of World, Hard Fork |
| Consumer Behavior | Quartz, Axios, Pivot |
| Regulation & Policy | Politico Tech, EFF, Medianama |
| Design & UX | Nielsen Norman Group, UX Collective |

Sources were chosen to balance AI-optimist vs. skeptic voices, US vs. global perspective, and primary sources vs. independent analysis.

## Hallucination Mitigation

Every claim in the synthesis is grounded in cited source items. The `source_index_lookup` maps each `[n]` citation back to the original article title and source name, making it easy to verify any claim in seconds.

The evaluator's **Grounding** dimension specifically checks whether cited sources actually contain evidence for each claim — not just whether citations are present. This caught a critical bug on day one: integer vs. string key mismatch in `source_index_lookup` caused the grounding judge to always return 1.00 (floor score) because lookups always failed silently.

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

### Configuration

All settings are in `.env`:

```
ANTHROPIC_API_KEY=        # Required
CLAUDE_MODEL=claude-sonnet-4-6
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
│   │   ├── summarizer.py   # Pass 1: per-item signal extraction
│   │   ├── synthesizer.py  # Pass 2: cross-source reasoning + citations
│   │   ├── evaluator.py    # Pass 3: LLM-as-judge quality scoring
│   │   └── cache.py        # SQLite date-based caching
│   └── templates/
│       ├── index.html      # Editorial frontend
│       ├── evals.html      # Evals dashboard
│       └── history.html    # Archive
├── config/
│   └── sources.json        # 39 curated sources
├── data/                   # SQLite digest + evals storage
├── validate_sources.py     # Source health checker
├── test_fetcher.py         # Pipeline test script
└── run.py                  # Entry point
```

## What I Learned Building This

**RSS feeds are unreliable at scale.** 30%+ of URLs I tried were dead, blocked, or redirected. Built a validation script to catch this systematically. Even with 39 configured sources, ~46% are silent on any given day — the digest runs on 20-25 active sources.

**LLM output needs token budgets.** The synthesizer was silently truncating JSON until I diagnosed it and increased `max_tokens` to 4000.

**Type mismatches cause silent eval failures.** The grounding score was 1.00 (floor) on day one because `source_index_lookup` used integer keys in the synthesizer but the evaluator looked up string keys. Every lookup failed silently, so the judge had no evidence to verify claims against. The fix was one character: `source_index_lookup[str(idx)]`.

**Citation grounding changes how you trust AI output.** Once every claim has a traceable source, hallucination becomes visible and checkable rather than hidden. The grounding eval dimension enforces this programmatically.

**Caching strategy matters for cost.** Without date-based SQLite caching, every page reload would re-run ~$0.10 in API calls. The evaluator alone makes 10-15 LLM calls per digest run.

**Prompt calibration requires an eval loop.** The topical breadth prompt went through 3 iterations — from "reward non-AI" to "penalize both extremes" to "score distance from 60/40 ideal" — before it matched what the synthesizer was actually producing. Without the eval framework, these mismatches would be invisible.

**Production deployment reveals bugs local testing misses.** Railway's ephemeral filesystem wiped the SQLite DB on every deploy until a persistent volume was mounted. The `digest_by_date` route crashed with `citation_index_map is undefined` because only the index route passed that variable to the template.

## Roadmap

* [ ] Email delivery option
* [ ] "Save to prep notes" — tag insights directly into interview prep
* [ ] Source health monitoring — auto-flag dead feeds
* [ ] Mobile-optimized layout
* [ ] Async pipeline — background refresh so `/refresh` returns immediately
* [ ] Timestamp localization — show IST instead of UTC

## Built With

| Tool | Role |
|---|---|
| Python / Flask | Web framework and request routing |
| Claude API (Anthropic) — `claude-sonnet-4-6` | Signal extraction, cross-source synthesis, and LLM-as-judge evaluation |
| feedparser | RSS and podcast feed parsing |
| APScheduler | Daily background pipeline execution at 7am IST |
| SQLite | Date-keyed digest and eval storage |
| Railway | Production deployment with persistent volume for SQLite |
| Cursor | AI-assisted development environment |