# PM Intelligence Brief

> A daily AI-powered intelligence digest built for senior product managers who want signal, not noise.

## What This Is

LinkedIn has become an AI echo chamber — FUD-driven posts, clickbait course promotions, and engagement farming. Most PM newsletters sit behind paid subscriptions or bury the insight under 800 words of preamble.

I built this tool to cut through that: stay current on industry trends, company moves, and product thinking without the noise tax.

The brief runs on demand via /refresh, pulls from 33 curated sources across 8 themes, and uses Claude to do two things most digest tools don't:

1. **Extract signal, not summaries** — each article is analyzed for what it means, not just what happened
2. **Reason across sources** — a second AI pass synthesizes patterns across all content, surfacing insights a sharp PM should have an opinion on

Every brief is automatically evaluated for quality across multiple dimensions and scored out of 100.

## What It Produces

Every day the brief generates:

- **What's Shifting** — 3-4 cross-source insights with inline citations, balanced across AI, business strategy, consumer behavior, regulation, and design
- **Interview Angle** — one specific thing to have a prepared opinion on before interviews this week
- **PM Craft Today** — the most actionable PM craft insight from the day's content, drawn from product craft sources
- **Company Watch** — signal for Google, Microsoft, Apple, Meta, Amazon, OpenAI, Anthropic, NVIDIA, and Uber
- **Startup Radar** — 2-3 disruption moves worth knowing about
- **Quality Score** — automated eval score (0-100) with full breakdown on the `/evals` page
- **Source Details** — utilized articles numbered to match inline citations, plus a separate collapsed view of filtered-out articles with drop reason
- **Archive** — browse any past brief at `/history`

## How It Works
```
RSS/Podcast Sources (33)
        ↓
    Fetcher
    (24hr filter)
        ↓
  Summarizer (Pass 1)
  Claude extracts signal per item,
  scores confidence + PM relevance
        ↓
  Two-step filter
  Drop low confidence → Drop low PM relevance
  (sequential: relevance never checked on low-confidence items)
        ↓
  Synthesizer (Pass 2)
  Claude reasons across filtered items,
  adds inline [n] citations
        ↓
  Evaluator
  LLM-as-judge scores 5 sections
  across 4 quality dimensions
        ↓
  SQLite Cache
  (digests + evals, keyed by date)
        ↓
  Flask Web App
  (on-demand via /refresh; optional
  daily scheduler via .env)
```

## Evals Framework

Every brief is automatically scored after generation using a two-layer system:

### Layer 1 — Pipeline Guardrails (deterministic, zero API cost)

Diagnostic metrics that measure pipeline health — not included in the quality score:

| Metric | What it measures |
|---|---|
| Silent % | Sources with no new articles in the lookback window / total configured sources |
| Fetched | Total articles collected across all active sources |
| Confident % | Articles summarized with high/medium confidence / total fetched |
| Relevant % | Articles with high/medium PM relevance / confident articles |
| Utilized % | Articles cited in the final synthesis / PM relevant articles |
| Weak % | Paragraphs scoring ≤ 2 on any quality dimension / total paragraphs scored |

### Layer 2 — Quality Score (LLM-as-judge, Claude Haiku, ~$0.02/day)

Five sections scored across four dimensions, weighted and summed to 100:

| Section | Weight | Dimensions scored |
|---|---|---|
| What's Shifting | 40 pts | Coherence, Insight Depth, Grounding, Topical Breadth |
| Company Watch | 25 pts | Coherence, Insight Depth, Grounding |
| Startup Radar | 20 pts | Coherence, Insight Depth, Grounding |
| PM Craft | 10 pts | Insight Depth |
| Interview Angle | 5 pts | Relevance |

**Dimension definitions:**
- **Coherence** — do all sentences support a single unified insight?
- **Insight Depth** — is this a genuine synthesis revealing something non-obvious, or just a summary?
- **Grounding** — does the cited source actually contain evidence for each claim?
- **Topical Breadth** — does What's Shifting cover both AI and non-AI themes, or is it AI-heavy?
- **Relevance** — how can a strong PM candidate use this insight to demonstrate strategic thinking in an interview?

### Running Evals
```bash
python run_evals.py              # today
python run_evals.py 2026-03-09   # specific date
python run_evals.py --all        # full archive
python run_evals.py --report     # trend table
```

### Baseline Scores (first 5 days)
```
Date          Overall  WS-Coh  WS-Ins  WS-Grd  WS-Brd  CW-Coh  CW-Ins  CW-Grd  SR-Coh  SR-Ins  SR-Grd  PC-Ins  IA-Rel
2026-03-09      75.4    4.00    3.75    3.00    2.00    4.60    3.40    4.60    4.33    4.00    4.33    4.00    4.00
2026-03-10      80.2    4.33    4.00    4.33    2.00    4.60    3.80    4.00    4.67    4.00    5.00    4.00    4.00
2026-03-11      77.3    4.00    4.00    3.33    2.00    4.50    4.00    4.00    4.33    4.00    5.00    4.00    4.00
2026-03-12      70.5    4.00    4.00    3.00    2.00    4.00    3.50    4.00    4.33    3.67    5.00    2.00    4.00
2026-03-13      79.7    4.00    3.75    3.00    4.00    4.57    3.57    4.14    4.33    4.00    5.00    4.00    4.00
```

WS-Grounding is the persistent weak signal (3.00–4.33 across all dates) — actively being addressed via tightened citation constraints in the synthesizer prompt.

## Source Design

33 sources across 8 themes, curated for balance:

| Theme | Sources |
|---|---|
| AI & Technology | Import AI, Simon Willison, Benedict Evans, AI Snake Oil, NVIDIA, a16z, Dwarkesh |
| Company Strategy | Google, Meta, Apple, Amazon, OpenAI, Microsoft Research, Verge Transportation |
| Product Craft | Shreyas Doshi, SVPG (Marty Cagan), Lenny's Podcast, Lenny's Newsletter, How I AI (Claire Vo) |
| Startup Disruption | TechCrunch, Y Combinator, Hacker News |
| Market Behavior | Platformer, MIT Technology Review, Rest of World, Hard Fork |
| Consumer Behavior | Quartz, Axios, Pivot |
| Regulation & Policy | Politico Tech, EFF |
| Design & UX | Nielsen Norman Group, UX Collective |

Sources were chosen to balance AI-optimist vs. skeptic voices, US vs. global perspective, and primary sources vs. independent analysis.

## Hallucination Mitigation

The pipeline has two layers of grounding protection:

1. **Citation enforcement** — every synthesis claim ends with an inline `[n]` citation referencing the source item. The synthesizer is explicitly instructed to only cite `[n]` if a specific insight bullet from that item directly supports the claim — not for thematic proximity.
2. **Grounding eval** — the LLM judge independently checks whether each cited source actually contains evidence for the claim made, scoring it 1–5. This makes hallucination visible and quantifiable rather than hidden.

The `source_index_lookup` maps each `[n]` back to the original article, and Source Details in the UI numbers utilized articles to match synthesis citations exactly.

## Setup

### Prerequisites
- Python 3.9+
- Anthropic API key (get one at console.anthropic.com)

### Installation
```bash
git clone https://github.com/mounicasirineni/pm-intelligence-digest.git
cd pm-intelligence-digest

pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

python run.py
```

Open `http://127.0.0.1:8000` — first load runs the full pipeline (~60 seconds).

### Configuration
```
ANTHROPIC_API_KEY=        # Required
CLAUDE_MODEL=claude-sonnet-4-5
LOOKBACK_HOURS=24         # Content lookback window in hours
DIGEST_SCHEDULE_HOUR=7    # Optional: auto-refresh hour (requires scheduler to be enabled)
DIGEST_SCHEDULE_MINUTE=0
DIGEST_TIMEZONE=Asia/Kolkata  # Timezone for scheduled refresh
```

The scheduler is optional. By default, run the pipeline manually via `http://127.0.0.1:8000/refresh`. To enable automatic daily refresh, set `ENABLE_SCHEDULER=true` in `.env` (disabled by default).

### Validating Sources
```bash
python validate_sources.py
```

## Project Structure
```
pm-intelligence-digest/
├── backend/app/
│   ├── main.py             # Flask app, routes, scheduler
│   ├── config.py           # Settings from .env
│   ├── services/
│   │   ├── rss.py          # RSS + podcast fetcher
│   │   ├── summarizer.py   # Pass 1: per-item signal extraction
│   │   ├── synthesizer.py  # Pass 2: cross-source reasoning + citation grounding
│   │   ├── cache.py        # SQLite date-based caching
│   │   └── evaluator.py    # Automated quality evals (guardrails + LLM judge)
│   └── templates/
│       ├── index.html      # Main brief + quality score bar
│       ├── history.html    # Archive browser
│       └── evals.html      # Eval scores + guardrails dashboard
├── config/
│   └── sources.json        # 33 curated sources
├── data/                   # SQLite storage (digests + evals)
├── validate_sources.py     # Source health checker
├── run_evals.py            # Evals CLI
├── test_fetcher.py         # Pipeline test script
└── run.py                  # Entry point
```

## What I Learned Building This

- RSS feeds are inconsistent — 30%+ of URLs I tried were dead, blocked, or redirected. Built a validation script to catch this systematically.
- LLM output needs token budgets — the synthesizer was silently truncating JSON until I diagnosed it and increased `max_tokens` to 4000.
- Citation grounding changes how you trust AI output — once every claim has a traceable source, hallucination becomes visible and checkable rather than hidden.
- Caching strategy matters for cost — without date-based SQLite caching, every page reload would re-run ~$0.10 in API calls.
- Evals thresholds should be derived from data, not guessed — I ran evals across 5 days of baseline data before setting any thresholds, avoiding the trap of speculative quality bars.
- Type mismatches are silent killers in LLM pipelines — a string/int mismatch in citation lookup caused the Grounding eval to score 1.0 for every paragraph until caught by inspecting the judge's reasoning.
- Pipeline filters should be sequential, not parallel — checking PM relevance on low-confidence summaries wastes tokens and produces unreliable signals. Confidence gates relevance.
- Scoring weights should reflect output complexity — Interview Angle is one derivative paragraph, so 5 pts is appropriate; What's Shifting requires cross-theme reasoning across the widest source pool, so 40 pts is appropriate.

## Roadmap

- [x] Two-pass Claude pipeline with citation grounding
- [x] 33 verified sources across 8 balanced themes
- [x] Two-step sequential filter (confidence → PM relevance)
- [x] Automated evals: 5-section weighted scoring + pipeline guardrails
- [x] LLM-as-judge across 4 quality dimensions
- [x] Source Details with citation-matched numbering + filtered-out visibility
- [x] Archive with full history browsing
- [x] Evals dashboard at `/evals`
- [ ] Deploy to Railway/Render for public URL
- [ ] Email delivery option
- [ ] Evals threshold alerts after 2-week baseline
- [ ] "Save to prep notes" — tag insights into interview prep
- [ ] Mobile-optimized layout

## Built With

- **Python / Flask** — lightweight server with minimal boilerplate; easy to extend routes as the pipeline grew
- **Claude API (Anthropic)** — `claude-sonnet-4-5` for pipeline reasoning, `claude-haiku-4-5-20251001` for evals (fast and cheap at ~$0.02/day)
- **feedparser, APScheduler, SQLite** — feedparser for RSS parsing, APScheduler for optional daily refresh, SQLite for zero-infra date-keyed caching of digests and evals
- **Cursor** — AI-assisted development environment; used for all code generation, iteration, and debugging throughout the build
