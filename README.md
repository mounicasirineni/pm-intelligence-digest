# PM Intelligence Brief

> A daily AI-powered intelligence digest built for senior product managers preparing for interviews at top tech companies.

## What This Is

I built this tool to solve a real problem during my job search: staying current on industry trends, company moves, and product thinking — without spending hours on LinkedIn or drowning in newsletter noise.

The brief runs on demand via /refresh, pulls from 33 curated sources across 8 themes, and uses Claude to do two things most digest tools don't:

1. **Extract signal, not summaries** — each article is analyzed for what it means, not just what happened
2. **Reason across sources** — a second AI pass synthesizes patterns across all content, surfacing insights a sharp PM should have an opinion on

Every brief is automatically evaluated for quality across four dimensions and scored out of 100.

## What It Produces

Every day the brief generates:

- **What's Shifting** — 3-4 cross-source insights with inline citations, balanced across AI, business strategy, consumer behavior, regulation, and design
- **Interview Angle** — one specific thing to have a prepared opinion on before interviews this week
- **PM Craft Today** — the most actionable PM craft insight from the day's content
- **Company Watch** — signal for Google, Microsoft, Apple, Meta, Amazon, OpenAI, Anthropic, NVIDIA, and Uber
- **Startup Radar** — 2-3 disruption moves worth knowing about
- **Quality Score** — automated eval score (0-100) with four metric dimensions visible in every brief
- **Source Details** — every underlying article with confidence score and insight bullets
- **Archive** — browse any past brief at `/history`

## How It Works
```
RSS/Podcast Sources (33)
        ↓
    Fetcher
    (24hr filter)
        ↓
  Summarizer (Pass 1)
  Claude extracts signal
  per item, scores confidence
        ↓
  Synthesizer (Pass 2)
  Claude reasons across all
  items, adds citations
        ↓
  Evaluator
  Scores coherence, insight depth,
  grounding, and theme balance
        ↓
  SQLite Cache
  (digests + evals, keyed by date)
        ↓
  Flask Web App
  (on-demand via /refresh; optional
  daily scheduler via .env)
```

## Evals Framework

Every brief is automatically scored after generation using a three-layer evaluation system:

### Layer 1 — Structural (deterministic, zero API cost)
- **Theme balance** — what % of synthesis citations come from non-AI sources
- **Citation coverage** — what % of synthesis sentences have inline citations (diagnostic)
- **Source utilization** — how many fetched sources contributed to synthesis

### Layer 2 — LLM-as-judge (Claude Haiku, ~$0.02/day)
For each *What's Shifting* paragraph, Claude scores three dimensions:

| Dimension | What it measures |
|---|---|
| Coherence | Do all sentences support a single unified insight? |
| Insight Depth | Is this genuine cross-source synthesis or restatement? |
| Grounding | Are claims actually supported by the cited sources? |

### Overall Score Formula
```
Overall = (Coherence/5 × 30) + (Insight Depth/5 × 30) + (Grounding/5 × 30) + (Theme Balance% × 10)
```

### Running Evals
```bash
python run_evals.py              # today
python run_evals.py 2026-03-09   # specific date
python run_evals.py --all        # full archive
python run_evals.py --report     # trend table
```

### Baseline Scores (first 4 days)
```
Date          Overall  Coherence  Insight  Grounding  ThemeBal%
2026-03-09      73.9       4.25     3.75       3.00       78.6
2026-03-10      82.3       4.33     4.00       4.00       83.3
2026-03-11      84.8       4.25     4.00       4.50       83.3
2026-03-12      81.0       4.33     4.00       4.00       70.0
```

## Source Design

33 sources across 8 themes, curated for balance:

| Theme | Sources |
|---|---|
| AI & Technology | Import AI, Simon Willison, Benedict Evans, AI Snake Oil, NVIDIA, a16z, Dwarkesh |
| Company Strategy | Google, Meta, Apple, Amazon, OpenAI, Microsoft Research, Verge Transportation |
| Product Craft | Shreyas Doshi, Gibson Biddle, Lenny's Podcast, Stratechery, Acquired |
| Startup Disruption | TechCrunch, Y Combinator, Hacker News |
| Market Behavior | Platformer, MIT Technology Review, Rest of World, Hard Fork |
| Consumer Behavior | Quartz, Axios, Pivot |
| Regulation & Policy | Politico Tech, EFF |
| Design & UX | Nielsen Norman Group, UX Collective |

Sources were chosen to balance AI-optimist vs. skeptic voices, US vs. global perspective, and primary sources vs. independent analysis.

## Hallucination Mitigation

Every synthesis claim is grounded in cited source items. The `source_index_lookup` maps each `[n]` citation back to the original article, and the Grounding eval dimension automatically checks whether each claim is actually supported by its cited source — flagging paragraphs where the model has gone beyond the evidence.

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
LOOKBACK_HOURS=24
DIGEST_SCHEDULE_HOUR=7
DIGEST_SCHEDULE_MINUTE=0
DIGEST_TIMEZONE=Asia/Kolkata
```

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
│   │   ├── synthesizer.py  # Pass 2: cross-source reasoning
│   │   ├── cache.py        # SQLite date-based caching
│   │   └── evaluator.py    # Automated quality evals
│   └── templates/
│       ├── index.html      # Main brief + quality score bar
│       └── history.html    # Archive browser
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
- Evals thresholds should be derived from data, not guessed — I ran evals across 4 days of baseline data before setting any thresholds, avoiding the trap of speculative quality bars.
- Type mismatches are silent killers in LLM pipelines — a string/int mismatch in citation lookup caused the Grounding eval to score 1.0 for every paragraph until caught by inspecting the judge's reasoning.

## Roadmap

- [x] Two-pass Claude pipeline with citation grounding
- [x] 33 verified sources across 8 balanced themes
- [x] Automated evals framework with LLM-as-judge
- [x] Archive with full history browsing
- [ ] Deploy to Railway/Render for public URL
- [ ] Email delivery option
- [ ] Evals threshold alerts after 2-week baseline
- [ ] "Save to prep notes" — tag insights into interview prep
- [ ] Mobile-optimized layout

## Built With

- Python / Flask
- Claude API (Anthropic) — `claude-sonnet-4-5` for pipeline, `claude-haiku-4-5-20251001` for evals
- feedparser, APScheduler, SQLite
