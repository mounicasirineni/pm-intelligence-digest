# PM Intelligence Brief

> A daily AI-powered intelligence digest built for senior product managers preparing for interviews at top tech companies.

## What This Is

I built this tool to solve a real problem during my job search: staying current on industry trends, company moves, and product thinking — without spending hours on LinkedIn or drowning in newsletter noise.

The brief runs every morning at 7am, pulls from 33 curated sources across 8 themes, and uses Claude to do two things most digest tools don't:

1. **Extract signal, not summaries** — each article is analyzed for what it means, not just what happened
2. **Reason across sources** — a second AI pass synthesizes patterns across all content, surfacing insights a sharp PM should have an opinion on

## What It Produces

Every day the brief generates:

- **What's Shifting** — 3-4 cross-source insights with inline citations, balanced across AI, business strategy, consumer behavior, regulation, and design
- **Interview Angle** — one specific thing to have a prepared opinion on before interviews this week
- **PM Craft Today** — the most actionable PM craft insight from the day's content
- **Company Watch** — signal for Google, Microsoft, Apple, Meta, Amazon, OpenAI, Anthropic, NVIDIA, and Uber
- **Startup Radar** — 2-3 disruption moves worth knowing about
- **Source Details** — every underlying article with confidence score and insight bullets

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
  SQLite Cache
  (keyed by date)
        ↓
  Flask Web App
  (daily auto-refresh at 7am)
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

Every claim in the synthesis is grounded in cited source items. The `source_index_lookup` maps each `[n]` citation back to the original article title and source name, making it easy to verify any claim in seconds.

Only high and medium confidence items (scored by Claude in Pass 1) are fed into the synthesis pass.

## Setup

### Prerequisites
- Python 3.9+
- Anthropic API key (get one at console.anthropic.com)

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

# Configure sources
# config/sources.json is already populated with 33 verified sources
# If you have a private Lenny's RSS URL, add it as LENNYS_PODCAST_RSS in .env

# Run
python run.py
```

Open `http://127.0.0.1:8000` — first load runs the full pipeline (~60 seconds).

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
│   ├── main.py           # Flask app, routes, scheduler
│   ├── config.py         # Settings from .env
│   ├── services/
│   │   ├── rss.py        # RSS + podcast fetcher
│   │   ├── summarizer.py # Pass 1: per-item signal extraction
│   │   ├── synthesizer.py# Pass 2: cross-source reasoning
│   │   └── cache.py      # SQLite date-based caching
│   └── templates/
│       └── index.html    # Editorial frontend
├── config/
│   └── sources.json      # 33 curated sources
├── data/                 # SQLite digest storage
├── validate_sources.py   # Source health checker
├── test_fetcher.py       # Pipeline test script
└── run.py               # Entry point
```

## What I Learned Building This

- RSS feeds are inconsistent — 30%+ of URLs I tried were dead, blocked, or redirected. Built a validation script to catch this systematically.
- LLM output needs token budgets — the synthesizer was silently truncating JSON until I diagnosed it and increased `max_tokens` to 4000.
- Citation grounding changes how you trust AI output — once every claim has a traceable source, hallucination becomes visible and checkable rather than hidden.
- Caching strategy matters for cost — without date-based SQLite caching, every page reload would re-run ~$0.10 in API calls.

## Roadmap

- [ ] Evals framework — theme balance scoring, citation validity checks, LLM-as-judge insight quality
- [ ] Email delivery option
- [ ] "Save to prep notes" feature — tag insights directly into interview prep
- [ ] Source health monitoring — auto-flag dead feeds
- [ ] Mobile-optimized layout

## Built With

- Python / Flask
- Claude API (Anthropic) — `claude-sonnet-4-5`
- feedparser
- APScheduler
- SQLite
```

*Save the file."*

---

Once saved, commit and push:
```
git add README.md
git commit -m "docs: add comprehensive README"
git push
