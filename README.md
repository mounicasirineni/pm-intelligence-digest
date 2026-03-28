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
* **PM Craft Today** — the most actionable PM craft insight from the day's content, grounded in a specific source; shows "Not available today." when no dedicated craft sources are in the pool
* **Company Watch** — strategic signal for Google, Microsoft, Apple, Meta, Amazon, Netflix, NVIDIA, OpenAI, and Anthropic — what's strategically shifting, not just what they announced, sourced exclusively from first-party official feeds; shows "Not available today." when no first-party signals are available
* **Startup Radar** — 2-3 disruption moves worth knowing about from early-stage or emerging companies only, each with a "so what" — the competitive threat or market pattern it reveals; established large-cap companies are filtered out at the pipeline level regardless of feed tag
* **Source Details** — every underlying article with confidence score, relevance score, and insight bullets. Utilized shows only articles actually cited in the synthesis; articles that passed filtering but were not selected by the synthesizer appear in Filtered Out as "Not Selected"

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
  company maturity assessed per item
  (startup / established / not_applicable)
        ↓
  Synthesizer (Pass 2)
  Sources partitioned by routing eligibility
  established companies dropped from startup_radar pool
  Claude reasons across all items
  adds inline citations [n]
  builds source_index_lookup
        ↓
  Evaluator (Pass 3)
  LLM-as-judge scores quality
  5 dimensions, 100pt weighted scale
  dynamic weight normalization
  (empty sections excluded from scoring)
        ↓
  Post-processing Validators
  multi-thread violations
  date warnings
  coherence warnings
  routing violations
  CW source integrity checks
  source concentration warnings
  split implication warnings
  theme audit warnings
        ↓
  SQLite Cache
  (date-keyed, Railway Volume)
        ↓
  Flask Web App
  (daily auto-refresh at 7am IST)
```

## Evals Framework

Every digest run is automatically evaluated by an LLM judge across 5 quality dimensions, producing a weighted score out of 100. Scoring is dynamic — sections with no output are excluded from both score components and score weights, and the overall score is normalized to 100 based only on the weights of sections that produced output. This handles all four scenarios: all 5 sections scored; PM Craft empty; Company Watch empty; both empty.

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
| **Coherence** | Do all sentences support a single unified insight, and is that unity emergent from the sources or constructed by the synthesizer? Paragraphs that ignore complicating source evidence to preserve narrative coherence score lower even if internally consistent. |
| **Insight Depth** | Is this a genuine synthesis revealing something non-obvious, does the closing implication commit to one sharp claim, and is that implication traceable to a source bullet rather than synthesizer reasoning? Closing sentences that add operational specificity not present in any source bullet (checklists, thresholds, implementation steps) are scored as inference boundary violations. |
| **Grounding** | Two-component check: (1) forward traceability — can every specific claim be traced to a source passage? (2) backward completeness — did the synthesis fairly represent the full source evidence, including contradictions and complicating bullets? Selective omission that distorts the conclusion is scored as a grounding failure even if every included claim is correctly sourced. The evaluator refers to sources by name only in justification text to prevent conflating its internal evidence reference numbers with the synthesis citation indices. |
| **Topical Breadth (What's Shifting)** | Does this section distribute central claims across the five eligible themes (AI & technology, market behavior, consumer behavior, regulation & policy, design & UX)? No single theme should anchor more than one paragraph. Scored against available source material — if a theme had 2+ eligible items and does not appear in any paragraph, that is a breadth failure. |
| **Relevance (Interview Angle)** | Can a PM use this insight to demonstrate strategic thinking in an interview? Domain specificity is not penalized if the underlying principle is transferable across PM roles. |

**Pipeline guardrails** (diagnostic, not scored):

| Metric | What it measures |
|---|---|
| Silent % | Sources with no new articles in the lookback window |
| Confident % | Articles summarized with high/medium confidence |
| Relevant % | Articles with high/medium PM relevance |
| Utilized % | Relevant articles actually cited in the synthesis (not just filtered pool size) |
| Weak % | Paragraphs scoring ≤2 on any quality dimension |
| Sections scored | Which sections contributed to the overall score that day — makes day-over-day score comparisons interpretable when section availability varies |

**Post-processing editorial warnings** (logged per run, not scored):

| Warning | What it catches |
|---|---|
| Multi-thread violations | Company Watch entries citing more than 2 sources or with high conjunction counts suggesting multiple disconnected claims |
| Date warnings | Milestone or timeline claims where the cited date is earlier than today — catches past dates stated as future events |
| Coherence warnings | Same source cited in multiple company entries (risk of contradictory framings) or shared between What's Shifting and Company Watch (routing violation) |
| Routing violations | What's Shifting paragraphs citing dedicated-section sources, or Company Watch citing What's Shifting sources |
| CW source integrity violations | Company Watch entries citing non-company_strategy sources (e.g. a third-party article about a major company routed incorrectly); entry is automatically cleared and logged |
| PM Craft source violations | PM Craft entries citing non-product_craft or non-design_ux sources; logged as a violation since PM Craft draws exclusively from dedicated craft sources |
| Source concentration warnings | Any single source contributing 3+ items to the filtered pool in a single run — flags potential source diversity risk without dropping items |
| Split implication warnings | Closing sentences containing conjunctions ("and", "as well as") that connect two distinct actionable consequences — flags potential implication focus violations |
| Theme audit warnings | What's Shifting paragraphs where a single theme (e.g. AI & technology) anchors more than one paragraph — triggers REWRITE_DUPLICATE_RECOMMENDED |

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

**Source routing:** Company Watch is sourced exclusively from first-party official feeds (company blogs and newsrooms). A post-processing validator enforces this at runtime — any Company Watch entry whose cited source is not tagged `company_strategy` is automatically cleared before the digest is stored or displayed. What's Shifting, Startup Radar, and PM Craft draw from third-party sources only — ensuring Company Watch and What's Shifting never recycle the same source across sections.

**Company maturity filtering:** Startup Radar is restricted to early-stage and emerging companies. The summarizer assesses company maturity per article (startup / established / not_applicable), and the synthesizer drops any `startup_disruption`-tagged item where the primary subject is an established large-cap or publicly traded company — regardless of feed tag. This prevents well-known companies from appearing in Startup Radar simply because they were covered by a startup-disruption feed.

Sources were chosen to balance AI-optimist vs. skeptic voices, US vs. global perspective, and primary sources vs. independent analysis.

## Hallucination Mitigation

Every claim in the synthesis is grounded in cited source items. The `source_index_lookup` maps each `[n]` citation back to the original article title and source name, making it easy to verify any claim in seconds.

The evaluator's **Grounding** dimension applies a two-component check. Forward traceability requires that every specific claim traces to a specific source passage or data point — plausibility and topic consistency are explicitly excluded as proxies for evidence, and specific numbers not appearing verbatim in the source are treated as unsourced. Backward completeness requires that the synthesis fairly represents the full source evidence: omitting a named expert contradiction, suppressing a bullet that challenges the paragraph's central claim, or building a conclusion from 1-2 bullets while ignoring stronger ones all constitute grounding failures regardless of how well-cited the included claims are.

Only high and medium confidence items (scored by Claude in Pass 1) are fed into the synthesis pass. Confidence is defined as source depth — how much of a bullet could be written from the content body itself — not topic interest. A thin source on an interesting topic scores low confidence.

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

The repo pins **Python 3.12.8** via `nixpacks.toml`, `mise.toml`, and `.python-version` so Railway's build does not pull **Python 3.13.x** builds that can fail under `mise` with *"Python installation is missing a `lib` directory"* (often tied to experimental freethreaded installers). If you deploy elsewhere, use Python **3.12.x** or **3.11.x** for the same reason.

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
│   │   ├── summarizer.py   # Pass 1: per-item signal extraction + PM actionability ranking + company maturity scoring
│   │   ├── synthesizer.py  # Pass 2: cross-source reasoning + citations + routing + company maturity filter + validators
│   │   ├── evaluator.py    # Pass 3: LLM-as-judge quality scoring + dynamic weight normalization
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

**Synthesizers optimize for narrative coherence over source fidelity.** Manual QA against raw source bullets revealed a consistent failure mode: the synthesizer selects 1-2 bullets that support its central thesis and drops the rest — including named expert contradictions, complicating evidence, and often stronger PM insights than the ones it leads with. Prompt rules alone don't fully fix this. The synthesizer needs an explicit omission check (review all bullets per source before finalizing, assess whether dropping any distorts the conclusion) and a contradiction mandate (named challenges to the central claim must appear). The evaluator needs a matching backward completeness check — otherwise it rewards well-constructed arguments built on selectively chosen evidence with perfect grounding scores.

**The highest-risk omissions are thesis-extending bullets, not thesis-supporting ones.** Manual QA identified a consistent pattern: the synthesizer drops bullets that contradict the central claim, extend it to a new domain, or name a product not yet mentioned — precisely the bullets that would make the paragraph more complete or force a thesis revision. Prompt rules targeting "review all bullets" don't catch this because the synthesizer can comply with the letter of the rule while still dropping the most challenging evidence. The fix requires naming the omission type explicitly: bullets that complicate your thesis are more valuable than bullets that support it.

**Thematic combination requires a traceable mechanism, not a shared category label.** The synthesizer consistently grouped mechanistically unrelated sources under broad category labels ("AI", "regulation", "platforms") and presented the combination as a genuine synthesis. The distinction that matters: a shared mechanism must be explicitly present in at least one source bullet — if it only emerges when you abstract across all bullets, it is a category label, not a mechanism. Two stories that share only a category label belong in separate paragraphs anchored to distinct themes, not forced into one paragraph under a synthesizer-constructed framing.

**Source routing prevents cross-section recycling but needs two enforcement layers.** Early versions of What's Shifting consistently recycled the strongest Company Watch insights at a higher abstraction level. The fix was architectural: partition the 40 sources into routing buckets at the synthesizer level. But prompt-level routing rules can still be violated when a third-party article about a major company reaches the synthesizer in the same call as first-party sources. A deterministic post-processing validator — checking that every Company Watch citation index maps to a `company_strategy`-themed source and clearing violations before storage — provides the second enforcement layer that prompt rules alone cannot guarantee.

**Post-processing validators catch what prompts miss.** Prompt rules alone don't reliably enforce structural constraints — a model under a long context will occasionally violate a rule it was given 3,000 tokens earlier. Adding deterministic post-processing validators (multi-thread detection, date validation, cross-source coherence checks, routing violation flags, CW source integrity checks, split implication detection, theme audit warnings) creates a second enforcement layer that runs after every synthesis pass and logs warnings before anything reaches the frontend.

**QA and prompt co-evolution requires source verification.** Automated eval scores can mask systematic omission bias — the grounding evaluator gave 5.0 to paragraphs that suppressed named contradictions and dropped stronger insights, because it only checked forward traceability. Manual source verification caught three failure categories the automated eval missed: contradiction suppression (named expert challenges dropped to preserve narrative), selective bullet use (1-2 bullets padded into a paragraph while 3-4 stronger bullets were ignored), and thematic forced combination (mechanistically unrelated stories grouped under a broad category label). The fix required both a synthesizer-side omission check and an evaluator-side backward completeness rubric — neither alone is sufficient.

**Roundup articles require per-story extraction, not lead-story focus.** Multi-story roundup articles (single URLs containing 4-5 unrelated stories) were producing systematic backward completeness failures — the summarizer's "focus on lead story" instruction caused 75%+ of each roundup's signal to be discarded before it reached the synthesizer. The fix was to extract insights from each distinct story separately in the summarizer, giving the synthesizer the full evidence pool to work with. This resolved the downstream pattern of the synthesizer combining mechanistically unrelated stories under forced category labels to compensate for thin evidence per bullet.

**Interview Angle must be anchored to What's Shifting content, not the broader source pool.** Manual QA identified a routing failure where the Interview Angle was sourced from a regulation_policy article that never appeared in What's Shifting — producing an angle with no connective tissue to the rest of the brief. The fix was a prompt-level restriction: the interview angle must derive from a source already cited in one of the whats_shifting paragraphs, not from any WS-eligible source. This ensures the angle feels like a natural extension of what the reader just consumed rather than an independent fifth story.

**Feed tags alone are insufficient for section routing.** A YourStory article about Intuit (a large public company) was tagged `startup_disruption` and reached Startup Radar because the synthesizer's routing rules only checked the feed tag, not the maturity of the company being covered. The fix required a two-layer approach: the summarizer now assesses company maturity per article, and the synthesizer filters out `startup_disruption` items where the primary subject is an established company before they reach the prompt — making the constraint deterministic rather than prompt-dependent.

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