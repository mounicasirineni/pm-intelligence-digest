# PM Intelligence Digest

A daily web app that:

- Pulls items from **RSS feeds** and **podcast transcript sources**
- Summarizes each item with the **Claude API**
- Runs a second **cross-source synthesis** pass to identify trends + insights
- Serves a simple **HTML/CSS** digest UI

## Quickstart

### 1) Create a virtualenv and install deps

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Configure environment + sources

- Copy `.env.example` to `.env` and fill `ANTHROPIC_API_KEY`
- Copy `config/sources.example.json` to `config/sources.json`

### 3) Run the web app

```bash
uvicorn backend.app.main:app --reload
```

Then open `http://127.0.0.1:8000`.

## How it’s structured

- `backend/app/main.py`: FastAPI app + routes
- `backend/app/workflows/digest.py`: Orchestration (fetch → summarize → synthesize → store)
- `backend/app/services/claude.py`: Claude API wrapper
- `backend/app/services/rss.py`: RSS fetching/parsing
- `backend/app/services/transcripts.py`: Transcript fetching (placeholder)
- `backend/app/services/storage.py`: SQLite persistence
- `backend/app/templates/`: HTML templates
- `backend/app/static/`: CSS
- `config/sources.json`: Your source list
- `data/`: Local SQLite DB (created on first run)

## Notes

- This is a **scaffold**: transcript ingestion varies a lot by provider. The `transcripts.py` layer is intentionally minimal and designed for you to swap in a real integration.
- The digest workflow supports “manual run” via the UI for now; you can later add a scheduled runner (Windows Task Scheduler / cron).

