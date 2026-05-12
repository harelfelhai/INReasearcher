# Running the Web Demo

The web demo is a React (Vite + Tailwind) frontend talking to a FastAPI
backend that wraps the existing `research_agent/` Python pipeline.

```
┌─────────────────┐   HTTP / SSE   ┌─────────────────────┐
│  Vite dev :5173 │ ─────────────► │  FastAPI :8000      │
│  React + Tailwind                │  research_agent/*   │
└─────────────────┘                └─────────────────────┘
```

## One-time setup

```bash
# 1. Python deps (adds fastapi + uvicorn to the existing requirements)
pip install -r requirements.txt

# 2. Node deps for the frontend
cd web
npm install
cd ..
```

Set environment variables in `.env` at the repo root (same file the CLI
already uses):

```
ANTHROPIC_API_KEY=sk-ant-...
SERPAPI_KEY=...          # only if you use the SerpAPI search engine
```

## Run

Open two terminals at the repo root.

**Terminal 1 — backend:**
```bash
uvicorn api.main:app --reload --port 8000
```

**Terminal 2 — frontend:**
```bash
cd web
npm run dev
```

Then open <http://localhost:5173>.

## The flow

1. **Setup** — type a research question, paste entities (one per line),
   pick a search engine. Click **Build Schema**.
2. **Review** — see the generated column schema, the field-clarity audit,
   and a mock-data preview. If anything is off, click **Refine question**
   to go back. Otherwise click **Approve & Run**.
3. **Results** — entities are streamed in via Server-Sent Events. Each row
   appears as soon as its entity finishes. Download CSV when done.

## Production notes

The FastAPI app in `api/` is the long-lived piece — the React frontend is
intentionally minimal and can be replaced/redesigned later without touching
the backend. The SSE-based `/api/run` endpoint is the contract that any
future frontend (mobile, marketing site, internal tool) will hit.
