"""Integration tests for /api/run and /api/discover-entities via TestClient.

The /api/run endpoint owns:
  - auth gating
  - ResearchSession row creation + completion
  - SSE event stream (session_started, entity_start, entity_done, done)
  - Anthropic token tracking → cost_used on the session
  - Excel export written to disk + ExcelExport row recorded
  - Optional seeded_probe → injected as Lane 1 probe hits

All of that is exercised here with zero LLM cost: MockClaude is patched in,
and a StubSearch instance replaces the search client.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


# ── Test DB + isolation setup ────────────────────────────────────────────────
_TEST_DB = Path(__file__).parent / "_test_api_run.db"
if _TEST_DB.exists():
    _TEST_DB.unlink()
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB}"
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")
_TEST_EXPORTS = Path(__file__).parent / "_test_exports"
_TEST_EXPORTS.mkdir(parents=True, exist_ok=True)
os.environ["EXPORTS_DIR"] = str(_TEST_EXPORTS)

from fastapi.testclient import TestClient  # noqa: E402

from auth.db import Base, SessionLocal, engine  # noqa: E402
from auth import crud  # noqa: E402
from api import main as api_main  # noqa: E402
from api.main import app  # noqa: E402
from research_agent.models import ColumnPlan, ResearchPlan  # noqa: E402

from tests.mocks import MockClaude, StubSearch  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        crud.ensure_default_admin(db)
    finally:
        db.close()
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client():
    return TestClient(app)


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _login(client, username="admin", password="admin") -> str:
    r = client.post("/api/auth/login", json={"username": username, "password": password})
    return r.json()["access_token"]


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    """Parse an SSE response body into a list of (event_name, payload) pairs."""
    events: list[tuple[str, dict]] = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        event = "message"
        data_parts = []
        for line in block.split("\n"):
            if line.startswith("event: "):
                event = line[len("event: "):].strip()
            elif line.startswith("data: "):
                data_parts.append(line[len("data: "):])
        if data_parts:
            try:
                payload = json.loads("".join(data_parts))
            except json.JSONDecodeError:
                payload = {"_raw": "".join(data_parts)}
            events.append((event, payload))
    return events


def _patch_clients(monkeypatch, claude: MockClaude, search: StubSearch):
    monkeypatch.setattr(api_main, "_claude", lambda: claude)
    monkeypatch.setattr(api_main, "_search_client", lambda _engine: search)


# ── /api/run: full happy-path stream ─────────────────────────────────────────

def test_api_run_streams_events_records_session_and_writes_excel(client, monkeypatch):
    page = (
        "Tel Aviv official site: the municipality operates www.tel-aviv.gov.il "
        "for citizens. Long enough page to clear minimum content length. "
        "Padding paragraph for the extractor min-length gate. " * 4
    )
    search = StubSearch().add(None, [
        {"url": "https://wiki.example/tlv", "raw_content": page, "content": page},
    ])

    # Single field → single-field extractor path → uses extract_field tool
    claude = MockClaude().on("extract_field", {
        "value": "www.tel-aviv.gov.il",
        "quote_original": "www.tel-aviv.gov.il",
        "confidence": 0.95,
    })
    _patch_clients(monkeypatch, claude, search)

    tok = _login(client)
    plan = ResearchPlan(
        entity_type="city",
        research_question_original="official website per city",
        columns=[ColumnPlan(
            id="website", label_he="אתר", label_en="Website", type="url",
            search_queries_he=["{entity} אתר רשמי"],
            search_queries_en=["{entity} official website"],
        )],
    )
    r = client.post(
        "/api/run",
        headers=_auth(tok),
        json={"plan": plan.model_dump(), "entities": ["Tel Aviv"],
              "search_engine": "duckduckgo"},
    )
    assert r.status_code == 200
    events = _parse_sse(r.text)
    event_names = [e for e, _ in events]
    assert "session_started" in event_names
    assert "entity_start" in event_names
    assert "entity_done" in event_names
    assert "done" in event_names
    assert "error" not in event_names

    # entity_done payload carries our cell value
    entity_done = next(p for e, p in events if e == "entity_done")
    assert entity_done["entity_name"] == "Tel Aviv"
    assert entity_done["cells"]["website"]["value"] == "www.tel-aviv.gov.il"

    # done payload tells us about the saved session + export
    done = next(p for e, p in events if e == "done")
    assert done["n"] == 1
    assert done["session_id"]
    assert done["export"]["export_id"]
    assert done["cost_used"] > 0
    assert done["tokens_in"] > 0
    assert done["tokens_out"] > 0

    # Session was persisted with status=completed and cost_used > 0
    sessions = client.get("/api/user/sessions", headers=_auth(tok)).json()
    assert len(sessions) == 1
    sess = sessions[0]
    assert sess["status"] == "completed"
    assert sess["cost_used"] > 0
    assert len(sess["exports"]) == 1

    # The exported .xlsx exists on disk
    from auth import crud as _crud
    db = SessionLocal()
    export = _crud.get_export(db, sess["exports"][0]["id"])
    db.close()
    assert export is not None
    assert Path(export.file_path).exists()
    assert Path(export.file_path).stat().st_size > 0


# ── /api/run: seeded probe lights up Lane 1 ─────────────────────────────────

def test_api_run_seeded_probe_emits_probe_hit_event(client, monkeypatch):
    """When seeded_probe is sent, the orchestrator should publish a probe_hit
    event with source='discovery' before any entity_start."""
    # Verification search after the probe seed needs SOME page; we don't
    # actually care whether it succeeds — just that the seeded path is
    # exercised and probe_hit fires.
    page = "Padding content padding. " * 30
    search = StubSearch().add(None, [
        {"url": "https://v.example/verify", "raw_content": page, "content": page},
    ])
    claude = MockClaude().on("extract_field", {
        "value": "1909", "quote_original": "1909", "confidence": 0.9,
    })
    _patch_clients(monkeypatch, claude, search)

    tok = _login(client)
    plan = ResearchPlan(
        entity_type="city", research_question_original="founded year",
        columns=[ColumnPlan(
            id="founded", label_he="ייסוד", label_en="Founded", type="number",
            search_queries_he=["{entity} שנת ייסוד"],
            search_queries_en=["{entity} founded"],
        )],
    )
    seeded = {
        "founded": {
            "Tel Aviv": {
                "value": "1909",
                "quote": "founded in 1909",
                "source_url": "https://wiki.example/cities",
            }
        }
    }
    r = client.post(
        "/api/run", headers=_auth(tok),
        json={"plan": plan.model_dump(), "entities": ["Tel Aviv"],
              "search_engine": "duckduckgo", "seeded_probe": seeded},
    )
    assert r.status_code == 200
    events = _parse_sse(r.text)
    probe_hits = [p for e, p in events if e == "probe_hit"]
    assert len(probe_hits) == 1
    assert probe_hits[0]["source"] == "discovery"
    assert probe_hits[0]["entities_found"] == 1


# ── Auth gate on /api/run ────────────────────────────────────────────────────

def test_api_run_requires_auth(client):
    r = client.post("/api/run", json={"plan": {}, "entities": []})
    assert r.status_code == 401


# ── /api/discover-entities/plan ──────────────────────────────────────────────

def test_discover_entities_plan_surfaces_audit_issues(client, monkeypatch):
    claude = MockClaude().on("plan_entity_discovery", {
        "query_he": "10 הערים הגדולות בישראל 2023",
        "query_en": "top 10 cities in Israel 2023",
        "expected_count": 10,
        "extraction_hint": "ordered list",
        "audit_issues": [{
            "issue_kind": "missing_anchor",
            "explanation_he": "אין שנה", "explanation_en": "no year",
            "suggested_fix_he": "ציין שנה", "suggested_fix_en": "specify a year",
        }],
    })
    monkeypatch.setattr(api_main, "_claude", lambda: claude)

    tok = _login(client)
    r = client.post(
        "/api/discover-entities/plan",
        headers=_auth(tok),
        json={"question": "the largest cities in Israel", "entity_type": "city"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["expected_count"] == 10
    assert len(body["audit_issues"]) == 1
    assert body["audit_issues"][0]["issue_kind"] == "missing_anchor"


def test_discover_entities_run_rejects_unresolved_audit_issues(client, monkeypatch):
    """If the discovery plan still has audit issues, /run must refuse."""
    # No Claude call expected — endpoint should reject before reaching it
    claude = MockClaude()
    monkeypatch.setattr(api_main, "_claude", lambda: claude)
    monkeypatch.setattr(api_main, "_search_client", lambda _e: StubSearch())

    tok = _login(client)
    plan = ResearchPlan(
        entity_type="city", research_question_original="x",
        columns=[ColumnPlan(id="founded", label_he="ייסוד",
                            label_en="Founded", type="number")],
    )
    r = client.post(
        "/api/discover-entities/run",
        headers=_auth(tok),
        json={
            "plan": plan.model_dump(),
            "discovery": {
                "query_he": "q", "query_en": "q",
                "extraction_hint": "list",
                "audit_issues": [{
                    "issue_kind": "missing_anchor",
                    "explanation_he": "x", "explanation_en": "x",
                    "suggested_fix_he": "x", "suggested_fix_en": "x",
                }],
            },
            "search_engine": "duckduckgo",
        },
    )
    assert r.status_code == 400
    assert len(claude.calls) == 0


def test_discover_entities_run_returns_entities_and_harvest(client, monkeypatch):
    page = (
        "Top cities in Israel (2023 census):\n"
        "1. ירושלים – 952,000, founded ancient times.\n"
        "2. תל אביב – 467,000, founded 1909 as a Jaffa suburb.\n"
        "3. חיפה – 285,000, founded 1761 as a port city. "
        + "Padding content. " * 10
    )
    search = StubSearch().add(None, [
        {"url": "https://wiki.example/cities", "raw_content": page, "content": page},
    ])
    claude = MockClaude().on("discover_entities", {
        "entities": [
            {"name": "ירושלים", "rank": 1, "quote": "1. ירושלים"},
            {"name": "תל אביב", "rank": 2, "quote": "2. תל אביב"},
            {"name": "חיפה",   "rank": 3, "quote": "3. חיפה"},
        ],
        "harvested": [
            {"entity_name": "תל אביב", "field_id": "founded",
             "value": "1909", "quote": "founded 1909"},
        ],
    })
    monkeypatch.setattr(api_main, "_claude", lambda: claude)
    monkeypatch.setattr(api_main, "_search_client", lambda _e: search)

    tok = _login(client)
    plan = ResearchPlan(
        entity_type="city", research_question_original="x",
        columns=[ColumnPlan(id="founded", label_he="ייסוד",
                            label_en="Founded", type="number")],
    )
    r = client.post(
        "/api/discover-entities/run",
        headers=_auth(tok),
        json={
            "plan": plan.model_dump(),
            "discovery": {
                "query_he": "q", "query_en": "q",
                "extraction_hint": "list",
                "audit_issues": [],
            },
            "search_engine": "duckduckgo",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert [e["name"] for e in body["entities"]] == ["ירושלים", "תל אביב", "חיפה"]
    assert len(body["harvested"]) == 1
    assert body["harvested"][0]["entity_name"] == "תל אביב"
    assert body["source_domain"] == "wiki.example"
