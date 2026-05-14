"""Golden run — the canary integration test.

Replays a hand-authored cassette through /api/run and asserts on the
end-to-end output. If this test breaks, the orchestrator has regressed in
a way that touches a realistic, multi-entity, multi-field scenario.

The cassette format lives in tests/cassettes/*.json and contains:
  - search   : entity-keyed list of search hits the StubSearch should return
  - claude   : tool-keyed queue of canned tool_use responses, in the order
               the orchestrator will request them

A real recorded cassette can be captured later by running the system once
against live APIs and dumping every (query → hits) and (tool → response).
Until then, the hand-authored fixture exercises the most-traveled path:
two entities, two fields, shared page per entity, cross-field dedup.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


# ── Test DB + exports isolation ──────────────────────────────────────────────
_TEST_DB = Path(__file__).parent / "_test_golden.db"
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

CASSETTE = Path(__file__).parent / "cassettes" / "two_cities_mayor_website.json"


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


def _build_clients_from_cassette(cassette: dict) -> tuple[MockClaude, StubSearch]:
    """Translate a cassette JSON dict into (MockClaude, StubSearch) ready to
    inject into /api/run. The translation is pure data."""
    search = StubSearch()
    for entity_substring, hits in cassette["search"].items():
        # Normalise hit dicts to the keys the extractor expects
        normalised = [{
            "url": h["url"],
            "title": h.get("title", ""),
            "content": h["raw_content"],
            "raw_content": h["raw_content"],
        } for h in hits]
        search.add(entity_substring, normalised)

    claude = MockClaude()
    # Each tool gets a FIFO queue of responses (entity order matters for
    # batch_extract_fields — Tel Aviv first, then Haifa).
    for tool_name, response_list in cassette["claude"].items():
        queue = [{"extractions": r["extractions"]} for r in response_list]
        claude.on(tool_name, queue)

    return claude, search


def test_golden_run_two_cities_two_fields(monkeypatch):
    cassette = json.loads(CASSETTE.read_text())
    claude, search = _build_clients_from_cassette(cassette)

    monkeypatch.setattr(api_main, "_claude", lambda: claude)
    monkeypatch.setattr(api_main, "_search_client", lambda _e: search)

    client = TestClient(app)
    tok = client.post("/api/auth/login",
                      json={"username": "admin", "password": "admin"}
                      ).json()["access_token"]

    plan = ResearchPlan(
        entity_type="Israeli city",
        research_question_original="For each city, find the mayor and the "
                                   "official municipal website.",
        columns=[
            ColumnPlan(id="mayor", label_he="ראש העיר", label_en="Mayor",
                       type="person_name",
                       search_queries_he=["ראש העיר של {entity}"],
                       search_queries_en=["mayor of {entity}"]),
            ColumnPlan(id="website", label_he="אתר רשמי", label_en="Website",
                       type="url",
                       search_queries_he=["{entity} אתר רשמי"],
                       search_queries_en=["{entity} official website"]),
        ],
    )

    r = client.post(
        "/api/run",
        headers={"Authorization": f"Bearer {tok}"},
        json={
            "plan": plan.model_dump(),
            "entities": ["Tel Aviv", "Haifa"],
            "search_engine": "duckduckgo",
        },
    )
    assert r.status_code == 200, r.text

    # Parse SSE
    events: list[tuple[str, dict]] = []
    for block in r.text.split("\n\n"):
        if not block.strip():
            continue
        event = "message"
        data = ""
        for line in block.split("\n"):
            if line.startswith("event: "):
                event = line[7:].strip()
            elif line.startswith("data: "):
                data += line[6:]
        if data:
            events.append((event, json.loads(data)))

    # ── Sequence
    names = [e for e, _ in events]
    assert names[0] == "session_started"
    assert names.count("entity_start") == 2
    assert names.count("entity_done") == 2
    assert names[-1] == "done"
    assert "error" not in names

    # ── Per-entity results — both fields HIGH/MEDIUM confidence, value set
    done_by_entity = {p["entity_name"]: p for e, p in events if e == "entity_done"}
    tlv = done_by_entity["Tel Aviv"]["cells"]
    hfa = done_by_entity["Haifa"]["cells"]

    assert tlv["mayor"]["value"] == "Ron Huldai"
    assert tlv["website"]["value"] == "www.tel-aviv.gov.il"
    assert tlv["mayor"]["confidence"] in {"HIGH", "MEDIUM"}
    assert tlv["website"]["confidence"] in {"HIGH", "MEDIUM"}

    assert hfa["mayor"]["value"] == "Yona Yahav"
    assert hfa["website"]["value"] == "www.haifa.muni.il"

    # ── Cross-field dedup property: exactly TWO batch_extract_fields calls
    # (one per entity), NOT four. If someone regresses the page-dedup
    # optimisation this assertion will catch it immediately.
    assert len(claude.calls_for("batch_extract_fields")) == 2

    # ── End-of-run payload + persistence
    done = events[-1][1]
    assert done["n"] == 2
    assert done["cost_used"] > 0
    assert done["export"]["export_id"]

    sessions = client.get("/api/user/sessions",
                          headers={"Authorization": f"Bearer {tok}"}).json()
    assert len(sessions) == 1
    sess = sessions[0]
    assert sess["status"] == "completed"
    assert sess["cost_used"] == done["cost_used"]
    assert len(sess["exports"]) == 1

    # Excel file actually written and non-empty
    db = SessionLocal()
    export = crud.get_export(db, sess["exports"][0]["id"])
    db.close()
    assert Path(export.file_path).exists()
    assert Path(export.file_path).stat().st_size > 0
