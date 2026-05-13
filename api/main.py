"""
FastAPI backend for the Autonomous Research Agent.

Endpoints map 1:1 to the CLI's Stage-0 phases plus a streaming run:

  POST /api/compile-schema  → preflight + bare schema      (Phase A + B1)
  POST /api/audit           → per-field clarity audit      (Phase C)
  POST /api/mock-preview    → fake rows for shape check    (Phase D)
  POST /api/enrich          → add search queries           (Phase B2)
  POST /api/run             → stream EntityResult via SSE  (Stage 1+)

The frontend should call these in order, mirroring the CLI flow.

Run:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import os
import sys
import json
import asyncio
import traceback
from typing import Literal, Optional

from dotenv import load_dotenv
import anthropic
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from research_agent.compiler import (
    compile_schema,
    audit_schema,
    generate_mock_rows,
    enrich_with_queries,
)
from research_agent.extractor import (
    search_and_extract,
    search_and_extract_batched,
    probe_field_list,
    verify_probe_extraction,
    MockTavilyClient,
    DuckDuckGoClient,
    WikipediaSearchClient,
    SerpApiClient,
)
from research_agent.verifier import verify_field
from research_agent.memory import SuccessMemory
from research_agent.models import (
    ClarificationRequest,
    ExecutableResearchPlan,
    ResearchPlan,
    FieldAuditReport,
    MockRow,
)

load_dotenv()

app = FastAPI(title="Autonomous Research Agent API", version="0.1.0")

# CORS for the Vite dev server (default port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Error translation ────────────────────────────────────────────────────────
#
# All endpoints rely on a single exception handler. It:
#   1. Logs the full traceback to the uvicorn terminal so the developer can
#      always see what went wrong.
#   2. Translates the common Anthropic errors into actionable messages with
#      sensible HTTP status codes.
#   3. Falls back to a generic 500 that still includes the exception class
#      name (e.g. "ConnectionError") rather than just "error".

def _translate(exc: Exception) -> tuple[int, str]:
    msg = str(exc)
    if isinstance(exc, anthropic.BadRequestError) and "credit balance" in msg.lower():
        return 402, (
            "Anthropic credits exhausted. "
            "Top up at console.anthropic.com → Plans & Billing, then retry."
        )
    if isinstance(exc, anthropic.AuthenticationError):
        return 401, "Invalid ANTHROPIC_API_KEY — check the .env file."
    if isinstance(exc, anthropic.RateLimitError):
        return 429, f"Anthropic rate limit hit. Wait a moment and retry. ({msg})"
    if isinstance(exc, anthropic.APIConnectionError):
        return 503, f"Could not reach Anthropic API: {msg}"
    if isinstance(exc, anthropic.APIError):
        return 502, f"Anthropic API error: {msg}"
    return 500, f"{type(exc).__name__}: {msg}"


@app.exception_handler(Exception)
async def all_exceptions_handler(request: Request, exc: Exception):
    # Always log the full traceback so the dev can debug from the terminal.
    print(
        f"\n[error] {request.method} {request.url.path} →",
        file=sys.stderr,
    )
    traceback.print_exc()

    # Pass HTTPException through unchanged so explicit raises keep their codes.
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

    status, detail = _translate(exc)
    return JSONResponse(status_code=status, content={"detail": detail})


# ── Shared Claude client ─────────────────────────────────────────────────────

def _claude() -> anthropic.Anthropic:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise HTTPException(500, "ANTHROPIC_API_KEY not set on the server")
    return anthropic.Anthropic(api_key=key)


SearchEngine = Literal["wikipedia", "serpapi", "duckduckgo", "mock"]


def _search_client(engine: SearchEngine):
    if engine == "wikipedia":
        return WikipediaSearchClient()
    if engine == "duckduckgo":
        return DuckDuckGoClient()
    if engine == "mock":
        return MockTavilyClient()
    if engine == "serpapi":
        key = os.getenv("SERPAPI_KEY", "")
        if not key:
            raise HTTPException(400, "SERPAPI_KEY not set on the server")
        return SerpApiClient(api_key=key)
    raise HTTPException(400, f"unknown search engine: {engine}")


# ── Request / Response models ────────────────────────────────────────────────

class CompileRequest(BaseModel):
    question: str
    entity_type: str = ""


class CompileResponse(BaseModel):
    kind: Literal["clarification", "plan"]
    clarification: Optional[ClarificationRequest] = None
    plan: Optional[ResearchPlan] = None


class PlanOnlyRequest(BaseModel):
    plan: ResearchPlan


class MockResponse(BaseModel):
    rows: list[MockRow]


class RunRequest(BaseModel):
    plan: ResearchPlan
    entities: list[str]
    search_engine: SearchEngine = "serpapi"


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/compile-schema", response_model=CompileResponse)
def api_compile_schema(req: CompileRequest) -> CompileResponse:
    """Phase A preflight + Phase B1 bare schema."""
    result = compile_schema(req.question, req.entity_type, _claude())
    if isinstance(result, ClarificationRequest):
        return CompileResponse(kind="clarification", clarification=result)
    return CompileResponse(kind="plan", plan=result.plan)


@app.post("/api/audit", response_model=FieldAuditReport)
def api_audit(req: PlanOnlyRequest) -> FieldAuditReport:
    """Phase C — per-field clarity audit."""
    return audit_schema(req.plan, _claude())


@app.post("/api/mock-preview", response_model=MockResponse)
def api_mock_preview(req: PlanOnlyRequest) -> MockResponse:
    """Phase D — 2-3 fake rows so the user can sanity-check shape."""
    return MockResponse(rows=generate_mock_rows(req.plan, _claude()))


@app.post("/api/enrich", response_model=ResearchPlan)
def api_enrich(req: PlanOnlyRequest) -> ResearchPlan:
    """Phase B2 — add search queries to an approved schema."""
    return enrich_with_queries(req.plan, _claude())


@app.post("/api/run")
def api_run(req: RunRequest):
    """
    Stage 1+ — run the research pipeline. Streams Server-Sent Events:

      event: entity_start    data: {entity}
      event: field_result    data: {entity, field_id, cell}
      event: entity_done     data: {entity_result}
      event: done            data: {n}
      event: error           data: {message}
    """
    claude = _claude()
    search = _search_client(req.search_engine)
    plan = req.plan
    entities = req.entities

    def _sse(event: str, payload: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    async def stream():
        try:
            # ── Phase 0: Probe each field for a directory/list page ──────────
            # For each field, run ONE entity-agnostic search to detect pages that
            # cover this field for many entities at once (the field-list axis).
            # Saves N-1 searches per field that has a good directory source.
            probe_results: dict[str, dict] = {}   # field_id → {entity → ExtractionResult}
            for field in plan.columns:
                if field.directory_probe_query_he:
                    result = probe_field_list(field, entities, search, claude)
                    if result:
                        probe_results[field.id] = result
                        found_n = sum(1 for r in result.values() if r.value)
                        yield _sse("probe_hit", {
                            "field_id": field.id,
                            "entities_found": found_n,
                            "entities_total": len(entities),
                        })
                await asyncio.sleep(0)

            # ── Phase 1: Entity loop ─────────────────────────────────────────
            # Per entity, we split fields into three lanes:
            #   (1) probe-covered    → individual extraction + verify (different path)
            #   (2) regular non-dep  → batched: one Claude call per unique page
            #   (3) deferred (dep)   → after deps resolve, batched again
            for entity in entities:
                yield _sse("entity_start", {"entity": entity})
                resolved: dict[str, str] = {}
                cells: dict = {}

                def _emit_field_result(fid: str):
                    return _sse("field_result", {
                        "entity": entity,
                        "field_id": fid,
                        "cell": cells[fid].model_dump(),
                    })

                # Lane 1 — probe-covered fields (one verification search each).
                probe_handled: set[str] = set()
                for field in plan.columns:
                    if field.depends_on:
                        continue
                    if field.id not in probe_results:
                        continue
                    probe_hit = probe_results[field.id].get(entity)
                    if not (probe_hit and probe_hit.value):
                        continue
                    extras = verify_probe_extraction(
                        field, entity, probe_hit, search, claude,
                    )
                    cell = verify_field(field, [probe_hit] + extras)
                    cells[field.id] = cell
                    if cell.value:
                        resolved[field.id] = cell.value
                    probe_handled.add(field.id)
                    yield _emit_field_result(field.id)
                    await asyncio.sleep(0)

                # Lane 2 — non-deferred fields not handled by probe: batched.
                lane2_fields = [
                    f for f in plan.columns
                    if not f.depends_on and f.id not in probe_handled
                ]
                deferred = [f for f in plan.columns if f.depends_on]

                if lane2_fields:
                    batched = search_and_extract_batched(
                        fields=lane2_fields, entity=entity,
                        resolved_deps=resolved, tavily=search,
                        claude=claude, memory=None,
                    )
                    for f in lane2_fields:
                        cell = verify_field(f, batched.get(f.id, []))
                        cells[f.id] = cell
                        if cell.value:
                            resolved[f.id] = cell.value
                        yield _emit_field_result(f.id)
                        await asyncio.sleep(0)

                # Lane 3 — deferred fields. If a dep still didn't resolve, fall
                # back to the entity name so the deferred field can still try.
                if deferred:
                    for f in deferred:
                        if f.depends_on and f.depends_on not in resolved:
                            resolved[f.depends_on] = entity
                    batched_def = search_and_extract_batched(
                        fields=deferred, entity=entity,
                        resolved_deps=resolved, tavily=search,
                        claude=claude, memory=None,
                    )
                    for f in deferred:
                        cell = verify_field(f, batched_def.get(f.id, []))
                        cells[f.id] = cell
                        if cell.value:
                            resolved[f.id] = cell.value
                        yield _emit_field_result(f.id)
                        await asyncio.sleep(0)

                row_flags = []
                not_found = sum(1 for c in cells.values() if c.confidence == "NOT_FOUND")
                if cells and not_found > len(cells) // 2:
                    row_flags.append("majority_not_found")

                yield _sse("entity_done", {
                    "entity_name": entity,
                    "cells": {k: v.model_dump() for k, v in cells.items()},
                    "row_flags": row_flags,
                })

            yield _sse("done", {"n": len(entities)})
        except Exception as exc:
            print(f"\n[error] /api/run stream failed →", file=sys.stderr)
            traceback.print_exc()
            _, detail = _translate(exc)
            yield _sse("error", {"message": detail})

    return StreamingResponse(stream(), media_type="text/event-stream")
