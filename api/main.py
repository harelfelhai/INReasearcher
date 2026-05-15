"""
FastAPI backend for the Autonomous Research Agent.

Endpoints map 1:1 to the CLI's Stage-0 phases plus a streaming run:

  POST /api/compile-schema  → preflight + bare schema      (Phase A + B1)
  POST /api/audit           → per-field clarity audit      (Phase C)
  POST /api/mock-preview    → fake rows for shape check    (Phase D)
  POST /api/enrich          → add search queries           (Phase B2)
  POST /api/run             → stream EntityResult via SSE  (Stage 1+)

User management lives in `auth/` and is mounted as:

  /api/auth/login, /api/auth/me
  /api/admin/users (admin), /api/admin/users/{id}/budget, ...
  /api/user/sessions, /api/user/exports/{export_id}

The frontend should call the research endpoints in order, mirroring the CLI flow.

Run:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import os
import sys
import json
import asyncio
import traceback
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
import anthropic
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from research_agent.compiler import (
    compile_schema,
    audit_schema,
    generate_mock_rows,
    enrich_with_queries,
    plan_entity_discovery,
)
from research_agent.memory import SuccessMemory
from research_agent.tracer import NullTracer, RunTracer
from research_agent.extractor import (
    search_and_extract_batched,
    probe_field_list,
    verify_probe_extraction,
    discover_entities,
    MockTavilyClient,
    DuckDuckGoClient,
    WikipediaSearchClient,
    SerpApiClient,
)
from research_agent.verifier import verify_field
from research_agent.models import (
    ClarificationRequest,
    ResearchPlan,
    EntityResult,
    EntityDiscoveryPlan,
    EntityDiscoveryResult,
    ExtractionResult,
    FieldAuditReport,
    MockRow,
)
from research_agent.output import write_xlsx

from auth import crud
from auth.db import SessionLocal, get_db, init_db
from auth.deps import get_current_user, require_admin
from auth.models import User
from auth.router import auth_router, admin_router, user_router
from auth.schemas import FeedbackRequest, MemorySeedRequest, MemoryStatsOut
from auth.usage_tracker import TrackedAnthropic

load_dotenv()

app = FastAPI(title="Autonomous Research Agent API", version="0.2.0")

_MEMORY_PATH = Path(os.getenv("MEMORY_PATH", "memory.json"))
_memory: SuccessMemory | None = None

# CORS for the Vite dev server (default port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _on_startup() -> None:
    global _memory
    init_db()
    db = SessionLocal()
    try:
        crud.ensure_default_admin(db)
    finally:
        db.close()
    _memory = SuccessMemory(_MEMORY_PATH)
    print(f"[memory] loaded {_MEMORY_PATH}: {_memory.stats()}", file=sys.stderr)


app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(user_router)


# ── Entity processing helper (module-level so threads can call it) ───────────

def _process_entity_sync(
    entity: str,
    plan: ResearchPlan,
    probe_results: dict,
    search,
    claude,
    memory: SuccessMemory | None = None,
    tracer=None,
) -> tuple[str, dict, list]:
    """Run the three-lane pipeline for one entity. Pure sync — safe to call
    from a thread pool worker alongside other entities."""
    import time as _time
    _t0 = _time.monotonic()
    tr = tracer or NullTracer()
    resolved: dict[str, str] = {}
    cells: dict = {}

    # Lane 1 — probe-covered fields
    probe_handled: set[str] = set()
    for field in plan.columns:
        if field.depends_on or field.id not in probe_results:
            continue
        probe_hit = probe_results[field.id].get(entity)
        if not (probe_hit and probe_hit.value):
            continue
        extras = verify_probe_extraction(field, entity, probe_hit, search, claude)
        all_extractions = [probe_hit] + extras
        cell = verify_field(field, all_extractions)
        cells[field.id] = cell
        tr.emit(
            "verification",
            entity=entity, field_id=field.id, field_type=field.type,
            lane="probe",
            candidates=len(all_extractions),
            grounded=sum(1 for e in all_extractions if e.is_grounded),
            winner_value=cell.value,
            confidence=cell.confidence,
            corroboration_count=cell.corroboration_count,
            flags=cell.flags,
        )
        if cell.value:
            resolved[field.id] = cell.value
        probe_handled.add(field.id)

    # Lane 2 — non-deferred fields not covered by probe
    lane2_fields = [f for f in plan.columns if not f.depends_on and f.id not in probe_handled]
    deferred = [f for f in plan.columns if f.depends_on]

    if lane2_fields:
        batched = search_and_extract_batched(
            fields=lane2_fields, entity=entity,
            resolved_deps=resolved, tavily=search,
            claude=claude, memory=memory, tracer=tr,
        )
        for f in lane2_fields:
            extractions = batched.get(f.id, [])
            cell = verify_field(f, extractions)
            cells[f.id] = cell
            tr.emit(
                "verification",
                entity=entity, field_id=f.id, field_type=f.type,
                candidates=len(extractions),
                grounded=sum(1 for e in extractions if e.is_grounded),
                winner_value=cell.value,
                confidence=cell.confidence,
                corroboration_count=cell.corroboration_count,
                flags=cell.flags,
            )
            if cell.value:
                resolved[f.id] = cell.value

    # Lane 3 — deferred fields
    if deferred:
        for f in deferred:
            if f.depends_on and f.depends_on not in resolved:
                resolved[f.depends_on] = entity
        batched_def = search_and_extract_batched(
            fields=deferred, entity=entity,
            resolved_deps=resolved, tavily=search,
            claude=claude, memory=memory, tracer=tr,
        )
        for f in deferred:
            extractions = batched_def.get(f.id, [])
            cell = verify_field(f, extractions)
            cells[f.id] = cell
            tr.emit(
                "verification",
                entity=entity, field_id=f.id, field_type=f.type,
                candidates=len(extractions),
                grounded=sum(1 for e in extractions if e.is_grounded),
                winner_value=cell.value,
                confidence=cell.confidence,
                corroboration_count=cell.corroboration_count,
                flags=cell.flags,
            )
            if cell.value:
                resolved[f.id] = cell.value

    row_flags = []
    not_found = sum(1 for c in cells.values() if c.confidence == "NOT_FOUND")
    if cells and not_found > len(cells) // 2:
        row_flags.append("majority_not_found")

    import time as _time2
    tr.emit(
        "entity_done",
        entity=entity,
        elapsed_sec=round(_time2.monotonic() - _t0, 2),
        found=sum(1 for c in cells.values() if c.value is not None),
        not_found_fields=[fid for fid, c in cells.items() if c.value is None],
        row_flags=row_flags,
    )
    return entity, cells, row_flags


# ── Error translation ────────────────────────────────────────────────────────

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
    print(f"\n[error] {request.method} {request.url.path} →", file=sys.stderr)
    traceback.print_exc()
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    status, detail = _translate(exc)
    return JSONResponse(status_code=status, content={"detail": detail})


# ── Shared Claude client ─────────────────────────────────────────────────────

def _domain_safe(url: str) -> str:
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc or ""
    except Exception:
        return ""


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
    # Optional pre-extracted (field_id, entity_name) → value+quote+source,
    # produced by /api/discover-entities/run. Seeds Lane 1 of the run.
    seeded_probe: Optional[dict] = None


class DiscoverPlanRequest(BaseModel):
    question: str
    entity_type: str = ""


class DiscoverRunRequest(BaseModel):
    plan: ResearchPlan
    discovery: EntityDiscoveryPlan
    search_engine: SearchEngine = "serpapi"


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/compile-schema", response_model=CompileResponse)
def api_compile_schema(
    req: CompileRequest,
    _user: User = Depends(get_current_user),
) -> CompileResponse:
    """Phase A preflight + Phase B1 bare schema."""
    result = compile_schema(req.question, req.entity_type, _claude(), memory=_memory)
    if isinstance(result, ClarificationRequest):
        return CompileResponse(kind="clarification", clarification=result)
    return CompileResponse(kind="plan", plan=result.plan)


@app.post("/api/audit", response_model=FieldAuditReport)
def api_audit(
    req: PlanOnlyRequest,
    _user: User = Depends(get_current_user),
) -> FieldAuditReport:
    """Phase C — per-field clarity audit."""
    return audit_schema(req.plan, _claude())


@app.post("/api/mock-preview", response_model=MockResponse)
def api_mock_preview(
    req: PlanOnlyRequest,
    _user: User = Depends(get_current_user),
) -> MockResponse:
    """Phase D — 2-3 fake rows so the user can sanity-check shape."""
    return MockResponse(rows=generate_mock_rows(req.plan, _claude()))


@app.post("/api/enrich", response_model=ResearchPlan)
def api_enrich(
    req: PlanOnlyRequest,
    _user: User = Depends(get_current_user),
) -> ResearchPlan:
    """Phase B2 — add search queries to an approved schema."""
    return enrich_with_queries(req.plan, _claude())


# ── Entity discovery ─────────────────────────────────────────────────────────
#
# Two-step flow, mirroring Stage-0 / Stage-1:
#   1. POST /api/discover-entities/plan  → returns query + audit issues.
#      The UI must surface any audit issues and let the user fix the question
#      before proceeding to step 2.
#   2. POST /api/discover-entities/run   → executes the discovery search,
#      returns the entity list (with source quotes) and any field values
#      already harvested from the same page. The UI then shows the
#      EntityReview screen for user approval before /api/run is called.

@app.post("/api/discover-entities/plan", response_model=EntityDiscoveryPlan)
def api_discover_plan(
    req: DiscoverPlanRequest,
    _user: User = Depends(get_current_user),
) -> EntityDiscoveryPlan:
    """Phase 0D — generate the discovery query and audit the question."""
    return plan_entity_discovery(req.question, req.entity_type, _claude())


@app.post("/api/discover-entities/run", response_model=Optional[EntityDiscoveryResult])
def api_discover_run(
    req: DiscoverRunRequest,
    _user: User = Depends(get_current_user),
):
    """Execute discovery search + LLM extraction. Returns None on failure."""
    if req.discovery.audit_issues:
        raise HTTPException(
            400,
            "Discovery question has unresolved audit issues — resolve them first.",
        )
    claude = _claude()
    search = _search_client(req.search_engine)
    return discover_entities(req.discovery, req.plan.columns, search, claude)


@app.post("/api/run")
def api_run(
    req: RunRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Stage 1+ — run the research pipeline. Streams Server-Sent Events.

    Side effects (no run-blocking yet):
      * Creates a ResearchSession row at start.
      * Tracks Anthropic token usage via TrackedAnthropic; the dollar estimate
        is written to ResearchSession.cost_used at completion.
      * Writes an .xlsx of the results and records an ExcelExport row.
    """
    claude = TrackedAnthropic(_claude())
    search = _search_client(req.search_engine)
    plan = req.plan
    entities = req.entities

    session_row = crud.create_session(
        db,
        user_id=user.id,
        question=plan.research_question_original,
        entity_type=plan.entity_type,
        entity_list=entities,
        plan_dict=plan.model_dump(),
    )
    session_id = session_row.id

    _LOGS_DIR = Path(os.getenv("LOGS_DIR", "logs"))
    run_tracer = RunTracer(session_id, _LOGS_DIR)
    run_tracer.emit(
        "run_start",
        question=plan.research_question_original,
        entity_type=plan.entity_type,
        entity_count=len(entities),
        entities=entities,
        columns=[{"id": c.id, "type": c.type, "label_en": c.label_en} for c in plan.columns],
        search_engine=req.search_engine,
    )

    def _sse(event: str, payload: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    async def stream():
        results: list[EntityResult] = []
        final_status = "completed"
        try:
            yield _sse("session_started", {
                "session_id": session_id,
                "credit_balance": user.credit_balance,
            })

            # ── Phase 0a: Seed probe_results from discovery harvest ──────────
            # Any (field, entity) values already lifted from the discovery
            # page are injected as synthetic probe hits. Lane 1 will then
            # treat them like normal probe hits — running ONE verification
            # search rather than the full per-field cycle.
            probe_results: dict[str, dict] = {}
            if req.seeded_probe:
                entity_set = set(entities)
                for fid, by_entity in req.seeded_probe.items():
                    if not isinstance(by_entity, dict):
                        continue
                    field_obj = next((c for c in plan.columns if c.id == fid), None)
                    if not field_obj:
                        continue
                    bucket: dict = {}
                    for ename, payload in by_entity.items():
                        if ename not in entity_set or not isinstance(payload, dict):
                            continue
                        if not payload.get("value"):
                            continue
                        src_url = payload.get("source_url") or ""
                        bucket[ename] = ExtractionResult(
                            field_id=fid,
                            value=payload["value"],
                            quote_original=payload.get("quote") or "",
                            source_url=src_url,
                            source_domain=_domain_safe(src_url),
                            is_grounded=True,
                            extractor_confidence=0.85,
                        )
                    if bucket:
                        probe_results[fid] = bucket
                        yield _sse("probe_hit", {
                            "field_id": fid,
                            "entities_found": len(bucket),
                            "entities_total": len(entities),
                            "source": "discovery",
                        })

            # ── Phase 0b: Standard probe per field (parallelised) ────────────
            # All probe searches are I/O-bound; run them concurrently via
            # asyncio.to_thread so fields don't block each other.
            fields_to_probe = [
                f for f in plan.columns
                if f.id not in probe_results and f.directory_probe_query_he
            ]
            if fields_to_probe:
                probe_outcomes = await asyncio.gather(
                    *[
                        asyncio.to_thread(probe_field_list, f, entities, search, claude)
                        for f in fields_to_probe
                    ],
                    return_exceptions=True,
                )
                for field, result in zip(fields_to_probe, probe_outcomes):
                    if isinstance(result, Exception):
                        continue
                    if result:
                        probe_results[field.id] = result
                        found_n = sum(1 for r in result.values() if r.value)
                        yield _sse("probe_hit", {
                            "field_id": field.id,
                            "entities_found": found_n,
                            "entities_total": len(entities),
                        })

            # ── Phase 1: Entity loop (parallel) ──────────────────────────────
            # Fire entity_start for every entity immediately, then run all
            # entity pipelines concurrently. entity_done events arrive as
            # each entity finishes — no waiting for the slowest to unblock
            # the fastest.
            for entity in entities:
                yield _sse("entity_start", {"entity": entity})
                await asyncio.sleep(0)

            entity_tasks = [
                asyncio.create_task(
                    asyncio.to_thread(
                        _process_entity_sync, entity, plan, probe_results, search, claude, _memory, run_tracer,
                    )
                )
                for entity in entities
            ]

            for finished in asyncio.as_completed(entity_tasks):
                try:
                    entity_name, cells, row_flags = await finished
                except Exception as exc:
                    print(f"\n[error] entity processing failed:", file=sys.stderr)
                    traceback.print_exc()
                    _, detail = _translate(exc)
                    final_status = "failed"
                    yield _sse("error", {"message": detail})
                    continue

                results.append(EntityResult(
                    entity_name=entity_name, cells=cells, row_flags=row_flags,
                ))
                yield _sse("entity_done", {
                    "entity_name": entity_name,
                    "cells": {k: v.model_dump() for k, v in cells.items()},
                    "row_flags": row_flags,
                })
                await asyncio.sleep(0)

        except Exception as exc:
            print(f"\n[error] /api/run stream failed →", file=sys.stderr)
            traceback.print_exc()
            _, detail = _translate(exc)
            final_status = "failed"
            yield _sse("error", {"message": detail})

        # ── Persist session + write Excel ───────────────────────────────────
        export_payload: dict | None = None
        try:
            db2 = SessionLocal()
            try:
                sess = crud.get_session(db2, session_id)
                if sess is not None:
                    results_dicts = [
                        {"entity_name": r.entity_name,
                         "cells": {k: v.model_dump() for k, v in r.cells.items()}}
                        for r in results
                    ]
                    crud.complete_session(
                        db2, sess,
                        status=final_status,
                        cost_used=claude.totals.cost_usd,
                        results=results_dicts,
                    )
                if results:
                    filename = f"{session_id}.xlsx"
                    file_path = crud.EXPORTS_DIR / filename
                    write_xlsx(results, str(file_path), plan.columns)
                    export = crud.record_export(
                        db2,
                        session_id=session_id,
                        user_id=user.id,
                        filename=f"research-{session_id[:8]}.xlsx",
                        file_path=str(file_path),
                    )
                    export_payload = {
                        "export_id": export.id,
                        "filename": export.filename,
                    }
            finally:
                db2.close()
        except Exception:
            print(f"\n[error] /api/run finalisation failed →", file=sys.stderr)
            traceback.print_exc()

        run_tracer.emit(
            "run_done",
            entity_count=len(results),
            cost_usd=claude.totals.cost_usd,
            tokens_in=claude.totals.input_tokens,
            tokens_out=claude.totals.output_tokens,
            trace_path=str(run_tracer.path),
        )
        yield _sse("done", {
            "n": len(results),
            "session_id": session_id,
            "cost_used": claude.totals.cost_usd,
            "tokens_in": claude.totals.input_tokens,
            "tokens_out": claude.totals.output_tokens,
            "export": export_payload,
            "trace_path": str(run_tracer.path),
        })

    return StreamingResponse(stream(), media_type="text/event-stream")


# ── Memory: post-run cell feedback ───────────────────────────────────────────

@app.post("/api/sessions/{session_id}/feedback")
def api_session_feedback(
    session_id: str,
    req: FeedbackRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """Record per-cell feedback for a completed session into the memory store.

    Each correct cell becomes a positive few-shot example for future runs of
    the same field type. Each incorrect cell becomes a negative warning.
    """
    if _memory is None:
        raise HTTPException(503, "Memory system not initialised")
    sess = crud.get_session(db, session_id)
    if not sess or sess.user_id != user.id:
        raise HTTPException(404, "Session not found")
    if not sess.results_json or not sess.plan_json:
        raise HTTPException(400, "Session has no stored results to review")

    try:
        results_data: list[dict] = json.loads(sess.results_json)
        plan_data: dict = json.loads(sess.plan_json)
    except (json.JSONDecodeError, TypeError):
        raise HTTPException(500, "Could not parse stored session data")

    # Build fast lookups
    cell_lookup: dict[str, dict[str, dict]] = {
        er["entity_name"]: er.get("cells", {})
        for er in results_data
    }
    field_meta: dict[str, dict] = {col["id"]: col for col in plan_data.get("columns", [])}

    n_successes = n_failures = 0
    any_correct = False

    for fb in req.cells:
        cell = cell_lookup.get(fb.entity_name, {}).get(fb.field_id, {})
        meta = field_meta.get(fb.field_id, {})
        value = cell.get("value") or ""
        quote = cell.get("quote_original") or ""
        source_url = cell.get("source_url") or ""
        source_domain = cell.get("source_domain") or ""
        ftype = meta.get("type", "text")
        flabel = meta.get("label_en") or meta.get("label_he") or fb.field_id

        if fb.is_correct and value:
            _memory.record_extraction_success(
                field_id=fb.field_id, field_label=flabel, field_type=ftype,
                entity=fb.entity_name, value=value, quote=quote,
                source_url=source_url, source_domain=source_domain,
            )
            n_successes += 1
            any_correct = True
        elif not fb.is_correct:
            reason = "User marked as incorrect"
            if fb.correct_value:
                reason += f" — correct value: {fb.correct_value}"
            _memory.record_extraction_failure(
                field_id=fb.field_id, field_type=ftype, entity=fb.entity_name,
                claimed_value=value or None, claimed_quote=quote or None,
                source_url=source_url, source_domain=source_domain,
                reason=reason,
            )
            n_failures += 1

    if any_correct:
        _memory.record_compiler_success(
            research_question=sess.question,
            entity_type=sess.entity_type or "",
            plan=plan_data,
        )

    return {
        "recorded_successes": n_successes,
        "recorded_failures": n_failures,
        "memory_stats": _memory.stats(),
    }


# ── Memory: admin seeding + stats ────────────────────────────────────────────

@app.post("/api/admin/memory/seed")
def api_memory_seed(
    req: MemorySeedRequest,
    _admin: User = Depends(require_admin),
) -> dict:
    """Manually inject a curated example into the memory store."""
    if _memory is None:
        raise HTTPException(503, "Memory system not initialised")
    from urllib.parse import urlparse
    domain = req.source_domain or urlparse(req.source_url).netloc or ""
    fid = req.field_label.lower().replace(" ", "_")

    if req.kind == "success":
        rid = _memory.record_extraction_success(
            field_id=fid, field_label=req.field_label, field_type=req.field_type,
            entity=req.entity, value=req.value, quote=req.quote,
            source_url=req.source_url, source_domain=domain,
        )
    else:
        if not req.reason:
            raise HTTPException(400, "reason is required for failure records")
        rid = _memory.record_extraction_failure(
            field_id=fid, field_type=req.field_type, entity=req.entity,
            claimed_value=req.value or None, claimed_quote=req.quote or None,
            source_url=req.source_url, source_domain=domain,
            reason=req.reason,
        )
    return {"id": rid, "stats": _memory.stats()}


@app.get("/api/admin/memory/stats", response_model=MemoryStatsOut)
def api_memory_stats(_admin: User = Depends(require_admin)) -> dict:
    """Return current memory store sizes."""
    if _memory is None:
        raise HTTPException(503, "Memory system not initialised")
    return {**_memory.stats(), "path": str(_memory.path)}
