"""Compiler tests with MockClaude.

Covers every phase + the post-processing logic (Pydantic mapping, defaults,
strategy parsing) without a real LLM. These tests catch:
  - Tool-output schema drift (renaming a field in the LLM tool but forgetting
    to update the Pydantic parsing)
  - Default-value regressions (e.g. min_corroborations falling back wrongly)
  - Optional-field handling (strategy = None when LLM returns empty dict)
"""
from __future__ import annotations

from research_agent.compiler import (
    compile_schema,
    audit_schema,
    generate_mock_rows,
    enrich_with_queries,
    plan_entity_discovery,
)
from research_agent.models import (
    ClarificationRequest,
    ColumnPlan,
    ExecutableResearchPlan,
    ResearchPlan,
)

from tests.mocks import MockClaude


def _plan(*columns: ColumnPlan) -> ResearchPlan:
    return ResearchPlan(
        entity_type="city",
        research_question_original="test",
        columns=list(columns),
    )


# ── Phase A → B1: compile_schema ─────────────────────────────────────────────

def test_compile_schema_returns_clarification_when_preflight_fails():
    claude = MockClaude().on("preflight_check", {
        "is_executable": False,
        "reason": "Missing time anchor",
        "missing_dimensions": [{
            "field": "temporal",
            "question_he": "באיזו שנה?", "question_en": "What year?",
            "example": "1990",
        }],
        "prompt_template": "For each [ENTITY], find the [FIELD] in [YEAR].",
    })
    out = compile_schema("Tell me about cities", "", claude)
    assert isinstance(out, ClarificationRequest)
    assert out.is_executable is False
    assert out.reason == "Missing time anchor"
    assert len(out.questions) == 1
    assert out.questions[0].question_he == "באיזו שנה?"
    # Only preflight was called — no schema phase when preflight fails
    assert len(claude.calls_for("create_schema")) == 0


def test_compile_schema_returns_executable_plan_when_preflight_passes():
    claude = (
        MockClaude()
          .on("preflight_check", {
              "is_executable": True, "reason": "ok",
              "missing_dimensions": [], "prompt_template": "",
          })
          .on("create_schema", {
              "entity_type": "Israeli municipality",
              "columns": [
                  {"id": "mayor_1990", "label_he": "ראש העיר 1990",
                   "label_en": "Mayor 1990", "type": "person_name",
                   "temporal_anchor": "1990"},
                  {"id": "website", "label_he": "אתר רשמי",
                   "label_en": "Website", "type": "url"},
              ],
          })
    )
    out = compile_schema("Q", "city", claude)
    assert isinstance(out, ExecutableResearchPlan)
    assert out.is_executable is True
    assert out.plan.entity_type == "Israeli municipality"
    assert [c.id for c in out.plan.columns] == ["mayor_1990", "website"]
    assert out.plan.columns[0].temporal_anchor == "1990"
    assert out.plan.columns[1].temporal_anchor is None


# ── Phase C: audit_schema ────────────────────────────────────────────────────

def test_audit_schema_propagates_issues_into_pydantic_objects():
    plan = _plan(ColumnPlan(id="career_path", label_he="מסלול",
                            label_en="Career Path", type="free_text"))
    claude = MockClaude().on("audit_schema", {
        "all_clear": False,
        "issues": [{
            "field_id": "career_path",
            "issue_kind": "unbounded",
            "explanation_he": "מסלול קריירה אינו תחום",
            "explanation_en": "Career path is unbounded",
            "suggested_fix_he": "בחר נקודה בזמן",
            "suggested_fix_en": "Pick a point in time",
        }],
    })
    report = audit_schema(plan, claude)
    assert report.all_clear is False
    assert len(report.issues) == 1
    assert report.issues[0].issue_kind == "unbounded"


# ── Phase D: generate_mock_rows ──────────────────────────────────────────────

def test_generate_mock_rows_maps_to_pydantic_models():
    plan = _plan(ColumnPlan(id="founded", label_he="ייסוד",
                            label_en="Founded", type="number"))
    claude = MockClaude().on("generate_mock_rows", {
        "rows": [
            {"entity_name": "Tel Aviv", "values": {"founded": "1909 (example)"}},
            {"entity_name": "Haifa",    "values": {"founded": "1761 (example)"}},
        ]
    })
    rows = generate_mock_rows(plan, claude)
    assert [r.entity_name for r in rows] == ["Tel Aviv", "Haifa"]
    assert rows[0].values["founded"] == "1909 (example)"


# ── Phase B2: enrich_with_queries ────────────────────────────────────────────

def test_enrich_with_queries_attaches_strategy_and_probe_query():
    plan = _plan(
        ColumnPlan(id="mayor", label_he="ראש עיר", label_en="Mayor",
                   type="person_name", temporal_anchor="1990"),
        ColumnPlan(id="website", label_he="אתר", label_en="Website",
                   type="url"),
    )
    claude = MockClaude().on("enrich_with_queries", {
        "columns": [
            {
                "id": "mayor",
                "search_queries_he": ["ראש העיר של {entity} 1990"],
                "search_queries_en": ["mayor of {entity} 1990"],
                "preferred_source_domains": ["wikipedia.org"],
                "min_corroborations": 2,
                "directory_probe_query_he": "רשימת ראשי ערים בישראל 1990",
                "extraction_strategy": {
                    "value_regex": r"[א-ת]+ [א-ת]+",
                    "value_anchors_he": ["ראש העיר"],
                    "value_anchors_en": [],   # empty list should be dropped
                },
            },
            {
                "id": "website",
                "search_queries_he": ["אתר רשמי {entity}"],
                "search_queries_en": ["{entity} official site"],
                "preferred_source_domains": [],
                "min_corroborations": 1,
                "directory_probe_query_he": "",   # falsy → must become None
                "extraction_strategy": {},        # empty → must become None
            },
        ]
    })
    enriched = enrich_with_queries(plan, claude)
    mayor, website = enriched.columns
    assert mayor.search_queries_he == ["ראש העיר של {entity} 1990"]
    assert mayor.preferred_source_domains == ["wikipedia.org"]
    assert mayor.min_corroborations == 2
    assert mayor.directory_probe_query_he == "רשימת ראשי ערים בישראל 1990"
    assert mayor.extraction_strategy is not None
    assert mayor.extraction_strategy.value_anchors_he == ["ראש העיר"]
    # Empty list collapsed to default ([]), not stored as []. The strategy
    # was built from the non-empty subset only.
    assert mayor.extraction_strategy.value_anchors_en == []

    # Empty strategy + empty probe query → None on the column
    assert website.directory_probe_query_he is None
    assert website.extraction_strategy is None
    # Original column metadata preserved
    assert website.label_he == "אתר"


# ── Phase 0D: plan_entity_discovery ──────────────────────────────────────────

def test_plan_entity_discovery_parses_audit_issues():
    claude = MockClaude().on("plan_entity_discovery", {
        "query_he": "10 הערים הגדולות בישראל לפי אוכלוסייה 2023",
        "query_en": "top 10 cities in Israel by population 2023",
        "expected_count": 10,
        "extraction_hint": "ordered list of city names",
        "audit_issues": [{
            "issue_kind": "missing_anchor",
            "explanation_he": "אין שנת ייחוס",
            "explanation_en": "No year anchor",
            "suggested_fix_he": "ציין שנה",
            "suggested_fix_en": "Specify a year",
        }],
    })
    out = plan_entity_discovery("largest cities in Israel", "city", claude)
    assert out.expected_count == 10
    assert out.query_he.startswith("10 הערים")
    assert len(out.audit_issues) == 1
    assert out.audit_issues[0].issue_kind == "missing_anchor"


def test_plan_entity_discovery_passes_when_question_is_clean():
    claude = MockClaude().on("plan_entity_discovery", {
        "query_he": "q", "query_en": "q",
        "expected_count": None, "extraction_hint": "list",
        "audit_issues": [],
    })
    out = plan_entity_discovery("clean question", "", claude)
    assert out.audit_issues == []
    assert out.expected_count is None
