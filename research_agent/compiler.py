"""
Stage 0: Research Compiler

Two-phase pipeline:
  Phase A — Preflight Check:  Is the prompt executable? If not, return
            a ClarificationRequest with targeted questions + a template.
  Phase B — Plan Generation:  Convert the executable prompt into a
            typed ResearchPlan that drives all downstream agents.

Both phases use Claude tool_use so the output is schema-enforced JSON,
not free-form text that might drift.

Model choice: claude-haiku-4-5-20251001
  Rationale: The compiler runs once per research session (cheap),
  and Haiku's instruction-following on structured output is sufficient
  for schema generation. Reserve Sonnet for per-cell extraction.
"""

import anthropic
from .models import (
    ColumnPlan,
    ResearchPlan,
    ClarificationQuestion,
    ClarificationRequest,
    ExecutableResearchPlan,
)

# ── Tool definitions (enforce JSON schema via tool_use) ───────────────────────

_PREFLIGHT_TOOL = {
    "name": "preflight_check",
    "description": (
        "Evaluate whether a research prompt is specific enough to execute. "
        "Return executable=true only when ALL required dimensions are clear."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "is_executable": {
                "type": "boolean",
                "description": "True only if the prompt is specific enough to build a research plan."
            },
            "reason": {
                "type": "string",
                "description": "One sentence: why the prompt is or is not executable."
            },
            "missing_dimensions": {
                "type": "array",
                "description": "List of missing/ambiguous dimensions. Empty if executable.",
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string"},
                        "question_he": {"type": "string"},
                        "question_en": {"type": "string"},
                        "example": {"type": "string"}
                    },
                    "required": ["field", "question_he", "question_en", "example"]
                }
            },
            "prompt_template": {
                "type": "string",
                "description": (
                    "A ready-to-use prompt template the user can copy and fill in. "
                    "Use [PLACEHOLDER] markers. Always in both Hebrew and English."
                )
            }
        },
        "required": ["is_executable", "reason", "missing_dimensions", "prompt_template"]
    }
}

_PLAN_TOOL = {
    "name": "create_research_plan",
    "description": "Convert an executable research question into a structured extraction plan.",
    "input_schema": {
        "type": "object",
        "properties": {
            "entity_type": {
                "type": "string",
                "description": "Type of entity (e.g. 'עיריה ישראלית', 'חבר כנסת', 'company')"
            },
            "columns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "snake_case identifier, e.g. 'mayor_1990'"
                        },
                        "label_he": {"type": "string"},
                        "label_en": {"type": "string"},
                        "type": {
                            "type": "string",
                            "enum": ["person_name", "url", "date", "free_text", "organization", "number"]
                        },
                        "temporal_anchor": {
                            "type": "string",
                            "description": "Year or date if time-specific, e.g. '1990'. Omit if not applicable."
                        },
                        "search_queries_he": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "3-5 Hebrew search queries. Use {entity} as placeholder. "
                                "Write as a real Israeli would type into Google in Hebrew."
                            )
                        },
                        "search_queries_en": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "2-3 English search queries with {entity} placeholder."
                        },
                        "preferred_source_domains": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "e.g. ['he.wikipedia.org', 'knesset.gov.il', 'data.gov.il']"
                        },
                        "min_corroborations": {
                            "type": "integer",
                            "description": (
                                "Independent sources needed: "
                                "2 for identity facts (names, dates), "
                                "1 for URLs and descriptive text."
                            )
                        },
                        "depends_on": {
                            "type": "string",
                            "description": (
                                "ID of a field that must resolve first. "
                                "E.g. IDF service depends on knowing the person's name."
                            )
                        }
                    },
                    "required": [
                        "id", "label_he", "label_en", "type",
                        "search_queries_he", "search_queries_en", "min_corroborations"
                    ]
                }
            }
        },
        "required": ["entity_type", "columns"]
    }
}

# ── System prompts ────────────────────────────────────────────────────────────

_PREFLIGHT_SYSTEM = """\
You are a research quality-control agent for Israeli academic data research.

Your job: decide if a research prompt is EXECUTABLE — specific enough that an
automated agent can produce reliable, sourced results.

An executable prompt must have:
  1. ENTITY TYPE   — what kind of thing is being researched (municipality, person, company…)
  2. SPECIFIC FIELDS — concrete data points to find (not vague like "tell me about them")
  3. TEMPORAL CONTEXT — if the question is historical, the year or period must be stated
  4. SCOPE — is this Israeli-specific? Global? Both?

If ANY of these are missing or too vague, the prompt is NOT executable.

When not executable, generate:
  - Precise clarifying questions in both Hebrew and English
  - A concrete prompt template with [PLACEHOLDER] markers the user can fill in

Be strict. "Tell me about Israeli mayors" is not executable.
"For each Israeli municipality, find who served as mayor in 1990" IS executable."""

_PLAN_SYSTEM = """\
You are a research planner for Israeli academic and government data research.

Your job: convert an executable research question into a structured extraction plan.

Rules for search queries:
  - Hebrew queries are PRIMARY. Write them exactly as an Israeli researcher would
    type into Google — colloquial, with correct Hebrew spelling, no transliteration.
  - Always include {entity} as a placeholder (will be substituted per entity).
  - If the field is historical, embed the year directly in the query.
  - For person-name fields: include role + location + year in queries to avoid disambiguation errors.

Rules for corroboration:
  - Identity fields (names, dates, positions): min_corroborations: 2
  - URLs and reference links: min_corroborations: 1 (just verify HTTP 200)
  - Free-text biographical fields: min_corroborations: 1

Rules for depends_on:
  - If extracting IDF service requires knowing the person's name first, set
    depends_on to the name field's id. The pipeline will resolve fields in order.

Preferred Israeli sources (use in preferred_source_domains where relevant):
  he.wikipedia.org, knesset.gov.il, data.gov.il, gov.il, nevo.co.il, ynet.co.il"""


# ── Public API ────────────────────────────────────────────────────────────────

def run_preflight(
    research_question: str,
    entity_type: str,
    client: anthropic.Anthropic,
) -> dict:
    """
    Phase A: Check if prompt is executable.
    Returns raw dict from the preflight_check tool.
    """
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=_PREFLIGHT_SYSTEM,
        tools=[_PREFLIGHT_TOOL],
        tool_choice={"type": "any"},
        messages=[{
            "role": "user",
            "content": (
                f"Research question: {research_question}\n"
                f"Entity type hint: {entity_type or 'not specified'}\n\n"
                "Evaluate if this is executable."
            )
        }]
    )
    tool_block = next(b for b in response.content if b.type == "tool_use")
    return tool_block.input


def compile_research_plan(
    research_question: str,
    entity_type: str,
    client: anthropic.Anthropic,
) -> ClarificationRequest | ExecutableResearchPlan:
    """
    Full Stage 0: preflight → plan (or clarification request).

    Returns either:
      ExecutableResearchPlan  — ready for Stage 1
      ClarificationRequest    — needs user input before proceeding
    """
    preflight = run_preflight(research_question, entity_type, client)

    if not preflight["is_executable"]:
        questions = [
            ClarificationQuestion(**q)
            for q in preflight.get("missing_dimensions", [])
        ]
        return ClarificationRequest(
            is_executable=False,
            reason=preflight["reason"],
            questions=questions,
            prompt_template=preflight.get("prompt_template", ""),
        )

    # Phase B: generate the full plan
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        system=_PLAN_SYSTEM,
        tools=[_PLAN_TOOL],
        tool_choice={"type": "any"},
        messages=[{
            "role": "user",
            "content": (
                f"Research question: {research_question}\n"
                f"Entity type: {entity_type or 'infer from question'}\n\n"
                "Create a detailed extraction plan."
            )
        }]
    )
    tool_block = next(b for b in response.content if b.type == "tool_use")
    data = tool_block.input

    columns = [
        ColumnPlan(**{k: v for k, v in col.items() if v is not None})
        for col in data["columns"]
    ]
    plan = ResearchPlan(
        entity_type=data["entity_type"],
        research_question_original=research_question,
        columns=columns,
    )
    return ExecutableResearchPlan(is_executable=True, plan=plan)
