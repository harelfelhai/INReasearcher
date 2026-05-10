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
from .memory import SuccessMemory, format_compiler_examples

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
You are a domain-agnostic research quality-control agent.

Your job: decide if a research prompt is EXECUTABLE — specific enough that
an automated agent can produce reliable, sourced results across ANY domain
(legal, corporate, academic, governmental, historical, scientific, etc.).

An executable prompt must have:
  1. ENTITY TYPE     — what kind of thing is being researched
                       (a municipality, a person, a company, a court case, a paper…)
  2. SPECIFIC FIELDS — concrete data points to extract
                       (not vague like "tell me about them")
  3. TEMPORAL CONTEXT — if the question is historical or time-sensitive,
                       the year or period must be stated
  4. GEOGRAPHIC / JURISDICTIONAL SCOPE — country, jurisdiction, or "global"
                       (matters because the same entity name may exist in
                       multiple jurisdictions)

If ANY of these are missing or too vague, the prompt is NOT executable.

When not executable, generate:
  - Precise clarifying questions in both Hebrew and English (the user base
    is bilingual; provide both regardless of input language)
  - A concrete prompt template with [PLACEHOLDER] markers to fill in

Be strict. Examples:
  "Tell me about mayors"                              → NOT executable
  "Find recent court cases"                           → NOT executable
  "Look up these companies"                           → NOT executable
  "For each Israeli municipality, find who served
   as mayor in 1990 and link the official record"    → executable
  "For each S&P 500 company in 2023, find the CEO,
   their tenure start date, and a primary source"    → executable"""

_PLAN_SYSTEM = """\
You are a domain-agnostic research planner. Your job is to convert any
executable research question into a structured extraction plan.

═══ SOURCE-AGNOSTIC REASONING (MOST IMPORTANT) ═══

You do NOT have a fixed list of "good sources". For every research question,
first reason about WHAT KIND of source would authoritatively answer it, then
generate queries that surface those sources.

Examples of source-type reasoning (apply analogously to ANY domain):
  - Legal precedent / case law       → court records, legal databases, official rulings
  - Corporate leadership / finances  → SEC/regulatory filings, official press releases, business press
  - Academic / scientific claims     → peer-reviewed journals, university pages, preprint servers
  - Government policy / officials    → official government domains, parliamentary records
  - Historical biographical facts    → encyclopedia entries, archival news, museum/library records
  - Sports / entertainment           → official league sites, established sports/entertainment press
  - Geographic / demographic data    → census bureaus, statistical agencies, mapping services

For each field, populate `preferred_source_domains` with 2-5 SPECIFIC domains
or domain patterns that fit THIS query. Do not default to a generic list.
If the question is about Israeli municipalities, you might choose Israeli
government and Hebrew Wikipedia domains. If it's about US Supreme Court
rulings, you'd choose supremecourt.gov, oyez.org, justia.com. Reason from
the question outward — never from a stock list inward.

═══ SEARCH QUERY RULES ═══

  - Match query language to where the answer most likely lives. Hebrew
    questions about Israeli topics → Hebrew queries primary. English
    questions about global topics → English queries primary. Multilingual
    topics → both, in proportion.
  - Always include {entity} as a placeholder (substituted per entity).
  - For historical questions, embed the year directly in the query.
  - For person-name fields, include role + context + period in queries to
    prevent disambiguation errors with similarly-named people.
  - Write queries the way a real researcher would type them into a search
    engine — natural, not keyword-stuffed.

═══ CORROBORATION RULES ═══

  - Identity facts (names, dates, positions, IDs):  min_corroborations: 2
  - URLs and reference links:                       min_corroborations: 1
  - Descriptive / biographical free text:           min_corroborations: 1
  - Numerical / statistical claims:                 min_corroborations: 2

═══ DEPENDS_ON ═══

If a field logically requires another to be resolved first (e.g. "their
military service" requires knowing the person's name), set depends_on to
that field's id. The pipeline resolves dependencies in order."""


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
    memory: SuccessMemory | None = None,
) -> ClarificationRequest | ExecutableResearchPlan:
    """
    Full Stage 0: preflight → plan (or clarification request).

    If `memory` is provided, retrieves up to 3 similar past validated plans
    and injects them as few-shot examples into the plan-generation prompt.

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

    # Few-shot: pull similar past validated plans from memory
    examples_block = ""
    if memory is not None:
        past = memory.get_compiler_examples(research_question, entity_type, k=3)
        examples_block = format_compiler_examples(past)

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
                f"Entity type: {entity_type or 'infer from question'}\n"
                f"{examples_block}\n\n"
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
