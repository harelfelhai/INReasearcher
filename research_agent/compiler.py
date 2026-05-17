"""
Stage 0: Research Compiler (multi-phase)

Pipeline:
  Phase A — Preflight Check     : Is the prompt executable at all?
  Phase B1 — Schema Generation  : Convert prompt → bare column schema
                                  (id, label, type, temporal_anchor only).
  Phase C — Field-clarity Audit : For each column, check if it's answerable.
                                  Flag unbounded / subjective / ambiguous fields.
  Phase D — Mock-data Preview   : Generate fake but plausible rows so the user
                                  can sanity-check the table shape before any
                                  real searches.
  Phase B2 — Query Enrichment   : Add search_queries_he/en, preferred_source_domains,
                                  min_corroborations. Only run AFTER user approves
                                  the schema, so we never spend tokens on a
                                  schema we'll throw away.

Model choice: claude-haiku-4-5-20251001 for all compiler phases.
"""

import anthropic
from .models import (
    ColumnPlan,
    ResearchPlan,
    ClarificationQuestion,
    ClarificationRequest,
    ExecutableResearchPlan,
    ExtractionStrategy,
    EntityDiscoveryAuditIssue,
    EntityDiscoveryPlan,
    FieldAuditIssue,
    FieldAuditReport,
    MockRow,
)
from .memory import SuccessMemory, format_compiler_examples


_HAIKU = "claude-haiku-4-5-20251001"


# ── Phase A — Preflight ──────────────────────────────────────────────────────

_PREFLIGHT_TOOL = {
    "name": "preflight_check",
    "description": (
        "Evaluate whether a research prompt is specific enough to execute. "
        "Return executable=true only when ALL required dimensions are clear."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "is_executable": {"type": "boolean"},
            "reason": {"type": "string"},
            "missing_dimensions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string"},
                        "question_he": {"type": "string"},
                        "question_en": {"type": "string"},
                        "example": {"type": "string"},
                    },
                    "required": ["field", "question_he", "question_en", "example"],
                },
            },
            "prompt_template": {"type": "string"},
        },
        "required": ["is_executable", "reason", "missing_dimensions", "prompt_template"],
    },
}

_PREFLIGHT_SYSTEM = """\
You are a domain-agnostic research quality-control agent.

Decide if a research prompt is EXECUTABLE — specific enough that an automated
agent can produce reliable sourced results.

An executable prompt must have:
  1. ENTITY TYPE — what kind of thing is being researched
  2. SPECIFIC FIELDS — concrete data points to extract
  3. TEMPORAL CONTEXT — year/period if historical or time-sensitive
  4. GEOGRAPHIC / JURISDICTIONAL SCOPE

IMPORTANT — entity list handling:
  - If the user prompt is accompanied by an explicit entity list (you will be
    told the count and shown a sample), treat the ENTITY TYPE and ENTITY
    SPECIFICATION dimensions as SATISFIED. The user has named the exact
    entities — do NOT ask which ones, do NOT ask for a list. You may still
    ask about TEMPORAL CONTEXT or FIELD AMBIGUITY if those are missing.
  - Only ask "which entities?" when no list is provided.

If any required dimension is missing, generate bilingual clarifying questions.
Each question MUST be written in Hebrew in `question_he` (clear, natural Hebrew
— do NOT leave English in the Hebrew field). The `field` value is a short
machine code (e.g. 'timeframe', 'field_ambiguity') — the frontend maps it to a
Hebrew label, so keep it stable.

Also provide a [PLACEHOLDER]-marked prompt template the user can fill in.

Examples:
  "Tell me about mayors"                              → NOT executable
  "For each Israeli municipality, find who served
   as mayor in 1990 and the official record URL"     → executable
  "Find the party of these MKs" + list of 20 names    → executable
                                                       (entities are supplied)"""


# ── Phase B1 — Schema only (no queries) ──────────────────────────────────────

_SCHEMA_TOOL = {
    "name": "create_schema",
    "description": (
        "Convert an executable research question into a BARE column schema. "
        "Do NOT generate search queries — that happens in a later phase."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "entity_type": {"type": "string"},
            "columns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "snake_case identifier"},
                        "label_he": {"type": "string"},
                        "label_en": {"type": "string"},
                        "type": {
                            "type": "string",
                            "enum": ["person_name", "url", "date", "free_text", "organization", "number"],
                        },
                        "volatility": {
                            "type": "string",
                            "enum": ["stable", "volatile"],
                            "description": (
                                "'stable' = historical or slowly-changing fact (founding year, "
                                "historical officeholder, country of origin). "
                                "'volatile' = can change over time (current CEO, price, "
                                "contact info, operating status, opening hours, current address). "
                                "When in doubt: if a source from 3 years ago could be wrong today, use 'volatile'."
                            ),
                        },
                        "temporal_anchor": {
                            "type": "string",
                            "description": (
                                "Year or period for historical fields, e.g. '1990'. "
                                "Omit only for current/timeless data."
                            ),
                        },
                        "depends_on": {"type": "string"},
                    },
                    "required": ["id", "label_he", "label_en", "type", "volatility"],
                },
            },
        },
        "required": ["entity_type", "columns"],
    },
}

_SCHEMA_SYSTEM = """\
You are a domain-agnostic research planner. Convert an executable research
question into a BARE column schema — column id, bilingual label, value type,
optional temporal anchor, optional depends_on. Do not generate search queries
or source domains — those come later.

Naming rules:
  - id: snake_case, descriptive, include temporal anchor if any
        (e.g. 'mayor_1990', not 'mayor')
  - label_he / label_en: human-readable column headers
  - volatility: REQUIRED. 'stable' for historical/slowly-changing facts;
        'volatile' for anything that could be wrong if the source is 3 years old
        (current officeholders, contact info, prices, hours, status, addresses).
        Fields with a temporal_anchor are almost always 'stable'.
  - temporal_anchor: REQUIRED for any historical or time-bounded field
  - depends_on: id of a field that must resolve first
        (e.g. 'idf_service' depends_on 'full_name')

Keep the schema minimal. Don't invent fields the user didn't ask for."""


# ── Phase C — Field-clarity Audit ────────────────────────────────────────────

_AUDIT_TOOL = {
    "name": "audit_schema",
    "description": (
        "Inspect each column in a schema and flag clarity issues that would "
        "make the field unanswerable, subjective, or ambiguously formatted."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "all_clear": {"type": "boolean"},
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "field_id": {"type": "string"},
                        "issue_kind": {
                            "type": "string",
                            "enum": [
                                "unbounded",
                                "subjective",
                                "missing_anchor",
                                "ambiguous_format",
                                "ambiguous_scope",
                                "no_canonical_source",
                            ],
                        },
                        "explanation_he": {"type": "string"},
                        "explanation_en": {"type": "string"},
                        "suggested_fix_he": {"type": "string"},
                        "suggested_fix_en": {"type": "string"},
                    },
                    "required": [
                        "field_id", "issue_kind",
                        "explanation_he", "explanation_en",
                        "suggested_fix_he", "suggested_fix_en",
                    ],
                },
            },
        },
        "required": ["all_clear", "issues"],
    },
}

_AUDIT_SYSTEM = """\
You audit a generated research schema for FIELD-LEVEL clarity. For each column,
ask:

  1. UNBOUNDED — is the answer a single value or a bounded list?
     FAIL: "career_path" (every job ever held), "achievements"
     PASS: "highest_degree", "previous_job_before_mayor"

  2. SUBJECTIVE — is there an objective answer that two researchers would agree on?
     FAIL: "best_policy", "most_influential_decision"
     PASS: "election_year", "vote_count"

  3. MISSING_ANCHOR — does a historical field have a year/period?
     FAIL: column "mayor" with no temporal_anchor
     PASS: column "mayor_1990" with temporal_anchor="1990"

  4. AMBIGUOUS_FORMAT — is the value format obvious?
     FAIL: "name" — full? nickname? Hebrew/English?
     PASS: "full_name_hebrew", "official_website_url"

     IMPORTANT EXCEPTION: do NOT flag AMBIGUOUS_FORMAT for a field that
     represents the entity's own name/identity when the entity_type is clearly
     a person (MK, politician, mayor, public figure, etc.). In that context
     "name" means the person's official registered name — unambiguous in practice.
     A researcher looking up a named individual always knows which name is meant.

  5. AMBIGUOUS_SCOPE — would two researchers extract the SAME granularity?
     Even if the answer is a single string and objective, the SEMANTIC SCOPE
     can be ambiguous: which slice of reality does this field point at?
     FAIL: "previous_career" — one job? a sector? years of experience?
           "education" — highest degree? full history? institution?
           "military_service" — unit? rank? years? role description?
     PASS: "most_recent_job_title_before_election",
           "highest_academic_degree",
           "idf_unit_name"
     Be strict here — vague nouns like "career", "background", "education",
     "service", "experience" almost always need a sharper anchor.

  6. NO_CANONICAL_SOURCE — would a researcher know where to look?
     FAIL: "personal_opinion_about_X"
     PASS: any field with a likely authoritative source

For EACH failing field emit a FieldAuditIssue with bilingual explanation and a
concrete suggested fix. Set all_clear=true ONLY if no issues. Be strict —
catching a bad field now saves dozens of wasted API calls later.

If the schema is well-formed, return all_clear=true with empty issues array."""


# ── Phase D — Mock-data Preview ──────────────────────────────────────────────

_MOCK_TOOL = {
    "name": "generate_mock_rows",
    "description": (
        "Generate 2-3 PLAUSIBLE BUT FAKE rows for the given schema, so the user "
        "can sanity-check the table shape before any real searches run."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "entity_name": {"type": "string"},
                        "values": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                            "description": "Map of field_id → mock value (string)",
                        },
                    },
                    "required": ["entity_name", "values"],
                },
            },
        },
        "required": ["rows"],
    },
}

_MOCK_SYSTEM = """\
You generate 2-3 MOCK (fake) rows so the user can see what their result table
will look like before any real searches run.

Rules:
  - Use realistic entity names from the relevant domain (e.g. real Israeli
    cities if the entity_type is 'עיריה ישראלית').
  - Values must be PLAUSIBLE but clearly synthetic. They are illustrations of
    FORMAT, not real facts. Suffix every value with " (דוגמה)" / "(example)"
    so the user knows it's mock.
  - Match every field_id in the schema. Format values consistent with the
    field type (URL → looks like a URL, person_name → looks like a name).
  - 2-3 rows is enough."""


# ── Phase B2 — Query Enrichment ──────────────────────────────────────────────

_QUERIES_TOOL = {
    "name": "enrich_with_queries",
    "description": (
        "Given an approved schema, add search queries, preferred source domains, "
        "and corroboration requirements to each column."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "columns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "search_queries_he": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "3-5 Hebrew queries with {entity} placeholder",
                        },
                        "search_queries_en": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "2-3 English queries with {entity} placeholder",
                        },
                        "preferred_source_domains": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "min_corroborations": {"type": "integer"},
                        "directory_probe_query_he": {
                            "type": "string",
                            "description": (
                                "ONE entity-agnostic Hebrew query to find a list/table page "
                                "covering this field for MANY entities at once. "
                                "No {entity} placeholder. "
                                "Example: 'רשימת ראשי ערים ישראל 1990' or 'ראשי עיריות ישראל 1990 טבלה'."
                            ),
                        },
                        "extraction_strategy": {
                            "type": "object",
                            "description": (
                                "Optional hints that help the page-windowing algorithm "
                                "find the relevant section of a long page. All fields "
                                "optional — leave a field out when no good hint exists."
                            ),
                            "properties": {
                                "value_regex": {
                                    "type": "string",
                                    "description": (
                                        "Python regex matching the literal value format. "
                                        "Be CONSERVATIVE — too-narrow regex hurts recall. "
                                        "Examples: r'https?://\\S+\\.(gov|muni|org)\\.il' for "
                                        "Israeli gov URLs; r'\\b\\d{4}\\b' for years. "
                                        "Leave blank for free_text and person_name."
                                    ),
                                },
                                "value_anchors_he": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": (
                                        "2-5 Hebrew phrases that typically appear within "
                                        "~100 chars of the value. CONTEXT cues, not the "
                                        "value itself. e.g. for mayor 1990: "
                                        "['ראש העיר', 'כיהן בין', 'נבחר לראש']"
                                    ),
                                },
                                "value_anchors_en": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": (
                                        "2-5 English context phrases. e.g. for founding "
                                        "year: ['founded in', 'established in', 'incorporated']"
                                    ),
                                },
                            },
                        },
                    },
                    "required": [
                        "id",
                        "search_queries_he",
                        "search_queries_en",
                        "min_corroborations",
                    ],
                },
            },
        },
        "required": ["columns"],
    },
}

_QUERIES_SYSTEM = """\
You enrich an APPROVED schema with search queries and source preferences.

═══ Source-type reasoning ═══
For each field, reason about what kind of source would authoritatively answer
it (court records / regulatory filings / encyclopedias / government domains /
…) and populate preferred_source_domains with 2-5 SPECIFIC domains for THIS
query. Do not default to a stock list.

═══ Query rules ═══
  - {entity} placeholder is required.
  - Hebrew queries primary for Israeli topics; English primary for global.
  - For historical fields, embed the year directly.
  - For person fields, include role + period to prevent disambiguation.
  - Write queries the way a real researcher would type — natural, not
    keyword-stuffed.

═══ Corroboration ═══
  - Identity facts (names, dates, IDs):  min_corroborations: 2
  - URLs / reference links:              min_corroborations: 1
  - Free text / descriptive:             min_corroborations: 1
  - Numerical / statistical:             min_corroborations: 2

═══ Extraction strategy ═══
For each field, optionally fill extraction_strategy to guide page windowing:
  - value_regex: a Python regex matching the value's literal format.
    BE CONSERVATIVE — bad regex tanks recall. Good cases:
      url (Israeli gov):  r'https?://\S+\.(gov|muni|org)\.il'
      year:               r'\b\d{4}\b'
      ID number:          r'\b\d{8,9}\b'
    Skip for free_text and person_name (formats too variable).
  - value_anchors_he / value_anchors_en: 2-5 short context phrases that
    typically appear NEAR the value (within ~100 chars). NOT the value itself.
    Examples for mayor in 1990: ['ראש העיר', 'כיהן בין', 'served as mayor']
    Examples for founding year: ['נוסדה', 'הוקמה בשנת', 'founded in']
  - Omit any sub-field you don't have a confident hint for.

═══ Directory probe query ═══
For EVERY field, also generate directory_probe_query_he — a single entity-agnostic
Hebrew search query that might find a list/table page covering this field for many
entities at once. This enables a major cost optimisation (one page → N entities).
  - Do NOT include {entity}.
  - Include the entity type, field label, and temporal anchor.
  - Use list-oriented language: "רשימת", "טבלה", "לפי שנה", "כל ה-".
  Examples:
    person_name + temporal_anchor=1990 → "רשימת ראשי ערים ישראל 1990"
    url, entity_type=municipality     → "אתרים רשמיים עיריות ישראל"
    number, population                → "אוכלוסיית ערים ישראל לפי שנה" """


# ── Helpers ──────────────────────────────────────────────────────────────────

def _call_tool(client: anthropic.Anthropic, system: str, tool: dict, user: str, max_tokens: int = 2048, temperature: float | None = None) -> dict:
    kwargs = dict(
        model=_HAIKU,
        max_tokens=max_tokens,
        system=system,
        tools=[tool],
        tool_choice={"type": "any"},
        messages=[{"role": "user", "content": user}],
    )
    if temperature is not None:
        kwargs["temperature"] = temperature
    response = client.messages.create(**kwargs)
    tool_block = next(b for b in response.content if b.type == "tool_use")
    return tool_block.input


# ── Public API ───────────────────────────────────────────────────────────────

def run_preflight(
    research_question: str,
    entity_type: str,
    client: anthropic.Anthropic,
    entities: list[str] | None = None,
) -> dict:
    """Phase A: returns raw preflight_check tool output."""
    entities = entities or []
    if entities:
        sample = ", ".join(entities[:5])
        more = f" (+{len(entities) - 5} more)" if len(entities) > 5 else ""
        entity_block = (
            f"Entity list provided by user: {len(entities)} items.\n"
            f"Sample: {sample}{more}\n"
            "→ ENTITY TYPE and ENTITY SPECIFICATION are SATISFIED. "
            "Do NOT ask which entities to research.\n"
        )
    else:
        entity_block = "No entity list provided.\n"
    return _call_tool(
        client,
        _PREFLIGHT_SYSTEM,
        _PREFLIGHT_TOOL,
        f"Research question: {research_question}\n"
        f"Entity type hint: {entity_type or 'not specified'}\n"
        f"{entity_block}\n"
        "Evaluate if this is executable.",
        max_tokens=1024,
    )


def compile_schema(
    research_question: str,
    entity_type: str,
    client: anthropic.Anthropic,
    memory: SuccessMemory | None = None,
    entities: list[str] | None = None,
) -> ClarificationRequest | ExecutableResearchPlan:
    """
    Phase A → Phase B1. Returns either:
      ClarificationRequest      — prompt failed preflight
      ExecutableResearchPlan    — bare schema (no queries yet)
    """
    preflight = run_preflight(research_question, entity_type, client, entities=entities)

    if not preflight["is_executable"]:
        return ClarificationRequest(
            is_executable=False,
            reason=preflight["reason"],
            questions=[
                ClarificationQuestion(**q)
                for q in preflight.get("missing_dimensions", [])
            ],
            prompt_template=preflight.get("prompt_template", ""),
        )

    examples_block = ""
    if memory is not None:
        past = memory.get_compiler_examples(research_question, entity_type, k=3)
        examples_block = format_compiler_examples(past)

    data = _call_tool(
        client,
        _SCHEMA_SYSTEM,
        _SCHEMA_TOOL,
        f"Research question: {research_question}\n"
        f"Entity type: {entity_type or 'infer from question'}\n"
        f"{examples_block}\n\n"
        "Create a bare column schema (no search queries yet).",
    )

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


def audit_schema(
    plan: ResearchPlan,
    client: anthropic.Anthropic,
) -> FieldAuditReport:
    """Phase C: per-field clarity audit."""
    columns_summary = "\n".join(
        f"  - id={c.id}, label_he={c.label_he!r}, label_en={c.label_en!r}, "
        f"type={c.type}, temporal_anchor={c.temporal_anchor!r}"
        for c in plan.columns
    )
    data = _call_tool(
        client,
        _AUDIT_SYSTEM,
        _AUDIT_TOOL,
        f"Research question: {plan.research_question_original}\n"
        f"Entity type: {plan.entity_type}\n"
        f"Columns:\n{columns_summary}\n\n"
        "Audit each column.",
        max_tokens=1500,
        temperature=0,
    )
    return FieldAuditReport(
        all_clear=data["all_clear"],
        issues=[FieldAuditIssue(**i) for i in data.get("issues", [])],
    )


def generate_mock_rows(
    plan: ResearchPlan,
    client: anthropic.Anthropic,
) -> list[MockRow]:
    """Phase D: 2-3 fake rows so the user can sanity-check the table shape."""
    columns_summary = "\n".join(
        f"  - id={c.id}, label_he={c.label_he!r}, type={c.type}, "
        f"temporal_anchor={c.temporal_anchor!r}"
        for c in plan.columns
    )
    data = _call_tool(
        client,
        _MOCK_SYSTEM,
        _MOCK_TOOL,
        f"Entity type: {plan.entity_type}\n"
        f"Research question: {plan.research_question_original}\n"
        f"Columns:\n{columns_summary}\n\n"
        "Generate 2-3 mock rows.",
        max_tokens=1200,
    )
    rows = []
    for r in data.get("rows", []):
        if not isinstance(r, dict):
            continue
        try:
            rows.append(MockRow(**r))
        except (TypeError, ValueError):
            continue
    return rows


def enrich_with_queries(
    plan: ResearchPlan,
    client: anthropic.Anthropic,
    clarifications: dict[str, str] | None = None,
) -> ResearchPlan:
    """
    Phase B2: add search queries, preferred domains, and corroboration counts
    to an APPROVED schema. Mutates a copy and returns it.

    clarifications: optional {field_id → user note} injected into the prompt
    so Claude generates more precise queries for flagged fields.
    """
    clarifications = clarifications or {}
    columns_summary = "\n".join(
        f"  - id={c.id}, label_he={c.label_he!r}, label_en={c.label_en!r}, "
        f"type={c.type}, temporal_anchor={c.temporal_anchor!r}"
        + (f"  [USER CLARIFICATION: {clarifications[c.id]}]" if c.id in clarifications else "")
        for c in plan.columns
    )
    data = _call_tool(
        client,
        _QUERIES_SYSTEM,
        _QUERIES_TOOL,
        f"Research question: {plan.research_question_original}\n"
        f"Entity type: {plan.entity_type}\n"
        f"Approved columns:\n{columns_summary}\n\n"
        "Enrich each column with search queries and source preferences.\n"
        "For columns with a USER CLARIFICATION note, generate queries that "
        "specifically target what the clarification describes.",
    )

    enrichment_by_id = {row["id"]: row for row in data["columns"]}
    enriched_columns: list[ColumnPlan] = []
    for col in plan.columns:
        enrich = enrichment_by_id.get(col.id, {})

        strategy_dict = enrich.get("extraction_strategy") or {}
        # Drop empty sub-fields so we don't store noise on the column.
        strategy_clean = {
            k: v for k, v in strategy_dict.items()
            if v not in ("", [], None)
        }
        strategy = ExtractionStrategy(**strategy_clean) if strategy_clean else None

        enriched_columns.append(
            col.model_copy(update={
                "search_queries_he": enrich.get("search_queries_he", []),
                "search_queries_en": enrich.get("search_queries_en", []),
                "preferred_source_domains": enrich.get("preferred_source_domains", []),
                "min_corroborations": enrich.get(
                    "min_corroborations", col.min_corroborations
                ),
                "directory_probe_query_he": enrich.get("directory_probe_query_he") or None,
                "extraction_strategy": strategy,
            })
        )
    return ResearchPlan(
        entity_type=plan.entity_type,
        research_question_original=plan.research_question_original,
        columns=enriched_columns,
    )


# ── Phase 0D — Entity Discovery planner + audit ──────────────────────────────
#
# Used only when the user opts in to auto-discovery. Generates the search query
# that will fetch the canonical list page AND audits the question for issues
# that would make discovery unreliable (unbounded count, ambiguous ranking,
# missing time anchor, etc.). If any audit issue is raised the caller MUST
# surface it before running the actual search — wrong entities are far more
# costly than wrong field values.

_DISCOVERY_PLAN_TOOL = {
    "name": "plan_entity_discovery",
    "description": (
        "Given a research question that asks for an open set of entities "
        "(\"the 10 largest cities in Israel\"), produce the search query that "
        "would surface the canonical ranked list, and audit the question for "
        "issues that would make discovery unreliable."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query_he":         {"type": "string"},
            "query_en":         {"type": "string"},
            "expected_count":   {"type": ["integer", "null"]},
            "extraction_hint":  {"type": "string"},
            "audit_issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "issue_kind": {
                            "type": "string",
                            "enum": [
                                "unbounded_count",
                                "ambiguous_ranking",
                                "missing_anchor",
                                "subjective_criterion",
                                "no_canonical_source",
                            ],
                        },
                        "explanation_he":   {"type": "string"},
                        "explanation_en":   {"type": "string"},
                        "suggested_fix_he": {"type": "string"},
                        "suggested_fix_en": {"type": "string"},
                    },
                    "required": [
                        "issue_kind", "explanation_he", "explanation_en",
                        "suggested_fix_he", "suggested_fix_en",
                    ],
                },
            },
        },
        "required": ["query_he", "query_en", "extraction_hint", "audit_issues"],
    },
}

_DISCOVERY_PLAN_SYSTEM = """\
You are planning an entity-discovery step for an automated research agent.

The user has given a question whose entity list is OPEN (must be discovered).
Your job has two parts:

1. AUDIT the question. Raise an issue when it is not safely answerable:
   - unbounded_count       — no explicit number ("the largest cities")
   - ambiguous_ranking     — "largest"/"best" by what metric?
   - missing_anchor        — rankings drift; needs a year or "as of YYYY"
   - subjective_criterion  — "best", "most popular" with no objective source
   - no_canonical_source   — no single authoritative ranked list exists

   If clean, return audit_issues = [].

2. Build the SEARCH QUERY (Hebrew + English) that will surface the canonical
   ranked list page (Wikipedia, government statistics bureau, etc.). Use
   the year anchor explicitly if present. Set expected_count when the
   question states one.

   The extraction_hint is a short natural-language description of what to
   extract from the resulting page (e.g. "ordered list of city names by
   2023 population"). The downstream extractor will see this.
"""


def plan_entity_discovery(
    research_question: str,
    entity_type: str,
    client: anthropic.Anthropic,
) -> EntityDiscoveryPlan:
    """Phase 0D: produce a search query + audit for the entity question."""
    data = _call_tool(
        client,
        _DISCOVERY_PLAN_SYSTEM,
        _DISCOVERY_PLAN_TOOL,
        f"Research question: {research_question}\n"
        f"Entity type: {entity_type or 'infer from question'}\n\n"
        "Plan the entity-discovery search and audit the question.",
        max_tokens=1024,
    )
    issues = [EntityDiscoveryAuditIssue(**i) for i in data.get("audit_issues", [])]
    return EntityDiscoveryPlan(
        query_he=data["query_he"],
        query_en=data["query_en"],
        expected_count=data.get("expected_count"),
        extraction_hint=data.get("extraction_hint"),
        audit_issues=issues,
    )


# ── Back-compat shim ─────────────────────────────────────────────────────────

def compile_research_plan(
    research_question: str,
    entity_type: str,
    client: anthropic.Anthropic,
    memory: SuccessMemory | None = None,
) -> ClarificationRequest | ExecutableResearchPlan:
    """
    Legacy single-shot compile: schema + queries in one go.
    New flows should call compile_schema → audit_schema → enrich_with_queries.
    """
    result = compile_schema(research_question, entity_type, client, memory)
    if isinstance(result, ClarificationRequest):
        return result
    enriched = enrich_with_queries(result.plan, client)
    return ExecutableResearchPlan(is_executable=True, plan=enriched)
