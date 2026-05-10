"""
Stage 1: Search → Extract with Provenance

For each (entity, field) pair:
  1. Generate targeted search queries (Hebrew primary, English fallback)
  2. Run Tavily search with raw content enabled
  3. For each result page, ask Claude to extract the value — grounded in the text
  4. Run the grounding check (is_grounded) to verify the quote actually exists
  5. Return all ExtractionResult objects to the verifier

Model: claude-sonnet-4-6
  Rationale: extraction requires precise instruction-following in Hebrew.
  Haiku drifts more on the "return null" rule; Sonnet is more disciplined.
  This is where most token cost lives — budget accordingly (~$0.02-0.05/field).

Cost control levers:
  - max_results caps how many pages we extract from per field
  - early_stop_on_high: stops searching once we hit N grounded results
  - Source content is truncated to 4000 chars (covers most useful content)
"""

import re
import anthropic
from tavily import TavilyClient

from .models import ColumnPlan, ExtractionResult
from .hebrew_utils import is_grounded

# ── Extractor tool (structured output enforcement) ────────────────────────────

_EXTRACTOR_TOOL = {
    "name": "extract_field",
    "description": "Extract a specific data field from provided source text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "value": {
                "type": ["string", "null"],
                "description": (
                    "The extracted value in the language of the source. "
                    "NULL if the information is not clearly present in this text."
                )
            },
            "quote_original": {
                "type": ["string", "null"],
                "description": (
                    "The exact sentence or short phrase from the source text that "
                    "proves this value. DIRECT COPY-PASTE — no paraphrasing. "
                    "NULL if value is null."
                )
            },
            "extractor_confidence": {
                "type": "number",
                "description": "0.0–1.0 confidence that value is correct and fully grounded.",
                "minimum": 0.0,
                "maximum": 1.0
            },
            "not_found_reason": {
                "type": ["string", "null"],
                "description": (
                    "If value is null, briefly explain why: "
                    "'wrong time period', 'entity not mentioned', 'ambiguous entity', "
                    "'information not in text', etc."
                )
            }
        },
        "required": ["value", "quote_original", "extractor_confidence"]
    }
}

_EXTRACTOR_SYSTEM = """\
You are a precision data extractor. You read arbitrary source text — from any
website, in any domain (legal, corporate, academic, governmental, news,
encyclopedic) — and extract specific facts. Nothing more, nothing less.

You are SOURCE-AGNOSTIC. The text may come from a court ruling, a Wikipedia
article, a SEC filing, a Hebrew news site, a PDF excerpt, or a personal blog.
Your job is identical regardless of source structure or domain.

ABSOLUTE RULES (violating them destroys data integrity):

1. GROUNDING ONLY
   You may return a value ONLY if it is explicitly stated in the provided text.
   No background knowledge. No inference. No "it's probably X because...".

2. VERBATIM QUOTE
   The quote_original field must be a direct copy-paste from the text — exactly
   as written, in the original language. Do NOT translate, summarize, or paraphrase.

3. NULL IS CORRECT
   If the information is not clearly in the text, return null for both value
   and quote_original. A confident null beats a hallucinated value every time.

4. ENTITY SPECIFICITY
   Confirm the text refers to the exact entity named in the question.
   If the text mentions a different person/place/case/company with a similar
   name, return null.

5. TEMPORAL SPECIFICITY
   When a year or period is specified, the extracted fact must refer to that
   period. A source saying "served 1995-2000" does NOT answer "served in 1990".

6. SOURCE QUALITY IS NOT YOUR JOB
   Do not refuse to extract based on perceived source quality. Extract what
   the text says; the verifier weighs source authority across multiple results."""


def extract_from_source(
    field: ColumnPlan,
    entity: str,
    source_url: str,
    source_content: str,
    resolved_deps: dict,
    client: anthropic.Anthropic,
) -> ExtractionResult:
    """
    Ask Claude to extract one field from one source page.
    Applies the grounding check before accepting the result.
    """
    dep_context = ""
    if field.depends_on and field.depends_on in resolved_deps:
        dep_val = resolved_deps[field.depends_on]
        dep_context = (
            f"\nContext: The '{field.depends_on}' for this entity has been "
            f"identified as: {dep_val}. Use this to disambiguate if needed."
        )

    temporal_note = (
        f"\nTEMPORAL CONSTRAINT: Extract information specifically about "
        f"the year/period '{field.temporal_anchor}'. Ignore data from other periods."
        if field.temporal_anchor else ""
    )

    user_prompt = (
        f"Entity: {entity}\n"
        f"Field to extract: {field.label_en} / {field.label_he}\n"
        f"Field type: {field.type}"
        f"{temporal_note}"
        f"{dep_context}\n\n"
        f"Source URL: {source_url}\n"
        f"Source text:\n---\n{source_content[:4000]}\n---\n\n"
        f"Extract '{field.label_en}' for entity '{entity}' from the text above. "
        f"Follow all grounding rules strictly. Return null if not clearly present."
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=_EXTRACTOR_SYSTEM,
        tools=[_EXTRACTOR_TOOL],
        tool_choice={"type": "any"},
        messages=[{"role": "user", "content": user_prompt}]
    )

    tool_block = next(b for b in response.content if b.type == "tool_use")
    data = tool_block.input

    raw_value = data.get("value")
    raw_quote = data.get("quote_original")
    llm_confidence = float(data.get("extractor_confidence", 0.0))

    # Grounding gate: reject if the quote or value can't be found in source
    grounded = False
    verified_quote = None
    if raw_value and raw_quote:
        grounded, verified_quote = is_grounded(raw_quote, source_content)
        if not grounded:
            # Secondary: maybe the quote has formatting artifacts — check value directly
            grounded, verified_quote = is_grounded(raw_value, source_content)
    elif raw_value:
        grounded, verified_quote = is_grounded(raw_value, source_content)

    return ExtractionResult(
        field_id=field.id,
        value=raw_value if grounded else None,
        quote_original=verified_quote if grounded else None,
        source_url=source_url,
        source_domain=_domain(source_url),
        is_grounded=grounded,
        extractor_confidence=llm_confidence if grounded else 0.0,
    )


def search_and_extract(
    field: ColumnPlan,
    entity: str,
    resolved_deps: dict,
    tavily: TavilyClient,
    claude: anthropic.Anthropic,
    max_results: int = 6,
    early_stop_on_high: int = 3,
) -> list[ExtractionResult]:
    """
    Run search queries for a field, extract from each result.

    Query order: Hebrew queries first (better for Israeli sources),
    then English. Stops early if we accumulate enough grounded results.
    """
    # Substitute entity name into query templates
    queries_he = [q.replace("{entity}", entity) for q in field.search_queries_he]
    queries_en = [q.replace("{entity}", entity) for q in field.search_queries_en[:2]]

    # If depends_on resolved to a person name, also search by that name
    if field.depends_on and field.depends_on in resolved_deps:
        dep_val = resolved_deps[field.depends_on]
        if dep_val:
            bonus = [q.replace("{entity}", dep_val) for q in field.search_queries_he[:2]]
            queries_he = queries_he + bonus

    all_queries = queries_he + queries_en

    results: list[ExtractionResult] = []
    seen_urls: set[str] = set()

    for query in all_queries[:6]:    # hard cap: 6 searches per field
        try:
            response = tavily.search(
                query=query,
                max_results=3,
                include_raw_content=True,
                search_depth="advanced",
            )
        except Exception as exc:
            print(f"    [search error] {query[:60]!r}: {exc}")
            continue

        for hit in response.get("results", []):
            url = hit.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            content = hit.get("raw_content") or hit.get("content", "")
            if not content or len(content) < 80:
                continue

            extraction = extract_from_source(
                field=field,
                entity=entity,
                source_url=url,
                source_content=content,
                resolved_deps=resolved_deps,
                client=claude,
            )

            if extraction.is_grounded and extraction.value:
                results.append(extraction)
                if len(results) >= early_stop_on_high:
                    return results   # enough corroborations, stop burning tokens

    return results


def _domain(url: str) -> str:
    m = re.search(r'https?://([^/]+)', url)
    return m.group(1) if m else url
