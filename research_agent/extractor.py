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

import io
import re
import time
import urllib.request
import anthropic
from concurrent.futures import ThreadPoolExecutor
from tavily import TavilyClient

from .models import ColumnPlan, ExtractionResult
from .hebrew_utils import is_grounded
from .memory import (
    SuccessMemory,
    format_extraction_examples,
    format_extraction_warnings,
)
from .tracer import NullTracer

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


# ── Smart content windowing ───────────────────────────────────────────────────

# Type-specific keywords used to locate relevant passages within long articles.
# Multilingual: Hebrew is primary for Israeli sources, English for global.
_TYPE_KEYWORDS: dict[str, list[str]] = {
    "person_name":   ["ראש העיר", "ראש עיר", "כיהן", "שימש", "נבחר", "מונה",
                      "mayor", "elected", "served", "appointed", "tenure"],
    "url":           ["www.", "http", "אתר", "אתר רשמי", ".gov", ".muni",
                      ".org", ".com", ".il", "website", "official site"],
    "organization":  ["עיריה", "עיריית", "מועצה", "municipality", "council",
                      "city hall"],
    "date":          ["שנת", "בשנת", "תאריך", "year", "date", "established"],
    "number":        ["אוכלוסייה", "תושבים", "population", "total"],
}

_INTRO_CHARS   = 1500    # always include the article intro for context
_CHUNK_SIZE    = 1400    # size of scored chunks for density windowing
_CHUNK_STEP    = 350     # step between chunks (overlapping)
_BUDGET        = 15000   # total chars sent to Claude per extraction
_ELLIPSIS      = "\n\n[...]\n\n"


def _field_keywords(field, entity: str) -> set[str]:
    """Build the keyword set used to score relevance of text chunks for one field."""
    keywords: set[str] = set()
    if entity and len(entity) > 2:
        keywords.add(entity)
    for word in field.label_he.split() + field.label_en.split():
        if len(word) > 3:
            keywords.add(word.lower())
    if field.temporal_anchor:
        keywords.add(field.temporal_anchor)
        try:
            yr = int(field.temporal_anchor)
            for delta in (-3, -2, -1, 1, 2, 3):
                keywords.add(str(yr + delta))
        except ValueError:
            pass
    keywords.update(_TYPE_KEYWORDS.get(field.type, []))
    return keywords


# ── Strategy-aware scoring ───────────────────────────────────────────────────
#
# Field extraction_strategy (from Phase B2) can boost chunks that:
#   - contain a value_regex match               → +5
#   - contain the entity name AND a value_anchor → +3
#
# A bad/over-narrow strategy can hurt recall, so we apply a safety net:
# if max chunk score after strategy boosts is below _STRATEGY_FALLBACK_THRESHOLD,
# we re-score with pure keyword density (existing behavior).

_REGEX_BOOST   = 5
_ANCHOR_BOOST  = 3
_STRATEGY_FALLBACK_THRESHOLD = 2


def _compile_strategies(strategies: list) -> tuple[list, list[str]]:
    """
    Compile regexes (skipping invalid ones) and union all anchors across the
    given strategies. Returns (compiled_regexes, anchors_lowercased).
    """
    compiled = []
    anchors: list[str] = []
    for s in strategies:
        if s is None:
            continue
        if s.value_regex:
            try:
                compiled.append(re.compile(s.value_regex, re.IGNORECASE))
            except re.error as exc:
                print(f"    [window] bad regex {s.value_regex!r}: {exc} — ignored")
        for a in s.value_anchors_he + s.value_anchors_en:
            if len(a) >= 2:
                anchors.append(a.lower())
    return compiled, anchors


def _score_chunk(
    chunk_lower: str,
    chunk_original: str,
    kw_list: list[str],
    regexes: list,
    anchors: list[str],
    entity_lower: str,
) -> int:
    """Compose keyword density + strategy boosts into a single chunk score."""
    score = sum(1 for kw in kw_list if kw in chunk_lower)
    if regexes and any(r.search(chunk_original) for r in regexes):
        score += _REGEX_BOOST
    if anchors and entity_lower and entity_lower in chunk_lower:
        if any(a in chunk_lower for a in anchors):
            score += _ANCHOR_BOOST
    return score


def _select_text_by_keywords(
    content: str,
    keywords: set[str],
    label: str,
    entity: str,
    strategies: list | None = None,
) -> str:
    """
    Density-scored windowing using a pre-built keyword set, optionally
    augmented by ExtractionStrategy hints (regex + anchors). Shared between
    single-field and multi-field extraction.
    """
    if not content:
        return ""
    if len(content) <= _BUDGET:
        return content

    kw_list = [kw.lower() for kw in keywords if len(kw) >= 2]
    content_lower = content.lower()
    entity_lower = entity.lower() if entity else ""

    if not kw_list:
        return content[:_BUDGET]

    regexes, anchors = _compile_strategies(strategies or [])

    def _score_pass(use_strategy: bool) -> list[tuple[int, int, int]]:
        out: list[tuple[int, int, int]] = []
        for start in range(0, len(content), _CHUNK_STEP):
            end = min(start + _CHUNK_SIZE, len(content))
            chunk_lower = content_lower[start:end]
            if use_strategy:
                chunk_original = content[start:end]
                s = _score_chunk(
                    chunk_lower, chunk_original, kw_list, regexes, anchors, entity_lower,
                )
            else:
                s = sum(1 for kw in kw_list if kw in chunk_lower)
            if s > 0:
                out.append((s, start, end))
        return out

    strategy_active = bool(regexes or anchors)
    scored = _score_pass(use_strategy=strategy_active)

    # Safety net: if a strategy was active but produced essentially no signal,
    # re-score with pure keyword density so a bad strategy can't tank recall.
    if strategy_active and (not scored or max(s[0] for s in scored) < _STRATEGY_FALLBACK_THRESHOLD):
        print(f"    [window] strategy gave weak signal for field={label!r}; "
              f"falling back to pure keyword density")
        scored = _score_pass(use_strategy=False)

    if not scored:
        return content[:_BUDGET]

    # Greedy top-K by score, rejecting overlap with already-picked chunks.
    # This lets a high-score chunk far in the document beat a swarm of
    # low-score filler chunks near the top.
    scored.sort(key=lambda x: -x[0])
    intro = content[:_INTRO_CHARS]
    used = len(intro)
    picked: list[tuple[int, int]] = []   # (start, end) — output order normalised at end

    for score, s, e in scored:
        if e <= _INTRO_CHARS:
            continue
        if s < _INTRO_CHARS:
            s = _INTRO_CHARS

        # Reject if this chunk overlaps a higher-scoring chunk we already took.
        if any(s < pe and ps < e for ps, pe in picked):
            continue

        section_len = e - s
        gap = len(_ELLIPSIS)
        if used + section_len + gap > _BUDGET:
            remaining_budget = _BUDGET - used - gap
            if remaining_budget > 200:
                picked.append((s, s + remaining_budget))
                used += remaining_budget + gap
            break

        picked.append((s, e))
        used += section_len + gap

    picked.sort()
    pieces: list[str] = [intro]
    for s, e in picked:
        pieces.append(_ELLIPSIS)
        pieces.append(content[s:e])

    result = "".join(pieces)
    entity_short = entity[:20]
    hit_kws = [kw for kw in kw_list if kw in content_lower][:5]
    print(f"    [window] {entity_short!r} field={label!r}: "
          f"{len(content):,}→{len(result):,} chars, "
          f"top keywords: {hit_kws}")
    return result


def _select_relevant_text(content: str, field, entity: str) -> str:
    """Single-field windowing — uses the field's extraction_strategy if present."""
    strategies = [field.extraction_strategy] if field.extraction_strategy else []
    return _select_text_by_keywords(
        content, _field_keywords(field, entity), field.id, entity,
        strategies=strategies,
    )


def _select_relevant_text_for_fields(content: str, fields: list, entity: str) -> str:
    """
    Multi-field windowing: union of all fields' keyword sets AND strategies, so
    chunks relevant to ANY field still surface. Used by batch_extract_fields_from_source.
    """
    keywords: set[str] = set()
    strategies = []
    for f in fields:
        keywords |= _field_keywords(f, entity)
        if f.extraction_strategy:
            strategies.append(f.extraction_strategy)
    label = ",".join(f.id for f in fields)
    return _select_text_by_keywords(
        content, keywords, label, entity, strategies=strategies,
    )


def extract_from_source(
    field: ColumnPlan,
    entity: str,
    source_url: str,
    source_content: str,
    resolved_deps: dict,
    client: anthropic.Anthropic,
    memory: SuccessMemory | None = None,
    tracer=None,
    publication_date: str | None = None,
) -> ExtractionResult:
    """
    Ask Claude to extract one field from one source page.
    Applies the grounding check before accepting the result.

    If `memory` is provided, injects:
      - up to 2 past validated extractions for this field type (positive few-shot)
      - up to 5 past hallucinations as AVOID warnings (negative few-shot)
    """
    dep_context = ""
    if field.depends_on and field.depends_on in resolved_deps:
        dep_val = resolved_deps[field.depends_on]
        dep_context = (
            f"\nContext: The '{field.depends_on}' for this entity has been "
            f"identified as: {dep_val}. Use this to disambiguate if needed."
        )

    temporal_note = (
        f"\nTEMPORAL CONSTRAINT: The answer must be valid for the year/period "
        f"'{field.temporal_anchor}'. A source stating a date RANGE that includes "
        f"'{field.temporal_anchor}' (e.g. 'served 1974–1993' covers 1990) IS a valid "
        f"answer — extract the value. Only return null if the source's dates clearly "
        f"EXCLUDE '{field.temporal_anchor}', or if no relevant date information exists."
        if field.temporal_anchor else ""
    )

    examples_block = ""
    warnings_block = ""
    if memory is not None:
        good = memory.get_extraction_examples(field.type, field.label_en, k=2)
        examples_block = format_extraction_examples(good)
        bad = memory.get_extraction_warnings(field.type, _domain(source_url))
        warnings_block = format_extraction_warnings(bad)

    user_prompt = (
        f"Entity: {entity}\n"
        f"Field to extract: {field.label_en} / {field.label_he}\n"
        f"Field type: {field.type}"
        f"{temporal_note}"
        f"{dep_context}"
        f"{examples_block}"
        f"{warnings_block}\n\n"
        f"Source URL: {source_url}\n"
        f"Source text:\n---\n{_select_relevant_text(source_content, field, entity)}\n---\n\n"
        f"Extract '{field.label_en}' for entity '{entity}' from the text above. "
        f"Follow all grounding rules strictly. Return null if not clearly present."
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=[{"type": "text", "text": _EXTRACTOR_SYSTEM,
                 "cache_control": {"type": "ephemeral"}}],
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

    result = ExtractionResult(
        field_id=field.id,
        value=raw_value if grounded else None,
        quote_original=verified_quote if grounded else None,
        source_url=source_url,
        source_domain=_domain(source_url),
        is_grounded=grounded,
        extractor_confidence=llm_confidence if grounded else 0.0,
        publication_date=publication_date,
    )

    tr = tracer or NullTracer()
    tr.emit(
        "extraction",
        entity=entity,
        field_id=field.id,
        field_type=field.type,
        url=source_url,
        domain=_domain(source_url),
        raw_value=raw_value,
        is_grounded=grounded,
        llm_confidence=llm_confidence,
        not_found_reason=data.get("not_found_reason"),
        quote_snippet=(verified_quote or "")[:120],
    )
    return result


# ── Probe model (field-list axis) ────────────────────────────────────────────
#
# For each field we run ONE entity-agnostic search ("רשימת ראשי ערים 1990")
# to check if a directory or table page exists that covers many entities at once.
# If found, we extract the field for ALL entities in a single Claude call instead
# of N individual searches.  Fall back to entity-by-entity search for any entity
# the probe page doesn't cover.

_MIN_PROBE_COVERAGE = 0.25    # probe page must mention ≥ 25% of entities to be used
_MIN_PROBE_ABSOLUTE = 2       # or at least 2 entities (for small entity lists)
_PROBE_KEYWORDS = [
    "רשימה", "רשימת", "טבלה", "לפי שנה", "כל ה", "list", "table", "directory",
    "index", "all cities", "all municipalities",
]

_BULK_EXTRACTOR_TOOL = {
    "name": "bulk_extract_field",
    "description": (
        "Extract one data field for MANY entities simultaneously from a directory "
        "or list page. Return one extraction object per entity."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "extractions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "entity_name": {
                            "type": "string",
                            "description": "Exact entity name from the input list.",
                        },
                        "value": {
                            "type": ["string", "null"],
                            "description": "Extracted value, or null if not found in text.",
                        },
                        "quote_original": {
                            "type": ["string", "null"],
                            "description": "Direct verbatim copy-paste from source proving the value.",
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                    },
                    "required": ["entity_name", "value", "quote_original", "confidence"],
                },
            },
        },
        "required": ["extractions"],
    },
}

_BULK_EXTRACTOR_SYSTEM = """\
You are a precision data extractor working on a DIRECTORY PAGE — a list, table,
or index that contains data for many entities.

For EACH entity in the provided list, find the requested field value in the source
text. All grounding rules still apply:

  1. Return a value ONLY if it is explicitly in the text.
  2. quote_original must be a direct verbatim copy-paste from the source.
  3. Return null for entities not mentioned — do not guess or infer.
  4. Match temporal constraints strictly if a year/period is specified.
  5. Return one extraction object per entity in the input list — no additions, no omissions."""


def _score_probe_coverage(content: str, entities: list[str]) -> float:
    """Fraction of entities (by partial name match) that appear in the page text."""
    if not entities:
        return 0.0
    content_lower = content.lower()
    found = 0
    for entity in entities:
        parts = [p for p in entity.split() if len(p) > 2]
        if parts and all(p.lower() in content_lower for p in parts[:2]):
            found += 1
        elif entity.lower() in content_lower:
            found += 1
    return found / len(entities)


def bulk_extract_from_source(
    field,
    entities: list[str],
    source_url: str,
    source_content: str,
    client: anthropic.Anthropic,
) -> dict[str, ExtractionResult]:
    """
    Extract one field for all entities from a single directory/list page.
    Returns a dict of entity_name → ExtractionResult (value may be None if not found).
    """
    temporal_note = (
        f"\nTEMPORAL CONSTRAINT: The answer must be valid for '{field.temporal_anchor}'."
        if field.temporal_anchor else ""
    )
    entities_list = "\n".join(f"  - {e}" for e in entities)
    user_prompt = (
        f"Field to extract: {field.label_en} / {field.label_he}\n"
        f"Field type: {field.type}{temporal_note}\n\n"
        f"Entities to look up ({len(entities)} total):\n{entities_list}\n\n"
        f"Source URL: {source_url}\n"
        f"Source text:\n---\n{source_content[:_BUDGET]}\n---\n\n"
        f"For EACH entity above, extract '{field.label_en}' if it appears in the text. "
        f"Return exactly {len(entities)} extraction objects — one per entity."
    )

    max_tokens = min(2048, 512 + 64 * len(entities))
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        system=[{"type": "text", "text": _BULK_EXTRACTOR_SYSTEM,
                 "cache_control": {"type": "ephemeral"}}],
        tools=[_BULK_EXTRACTOR_TOOL],
        tool_choice={"type": "any"},
        messages=[{"role": "user", "content": user_prompt}],
    )

    tool_block = next(b for b in response.content if b.type == "tool_use")
    extractions_raw = tool_block.input.get("extractions", [])

    results: dict[str, ExtractionResult] = {}
    for ext in extractions_raw:
        entity_name = ext.get("entity_name", "")
        raw_value = ext.get("value")
        raw_quote = ext.get("quote_original")
        confidence = float(ext.get("confidence", 0.0))

        grounded = False
        verified_quote = None
        if raw_value and raw_quote:
            grounded, verified_quote = is_grounded(raw_quote, source_content)
            if not grounded:
                grounded, verified_quote = is_grounded(raw_value, source_content)
        elif raw_value:
            grounded, verified_quote = is_grounded(raw_value, source_content)

        results[entity_name] = ExtractionResult(
            field_id=field.id,
            value=raw_value if grounded else None,
            quote_original=verified_quote if grounded else None,
            source_url=source_url,
            source_domain=_domain(source_url),
            is_grounded=grounded,
            extractor_confidence=confidence if grounded else 0.0,
        )

    # Fill in missing entities as NOT_FOUND
    for entity in entities:
        if entity not in results:
            results[entity] = ExtractionResult(
                field_id=field.id,
                value=None,
                quote_original=None,
                source_url=source_url,
                source_domain=_domain(source_url),
                is_grounded=False,
                extractor_confidence=0.0,
            )

    return results


# ── Entity-page batch extraction (many fields, one page, one entity) ─────────
#
# When the same page appears in the top results for multiple fields of the
# same entity (e.g. the Wikipedia article for Tel Aviv answers BOTH
# "mayor_1990" AND "founding_year"), we extract everything in one Claude call
# instead of one call per field. Cost: same input tokens (content dominates),
# but one round-trip and one set of output tokens instead of N.

_BATCH_FIELDS_TOOL = {
    "name": "batch_extract_fields",
    "description": (
        "Extract MULTIPLE data fields for ONE entity from a single source page. "
        "Return one extraction object per requested field_id."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "extractions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "field_id": {
                            "type": "string",
                            "description": "Must match one of the requested field IDs exactly.",
                        },
                        "value": {
                            "type": ["string", "null"],
                            "description": "Extracted value, or null if not in text.",
                        },
                        "quote_original": {
                            "type": ["string", "null"],
                            "description": "Direct verbatim copy-paste proving the value.",
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                    },
                    "required": ["field_id", "value", "quote_original", "confidence"],
                },
            },
        },
        "required": ["extractions"],
    },
}

_BATCH_FIELDS_SYSTEM = """\
You extract MULTIPLE data fields for ONE entity from a single source page.

For EACH field in the provided list, find the value if it appears in the source.
All grounding rules apply, per field:

  1. Return a value ONLY if explicitly in the text. No inference, no background knowledge.
  2. quote_original must be a direct verbatim copy-paste from the source.
  3. Return null for fields not found — a confident null beats a hallucinated value.
  4. Match entity specificity — value must refer to the SAME entity requested.
  5. Match temporal constraints strictly when a year/period is given for the field.
  6. Return EXACTLY one extraction object per requested field_id — no additions, no omissions."""


def batch_extract_fields_from_source(
    fields: list,
    entity: str,
    source_url: str,
    source_content: str,
    client: anthropic.Anthropic,
    tracer=None,
    publication_date: str | None = None,
) -> dict[str, ExtractionResult]:
    """
    Extract multiple fields for one entity from a single page in ONE Claude call.
    Returns {field_id → ExtractionResult}, including not-found results so the
    caller can record sources for every field.
    """
    fields_desc_lines = []
    for f in fields:
        anchor = f" (must be valid for {f.temporal_anchor!r})" if f.temporal_anchor else ""
        fields_desc_lines.append(
            f"  - field_id={f.id!r}: {f.label_en} / {f.label_he} "
            f"[type={f.type}]{anchor}"
        )
    fields_block = "\n".join(fields_desc_lines)

    selected = _select_relevant_text_for_fields(source_content, fields, entity)

    user_prompt = (
        f"Entity: {entity}\n\n"
        f"Fields to extract ({len(fields)}):\n{fields_block}\n\n"
        f"Source URL: {source_url}\n"
        f"Source text:\n---\n{selected}\n---\n\n"
        f"For EACH field above, extract the value for entity '{entity}' from the text. "
        f"Return exactly {len(fields)} extraction objects — one per field_id."
    )

    max_tokens = min(2048, 384 + 256 * len(fields))
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        system=[{"type": "text", "text": _BATCH_FIELDS_SYSTEM,
                 "cache_control": {"type": "ephemeral"}}],
        tools=[_BATCH_FIELDS_TOOL],
        tool_choice={"type": "any"},
        messages=[{"role": "user", "content": user_prompt}],
    )

    tool_block = next(b for b in response.content if b.type == "tool_use")
    extractions_raw = tool_block.input.get("extractions", [])

    field_by_id = {f.id: f for f in fields}
    results: dict[str, ExtractionResult] = {}

    for ext in extractions_raw:
        fid = ext.get("field_id", "")
        if fid not in field_by_id:
            continue
        raw_value = ext.get("value")
        raw_quote = ext.get("quote_original")
        confidence = float(ext.get("confidence", 0.0))

        grounded = False
        verified_quote = None
        if raw_value and raw_quote:
            grounded, verified_quote = is_grounded(raw_quote, source_content)
            if not grounded:
                grounded, verified_quote = is_grounded(raw_value, source_content)
        elif raw_value:
            grounded, verified_quote = is_grounded(raw_value, source_content)

        results[fid] = ExtractionResult(
            field_id=fid,
            value=raw_value if grounded else None,
            quote_original=verified_quote if grounded else None,
            source_url=source_url,
            source_domain=_domain(source_url),
            is_grounded=grounded,
            extractor_confidence=confidence if grounded else 0.0,
            publication_date=publication_date,
        )

    # Fill in any field the LLM forgot to return.
    for f in fields:
        if f.id not in results:
            results[f.id] = ExtractionResult(
                field_id=f.id,
                value=None, quote_original=None,
                source_url=source_url, source_domain=_domain(source_url),
                is_grounded=False, extractor_confidence=0.0,
                publication_date=publication_date,
            )

    tr = tracer or NullTracer()
    tr.emit(
        "batch_extraction",
        entity=entity,
        url=source_url,
        domain=_domain(source_url),
        field_results=[
            {"field_id": fid, "value": r.value, "is_grounded": r.is_grounded}
            for fid, r in results.items()
        ],
    )
    return results


def verify_probe_extraction(
    field,
    entity: str,
    probe_extraction: ExtractionResult,
    search_client,
    claude: anthropic.Anthropic,
) -> list[ExtractionResult]:
    """
    When the probe found a value for (field, entity), run ONE focused search
    using that value as the anchor — to discover independent sources that
    confirm or contradict it.

    Why this matters: the probe gives us 1 source. Identity facts need
    min_corroborations: 2 to reach HIGH confidence. A query like
    '"Shlomo Lahat" Tel Aviv 1990' is far more precise than a generic
    "Tel Aviv mayor 1990" query — and (importantly) will NOT return the
    same directory page again, so we get independent evidence.

    Returns 0-2 additional ExtractionResult objects. The caller combines
    these with the probe extraction and passes the merged list to
    verify_field, which handles corroboration_count and confidence.
    """
    value = probe_extraction.value
    if not value:
        return []

    # Build a high-precision query: the found value is the anchor.
    # For URL fields the URL itself is unique enough — no need to add context.
    if field.type == "url":
        query = value
    else:
        query = f'"{value}" {entity}'
        if field.temporal_anchor:
            query += f" {field.temporal_anchor}"

    print(f"  [probe-verify] entity={entity!r} field={field.id!r} q={query!r}")

    try:
        response = search_client.search(
            query, max_results=2, include_raw_content=True
        )
    except Exception as exc:
        print(f"  [probe-verify error] {exc}")
        return []

    probe_domain = probe_extraction.source_domain
    extras: list[ExtractionResult] = []

    for hit in response.get("results", []):
        url = hit.get("url", "")
        content = hit.get("raw_content") or hit.get("content", "")
        if not url or not content or len(content) < 80:
            continue
        if _domain(url) == probe_domain:
            # Same source as the probe — would inflate corroboration_count.
            continue

        extraction = extract_from_source(
            field=field, entity=entity,
            source_url=url, source_content=content,
            resolved_deps={}, client=claude, memory=None,
        )
        if extraction.value:
            extras.append(extraction)

    return extras


def probe_field_list(
    field,
    entities: list[str],
    search_client,
    claude: anthropic.Anthropic,
    min_coverage: float = _MIN_PROBE_COVERAGE,
    prefetched_page: tuple[str, str] | None = None,
    found_page_out: list | None = None,
) -> dict[str, ExtractionResult] | None:
    """
    Try to find a directory/list page covering `field` for many entities at once.

    Returns a dict of entity_name → ExtractionResult when a good directory page is
    found.  Returns None if no suitable page exists (caller falls back to per-entity
    search).

    Cost savings: replaces N search calls + N Claude extraction calls with
    1 search call + 1 Claude bulk-extraction call.

    prefetched_page: (url, content) — when provided, skip the SerpAPI search
        entirely and go straight to bulk extraction.  Used when a sibling field
        shares the same directory_probe_query_he so the search doesn't run twice.

    found_page_out: caller-provided list — when a qualifying page is found, its
        (url, content) is appended so the caller can share it with sibling fields.
    """
    probe_query = field.directory_probe_query_he
    if not probe_query:
        return None

    # ── Fast path: caller already found a qualifying page for this query ────
    if prefetched_page is not None:
        url, content = prefetched_page
        print(f"  [probe] field={field.id!r} CACHE-HIT query={probe_query!r} "
              f"— skipping search, reusing {url[:60]!r}")
        bulk = bulk_extract_from_source(field, entities, url, content, claude)
        found_count = sum(1 for r in bulk.values() if r.value)
        print(f"  [probe] result: {found_count}/{len(entities)} entities filled")
        if found_count >= max(1, len(entities) * 0.10):
            return bulk
        return None

    # ── Normal path: search for the directory page ──────────────────────────
    print(f"  [probe] field={field.id!r} query={probe_query!r}")

    try:
        response = search_client.search(probe_query, max_results=3, include_raw_content=True)
    except Exception as exc:
        print(f"  [probe error] search failed: {exc}")
        return None

    for hit in response.get("results", []):
        url = hit.get("url", "")
        content = hit.get("raw_content") or hit.get("content", "")
        if not content or len(content) < 200:
            continue

        coverage = _score_probe_coverage(content, entities)
        found_abs = int(coverage * len(entities))
        has_dir_kw = any(kw in content.lower() for kw in _PROBE_KEYWORDS)

        print(
            f"  [probe] {url[:70]!r}: "
            f"coverage={coverage:.0%} ({found_abs}/{len(entities)}), "
            f"dir_kw={has_dir_kw}"
        )

        qualifies = (
            coverage >= min_coverage
            or found_abs >= _MIN_PROBE_ABSOLUTE
            or (coverage >= 0.10 and has_dir_kw)
        )
        if not qualifies:
            continue

        print(f"  [probe] HIT — bulk-extracting from {url[:70]!r}")
        bulk = bulk_extract_from_source(field, entities, url, content, claude)
        found_count = sum(1 for r in bulk.values() if r.value)
        print(f"  [probe] result: {found_count}/{len(entities)} entities filled")

        if found_count >= max(1, len(entities) * 0.10):
            # Share the qualifying page with sibling fields that have the same query
            if found_page_out is not None:
                found_page_out.append((url, content))
            return bulk

    return None


# ── Entity Discovery ─────────────────────────────────────────────────────────
#
# When the entity list itself is open ("the 10 largest cities in Israel") we
# run ONE focused search + ONE LLM call that does double duty:
#   1. Extracts the ordered entity list.
#   2. Harvests any column-plan field values already visible on the same page
#      (a city-by-population table will typically also include founding year,
#      district, area, etc.).
#
# The harvested values are returned alongside the entity list and feed straight
# into probe_results in /api/run, so verification-search + Lane 1 logic is
# reused as-is. There is NO automatic acceptance — the orchestrator MUST gate
# on user approval before passing the list to the entity loop.

_DISCOVER_TOOL = {
    "name": "discover_entities",
    "description": (
        "From a directory/ranking page, extract the ordered list of entities "
        "that match the research criterion. Additionally, for each entity, "
        "harvest any of the requested column-plan field values that are "
        "explicitly stated on the same page."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name":  {"type": "string"},
                        "rank":  {"type": ["integer", "null"]},
                        "quote": {"type": "string"},
                    },
                    "required": ["name", "quote"],
                },
            },
            "harvested": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "entity_name": {"type": "string"},
                        "field_id":    {"type": "string"},
                        "value":       {"type": ["string", "null"]},
                        "quote":       {"type": "string"},
                    },
                    "required": ["entity_name", "field_id", "value", "quote"],
                },
            },
        },
        "required": ["entities", "harvested"],
    },
}

_DISCOVER_SYSTEM = """\
You are an entity-discovery agent reading a ranked-list page.

Two outputs in one call:

A. ENTITIES — produce the ordered list that matches the research criterion.
   - Use the exact names as written on the page (canonical form).
   - Set rank = 1-based ordinal when the list is ordered.
   - quote MUST be a verbatim substring from the source proving the name appears.
   - If the question specifies a count N, return up to N entities, in order.
   - If the page does not yield a confident list, return an empty entities array.

B. HARVESTED — for EACH entity returned in (A), and EACH column-plan field
   provided in the prompt, if the value is EXPLICITLY stated on the page,
   emit one harvested row with the exact value + verbatim quote. Otherwise
   skip that (entity, field) — do NOT guess.

All grounding rules from the standard extractor apply: every value must have
a verbatim quote; never fabricate.
"""


def discover_entities(
    discovery_plan,
    plan_columns: list,
    search_client,
    claude: anthropic.Anthropic,
):
    """Run discovery search + LLM extraction. Returns EntityDiscoveryResult or None."""
    from .models import (
        DiscoveredEntity, HarvestedValue, EntityDiscoveryResult,
    )

    query = discovery_plan.query_he or discovery_plan.query_en
    if not query:
        return None

    print(f"[discover] query={query!r}")
    try:
        response = search_client.search(query, max_results=3, include_raw_content=True)
    except Exception as exc:
        print(f"[discover error] search failed: {exc}")
        return None

    columns_summary = "\n".join(
        f"  - id={c.id}, label_he={c.label_he!r}, label_en={c.label_en!r}, "
        f"type={c.type}"
        + (f", temporal_anchor={c.temporal_anchor!r}" if c.temporal_anchor else "")
        for c in plan_columns
    )
    expected = (
        f"Expected count: {discovery_plan.expected_count}\n"
        if discovery_plan.expected_count else ""
    )
    hint = (
        f"Extraction hint: {discovery_plan.extraction_hint}\n"
        if discovery_plan.extraction_hint else ""
    )

    for hit in response.get("results", []):
        url = hit.get("url", "")
        content = hit.get("raw_content") or hit.get("content", "")
        if not content or len(content) < 200:
            continue

        user_prompt = (
            f"{expected}{hint}"
            f"Research question criterion: {query}\n\n"
            f"Column-plan fields to also harvest if visible:\n{columns_summary}\n\n"
            f"Source URL: {url}\n"
            f"Source text:\n---\n{content[:_BUDGET]}\n---\n\n"
            "Return the ordered entity list AND any harvested field values."
        )

        n_exp = (discovery_plan.expected_count or 30)
        max_tokens = min(2048, 256 + 80 * n_exp + 60 * n_exp * len(plan_columns))
        try:
            resp = claude.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=max_tokens,
                system=[{"type": "text", "text": _DISCOVER_SYSTEM,
                         "cache_control": {"type": "ephemeral"}}],
                tools=[_DISCOVER_TOOL],
                tool_choice={"type": "tool", "name": "discover_entities"},
                messages=[{"role": "user", "content": user_prompt}],
            )
            tool_block = next(b for b in resp.content if b.type == "tool_use")
            data = tool_block.input
        except Exception as exc:
            print(f"[discover error] LLM failed on {url[:70]!r}: {exc}")
            continue

        raw_entities = data.get("entities", []) or []
        if not raw_entities:
            print(f"[discover] {url[:70]!r}: empty list, trying next hit")
            continue

        # Ground every entity quote in the page text (substring check).
        content_lower = content.lower()
        entities = []
        for i, e in enumerate(raw_entities):
            name = (e.get("name") or "").strip()
            if not name:
                continue
            quote = e.get("quote") or ""
            grounded = quote and quote[:80].lower() in content_lower
            entities.append(DiscoveredEntity(
                name=name,
                rank=e.get("rank") if e.get("rank") is not None else i + 1,
                quote=quote if grounded else None,
            ))
        if not entities:
            continue

        # Filter harvest: only keep (entity, field) pairs we can ground.
        valid_entity_names = {e.name for e in entities}
        valid_field_ids = {c.id for c in plan_columns}
        harvested = []
        for h in data.get("harvested", []) or []:
            ename = (h.get("entity_name") or "").strip()
            fid   = (h.get("field_id") or "").strip()
            val   = h.get("value")
            quote = h.get("quote") or ""
            if ename not in valid_entity_names or fid not in valid_field_ids:
                continue
            if val is None:
                continue
            if not quote or quote[:80].lower() not in content_lower:
                continue
            harvested.append(HarvestedValue(
                entity_name=ename, field_id=fid, value=val, quote=quote,
            ))

        print(
            f"[discover] HIT {url[:70]!r}: "
            f"{len(entities)} entities, {len(harvested)} harvested cells"
        )
        return EntityDiscoveryResult(
            entities=entities,
            harvested=harvested,
            source_url=url,
            source_domain=_domain(url),
        )

    return None


def _wikidata_year(qualifier_list: list) -> int | None:
    """Extract a year integer from a Wikidata time-qualifier list."""
    for q in qualifier_list:
        time_val = q.get("datavalue", {}).get("value", {}).get("time", "")
        if time_val:
            try:
                return int(time_val[1:5])   # "+1974-01-01T..." → 1974
            except ValueError:
                pass
    return None


def _wikidata_fetch(params: dict) -> dict:
    """Single Wikidata API call with rate-limit sleep."""
    import urllib.request, urllib.parse, json as _json, time
    qs = urllib.parse.urlencode({**params, "format": "json", "utf8": 1})
    req = urllib.request.Request(
        f"https://www.wikidata.org/w/api.php?{qs}",
        headers={"User-Agent": WikipediaSearchClient._UA},
    )
    time.sleep(1)
    with urllib.request.urlopen(req, timeout=12) as resp:
        return _json.loads(resp.read().decode("utf-8"))


def _inject_wikidata(entity: str, field, results: list, seen_urls: set) -> None:
    """
    Query Wikidata for structured facts that Wikipedia plaintext doesn't carry.

    P856  (official website)    → injected for url fields
    P6    (head of government)  → injected for person_name fields with a
                                   temporal_anchor; filters by tenure dates
    """
    he_chars = sum(1 for c in entity if 'א' <= c <= 'ת')
    site = "hewiki" if he_chars > 0 else "enwiki"

    try:
        # Step 1: resolve entity → Wikidata QID + claims
        data = _wikidata_fetch({
            "action": "wbgetentities", "sites": site, "titles": entity,
            "props": "claims",
        })
        wd_entities = data.get("entities", {})

        for qid, ent in wd_entities.items():
            if ent.get("missing"):
                continue
            claims = ent.get("claims", {})
            entity_wd_url = f"https://www.wikidata.org/wiki/{qid}"

            # ── P856: official website ────────────────────────────────────
            if field.type == "url":
                for claim in claims.get("P856", []):
                    url_val = (claim.get("mainsnak", {})
                                    .get("datavalue", {})
                                    .get("value"))
                    if url_val and url_val not in seen_urls:
                        seen_urls.add(url_val)
                        print(f"    [wikidata] P856 official website: {url_val}")
                        results.append(ExtractionResult(
                            field_id=field.id,
                            value=url_val,
                            quote_original=(
                                f"Wikidata {qid} P856 (official website): {url_val}"
                            ),
                            source_url=entity_wd_url,
                            source_domain="www.wikidata.org",
                            is_grounded=True,
                            extractor_confidence=0.95,
                        ))

            # ── P6: head of government (historical mayors etc.) ───────────
            elif field.type == "person_name" and field.temporal_anchor:
                try:
                    target_year = int(field.temporal_anchor)
                except ValueError:
                    continue

                person_qids: list[tuple[str, int | None, int | None]] = []
                for claim in claims.get("P6", []):
                    person_qid = (claim.get("mainsnak", {})
                                       .get("datavalue", {})
                                       .get("value", {})
                                       .get("id"))
                    if not person_qid:
                        continue
                    quals = claim.get("qualifiers", {})
                    start = _wikidata_year(quals.get("P580", []))
                    end   = _wikidata_year(quals.get("P582", []))
                    # Accept if target year is within [start, end] (None = open)
                    if (start is None or start <= target_year) and \
                       (end   is None or end   >= target_year):
                        person_qids.append((person_qid, start, end))

                if not person_qids:
                    continue

                # Step 2: resolve person QIDs → Hebrew labels
                all_qids = [p[0] for p in person_qids]
                labels_data = _wikidata_fetch({
                    "action": "wbgetentities",
                    "ids": "|".join(all_qids),
                    "props": "labels",
                    "languages": "he|en",
                })
                label_ents = labels_data.get("entities", {})

                for person_qid, start, end in person_qids:
                    lent   = label_ents.get(person_qid, {}).get("labels", {})
                    name   = (lent.get("he", {}).get("value") or
                              lent.get("en", {}).get("value", person_qid))
                    tenure = (f"{start}–{end}" if start and end
                              else f"from {start}" if start
                              else f"until {end}" if end
                              else "unknown tenure")
                    print(f"    [wikidata] P6 head of government "
                          f"in {target_year}: {name} ({tenure})")
                    results.append(ExtractionResult(
                        field_id=field.id,
                        value=name,
                        quote_original=(
                            f"Wikidata {qid} P6 (head of government): "
                            f"{name}, tenure {tenure}, covers {target_year}"
                        ),
                        source_url=f"https://www.wikidata.org/wiki/{person_qid}",
                        source_domain="www.wikidata.org",
                        is_grounded=True,
                        extractor_confidence=0.95,
                    ))

    except Exception as exc:
        print(f"    [wikidata error] {exc}")


_DATE_FROM_URL_RE = re.compile(r'/(\d{4})/(\d{1,2})/(\d{1,2})/')
_DATE_FROM_URL_SHORT_RE = re.compile(r'[/_-](\d{4})(\d{2})(\d{2})[/_.-]')
_DATE_FROM_TEXT_RE = re.compile(
    # ISO: 2023-04-15 or 2023/04/15
    r'\b(20\d{2})[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b'
)


def _extract_source_date(hit: dict, content: str) -> str | None:
    """
    Best-effort extraction of a publication/update date for a search result.
    Returns an ISO-format string (YYYY-MM-DD or YYYY-MM) or None.

    Priority:
      1. Tavily's published_date field (most reliable when present)
      2. Date pattern in the URL path (/2023/04/15/)
      3. ISO date pattern near the start of the content text
    """
    # 1. Tavily metadata
    raw = hit.get("published_date") or hit.get("publishedDate") or ""
    if raw:
        # Normalise to YYYY-MM-DD or YYYY-MM
        m = re.match(r'(\d{4}-\d{2}-\d{2})', str(raw))
        if m:
            return m.group(1)
        m = re.match(r'(\d{4}-\d{2})', str(raw))
        if m:
            return m.group(1)

    # 2. URL date pattern
    url = hit.get("url", "")
    m = _DATE_FROM_URL_RE.search(url)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = _DATE_FROM_URL_SHORT_RE.search(url)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # 3. ISO date in the first 1000 chars of content (article datelines, bylines)
    snippet = (content or "")[:1000]
    m = _DATE_FROM_TEXT_RE.search(snippet)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    return None


def _extract_pdf_text(url: str, timeout: int = 15) -> str | None:
    """
    Fetch a PDF URL and extract its text via pdfplumber.

    Called as a fallback when Tavily returns a PDF link with no usable content.
    Returns extracted text, or None on any failure (network error, encrypted
    PDF, image-only scan, etc.). Failures are non-fatal — the URL is simply
    skipped, same as today.

    Limits:
      - First 30 pages only (avoids huge documents)
      - Text capped at _BUDGET chars before returning (windowing runs later)
    """
    try:
        import pdfplumber
    except ImportError:
        return None

    _MAX_PAGES = 30
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; INResearcher/1.0; "
                    "+https://github.com/harelfelhai/inreasearcher)"
                )
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except Exception as exc:
        print(f"    [pdf-fetch] {url[:70]!r}: {exc}")
        return None

    try:
        pages_text: list[str] = []
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for page in pdf.pages[:_MAX_PAGES]:
                text = page.extract_text() or ""
                if text.strip():
                    pages_text.append(text)
        result = "\n\n".join(pages_text)
        if len(result) < 80:
            return None
        print(
            f"    [pdf] extracted {len(result):,} chars "
            f"from {len(pages_text)} page(s): {url[:70]!r}"
        )
        return result[:_BUDGET * 3]   # windowing trims further
    except Exception as exc:
        print(f"    [pdf] extraction failed for {url[:70]!r}: {exc}")
        return None


def _is_pdf_url(url: str) -> bool:
    """Heuristic: URL path ends in .pdf or contains /pdf/ segment."""
    lower = url.lower().split("?")[0]
    return lower.endswith(".pdf") or "/pdf/" in lower


def _gather_pages_for_field(
    field: ColumnPlan,
    entity: str,
    resolved_deps: dict,
    tavily: TavilyClient,
    seen_urls: set[str] | None = None,
    tracer=None,
    search_counter: list[int] | None = None,
    search_budget: int | None = None,
    max_results: int = 5,
) -> tuple[list[tuple[str, str, str | None]], list[ExtractionResult]]:
    """Returns (pages, wikidata_results) where each page is (url, content, date)."""
    """
    Run search queries for a field and return raw candidate pages
    (no LLM extraction yet).

    Returns:
      pages: list of (url, content) tuples — candidates for extraction.
      wikidata_results: pre-built ExtractionResults injected from Wikidata
        (structured facts; do not need LLM extraction).

    The caller can dedupe pages across multiple fields before extracting.
    """
    if seen_urls is None:
        seen_urls = set()

    # Strip queries that lack {entity} — they return the same page for every
    # entity, wasting SerpAPI quota and producing duplicate content.
    queries_he = [
        q.replace("{entity}", entity)
        for q in field.search_queries_he
        if "{entity}" in q
    ]
    queries_en = [
        q.replace("{entity}", entity)
        for q in field.search_queries_en[:2]
        if "{entity}" in q
    ]
    if not queries_he and not queries_en and field.search_queries_he:
        print(f"    [search] warning: field={field.id!r} has no queries with "
              f"{{entity}} placeholder — skipping per-entity search")

    if field.depends_on and field.depends_on in resolved_deps:
        dep_val = resolved_deps[field.depends_on]
        if dep_val:
            bonus = [
                q.replace("{entity}", dep_val)
                for q in field.search_queries_he[:2]
                if "{entity}" in q
            ]
            queries_he = queries_he + bonus

    pages: list[tuple[str, str, str | None]] = []
    wikidata_results: list[ExtractionResult] = []
    tr = tracer or NullTracer()

    is_wikipedia = isinstance(tavily, WikipediaSearchClient)
    query_cap = 3 if is_wikipedia else 1

    if is_wikipedia:
        extra = []
        if field.type == "person_name" and field.temporal_anchor:
            extra.append(f"ראשי עיר {entity}")

        direct = tavily.fetch_entity_article(entity, extra_titles=extra)

        if field.type in ("url", "person_name"):
            canonical = (direct["results"][0]["title"]
                         if direct.get("results") else entity)
            _inject_wikidata(canonical, field, wikidata_results, seen_urls)

        for hit in direct.get("results", []):
            url = hit.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                content = hit.get("raw_content") or hit.get("content", "")
                if content and len(content) >= 80:
                    pages.append((url, content, _extract_source_date(hit, content)))

    # Fan out all search queries concurrently — each is independent I/O.
    # SerpAPI path: Hebrew-only. For deferred fields the bonus dep-resolution
    # query is appended to queries_he — allow cap=2 so it is not discarded.
    # Wikipedia path: He+En up to cap.
    if is_wikipedia:
        all_queries = list((queries_he + queries_en)[:query_cap])
    else:
        serp_cap = 2 if (field.depends_on and len(queries_he) > 1) else query_cap
        all_queries = list(queries_he[:serp_cap])

    # Trim to remaining per-entity search budget so early entities don't
    # consume all SerpAPI quota before later entities get a turn.
    if search_budget is not None and search_counter is not None:
        remaining = search_budget - search_counter[0]
        if remaining <= 0:
            print(f"    [budget] entity={entity!r} field={field.id!r}: "
                  f"budget exhausted ({search_budget} calls) — skipping")
            all_queries = []
        elif len(all_queries) > remaining:
            print(f"    [budget] entity={entity!r} field={field.id!r}: "
                  f"trimming to {remaining} query/ies (budget={search_budget})")
            all_queries = all_queries[:remaining]

    tr.emit(
        "field_search",
        entity=entity,
        field_id=field.id,
        field_type=field.type,
        queries=all_queries,
        engine=type(tavily).__name__,
    )

    def _one_search(q: str) -> tuple[str, list]:
        _MAX_RETRIES = 2
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return q, tavily.search(
                    q, max_results=max_results, include_raw_content=True,
                ).get("results", [])
            except Exception as exc:
                if attempt < _MAX_RETRIES:
                    wait = 2.0 * (attempt + 1)
                    print(f"    [search-retry] attempt {attempt+1}/{_MAX_RETRIES+1} "
                          f"{q[:50]!r}: {exc} — retry in {wait:.0f}s")
                    time.sleep(wait)
                else:
                    print(f"    [search error] {q[:60]!r}: {exc}")
                    return q, []
        return q, []   # unreachable but satisfies type checker

    if all_queries:
        with ThreadPoolExecutor(max_workers=min(len(all_queries), 4)) as pool:
            query_results = list(pool.map(_one_search, all_queries))
    else:
        query_results = []

    # Track search calls consumed for budget accounting.
    if search_counter is not None:
        search_counter[0] += len(all_queries)

    for query, hits in query_results:
        if hits:
            print(f"    [search] '{query[:55]}' → {len(hits)} result(s): "
                  + ", ".join(h.get("title", h.get("url", "?"))[:30] for h in hits[:3]))
        for hit in hits:
            url = hit.get("url", "")
            is_new = url not in seen_urls
            content = hit.get("raw_content") or hit.get("content", "")

            # PDF fallback: Tavily often returns empty raw_content for PDF
            # links. Attempt direct fetch + pdfplumber extraction before
            # discarding the URL — these are frequently the most authoritative
            # sources for government/legal research.
            if is_new and _is_pdf_url(url) and (not content or len(content) < 80):
                pdf_text = _extract_pdf_text(url)
                if pdf_text:
                    content = pdf_text

            pub_date = _extract_source_date(hit, content)
            tr.emit(
                "search_hit",
                entity=entity,
                field_id=field.id,
                query=query,
                url=url,
                domain=_domain(url),
                content_length=len(content) if content else 0,
                is_new=is_new,
                accepted=is_new and bool(content) and len(content) >= 80,
                pdf_fallback=is_new and _is_pdf_url(url),
                publication_date=pub_date,
            )
            if not url or not is_new:
                continue
            seen_urls.add(url)
            if not content or len(content) < 80:
                continue
            pages.append((url, content, pub_date))

    return pages, wikidata_results


def search_and_extract(
    field: ColumnPlan,
    entity: str,
    resolved_deps: dict,
    tavily: TavilyClient,
    claude: anthropic.Anthropic,
    memory: SuccessMemory | None = None,
    max_results: int = 6,
    early_stop_on_high: int = 3,
) -> list[ExtractionResult]:
    """
    Single-field search + extract.  Stops early when we accumulate
    `early_stop_on_high` grounded results.

    For multi-field calls on the same entity prefer search_and_extract_batched,
    which deduplicates pages that appear in multiple fields' search results.
    """
    pages, results = _gather_pages_for_field(field, entity, resolved_deps, tavily)

    for url, content, pub_date in pages:
        extraction = extract_from_source(
            field=field, entity=entity,
            source_url=url, source_content=content,
            resolved_deps=resolved_deps, client=claude, memory=memory,
            publication_date=pub_date,
        )
        if extraction.is_grounded and extraction.value:
            results.append(extraction)
            if len(results) >= early_stop_on_high:
                return results

    return results


def search_and_extract_batched(
    fields: list[ColumnPlan],
    entity: str,
    resolved_deps: dict,
    tavily: TavilyClient,
    claude: anthropic.Anthropic,
    memory: SuccessMemory | None = None,
    tracer=None,
    max_results: int = 5,
    search_budget: int | None = None,
    seed_pages: dict[str, tuple[str, str | None]] | None = None,
    page_sink: dict[str, tuple[str, str | None]] | None = None,
) -> dict[str, list[ExtractionResult]]:
    """
    Process MULTIPLE fields for one entity with cross-field page deduplication.

      Phase 1: Gather candidate pages for every field (search calls only).
      Phase 2: Union all unique pages across fields.
      Phase 3: One multi-field Claude call per unique page (instead of N calls,
               one per field). For a single-field call we fall back to the
               existing single-field extractor so memory examples are used.
      Phase 4: Wikidata-injected results are merged in per field unchanged.

    seed_pages: pre-fetched {url → (content, pub_date)} from a prior lane.
      URLs in this dict are pre-loaded into shared_seen so they are not
      re-fetched, and their content is included in extraction.
    page_sink: if provided, every newly fetched page is added here so the
      next lane can pass it as seed_pages.

    Returns {field_id → list[ExtractionResult]} — same shape as N calls to
    search_and_extract, but each page is processed once, regardless of how
    many fields surfaced it.
    """
    if not fields:
        return {}

    results: dict[str, list[ExtractionResult]] = {f.id: [] for f in fields}

    # Phase 1 — gather pages per field. SHARED seen_urls avoids re-fetching
    # the same URL across fields' search phases. A shared counter enforces a
    # per-entity search budget so early entities can't starve later ones.
    # Pre-seed seen_urls with pages already fetched by a prior lane so those
    # URLs are skipped in searches (they'll still be extracted below).
    shared_seen: set[str] = set(seed_pages.keys()) if seed_pages else set()
    entity_search_count: list[int] = [0]
    # Start with seeded pages so they participate in extraction.
    all_urls_with_content: dict[str, tuple[str, str | None]] = dict(seed_pages) if seed_pages else {}
    for field in fields:
        pages, wikidata_results = _gather_pages_for_field(
            field, entity, resolved_deps, tavily, seen_urls=shared_seen, tracer=tracer,
            search_counter=entity_search_count, search_budget=search_budget,
            max_results=max_results,
        )
        results[field.id].extend(wikidata_results)
        for url, content, pub_date in pages:
            is_new = url not in all_urls_with_content
            all_urls_with_content.setdefault(url, (content, pub_date))
            if is_new and page_sink is not None and url not in (seed_pages or {}):
                page_sink[url] = (content, pub_date)

    if not all_urls_with_content:
        return results

    # Phase 2-3 — single batched extract per page if ≥ 2 fields; else
    # single-field extractor (uses memory examples).
    n_pages = len(all_urls_with_content)
    if len(fields) >= 2:
        print(f"    [batch] entity={entity!r}: {n_pages} unique page(s) × "
              f"{len(fields)} field(s) — one batched extract per page")
        for url, (content, pub_date) in all_urls_with_content.items():
            batch = batch_extract_fields_from_source(
                fields=fields, entity=entity,
                source_url=url, source_content=content,
                client=claude, tracer=tracer, publication_date=pub_date,
            )
            for fid, ext in batch.items():
                if ext.is_grounded and ext.value:
                    results[fid].append(ext)
    else:
        field = fields[0]
        for url, (content, pub_date) in all_urls_with_content.items():
            ext = extract_from_source(
                field=field, entity=entity,
                source_url=url, source_content=content,
                resolved_deps=resolved_deps, client=claude, memory=memory,
                tracer=tracer, publication_date=pub_date,
            )
            if ext.is_grounded and ext.value:
                results[field.id].append(ext)

    return results


def _domain(url: str) -> str:
    m = re.search(r'https?://([^/]+)', url)
    return m.group(1) if m else url


# ── Mock search layer (for testing without a live Tavily key) ─────────────────

_MOCK_CONTENT_TEL_AVIV = """\
תל אביב-יפו היא עיר בישראל, הגדולה בישראל מבחינת אוכלוסייה עירונית.
שלמה להט (צ'יץ') כיהן כראש עיריית תל אביב-יפו בין השנים 1974 ל-1993.
בשנת 1990 כיהן שלמה להט בתפקיד ראש העיר.
האתר הרשמי של עיריית תל אביב הוא www.tel-aviv.gov.il.
תל אביב הוכרזה כעיר בשנת 1950 לאחר איחודה עם יפו.
"""

_MOCK_CONTENT_HAIFA = """\
חיפה היא עיר בצפון ישראל הממוקמת על הר הכרמל.
עריאל שרון ואחרים שימשו בתפקידים שונים בחיפה.
גוריון ביינארט כיהן כראש עיריית חיפה בתחילת שנות התשעים.
אריה גוראל היה ראש עיריית חיפה בין השנים 1983 ל-1993.
בשנת 1990 כיהן אריה גוראל כראש עיריית חיפה.
האתר הרשמי של עיריית חיפה הוא www.haifa.muni.il.
"""

_MOCK_RESULTS = {
    "תל אביב": [
        {
            "url": "https://he.wikipedia.org/wiki/תל_אביב-יפו",
            "title": "תל אביב-יפו — ויקיפדיה",
            "content": _MOCK_CONTENT_TEL_AVIV,
            "raw_content": _MOCK_CONTENT_TEL_AVIV,
        },
        {
            "url": "https://www.tel-aviv.gov.il/about",
            "title": "אודות עיריית תל אביב-יפו",
            "content": "שלמה להט כיהן כראש עיריית תל אביב בשנת 1990. האתר הרשמי: www.tel-aviv.gov.il",
            "raw_content": "שלמה להט כיהן כראש עיריית תל אביב בשנת 1990. האתר הרשמי: www.tel-aviv.gov.il",
        },
    ],
    "tel aviv": [
        {
            "url": "https://en.wikipedia.org/wiki/Tel_Aviv",
            "title": "Tel Aviv — Wikipedia",
            "content": "Shlomo Lahat (Chich) served as mayor of Tel Aviv from 1974 to 1993. In 1990 the mayor was Shlomo Lahat. Official site: www.tel-aviv.gov.il",
            "raw_content": "Shlomo Lahat (Chich) served as mayor of Tel Aviv from 1974 to 1993. In 1990 the mayor was Shlomo Lahat. Official site: www.tel-aviv.gov.il",
        }
    ],
    "חיפה": [
        {
            "url": "https://he.wikipedia.org/wiki/חיפה",
            "title": "חיפה — ויקיפדיה",
            "content": _MOCK_CONTENT_HAIFA,
            "raw_content": _MOCK_CONTENT_HAIFA,
        }
    ],
    "default": [
        {
            "url": "https://he.wikipedia.org/wiki/רשימת_ראשי_עיר_בישראל",
            "title": "רשימת ראשי עיר בישראל — ויקיפדיה",
            "content": "רשימת ראשי עיר ומועצות מקומיות בישראל לפי שנים. בשנת 1990 כיהנו ראשי עיר שונים ברחבי הארץ.",
            "raw_content": "רשימת ראשי עיר ומועצות מקומיות בישראל לפי שנים. בשנת 1990 כיהנו ראשי עיר שונים ברחבי הארץ.",
        }
    ],
}


class MockTavilyClient:
    """Drop-in replacement for TavilyClient that returns canned results."""

    def __init__(self, results: dict | None = None):
        self._results = results or _MOCK_RESULTS

    def search(self, query: str, **kwargs) -> dict:
        # Return domain-specific mock if available, else default
        for key, hits in self._results.items():
            if key != "default" and key in query.lower():
                return {"results": hits}
        return {"results": self._results.get("default", [])}


# ── DuckDuckGo search client (no API key required) ────────────────────────────

class WikipediaSearchClient:
    """
    Tavily-compatible client that searches and fetches full article content
    from Wikipedia (Hebrew-first, English fallback).

    Why Wikipedia instead of DuckDuckGo/Google:
      - No API key, no rate limits, no IP blocks
      - Returns full structured article text — ideal for grounding
      - Hebrew Wikipedia (he.wikipedia.org) has good coverage of Israeli
        municipalities, politicians, and public figures
      - The MediaWiki API is stable and officially supported

    Strategy per query:
      1. Detect language (Hebrew chars → he.wikipedia.org, else en)
      2. Call the Wikipedia search API to find matching article titles
      3. Fetch each article's full plaintext via the extracts API
    """

    _UA = "INResearcher/1.0 (autonomous research agent; github.com/harelfelhai/inreasearcher)"
    _RATE_DELAY  = 2.0   # seconds between every Wikipedia API call

    def search(self, query: str, max_results: int = 3, **kwargs) -> dict:
        import urllib.request, urllib.parse, json as _json

        he_chars = sum(1 for c in query if 'א' <= c <= 'ת')
        lang = "he" if he_chars > 2 else "en"
        base = f"https://{lang}.wikipedia.org/w/api.php"

        search_params = urllib.parse.urlencode({
            "action": "query", "list": "search",
            "srsearch": query, "srlimit": max_results,
            "format": "json", "utf8": 1,
        })
        import time

        titles = []
        try:
            titles = self._api_get(base, search_params)["query"]["search"]
            titles = [r["title"] for r in titles]
        except Exception as exc:
            print(f"    [wikipedia search error] {exc}")
            return {"results": []}

        if not titles:
            return {"results": []}

        extract_params = urllib.parse.urlencode({
            "action": "query", "prop": "extracts",
            "titles": "|".join(titles),
            "explaintext": 1, "exsectionformat": "plain",
            "format": "json", "utf8": 1,
        })
        results = []
        try:
            pages = self._api_get(base, extract_params)["query"]["pages"]
            for page in pages.values():
                title   = page.get("title", "")
                extract = page.get("extract", "")
                if not extract:
                    continue
                url = (
                    f"https://{lang}.wikipedia.org/wiki/"
                    + urllib.parse.quote(title.replace(" ", "_"))
                )
                results.append({
                    "url": url, "title": title,
                    "content": extract[:400],
                    "raw_content": extract,   # full text — windowing done by _select_relevant_text
                })
        except Exception as exc:
            print(f"    [wikipedia fetch error] {exc}")

        return {"results": results}

    def fetch_entity_article(self, entity: str, extra_titles: list[str] | None = None) -> dict:
        """
        Directly fetch Wikipedia articles for `entity` (and optional extra titles).
        Follows redirects (e.g. 'תל אביב' → 'תל אביב-יפו').
        Returns a Tavily-compatible results dict.

        `extra_titles` lets callers fetch related articles in one round-trip,
        e.g. ["ראשי עיר תל אביב-יפו"] for the dedicated mayors list page.
        """
        import urllib.parse

        he_chars = sum(1 for c in entity if 'א' <= c <= 'ת')
        lang = "he" if he_chars > 0 else "en"
        base = f"https://{lang}.wikipedia.org/w/api.php"

        titles_to_fetch = [entity] + (extra_titles or [])

        params = urllib.parse.urlencode({
            "action": "query", "prop": "extracts",
            "titles": "|".join(titles_to_fetch),
            "explaintext": 1, "exsectionformat": "plain",
            "redirects": 1,
            "format": "json", "utf8": 1,
        })
        try:
            data = self._api_get(base, params)
            pages = data.get("query", {}).get("pages", {})
            results = []
            for page in pages.values():
                if page.get("missing") is not None:
                    continue
                title   = page.get("title", entity)
                extract = page.get("extract", "")
                if not extract:
                    continue
                url = (
                    f"https://{lang}.wikipedia.org/wiki/"
                    + urllib.parse.quote(title.replace(" ", "_"))
                )
                results.append({
                    "url": url, "title": title,
                    "content": extract[:400],
                    "raw_content": extract,
                })
                print(f"    [direct fetch] '{title}' ({len(extract):,} chars)")
            return {"results": results}
        except Exception as exc:
            print(f"    [direct fetch error] {exc}")
            return {"results": []}

    def _api_get(self, base: str, params: str) -> dict:
        """
        Make one Wikipedia API call with rate-limiting and a single 429 retry.
        Sleeps _RATE_DELAY seconds BEFORE and AFTER the call so successive
        queries within a field don't hammer the endpoint.
        """
        import time, urllib.request, json as _json

        time.sleep(self._RATE_DELAY)
        req = urllib.request.Request(
            f"{base}?{params}",
            headers={"User-Agent": self._UA}
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = _json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            # Retry once after a longer back-off on 429
            if "429" in str(exc):
                print(f"    [wikipedia 429 — waiting 10s before retry]")
                time.sleep(10)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = _json.loads(resp.read().decode("utf-8"))
            else:
                raise
        time.sleep(self._RATE_DELAY)
        return data


class DuckDuckGoClient:
    """
    Tavily-compatible client backed by DuckDuckGo (ddgs package).
    Fetches full page content for better grounding. Falls back to snippet.
    Note: may rate-limit in server environments — prefer WikipediaSearchClient
    for research tasks, use DuckDuckGo only when broader web coverage is needed.
    """

    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
        "Accept-Language": "he,en;q=0.9",
    }
    _FETCH_TIMEOUT = 8
    _MAX_CONTENT = 6000

    def __init__(self):
        # Session-level page cache: a single URL fetched once per run, even if
        # it appears in search results for multiple entities. Re-windowing per
        # entity is done downstream by _select_relevant_text.
        self._page_cache: dict[str, str] = {}

    def search(self, query: str, max_results: int = 5, **kwargs) -> dict:
        import urllib.request, time

        raw_hits: list[dict] = []
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                raw_hits = list(ddgs.text(query, max_results=max_results))
        except ImportError:
            try:
                from duckduckgo_search import DDGS
                with DDGS() as ddgs:
                    raw_hits = list(ddgs.text(query, max_results=max_results))
            except Exception as exc:
                print(f"    [ddg error] {exc}")
                return {"results": []}
        except Exception as exc:
            print(f"    [ddg search error] {exc}")
            return {"results": []}

        results = []
        for hit in raw_hits:
            url     = hit.get("href", "")
            snippet = hit.get("body", "")
            title   = hit.get("title", "")
            raw_content = self._fetch(url) or snippet
            results.append({"url": url, "title": title,
                             "content": snippet, "raw_content": raw_content})
            time.sleep(0.5)   # gentle rate-limit buffer

        return {"results": results}

    def _fetch(self, url: str) -> str | None:
        import urllib.request
        if not url.startswith("http"):
            return None
        if url in self._page_cache:
            print(f"    [page-cache hit] {url[:70]}")
            return self._page_cache[url]
        try:
            req = urllib.request.Request(url, headers=self._HEADERS)
            with urllib.request.urlopen(req, timeout=self._FETCH_TIMEOUT) as resp:
                raw = resp.read()
                charset = resp.headers.get_content_charset() or "utf-8"
                try:
                    html = raw.decode(charset)
                except (UnicodeDecodeError, LookupError):
                    try:
                        html = raw.decode("windows-1255")
                    except Exception:
                        html = raw.decode("utf-8", errors="replace")
                content = _strip_html(html)[: self._MAX_CONTENT]
                if content:
                    self._page_cache[url] = content
                return content
        except Exception:
            return None


class SerpApiClient:
    """
    Tavily-compatible client backed by SerpAPI (Google Search).
    Free tier: 100 searches/month, no credit card required.

    Setup:
      1. Sign up at serpapi.com (free, no card)
      2. Copy your API key from the dashboard
      3. Add to .env:  SERPAPI_KEY=your_key_here

    Usage: py main.py ... --search-engine serpapi
    """

    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
        "Accept-Language": "he,en;q=0.9",
    }
    _FETCH_TIMEOUT = 10
    _MAX_CONTENT   = 8000

    def __init__(self, api_key: str):
        self.api_key = api_key
        # Session-level page cache (see DuckDuckGoClient for rationale).
        self._page_cache: dict[str, str] = {}

    def search(self, query: str, max_results: int = 5, **kwargs) -> dict:
        import urllib.request, urllib.parse, json as _json

        params = urllib.parse.urlencode({
            "q":       query,
            "api_key": self.api_key,
            "engine":  "google",
            "hl":      "iw",    # Hebrew interface
            "gl":      "il",    # Israel region
            "num":     min(max_results, 10),
        })
        try:
            req = urllib.request.Request(
                f"https://serpapi.com/search.json?{params}",
                headers={"User-Agent": self._HEADERS["User-Agent"]},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = _json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            body = getattr(exc, "read", lambda: b"")()
            if body:
                try:
                    msg = _json.loads(body).get("error", str(exc))
                except Exception:
                    msg = str(exc)
            else:
                msg = str(exc)
            print(f"    [serpapi error] {msg}")
            return {"results": []}

        results = []
        for item in data.get("organic_results", []):
            url     = item.get("link", "")
            title   = item.get("title", "")
            snippet = item.get("snippet", "")
            raw_content = self._fetch(url) or snippet
            results.append({
                "url": url, "title": title,
                "content": snippet, "raw_content": raw_content,
            })
        return {"results": results}

    def _fetch(self, url: str) -> str | None:
        import urllib.request
        if not url.startswith("http"):
            return None
        if url in self._page_cache:
            print(f"    [page-cache hit] {url[:70]}")
            return self._page_cache[url]
        try:
            req = urllib.request.Request(url, headers=self._HEADERS)
            with urllib.request.urlopen(req, timeout=self._FETCH_TIMEOUT) as resp:
                raw = resp.read()
                charset = resp.headers.get_content_charset() or "utf-8"
                try:
                    html = raw.decode(charset)
                except (UnicodeDecodeError, LookupError):
                    try:
                        html = raw.decode("windows-1255")
                    except Exception:
                        html = raw.decode("utf-8", errors="replace")
                content = _strip_html(html)[: self._MAX_CONTENT]
                if content:
                    self._page_cache[url] = content
                return content
        except Exception:
            return None


class GoogleSearchClient:
    """
    Tavily-compatible client backed by Google Custom Search JSON API.
    Fetches full page content for each result (same approach as DuckDuckGoClient).

    Free tier: 100 queries/day.  Paid: $5 per 1,000 queries.

    Setup (one-time):
      1. console.cloud.google.com → New project → Enable "Custom Search API"
      2. APIs & Services → Credentials → Create API key  → set as GOOGLE_API_KEY
      3. programmablesearchengine.google.com → New engine → "Search the entire web"
         → copy the cx value                              → set as GOOGLE_CSE_ID
      4. Add both to your .env file

    Why Google over Wikipedia for historical political data:
      - Finds news archives, government PDFs, and niche Hebrew sites that
        Wikipedia/Wikidata don't index as structured data.
      - Returns the actual pages; _select_relevant_text handles the windowing.
    """

    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
        "Accept-Language": "he,en;q=0.9",
    }
    _FETCH_TIMEOUT = 10
    _MAX_CONTENT   = 8000

    def __init__(self, api_key: str, cse_id: str):
        self.api_key = api_key
        self.cse_id  = cse_id
        # Session-level page cache (see DuckDuckGoClient for rationale).
        self._page_cache: dict[str, str] = {}

    def search(self, query: str, max_results: int = 5, **kwargs) -> dict:
        import urllib.request, urllib.parse, json as _json

        params = urllib.parse.urlencode({
            "key": self.api_key,
            "cx":  self.cse_id,
            "q":   query,
            "num": min(max_results, 10),
        })
        try:
            req = urllib.request.Request(
                f"https://www.googleapis.com/customsearch/v1?{params}",
                headers={"User-Agent": self._HEADERS["User-Agent"]},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = _json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            # Print response body if available (helps diagnose 400/403 errors)
            body = getattr(exc, "read", lambda: b"")()
            if body:
                try:
                    err = _json.loads(body)
                    msg = err.get("error", {}).get("message", str(exc))
                except Exception:
                    msg = str(exc)
            else:
                msg = str(exc)
            print(f"    [google search error] {msg}")
            return {"results": []}

        results = []
        for item in data.get("items", []):
            url     = item.get("link", "")
            title   = item.get("title", "")
            snippet = item.get("snippet", "")
            raw_content = self._fetch(url) or snippet
            results.append({
                "url": url, "title": title,
                "content": snippet, "raw_content": raw_content,
            })
        return {"results": results}

    def _fetch(self, url: str) -> str | None:
        import urllib.request
        if not url.startswith("http"):
            return None
        if url in self._page_cache:
            print(f"    [page-cache hit] {url[:70]}")
            return self._page_cache[url]
        try:
            req = urllib.request.Request(url, headers=self._HEADERS)
            with urllib.request.urlopen(req, timeout=self._FETCH_TIMEOUT) as resp:
                raw = resp.read()
                charset = resp.headers.get_content_charset() or "utf-8"
                try:
                    html = raw.decode(charset)
                except (UnicodeDecodeError, LookupError):
                    try:
                        html = raw.decode("windows-1255")
                    except Exception:
                        html = raw.decode("utf-8", errors="replace")
                content = _strip_html(html)[: self._MAX_CONTENT]
                if content:
                    self._page_cache[url] = content
                return content
        except Exception:
            return None


def _strip_html(html: str) -> str:
    """
    Extract readable text from HTML using stdlib html.parser.
    Strips tags, decodes entities, collapses whitespace.
    """
    from html.parser import HTMLParser

    class _Stripper(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self._buf: list[str] = []
            self._skip = False

        def handle_starttag(self, tag, attrs):
            if tag in {"script", "style", "nav", "footer", "header"}:
                self._skip = True

        def handle_endtag(self, tag):
            if tag in {"script", "style", "nav", "footer", "header"}:
                self._skip = False

        def handle_data(self, data):
            if not self._skip:
                stripped = data.strip()
                if stripped:
                    self._buf.append(stripped)

        def get_text(self) -> str:
            return "\n".join(self._buf)

    parser = _Stripper()
    parser.feed(html)
    text = parser.get_text()
    # Collapse runs of blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()