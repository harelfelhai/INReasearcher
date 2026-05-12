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
from .memory import (
    SuccessMemory,
    format_extraction_examples,
    format_extraction_warnings,
)

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


def _select_relevant_text(content: str, field, entity: str) -> str:
    """
    For short content (≤ _BUDGET chars) → returns as-is.
    For long content → density-scored windowing:
      1. Divide the article into overlapping chunks
      2. Score each chunk by how many DISTINCT field-relevant keywords appear in it
      3. Take the highest-scoring chunks (most relevant sections) until budget is full
      4. Always prepend the article intro for context

    Density scoring beats naive per-hit windowing because common words like
    'ראש העיר' appear hundreds of times and would fill the budget with irrelevant
    passages. A chunk containing role + year + name scores higher than one that
    only contains the role keyword once.
    """
    if not content:
        return ""
    if len(content) <= _BUDGET:
        return content

    # 1. Build keyword set
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
    kw_list = [kw.lower() for kw in keywords if len(kw) >= 2]

    content_lower = content.lower()

    if not kw_list:
        return content[:_BUDGET]

    # 2. Score overlapping chunks by distinct keyword count
    scored: list[tuple[int, int, int]] = []  # (score, start, end)
    for start in range(0, len(content), _CHUNK_STEP):
        end = min(start + _CHUNK_SIZE, len(content))
        chunk = content_lower[start:end]
        score = sum(1 for kw in kw_list if kw in chunk)
        if score > 0:
            scored.append((score, start, end))

    if not scored:
        return content[:_BUDGET]

    # 3. Pick chunks greedily by score (highest first), merge overlapping
    scored.sort(key=lambda x: -x[0])
    selected: list[tuple[int, int]] = []
    for _, s, e in scored:
        # Merge with any already-selected overlapping interval
        merged_s, merged_e = s, e
        remaining = []
        for ps, pe in selected:
            if merged_s <= pe and ps <= merged_e:
                merged_s = min(merged_s, ps)
                merged_e = max(merged_e, pe)
            else:
                remaining.append((ps, pe))
        remaining.append((merged_s, merged_e))
        selected = remaining

    # Sort selected windows by position for output
    selected.sort()

    # 4. Assemble: intro + top windows, up to budget
    intro = content[:_INTRO_CHARS]
    pieces: list[str] = [intro]
    used = len(intro)

    for s, e in selected:
        if e <= _INTRO_CHARS:
            continue
        if s < _INTRO_CHARS:
            s = _INTRO_CHARS
        section = content[s:e]
        gap = len(_ELLIPSIS)
        if used + len(section) + gap > _BUDGET:
            remaining_budget = _BUDGET - used - gap
            if remaining_budget > 200:
                pieces.append(_ELLIPSIS)
                pieces.append(section[:remaining_budget])
            break
        pieces.append(_ELLIPSIS)
        pieces.append(section)
        used += len(section) + gap

    result = "".join(pieces)
    entity_short = entity[:20]
    hit_kws = [kw for kw in kw_list if kw in content_lower][:5]
    print(f"    [window] {entity_short!r} field={field.id!r}: "
          f"{len(content):,}→{len(result):,} chars, "
          f"top keywords: {hit_kws}")
    return result


def extract_from_source(
    field: ColumnPlan,
    entity: str,
    source_url: str,
    source_content: str,
    resolved_deps: dict,
    client: anthropic.Anthropic,
    memory: SuccessMemory | None = None,
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

    max_tokens = min(4096, 512 + 64 * len(entities))
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        system=_BULK_EXTRACTOR_SYSTEM,
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


def probe_field_list(
    field,
    entities: list[str],
    search_client,
    claude: anthropic.Anthropic,
    min_coverage: float = _MIN_PROBE_COVERAGE,
) -> dict[str, ExtractionResult] | None:
    """
    Try to find a directory/list page covering `field` for many entities at once.

    Returns a dict of entity_name → ExtractionResult when a good directory page is
    found.  Returns None if no suitable page exists (caller falls back to per-entity
    search).

    Cost savings: replaces N search calls + N Claude extraction calls with
    1 search call + 1 Claude bulk-extraction call.
    """
    probe_query = field.directory_probe_query_he
    if not probe_query:
        return None

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

        # Only accept if we actually extracted something useful
        if found_count >= max(1, len(entities) * 0.10):
            return bulk

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

    is_wikipedia = isinstance(tavily, WikipediaSearchClient)
    query_cap = 3 if is_wikipedia else 6

    # For Wikipedia: fetch the entity's own article first (most reliable source).
    # For person_name fields also fetch the dedicated "ראשי עיר {entity}" page
    # (Wikipedia maintains this separately from the main city article).
    # For url fields also query Wikidata (P856 = official website property).
    if is_wikipedia:
        extra = []
        if field.type == "person_name" and field.temporal_anchor:
            extra.append(f"ראשי עיר {entity}")

        direct = tavily.fetch_entity_article(entity, extra_titles=extra)

        if field.type in ("url", "person_name"):
            # Use the canonical Wikipedia title (post-redirect) for Wikidata lookup,
            # because Wikidata sitelinks use canonical titles not redirect aliases.
            # e.g. "תל אביב" → Wikidata: not found; "תל אביב-יפו" → Q33935: found
            canonical = (direct["results"][0]["title"]
                         if direct.get("results") else entity)
            _inject_wikidata(canonical, field, results, seen_urls)
        for hit in direct.get("results", []):
            url = hit.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                content = hit.get("raw_content") or hit.get("content", "")
                if content and len(content) >= 80:
                    extraction = extract_from_source(
                        field=field, entity=entity,
                        source_url=url, source_content=content,
                        resolved_deps=resolved_deps, client=claude, memory=memory,
                    )
                    if extraction.value:
                        results.append(extraction)

    for query in all_queries[:query_cap]:
        try:
            response = tavily.search(
                query=query,
                max_results=3,
                include_raw_content=True,
            )
        except Exception as exc:
            print(f"    [search error] {query[:60]!r}: {exc}")
            continue

        hits = response.get("results", [])
        if hits:
            print(f"    [search] '{query[:55]}' → {len(hits)} result(s): "
                  + ", ".join(h.get("title", h.get("url","?"))[:30] for h in hits[:3]))
        for hit in hits:
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
                memory=memory,
            )

            if extraction.is_grounded and extraction.value:
                results.append(extraction)
                if len(results) >= early_stop_on_high:
                    return results   # enough corroborations, stop burning tokens

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
                return _strip_html(html)[: self._MAX_CONTENT]
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
                return _strip_html(html)[: self._MAX_CONTENT]
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
                return _strip_html(html)[: self._MAX_CONTENT]
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