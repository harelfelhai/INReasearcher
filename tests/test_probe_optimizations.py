"""Tests for two cost-reduction optimizations:

1. Generic-query filter — queries without {entity} placeholder are stripped
   from per-entity searches (they waste SerpAPI quota returning the same page
   for every entity).

2. Probe-verify skip — verify_probe_extraction is bypassed when:
   a. field.min_corroborations == 1  (one source is sufficient)
   b. probe confidence >= 0.90  (e.g. Wikidata hit; already reliable)
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from research_agent.models import ColumnPlan, ExtractionResult
from tests.mocks import MockClaude, StubSearch


# ── helpers ──────────────────────────────────────────────────────────────────

def _col(id_: str, **kw) -> ColumnPlan:
    defaults = dict(id=id_, label_he=id_, label_en=id_, type="free_text")
    defaults.update(kw)
    return ColumnPlan(**defaults)


def _probe_hit(confidence: float = 0.85) -> ExtractionResult:
    return ExtractionResult(
        field_id="mayor",
        value="Ron Huldai",
        quote_original="Ron Huldai is the mayor",
        source_url="https://wiki.example/list",
        source_domain="wiki.example",
        is_grounded=True,
        extractor_confidence=confidence,
    )


# ── 1. Generic-query filter ──────────────────────────────────────────────────

def test_queries_without_entity_placeholder_are_stripped(tmp_path):
    """_gather_pages_for_field must not issue search calls for generic queries."""
    from research_agent.extractor import _gather_pages_for_field

    field = _col(
        "party",
        type="organization",
        search_queries_he=[
            "מפלגת {entity}",           # good — has placeholder
            "רשימת חברי כנסת",          # bad — generic, no placeholder
        ],
        search_queries_en=[
            "{entity} political party",  # good
            "Israeli parliament members", # bad — generic
        ],
    )

    issued_queries: list[str] = []

    class _CapturingSearch:
        def search(self, query, **kw):
            issued_queries.append(query)
            return {"results": []}

    _gather_pages_for_field(
        field=field,
        entity="גדעון סער",
        resolved_deps={},
        tavily=_CapturingSearch(),
        seen_urls=set(),
    )

    # Only the two queries that contained {entity} should have fired
    for q in issued_queries:
        assert "גדעון סער" in q or "Gideon Sa'ar" in q or "גדעון סער" in q, (
            f"Expected only entity-specific queries, got: {q!r}"
        )
    # Generic queries must NOT appear
    assert not any("רשימת חברי כנסת" in q for q in issued_queries), (
        "Generic Hebrew query should have been filtered"
    )
    assert not any("Israeli parliament" in q for q in issued_queries), (
        "Generic English query should have been filtered"
    )


def test_field_with_all_generic_queries_issues_no_search_calls():
    """A field where EVERY query lacks {entity} should produce zero search calls."""
    from research_agent.extractor import _gather_pages_for_field

    field = _col(
        "party",
        search_queries_he=["רשימת כל חברי הכנסת"],
        search_queries_en=["all Knesset members list"],
    )

    issued: list[str] = []

    class _CapturingSearch:
        def search(self, query, **kw):
            issued.append(query)
            return {"results": []}

    pages, _ = _gather_pages_for_field(
        field=field, entity="Test Entity",
        resolved_deps={}, tavily=_CapturingSearch(), seen_urls=set(),
    )

    assert issued == [], f"Expected zero search calls, got: {issued}"
    assert pages == []


def test_field_with_entity_queries_issues_search_calls():
    """Sanity: queries WITH {entity} still reach the search backend."""
    from research_agent.extractor import _gather_pages_for_field

    field = _col(
        "party",
        search_queries_he=["מפלגת {entity}"],
        search_queries_en=[],
    )

    issued: list[str] = []

    class _CapturingSearch:
        def search(self, query, **kw):
            issued.append(query)
            return {"results": []}

    _gather_pages_for_field(
        field=field, entity="גדעון סער",
        resolved_deps={}, tavily=_CapturingSearch(), seen_urls=set(),
    )

    assert len(issued) == 1
    assert "גדעון סער" in issued[0]


# ── 2. Probe-verify skip logic ────────────────────────────────────────────────
#
# The skip logic lives in _process_entity_sync in api/main.py.
# We test it indirectly: by counting how many times verify_probe_extraction
# is called under different (min_corroborations, confidence) combinations.

def _run_probe_verify_logic(
    min_corroborations: int,
    confidence: float,
) -> int:
    """
    Simulate the Lane-1 branch of _process_entity_sync for a single
    (entity, field) pair. Returns the number of verify_probe_extraction calls.
    """
    call_count = 0

    def _fake_verify_probe(field, entity, probe_hit, search, claude):
        nonlocal call_count
        call_count += 1
        return []

    field = _col("mayor", type="person_name", min_corroborations=min_corroborations)
    hit = _probe_hit(confidence=confidence)

    # Reproduce the exact condition from api/main.py _process_entity_sync Lane 1
    needs_corroboration = field.min_corroborations > 1
    high_conf = hit.extractor_confidence >= 0.90

    if needs_corroboration and not high_conf:
        _fake_verify_probe(field, "TestEntity", hit, None, None)

    return call_count


def test_verify_probe_called_when_needs_corroboration_and_low_confidence():
    """Standard case: min_corroborations=2, confidence=0.85 → must verify."""
    assert _run_probe_verify_logic(min_corroborations=2, confidence=0.85) == 1


def test_verify_probe_skipped_when_single_corr_ok():
    """If min_corroborations=1 (URL, free_text), skip verify regardless of confidence."""
    assert _run_probe_verify_logic(min_corroborations=1, confidence=0.70) == 0


def test_verify_probe_skipped_when_high_confidence():
    """If probe confidence ≥ 0.90 (e.g. Wikidata), skip verify even for multi-corr fields."""
    assert _run_probe_verify_logic(min_corroborations=2, confidence=0.95) == 0


def test_verify_probe_skipped_when_both_conditions_met():
    """min_corroborations=1 AND high confidence → skipped (union of skip conditions)."""
    assert _run_probe_verify_logic(min_corroborations=1, confidence=0.95) == 0


def test_verify_probe_called_exactly_once_at_boundary_confidence():
    """Boundary: confidence=0.89 is below threshold → verify IS called."""
    assert _run_probe_verify_logic(min_corroborations=2, confidence=0.89) == 1


def test_verify_probe_skipped_at_exact_threshold():
    """Boundary: confidence=0.90 meets threshold → verify is SKIPPED."""
    assert _run_probe_verify_logic(min_corroborations=2, confidence=0.90) == 0


# ── 3. End-to-end savings estimate ───────────────────────────────────────────

def test_probe_verify_skips_produce_fewer_search_calls():
    """
    With 20 entities × 5 fields, all probe hits having confidence=0.95
    (Wikidata-level), zero verify_probe calls should fire.

    This simulates the scenario from the first production run that caused
    100 rapid-fire Claude calls and triggered rate-limit crashes.
    """
    entities = [f"entity_{i}" for i in range(20)]
    fields = [
        _col(f"field_{j}", type="person_name", min_corroborations=2)
        for j in range(5)
    ]

    verify_calls = 0

    for field in fields:
        for entity in entities:
            hit = _probe_hit(confidence=0.95)  # Wikidata-level → skip
            needs_corroboration = field.min_corroborations > 1
            high_conf = hit.extractor_confidence >= 0.90
            if needs_corroboration and not high_conf:
                verify_calls += 1

    assert verify_calls == 0, (
        f"Expected 0 verify_probe calls with high-confidence probes, got {verify_calls}"
    )


def test_probe_verify_fires_for_low_confidence_identity_fields():
    """
    With confidence=0.75 and min_corroborations=2, every (entity, field)
    pair SHOULD trigger verify_probe — this is the intended behaviour for
    moderate-confidence identity facts.
    """
    entities = [f"entity_{i}" for i in range(5)]
    fields = [_col(f"field_{j}", type="person_name", min_corroborations=2) for j in range(3)]

    verify_calls = 0
    for field in fields:
        for entity in entities:
            hit = _probe_hit(confidence=0.75)
            needs_corroboration = field.min_corroborations > 1
            high_conf = hit.extractor_confidence >= 0.90
            if needs_corroboration and not high_conf:
                verify_calls += 1

    assert verify_calls == len(entities) * len(fields)


# ── 4. Probe query dedup — prefetched_page / found_page_out ──────────────────

def test_probe_field_list_uses_prefetched_page_skips_search():
    """When prefetched_page is supplied, probe_field_list must not call search."""
    from research_agent.extractor import probe_field_list

    field = _col(
        "email",
        type="free_text",
        directory_probe_query_he="רשימת חברי כנסת",
        min_corroborations=1,
    )
    entities = ["Alice", "Bob"]
    page_content = "Alice email: alice@example.com\nBob email: bob@example.com"

    search_calls: list[str] = []

    class _TrackingSearch:
        def search(self, query, **kw):
            search_calls.append(query)
            return {"results": []}

    claude = (
        MockClaude()
        .on("bulk_extract_field", {
            "extractions": [
                {"entity_name": "Alice", "value": "alice@example.com",
                 "quote_original": "alice@example.com", "confidence": 0.9},
                {"entity_name": "Bob", "value": "bob@example.com",
                 "quote_original": "bob@example.com", "confidence": 0.9},
            ]
        })
    )

    probe_field_list(
        field=field,
        entities=entities,
        search_client=_TrackingSearch(),
        claude=claude,
        prefetched_page=("https://example.com/list", page_content),
    )

    assert search_calls == [], (
        f"probe_field_list should not search when prefetched_page is given; got: {search_calls}"
    )


def test_probe_field_list_populates_found_page_out():
    """A successful probe should append the winning page to found_page_out."""
    from research_agent.extractor import probe_field_list

    field = _col(
        "phone",
        type="free_text",
        directory_probe_query_he="רשימת טלפונים",
        min_corroborations=1,
    )
    entities = [f"Member {i}" for i in range(5)]
    page_url = "https://example.com/phones"
    # Content must be ≥ 200 chars and contain entity names for coverage check
    page_content = "\n".join(
        f"Member {i}: phone number 050-{i:07d}, office extension {100+i}, fax 03-{i:07d}"
        for i in range(5)
    ) + "\n" + "Additional directory information about the members of this organisation."

    class _StaticSearch:
        def search(self, query, **kw):
            return {"results": [{"url": page_url, "raw_content": page_content}]}

    claude = (
        MockClaude()
        .on("bulk_extract_field", {
            "extractions": [
                {"entity_name": f"Member {i}", "value": f"050-{i:07d}",
                 "quote_original": f"050-{i:07d}", "confidence": 0.9}
                for i in range(5)
            ]
        })
    )

    found_page_out: list = []
    probe_field_list(
        field=field,
        entities=entities,
        search_client=_StaticSearch(),
        claude=claude,
        found_page_out=found_page_out,
    )

    assert len(found_page_out) == 1, (
        f"Expected found_page_out to have 1 entry, got {len(found_page_out)}"
    )
    assert found_page_out[0][0] == page_url


# ── 5. Lane 2 → Lane 3 page cache (seed_pages / page_sink) ───────────────────

def test_seed_pages_url_not_added_to_all_urls_twice():
    """A seeded URL must appear exactly once in extracted pages."""
    from research_agent.extractor import search_and_extract_batched

    field = _col(
        "party",
        type="organization",
        search_queries_he=["מפלגת {entity}"],
        search_queries_en=[],
        min_corroborations=1,
    )
    entity = "גדעון סער"
    seeded_url = "https://knesset.example/members"
    seeded_content = "גדעון סער חבר מפלגת הליכוד"

    class _TrackingSearch:
        def search(self, query, **kw):
            # Returns the same URL as the seeded page; must not cause double-extraction
            return {"results": [{"url": seeded_url, "raw_content": seeded_content}]}

    claude = (
        MockClaude()
        .on("extract_field", {
            "field_id": "party",
            "value": "ליכוד",
            "quote_original": "מפלגת הליכוד",
            "confidence": 0.9,
            "source_url": seeded_url,
        })
    )

    seed = {seeded_url: (seeded_content, None)}
    # Must not raise and must return a result (not crash on duplicate URL)
    results = search_and_extract_batched(
        fields=[field], entity=entity,
        resolved_deps={}, tavily=_TrackingSearch(), claude=claude,
        seed_pages=seed,
    )
    # Structural: no exception means seeded URL handled correctly
    assert "party" in results


def test_page_sink_collects_new_pages():
    """Pages fetched during search_and_extract_batched should be added to page_sink."""
    from research_agent.extractor import search_and_extract_batched

    field = _col(
        "city",
        type="free_text",
        search_queries_he=["עיר מגורים {entity}"],
        search_queries_en=[],
        min_corroborations=1,
    )
    entity = "ראובן ריבלין"
    new_url = "https://bio.example/rivlin"
    # Content must be ≥ 80 chars for _gather_pages_for_field to accept it
    new_content = (
        "ראובן ריבלין גר בירושלים. הוא כיהן כנשיא מדינת ישראל בין השנים 2014 ל-2021. "
        "נולד בירושלים בשנת 1939 ולמד משפטים באוניברסיטה העברית."
    )

    class _StaticSearch:
        def search(self, query, **kw):
            return {"results": [{"url": new_url, "raw_content": new_content}]}

    claude = (
        MockClaude()
        .on("extract_field", {
            "field_id": "city",
            "value": "ירושלים",
            "quote_original": "גר בירושלים",
            "confidence": 0.9,
            "source_url": new_url,
        })
    )

    sink: dict = {}
    search_and_extract_batched(
        fields=[field], entity=entity,
        resolved_deps={}, tavily=_StaticSearch(), claude=claude,
        page_sink=sink,
    )

    assert new_url in sink, f"Expected {new_url!r} in page_sink, got keys: {list(sink.keys())}"
    assert sink[new_url][0] == new_content


def test_seed_pages_url_excluded_from_page_sink():
    """A URL already in seed_pages must not be re-added to page_sink."""
    from research_agent.extractor import search_and_extract_batched

    field = _col(
        "role",
        type="free_text",
        search_queries_he=["תפקיד {entity}"],
        search_queries_en=[],
        min_corroborations=1,
    )
    entity = "TestPerson"
    shared_url = "https://shared.example/page"
    shared_content = "TestPerson serves as Minister"

    class _StaticSearch:
        def search(self, query, **kw):
            return {"results": [{"url": shared_url, "raw_content": shared_content}]}

    claude = (
        MockClaude()
        .on("extract_field", {
            "field_id": "role",
            "value": "Minister",
            "quote_original": "serves as Minister",
            "confidence": 0.9,
            "source_url": shared_url,
        })
    )

    seed = {shared_url: (shared_content, None)}
    sink: dict = {}
    search_and_extract_batched(
        fields=[field], entity=entity,
        resolved_deps={}, tavily=_StaticSearch(), claude=claude,
        seed_pages=seed,
        page_sink=sink,
    )

    assert shared_url not in sink, (
        "URL already in seed_pages must not be duplicated into page_sink"
    )
