"""Integration tests for extractor primitives with MockClaude + StubSearch.

Covers (1 LLM call per test, all faked):
  - bulk_extract_from_source       — probe page → N entities
  - probe_field_list               — search + coverage filter + bulk extract
  - discover_entities              — discovery search → list + harvest, with
                                     substring grounding of every quote
  - batch_extract_fields_from_source — multi-field one-page Claude call
  - verify_probe_extraction        — value-anchored search after a probe hit
"""
from __future__ import annotations

from research_agent.extractor import (
    bulk_extract_from_source,
    probe_field_list,
    discover_entities,
    batch_extract_fields_from_source,
    verify_probe_extraction,
)
from research_agent.models import (
    ColumnPlan,
    EntityDiscoveryPlan,
    ExtractionResult,
)

from tests.mocks import MockClaude, StubSearch


def _col(id_: str, **kwargs) -> ColumnPlan:
    defaults = dict(
        id=id_,
        label_he=id_,
        label_en=id_,
        type="free_text",
    )
    defaults.update(kwargs)
    return ColumnPlan(**defaults)


# ── bulk_extract_from_source ─────────────────────────────────────────────────

def test_bulk_extract_returns_one_result_per_entity_and_grounds_quotes():
    field = _col("year_est", type="number")
    entities = ["תל אביב", "חיפה", "ירושלים"]
    content = (
        "Tel Aviv was founded in 1909. תל אביב נוסדה ב-1909. "
        "Haifa was founded in 1761. חיפה נוסדה ב-1761. "
        "Jerusalem's age is disputed."
    )

    claude = MockClaude().on("bulk_extract_field", {
        "extractions": [
            {"entity_name": "תל אביב", "value": "1909",
             "quote_original": "תל אביב נוסדה ב-1909", "confidence": 0.95},
            {"entity_name": "חיפה", "value": "1761",
             "quote_original": "חיפה נוסדה ב-1761", "confidence": 0.90},
            # Last one returns null — must propagate as not-found, not crash.
            {"entity_name": "ירושלים", "value": None,
             "quote_original": None, "confidence": 0.0},
        ]
    })

    out = bulk_extract_from_source(
        field=field, entities=entities,
        source_url="https://example.com/list",
        source_content=content,
        client=claude,
    )

    assert set(out.keys()) == set(entities)
    assert out["תל אביב"].value == "1909"
    assert out["תל אביב"].is_grounded is True
    assert out["חיפה"].value == "1761"
    assert out["ירושלים"].value is None

    # Exactly one LLM call regardless of entity count
    assert len(claude.calls_for("bulk_extract_field")) == 1


def test_bulk_extract_marks_ungrounded_when_value_not_in_source():
    """If both value and quote are absent from the source, is_grounded must
    be False — primary hallucination safety rail."""
    field = _col("mayor", type="person_name")
    claude = MockClaude().on("bulk_extract_field", {
        "extractions": [
            {"entity_name": "תל אביב", "value": "Someone Not Real",
             "quote_original": "TOTALLY FABRICATED QUOTE",
             "confidence": 0.99},
        ]
    })
    out = bulk_extract_from_source(
        field=field, entities=["תל אביב"],
        source_url="https://example.com/mayors",
        source_content="A long page about mayors. Mayor X did things. " * 10,
        client=claude,
    )
    assert out["תל אביב"].is_grounded is False


# ── probe_field_list ─────────────────────────────────────────────────────────

def test_probe_field_list_uses_directory_page_when_coverage_passes():
    field = _col(
        "founded", type="number",
        directory_probe_query_he="רשימת ערים בישראל לפי שנת ייסוד",
    )
    entities = ["תל אביב", "חיפה", "ירושלים", "באר שבע"]
    # Directory page that mentions every entity (high coverage) and has a
    # list-keyword so the probe accepts it.
    dir_content = (
        "רשימה מקיפה של ערים בישראל לפי שנת הייסוד הרשמית. "
        "הרשימה כוללת את כל המוקדים העירוניים הגדולים במדינה. "
        "תל אביב נוסדה ב-1909 כשכונה צפונית ליפו ועד מהרה הפכה למרכז הכלכלי. "
        "חיפה נוסדה ב-1761 על ידי דאהר אל-עומר ושימשה כנמל מרכזי. "
        "ירושלים היא עיר עתיקה שמועדי ייסודה הראשונים אינם ידועים בוודאות. "
        "באר שבע נוסדה מחדש ב-1900 על ידי השלטון העות'מאני כעיר מחוז."
    )

    search = StubSearch().add(None, [
        {"url": "https://wiki.example/list", "raw_content": dir_content,
         "content": dir_content},
    ])
    claude = MockClaude().on("bulk_extract_field", {
        "extractions": [
            {"entity_name": e, "value": "1909",
             "quote_original": "תל אביב 1909" if e == "תל אביב"
                               else "חיפה 1761",
             "confidence": 0.9}
            for e in entities
        ]
    })

    out = probe_field_list(field, entities, search, claude)
    assert out is not None
    assert set(out.keys()) == set(entities)
    # Exactly one search + one bulk extract call — the whole point of probe
    assert len(search.search_log) == 1
    assert len(claude.calls_for("bulk_extract_field")) == 1


def test_probe_field_list_returns_none_when_no_query():
    field = _col("x")
    out = probe_field_list(field, ["e1"], StubSearch(), MockClaude())
    assert out is None


def test_probe_field_list_returns_none_when_coverage_too_low():
    field = _col(
        "x", directory_probe_query_he="some directory query",
    )
    # Page mentions only one of 10 entities and has no directory keyword
    entities = [f"entity_{i}" for i in range(10)]
    bad_page = "this page only talks about entity_0 in passing"
    search = StubSearch().add(None, [
        {"url": "u", "raw_content": bad_page, "content": bad_page},
    ])
    claude = MockClaude()  # never called
    out = probe_field_list(field, entities, search, claude)
    assert out is None
    assert len(claude.calls) == 0


# ── discover_entities ────────────────────────────────────────────────────────

def test_discover_entities_grounds_quotes_and_filters_invalid_harvest():
    plan = EntityDiscoveryPlan(
        query_he="10 הערים הגדולות בישראל לפי אוכלוסייה",
        query_en="top 10 cities in Israel by population",
        expected_count=3,
        extraction_hint="ordered list of city names",
    )
    page = (
        "Top cities in Israel by population (2023 census data):\n"
        "This is the official ranked list maintained by the central bureau "
        "of statistics. Population figures are end-of-year estimates.\n"
        "1. ירושלים – 952,000, founded ancient times. The capital city.\n"
        "2. תל אביב – 467,000, founded 1909 as a northern suburb of Jaffa.\n"
        "3. חיפה – 285,000, founded 1761 by Dahir al-Umar as a port city."
    )
    columns = [_col("year_est", type="number")]

    search = StubSearch().add(None, [
        {"url": "https://wiki.example/cities", "raw_content": page, "content": page},
    ])

    claude = MockClaude().on("discover_entities", {
        "entities": [
            {"name": "ירושלים", "rank": 1, "quote": "1. ירושלים – 952,000"},
            {"name": "תל אביב", "rank": 2, "quote": "2. תל אביב – 467,000"},
            {"name": "חיפה",   "rank": 3, "quote": "3. חיפה – 285,000"},
        ],
        "harvested": [
            # Valid — quote substring is on the page
            {"entity_name": "תל אביב", "field_id": "year_est",
             "value": "1909", "quote": "founded 1909"},
            # Bogus field_id — must be dropped
            {"entity_name": "תל אביב", "field_id": "doesnt_exist",
             "value": "x", "quote": "founded 1909"},
            # Quote NOT on page — must be dropped
            {"entity_name": "חיפה", "field_id": "year_est",
             "value": "1761", "quote": "this exact string is not in the page"},
        ],
    })

    out = discover_entities(plan, columns, search, claude)
    assert out is not None
    assert [e.name for e in out.entities] == ["ירושלים", "תל אביב", "חיפה"]
    # Only the one valid harvest row survives
    assert len(out.harvested) == 1
    assert out.harvested[0].entity_name == "תל אביב"
    assert out.harvested[0].field_id == "year_est"
    assert out.source_domain == "wiki.example"


def test_discover_entities_returns_none_on_empty_search():
    plan = EntityDiscoveryPlan(query_he="q", query_en="q")
    search = StubSearch()  # no results
    claude = MockClaude()
    assert discover_entities(plan, [_col("x")], search, claude) is None
    assert len(claude.calls) == 0


# ── batch_extract_fields_from_source ─────────────────────────────────────────

def test_batch_extract_returns_one_result_per_field_from_one_llm_call():
    fields = [_col("mayor", type="person_name"), _col("website", type="url")]
    page = "Tel Aviv: mayor Ron Huldai, official site www.tel-aviv.gov.il"

    claude = MockClaude().on("batch_extract_fields", {
        "extractions": [
            {"field_id": "mayor", "value": "Ron Huldai",
             "quote_original": "mayor Ron Huldai", "confidence": 0.95},
            {"field_id": "website", "value": "www.tel-aviv.gov.il",
             "quote_original": "official site www.tel-aviv.gov.il",
             "confidence": 0.9},
        ]
    })

    out = batch_extract_fields_from_source(
        fields=fields, entity="תל אביב",
        source_url="https://example/tlv", source_content=page, client=claude,
    )
    assert out["mayor"].value == "Ron Huldai"
    assert out["website"].value == "www.tel-aviv.gov.il"
    assert out["mayor"].is_grounded and out["website"].is_grounded
    assert len(claude.calls_for("batch_extract_fields")) == 1


# ── verify_probe_extraction ──────────────────────────────────────────────────

def test_verify_probe_extraction_runs_value_anchored_search():
    field = _col("mayor", type="person_name",
                 search_queries_he=["ראש העיר של {entity}"],
                 search_queries_en=["mayor of {entity}"])
    probe_hit = ExtractionResult(
        field_id="mayor", value="Ron Huldai",
        quote_original="Ron Huldai serves as mayor",
        source_url="https://wiki.example/list",
        source_domain="wiki.example", is_grounded=True,
        extractor_confidence=0.85,
    )
    # Stub a verification page that confirms the value
    verify_page = "Ron Huldai is the mayor of Tel Aviv since 1998."
    search = (
        StubSearch().add(None, [
            {"url": "https://verify.example/huldai", "raw_content": verify_page,
             "content": verify_page},
        ])
    )
    claude = MockClaude().on("extract_field", {
        "value": "Ron Huldai",
        "quote_original": "Ron Huldai is the mayor of Tel Aviv",
        "confidence": 0.95,
    })

    extras = verify_probe_extraction(field, "Tel Aviv", probe_hit, search, claude)
    assert isinstance(extras, list)
    # Must have done at least one verification search
    assert len(search.search_log) >= 1
    # And the search query should reference the probed value (value-anchored)
    assert any("Ron Huldai" in q for q in search.search_log)
