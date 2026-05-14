"""End-to-end orchestration tests for `research_entity` (main.py).

These exercise the full three-lane dispatch with zero network and zero LLM
cost: a MockClaude returns canned tool outputs for `extract_field`,
`bulk_extract_field`, and `batch_extract_fields`; a StubSearch returns
canned pages per query.

The tests assert call counts as well as cell values — the call counts are
where regressions in the lane router would show up first.
"""
from __future__ import annotations

from main import research_entity
from research_agent.models import (
    ColumnPlan,
    ExtractionResult,
    ResearchPlan,
)

from tests.mocks import MockClaude, StubSearch


def _col(id_: str, **kwargs) -> ColumnPlan:
    defaults = dict(
        id=id_, label_he=id_, label_en=id_, type="free_text",
        search_queries_he=[f"{id_} of {{entity}}"],
        search_queries_en=[f"{id_} of {{entity}}"],
    )
    defaults.update(kwargs)
    return ColumnPlan(**defaults)


def _long_page(prefix: str, *snippets: str) -> str:
    """Pad page content so it clears the extractor's 200-char minimum."""
    filler = " " + "Padding paragraph to clear the minimum content length. " * 6
    return prefix + " " + " ".join(snippets) + filler


# ── Three-lane routing: probe hit + plain field + deferred field ─────────────

def test_three_lane_orchestration_routes_each_field_correctly():
    mayor = _col("mayor", type="person_name")
    website = _col("website", type="url",
                   depends_on="mayor")     # deferred — needs mayor first
    year = _col("year_est", type="number")  # will arrive via probe hit
    plan = ResearchPlan(
        entity_type="city",
        research_question_original="test",
        columns=[year, mayor, website],
    )

    # ── Lane 1 seed: probe already supplied `year_est` for this entity.
    probe_seed = {
        "year_est": {
            "Tel Aviv": ExtractionResult(
                field_id="year_est", value="1909",
                quote_original="Tel Aviv was founded in 1909",
                source_url="https://wiki.example/cities",
                source_domain="wiki.example",
                is_grounded=True,
                extractor_confidence=0.9,
            )
        }
    }

    # ── Stub search: returns one page per field query.
    mayor_page = _long_page(
        "Tel Aviv mayor page:",
        "Ron Huldai served as mayor of Tel Aviv from 1998 onwards.",
    )
    website_page = _long_page(
        "Tel Aviv official site:",
        "The municipality of Tel Aviv operates www.tel-aviv.gov.il.",
    )
    verify_page = _long_page(
        "Founding context:",
        "Tel Aviv was officially founded in 1909 as a new district.",
    )
    search = (
        StubSearch()
          .add("mayor",   [{"url": "https://m.ex/tlv", "raw_content": mayor_page,
                            "content": mayor_page}])
          .add("website", [{"url": "https://w.ex/tlv", "raw_content": website_page,
                            "content": website_page}])
          # Lane 1 verification search: value-anchored, contains "1909".
          .add("1909",    [{"url": "https://v.ex/year", "raw_content": verify_page,
                            "content": verify_page}])
    )

    # ── Mock Claude responses for every tool the orchestrator may invoke.
    claude = (
        MockClaude()
          # Lane 1 verification: single-field extractor (extract_field).
          .on("extract_field", {
              "value": "1909",
              "quote_original": "Tel Aviv was officially founded in 1909",
              "confidence": 0.95,
          })
          # Lane 2: only `mayor` (single-field path).
          .on("batch_extract_fields", lambda call: {
              "extractions": [
                  {"field_id": "mayor", "value": "Ron Huldai",
                   "quote_original": "Ron Huldai served as mayor of Tel Aviv",
                   "confidence": 0.95},
              ]
          })
    )

    # search_and_extract_batched with a single field falls back to
    # extract_from_source (uses `extract_field` tool). Lane 2 has only `mayor`
    # and Lane 3 has only `website` — so both go through extract_field. The
    # queue lets us return different values per call.
    claude.responses["extract_field"] = [
        # Lane 1 verification (called first)
        {"value": "1909",
         "quote_original": "Tel Aviv was officially founded in 1909",
         "confidence": 0.95},
        # Lane 2: mayor
        {"value": "Ron Huldai",
         "quote_original": "Ron Huldai served as mayor of Tel Aviv",
         "confidence": 0.95},
        # Lane 3: website — must see "Ron Huldai" anywhere in the prompt to
        # confirm the dep was resolved.
        {"value": "www.tel-aviv.gov.il",
         "quote_original": "operates www.tel-aviv.gov.il",
         "confidence": 0.9},
    ]

    result = research_entity(
        entity="Tel Aviv", plan=plan,
        tavily=search, claude=claude, memory=None,
        probe_results=probe_seed,
    )

    # ── All three lanes succeeded
    assert result.cells["year_est"].value == "1909"
    assert result.cells["mayor"].value == "Ron Huldai"
    assert result.cells["website"].value == "www.tel-aviv.gov.il"
    assert "majority_not_found" not in result.row_flags

    # ── Lane 3 saw the resolved dep: the website extractor's prompt must
    # contain "Ron Huldai" because _gather_pages_for_field injects the
    # resolved dep value into the bonus queries. We can also check the
    # search log for it.
    assert any("Ron Huldai" in q for q in search.search_log), (
        f"Lane 3 should have used resolved dep in queries. Log: {search.search_log}"
    )

    # ── Lane 1 must have triggered exactly ONE verification search +
    # ONE extract_field call before Lane 2 ran.
    # Total extract_field calls = 3 (verify + mayor + website).
    assert len(claude.calls_for("extract_field")) == 3


# ── majority_not_found flag ──────────────────────────────────────────────────

def test_majority_not_found_flag_set_when_most_cells_are_empty():
    plan = ResearchPlan(
        entity_type="city",
        research_question_original="test",
        columns=[_col("a"), _col("b"), _col("c")],
    )
    # Search returns nothing for every query → all cells empty.
    search = StubSearch()  # empty rules → empty results
    claude = MockClaude()  # never called because no pages reach Claude

    result = research_entity(
        entity="X", plan=plan,
        tavily=search, claude=claude, memory=None,
    )
    not_found = sum(1 for c in result.cells.values() if c.confidence == "NOT_FOUND")
    assert not_found == 3
    assert "majority_not_found" in result.row_flags
    # No LLM calls should have been made — no pages to extract from
    assert len(claude.calls) == 0


# ── Lane 2 only (no probe, no deps) → single batched call per page ──────────

def test_lane2_dedups_two_fields_sharing_one_page():
    """When two fields' searches return the same URL, the batched extractor
    must call Claude exactly ONCE for that page (one batch_extract_fields
    call), not once per field."""
    mayor = _col("mayor", type="person_name")
    website = _col("website", type="url")
    plan = ResearchPlan(
        entity_type="city", research_question_original="t",
        columns=[mayor, website],
    )

    shared_page = _long_page(
        "Tel Aviv wiki:",
        "Ron Huldai serves as mayor and the official site is "
        "www.tel-aviv.gov.il for all communications.",
    )
    # Both fields' queries return the SAME URL → must be deduped.
    search = (
        StubSearch()
          .add(None, [{"url": "https://wiki.example/tlv",
                       "raw_content": shared_page, "content": shared_page}])
    )

    claude = MockClaude().on("batch_extract_fields", {
        "extractions": [
            {"field_id": "mayor", "value": "Ron Huldai",
             "quote_original": "Ron Huldai serves as mayor", "confidence": 0.95},
            {"field_id": "website", "value": "www.tel-aviv.gov.il",
             "quote_original": "official site is www.tel-aviv.gov.il",
             "confidence": 0.95},
        ]
    })

    result = research_entity(
        entity="Tel Aviv", plan=plan,
        tavily=search, claude=claude, memory=None,
    )

    # The page-dedup optimisation: only ONE batch call for the shared page.
    assert len(claude.calls_for("batch_extract_fields")) == 1
    assert result.cells["mayor"].value == "Ron Huldai"
    assert result.cells["website"].value == "www.tel-aviv.gov.il"
