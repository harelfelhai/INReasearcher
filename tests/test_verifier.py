"""Tests for research_agent.verifier — consensus + confidence logic."""
import pytest
from research_agent.models import ColumnPlan, ExtractionResult
from research_agent.verifier import (
    verify_field, _is_preferred, _is_structured_authoritative,
)


# ── _is_structured_authoritative ──────────────────────────────────────────────

def test_structured_authoritative_knesset():
    assert _is_structured_authoritative("knesset.gov.il")
    assert _is_structured_authoritative("www.knesset.gov.il")


def test_structured_authoritative_wikidata():
    assert _is_structured_authoritative("wikidata.org")
    assert _is_structured_authoritative("www.wikidata.org")


def test_structured_authoritative_negative():
    assert not _is_structured_authoritative("example.com")
    assert not _is_structured_authoritative("wikipedia.org")
    assert not _is_structured_authoritative("")
    assert not _is_structured_authoritative("notknesset.gov.il")


# ── Single-source structured-authoritative override ───────────────────────────

def test_knesset_single_source_promotes_low_to_high():
    """One hit from knesset.gov.il is HIGH even with min_corroborations=2."""
    field = make_field(min_corroborations=2)
    ext = make_extraction(value="הליכוד", domain="knesset.gov.il")
    cell = verify_field(field, [ext])
    assert cell.confidence == "HIGH"
    assert "structured_authoritative_source" in cell.flags
    assert not any(f.startswith("below_min_corroborations") for f in cell.flags)


def test_wikidata_single_source_promotes_low_to_high():
    field = make_field(min_corroborations=2)
    ext = make_extraction(value="https://example.gov.il", domain="www.wikidata.org")
    cell = verify_field(field, [ext])
    assert cell.confidence == "HIGH"
    assert "structured_authoritative_source" in cell.flags


def test_regular_domain_does_not_get_structured_promotion():
    """No promotion for non-structured sources — must use normal corroboration."""
    field = make_field(min_corroborations=2)
    ext = make_extraction(value="הליכוד", domain="he.wikipedia.org")
    cell = verify_field(field, [ext])
    assert cell.confidence == "LOW"
    assert any(f.startswith("below_min_corroborations") for f in cell.flags)


# ── helpers ──────────────────────────────────────────────────────────────────

def make_field(
    field_id: str = "test_field",
    min_corroborations: int = 2,
    preferred: list[str] | None = None,
) -> ColumnPlan:
    return ColumnPlan(
        id=field_id,
        label_he="תיאור",
        label_en="Field",
        type="person_name",
        search_queries_he=["{entity}"],
        search_queries_en=["{entity}"],
        preferred_source_domains=preferred or [],
        min_corroborations=min_corroborations,
    )


def make_extraction(
    value: str | None = "John Smith",
    domain: str = "example.com",
    is_grounded: bool = True,
    quote: str = "John Smith was the mayor",
    confidence: float = 0.9,
) -> ExtractionResult:
    return ExtractionResult(
        field_id="test_field",
        value=value,
        quote_original=quote,
        source_url=f"http://{domain}/page",
        source_domain=domain,
        is_grounded=is_grounded,
        extractor_confidence=confidence,
    )


# ── _is_preferred ────────────────────────────────────────────────────────────

def test_preferred_exact_match():
    assert _is_preferred("gov.il", ["gov.il"]) is True


def test_preferred_suffix_match():
    assert _is_preferred("foo.muni.gov.il", ["gov.il"]) is True


def test_preferred_strips_www():
    assert _is_preferred("www.sec.gov", ["sec.gov"]) is True


def test_preferred_no_match():
    assert _is_preferred("randomsite.com", ["gov.il", "sec.gov"]) is False


def test_preferred_empty_list():
    assert _is_preferred("anything.com", []) is False


def test_preferred_substring_does_not_match():
    # "gov.il" should not match "notgov.il" (suffix logic must be strict)
    assert _is_preferred("notgov.il", ["gov.il"]) is False


# ── NOT_FOUND scenarios ──────────────────────────────────────────────────────

def test_not_found_when_no_extractions():
    field = make_field()
    cell = verify_field(field, [])
    assert cell.confidence == "NOT_FOUND"
    assert cell.value is None
    assert cell.corroboration_count == 0
    assert "no_grounded_sources" in cell.flags


def test_not_found_when_all_ungrounded():
    field = make_field()
    cell = verify_field(field, [
        make_extraction(is_grounded=False),
        make_extraction(is_grounded=False, domain="other.com"),
    ])
    assert cell.confidence == "NOT_FOUND"


def test_not_found_when_all_values_null():
    field = make_field()
    cell = verify_field(field, [
        make_extraction(value=None),
        make_extraction(value=None, domain="other.com"),
    ])
    assert cell.confidence == "NOT_FOUND"


# ── Consensus & confidence ───────────────────────────────────────────────────

def test_two_independent_domains_agree_high_confidence():
    field = make_field(min_corroborations=2)
    cell = verify_field(field, [
        make_extraction(value="John Smith", domain="wikipedia.org"),
        make_extraction(value="John Smith", domain="news.com"),
    ])
    assert cell.confidence == "HIGH"
    assert cell.value == "John Smith"
    assert cell.corroboration_count == 2


def test_single_grounded_below_min_corroborations_low():
    field = make_field(min_corroborations=2)
    cell = verify_field(field, [
        make_extraction(value="John Smith", domain="wikipedia.org"),
    ])
    assert cell.confidence == "LOW"
    assert any("below_min_corroborations" in f for f in cell.flags)


def test_single_grounded_with_min_corroborations_one_medium_or_high():
    field = make_field(min_corroborations=1)
    cell = verify_field(field, [
        make_extraction(value="John Smith", domain="randomsite.com"),
    ])
    # Single non-preferred source with min_corr=1 → MEDIUM
    assert cell.confidence == "MEDIUM"


# ── Mirror dedup ─────────────────────────────────────────────────────────────

def test_mirrors_of_same_domain_count_once():
    field = make_field(min_corroborations=2)
    # Three URLs but all from the same domain — should count as 1 vote
    cell = verify_field(field, [
        make_extraction(value="John Smith", domain="wikipedia.org"),
        make_extraction(value="John Smith", domain="wikipedia.org"),
        make_extraction(value="John Smith", domain="wikipedia.org"),
    ])
    # Single domain → count of 1, below min_corroborations of 2 → LOW
    assert cell.corroboration_count == 1
    assert cell.confidence == "LOW"


# ── Conflict detection ───────────────────────────────────────────────────────

def test_conflict_flagged_when_domains_disagree():
    field = make_field(min_corroborations=1)
    cell = verify_field(field, [
        make_extraction(value="John Smith", domain="wikipedia.org"),
        make_extraction(value="Jane Doe", domain="news.com"),
    ])
    assert any("conflict" in f for f in cell.flags)


def test_conflict_majority_value_wins():
    field = make_field(min_corroborations=2)
    cell = verify_field(field, [
        make_extraction(value="John Smith", domain="a.com"),
        make_extraction(value="John Smith", domain="b.com"),
        make_extraction(value="Jane Doe", domain="c.com"),
    ])
    assert cell.value == "John Smith"
    assert cell.corroboration_count == 2
    assert cell.confidence == "HIGH"
    assert any("conflict" in f for f in cell.flags)


# ── Preferred-source boost ───────────────────────────────────────────────────

def test_preferred_source_lifts_medium_to_high():
    """Single preferred-source result with min_corr=1 should be HIGH."""
    field = make_field(min_corroborations=1, preferred=["gov.il"])
    cell = verify_field(field, [
        make_extraction(value="John Smith", domain="muni.gov.il"),
    ])
    assert cell.confidence == "HIGH"


def test_no_preferred_boost_when_domain_not_in_list():
    field = make_field(min_corroborations=1, preferred=["gov.il"])
    cell = verify_field(field, [
        make_extraction(value="John Smith", domain="randomsite.com"),
    ])
    assert cell.confidence == "MEDIUM"


def test_winning_extraction_prefers_authoritative_domain():
    """When two domains tie, the preferred one should be picked as primary."""
    field = make_field(min_corroborations=1, preferred=["gov.il"])
    cell = verify_field(field, [
        make_extraction(value="John Smith", domain="randomsite.com", confidence=0.5),
        make_extraction(value="John Smith", domain="muni.gov.il", confidence=0.5),
    ])
    assert cell.primary_source["domain"] == "muni.gov.il"


# ── Provenance attachment ────────────────────────────────────────────────────

def test_primary_source_includes_url_quote_domain():
    field = make_field(min_corroborations=1)
    cell = verify_field(field, [
        make_extraction(
            value="John Smith",
            domain="wikipedia.org",
            quote="John Smith was elected in 1990",
        ),
    ])
    assert cell.primary_source is not None
    assert cell.primary_source["url"] == "http://wikipedia.org/page"
    assert cell.primary_source["quote"] == "John Smith was elected in 1990"
    assert cell.primary_source["domain"] == "wikipedia.org"


def test_all_sources_lists_every_grounded_extraction():
    field = make_field(min_corroborations=1)
    cell = verify_field(field, [
        make_extraction(value="John Smith", domain="a.com"),
        make_extraction(value="John Smith", domain="b.com"),
        make_extraction(value="John Smith", domain="c.com"),
    ])
    assert len(cell.all_sources) == 3
    domains = {s["domain"] for s in cell.all_sources}
    assert domains == {"a.com", "b.com", "c.com"}


# ── Hebrew normalization in consensus ────────────────────────────────────────

def test_hebrew_values_with_niqqud_treated_as_same():
    """Two sources spelling the name with/without niqqud should agree."""
    field = make_field(min_corroborations=2)
    cell = verify_field(field, [
        make_extraction(value="שָׁלֹמֹה לָהָט", domain="a.co.il", quote="שָׁלֹמֹה לָהָט"),
        make_extraction(value="שלמה להט", domain="b.co.il", quote="שלמה להט"),
    ])
    # Both should be normalized to the same value → consensus
    assert cell.corroboration_count == 2
    assert cell.confidence == "HIGH"


def test_hebrew_final_form_difference_treated_as_same():
    field = make_field(min_corroborations=2)
    cell = verify_field(field, [
        make_extraction(value="דוד כהן", domain="a.co.il", quote="דוד כהן"),
        make_extraction(value="דוד כהנ", domain="b.co.il", quote="דוד כהנ"),  # non-final
    ])
    assert cell.corroboration_count == 2
    assert cell.confidence == "HIGH"
