"""Tests for the smart content-windowing helper used by the extractor."""
from research_agent.extractor import _select_relevant_text
from research_agent.models import ColumnPlan


def _field(field_type="person_name", temporal=None, label_he="X", label_en="X"):
    return ColumnPlan(
        id="f", label_he=label_he, label_en=label_en, type=field_type,
        search_queries_he=[], search_queries_en=[],
        preferred_source_domains=[], min_corroborations=1,
        temporal_anchor=temporal,
    )


def test_short_content_passes_through_unchanged():
    text = "short content with תל אביב mentioned once"
    field = _field()
    assert _select_relevant_text(text, field, "תל אביב") == text


def test_long_content_includes_intro():
    # 30K chars of filler — well above the 15K budget
    intro = "INTRO_MARKER " + "a" * 1400
    body  = "b" * 30000
    text  = intro + body
    out = _select_relevant_text(text, _field(), "missing_entity")
    assert "INTRO_MARKER" in out
    assert len(out) <= 16000   # budget + small overhead


def test_long_content_finds_keyword_window():
    intro = "intro filler " + "x" * 1500
    middle_filler = "y" * 20000
    target_section = "ראש העיר היה שלמה להט בשנת 1990. " + "z" * 500
    tail = "z" * 10000
    text = intro + middle_filler + target_section + tail

    field = _field(field_type="person_name", temporal="1990")
    out = _select_relevant_text(text, field, "תל אביב")

    assert "שלמה להט" in out
    assert "1990" in out


def test_url_field_finds_website_section():
    intro = "intro " + "x" * 1400
    middle = "y" * 25000
    section = "האתר הרשמי של העיריה הוא www.tel-aviv.gov.il לפרטים נוספים"
    text = intro + middle + section + ("z" * 5000)

    field = _field(field_type="url", label_he="אתר רשמי", label_en="Website")
    out = _select_relevant_text(text, field, "תל אביב")

    assert "www.tel-aviv.gov.il" in out


def test_temporal_anchor_includes_adjacent_years():
    # Article only mentions adjacent year (1989), not 1990 — should still
    # be located because the helper adds neighboring years to keywords
    intro = "x" * 1500
    middle = "y" * 25000
    section = "המאיר שלום כיהן כראש עיר עד 1989."
    text = intro + middle + section + ("z" * 5000)

    field = _field(field_type="person_name", temporal="1990")
    out = _select_relevant_text(text, field, "תל אביב")
    assert "1989" in out
    assert "המאיר שלום" in out


def test_no_keyword_matches_returns_intro_only():
    text = "x" * 50000
    field = _field(field_type="person_name", temporal="1990")
    out = _select_relevant_text(text, field, "completely_absent_entity")
    assert len(out) <= 16000
    assert out.startswith("x")


def test_empty_content():
    assert _select_relevant_text("", _field(), "x") == ""


def test_respects_budget_cap():
    # Many keyword matches throughout the article — output must still cap
    text = ("ראש העיר תל אביב 1990 " * 5000)   # ~100K chars, every chunk relevant
    field = _field(field_type="person_name", temporal="1990")
    out = _select_relevant_text(text, field, "תל אביב")
    assert len(out) <= 16000
