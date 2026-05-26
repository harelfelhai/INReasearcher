"""
Tests for the Israeli political-domain catalog. Verifies in isolation,
before any integration with the pipeline:

  - Loading (file presence, JSON validity, entity counts)
  - Lookup by canonical Hebrew / English / alias variants
  - Lookup with and without category hint
  - Hebrew name-variant cases (Tel Aviv ↔ Tel Aviv-Yafo, gershayim variants)
  - Category fuzzy-matching from natural Hebrew/English phrases
  - The normalize_entity_for_search wrapper (the actual integration entry point)
  - The IL_CATALOG_DISABLED kill-switch — must make every lookup a graceful MISS
"""
import os

import pytest

from research_agent import il_catalog


@pytest.fixture(autouse=True)
def _fresh_cache():
    """Each test gets a freshly-loaded catalog (the kill-switch tests need this)."""
    il_catalog.reset_cache()
    yield
    il_catalog.reset_cache()


# ── Loading ──────────────────────────────────────────────────────────────────

def test_catalog_loads_both_categories():
    cat = il_catalog.load_catalog()
    assert "municipality" in cat.categories
    assert "ministry" in cat.categories
    # Sanity counts — match the seed dataset we curated.
    assert len(cat.categories["municipality"].entities) >= 30
    assert len(cat.categories["ministry"].entities) >= 20


def test_catalog_categories_have_versions_and_labels():
    cat = il_catalog.load_catalog()
    for c in cat.categories.values():
        assert c.version
        assert c.labels_he, f"{c.name} missing he labels"
        assert c.labels_en, f"{c.name} missing en labels"


# ── Basic lookup ─────────────────────────────────────────────────────────────

def test_lookup_by_canonical_hebrew():
    e = il_catalog.find_entity("תל אביב-יפו")
    assert e is not None
    assert e.canonical_he == "תל אביב-יפו"
    assert e.canonical_en == "Tel Aviv-Yafo"


def test_lookup_by_canonical_english():
    e = il_catalog.find_entity("Tel Aviv-Yafo")
    assert e is not None
    assert e.canonical_he == "תל אביב-יפו"


def test_lookup_returns_none_for_unknown():
    assert il_catalog.find_entity("מיקום לא קיים בעולם") is None
    assert il_catalog.find_entity("") is None
    assert il_catalog.find_entity("    ") is None


# ── Hebrew name-variant cases (the real-world ones that broke us before) ─────

def test_tel_aviv_short_form_normalizes_to_canonical():
    """'תל אביב' (the common short form) should resolve to 'תל אביב-יפו'."""
    e = il_catalog.find_entity("תל אביב")
    assert e is not None
    assert e.canonical_he == "תל אביב-יפו"


def test_tel_aviv_abbreviation_with_gershayim():
    """ת״א with curly gershayim — common in formal Hebrew text."""
    e = il_catalog.find_entity("ת״א")
    assert e is not None
    assert e.canonical_he == "תל אביב-יפו"


def test_tel_aviv_abbreviation_with_straight_quote():
    """ת\"א with straight quote — common in informal typing."""
    e = il_catalog.find_entity('ת"א')
    assert e is not None
    assert e.canonical_he == "תל אביב-יפו"


def test_petah_tikva_with_hyphen():
    """'פתח-תקווה' (hyphenated) should still hit."""
    e = il_catalog.find_entity("פתח-תקווה")
    assert e is not None
    assert e.canonical_he == "פתח תקווה"


def test_english_transliteration_variants_hit():
    """Different English spellings of the same Hebrew name should all resolve."""
    for spelling in ["Be'er Sheva", "Beer Sheva", "Beersheba"]:
        e = il_catalog.find_entity(spelling)
        assert e is not None, f"variant {spelling!r} did not match"
        assert e.canonical_he == "באר שבע"


def test_case_insensitive_english():
    """English lookups should ignore case."""
    e = il_catalog.find_entity("HAIFA")
    assert e is not None
    assert e.canonical_he == "חיפה"


# ── Lookup with category hint ────────────────────────────────────────────────

def test_lookup_with_correct_category_hint():
    e = il_catalog.find_entity("חיפה", category="municipality")
    assert e is not None
    assert e.canonical_he == "חיפה"


def test_lookup_with_wrong_category_hint_returns_none():
    # Haifa is a municipality, not a ministry. Hinting 'ministry' should miss.
    e = il_catalog.find_entity("חיפה", category="ministry")
    assert e is None


def test_lookup_unknown_category_falls_back_to_global():
    # An unknown category hint shouldn't break — it falls back to scanning all.
    e = il_catalog.find_entity("חיפה", category="nonexistent_category")
    assert e is not None
    assert e.canonical_he == "חיפה"


# ── Ministry lookups ─────────────────────────────────────────────────────────

def test_ministry_lookup_by_short_alias():
    e = il_catalog.find_entity("ביטחון", category="ministry")
    assert e is not None
    assert e.canonical_he == "משרד הביטחון"


def test_ministry_lookup_by_english():
    e = il_catalog.find_entity("Ministry of Finance")
    assert e is not None
    assert e.canonical_he == "משרד האוצר"


# ── Category fuzzy-matching ──────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("ראשי ערים",        "municipality"),
    ("עיריות",           "municipality"),
    ("רשויות מקומיות",   "municipality"),
    ("cities",            "municipality"),
    ("Israeli cities",    "municipality"),
    ("municipalities",    "municipality"),
    ("משרדי הממשלה",     "ministry"),
    ("ministries",        "ministry"),
    ("Israeli ministry",  "ministry"),
])
def test_match_category(text, expected):
    assert il_catalog.match_category(text) == expected


def test_match_category_unknown_returns_none():
    assert il_catalog.match_category("פיצריות בתל אביב") is None
    assert il_catalog.match_category("") is None


# ── list_entities ────────────────────────────────────────────────────────────

def test_list_entities_municipality():
    items = il_catalog.list_entities("municipality")
    assert len(items) >= 30
    # Make sure each entry round-trips its category.
    for e in items:
        assert e.category == "municipality"


def test_list_entities_unknown_category_is_empty():
    assert il_catalog.list_entities("nonexistent") == []


# ── normalize_entity_for_search (the integration entry point) ────────────────

def test_normalize_hit_returns_canonical_and_aliases():
    canonical, aliases = il_catalog.normalize_entity_for_search(
        "תל אביב", entity_type="ראשי ערים",
    )
    assert canonical == "תל אביב-יפו"
    # Aliases should include English canonical + Hebrew variants, NOT the canonical itself.
    assert "Tel Aviv-Yafo" in aliases
    assert "Tel Aviv" in aliases
    assert "תל אביב-יפו" not in aliases


def test_normalize_miss_returns_input_unchanged():
    """No-regression guarantee: if the catalog doesn't recognize the entity,
    the function must return the input verbatim with an empty alias list.
    The whole pipeline relies on this fall-through behavior."""
    canonical, aliases = il_catalog.normalize_entity_for_search(
        "גן יבנה הקטנטונת", entity_type="ראשי ערים",
    )
    assert canonical == "גן יבנה הקטנטונת"
    assert aliases == []


def test_normalize_without_entity_type_still_works():
    canonical, _ = il_catalog.normalize_entity_for_search("Tel Aviv")
    assert canonical == "תל אביב-יפו"


# ── Kill-switch ──────────────────────────────────────────────────────────────

def test_kill_switch_disables_catalog(monkeypatch):
    """IL_CATALOG_DISABLED=1 must turn every lookup into a graceful MISS,
    so a regression in catalog behavior can be rolled back via env var alone."""
    monkeypatch.setenv("IL_CATALOG_DISABLED", "1")
    il_catalog.reset_cache()

    # Loading returns an empty catalog.
    cat = il_catalog.load_catalog()
    assert cat.categories == {}

    # Every lookup misses regardless of input.
    assert il_catalog.find_entity("תל אביב") is None
    assert il_catalog.find_entity("Tel Aviv-Yafo") is None
    assert il_catalog.match_category("ראשי ערים") is None

    # normalize_entity_for_search returns input unchanged.
    canonical, aliases = il_catalog.normalize_entity_for_search(
        "תל אביב", entity_type="ראשי ערים",
    )
    assert canonical == "תל אביב"
    assert aliases == []


def test_kill_switch_accepts_various_truthy_strings(monkeypatch):
    for val in ["1", "true", "TRUE", "yes", "Yes"]:
        monkeypatch.setenv("IL_CATALOG_DISABLED", val)
        il_catalog.reset_cache()
        assert il_catalog.load_catalog().categories == {}
