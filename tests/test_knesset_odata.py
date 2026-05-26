"""
Unit tests for the Knesset OData adapter (research_agent/knesset_odata.py).

All HTTP calls are mocked via patch.object on _knesset_fetch — no real
network traffic. Tests verify:
  - MK context detection (is_mk_context)
  - Person lookup (find_mk_by_name)
  - Position fetch (fetch_mk_positions)
  - Faction name fetch (fetch_faction_name)
  - Text formatting (format_mk_as_text)
  - Full injection (inject_knesset_mk_data)
  - Kill switch (KNESSET_ODATA_DISABLED=1)
"""
import pytest
from unittest.mock import patch

from research_agent import knesset_odata


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure env kill-switches are unset and OData caches are cleared."""
    monkeypatch.delenv("KNESSET_ODATA_DISABLED", raising=False)
    monkeypatch.delenv("KNESSET_ODATA_BASE_URL", raising=False)
    knesset_odata.clear_caches()
    yield
    knesset_odata.clear_caches()


# ── Fixtures ──────────────────────────────────────────────────────────────────

_PERSON = {
    "PersonID": 877,
    "FirstName": "בנימין",
    "LastName": "נתניהו",
    "GenderID": 251,
    "GenderDesc": "זכר",
    "Email": "netanyahu@knesset.gov.il",
    "IsCurrent": True,
    "LastUpdatedDate": "2024-07-28T21:41:35.37",
}

_POSITIONS = [
    {
        "PersonID": 877,
        "PositionID": 1,
        "FactionID": 23,
        "KnessetNum": 25,
        "StartDate": "2022-11-15T00:00:00",
        "EndDate": None,
    },
    {
        "PersonID": 877,
        "PositionID": 1,
        "FactionID": 21,
        "KnessetNum": 24,
        "StartDate": "2021-03-15T00:00:00",
        "EndDate": "2022-11-10T00:00:00",
    },
]


def _make_sequential_mock(*responses):
    """Return a side_effect function that yields responses in order."""
    it = iter(responses)

    def _fetch(url):
        try:
            return next(it)
        except StopIteration:
            return {}

    return _fetch


# ── is_mk_context ─────────────────────────────────────────────────────────────

def test_is_mk_context_hebrew_variants():
    assert knesset_odata.is_mk_context("חברי כנסת")
    assert knesset_odata.is_mk_context("חבר כנסת")
    assert knesset_odata.is_mk_context("חברת כנסת")
    assert knesset_odata.is_mk_context('ח"כ')
    assert knesset_odata.is_mk_context("חברי כנסת ה-25")


def test_is_mk_context_english_variants():
    assert knesset_odata.is_mk_context("Members of Knesset")
    assert knesset_odata.is_mk_context("Knesset Members")
    assert knesset_odata.is_mk_context("MK")
    assert knesset_odata.is_mk_context("MKs")
    assert knesset_odata.is_mk_context("knesset member")


def test_is_mk_context_false_for_non_mk():
    assert not knesset_odata.is_mk_context(None)
    assert not knesset_odata.is_mk_context("")
    assert not knesset_odata.is_mk_context("ראשי ערים")
    assert not knesset_odata.is_mk_context("עיריות")
    assert not knesset_odata.is_mk_context("משרדי ממשלה")
    assert not knesset_odata.is_mk_context("municipalities")


# ── find_mk_by_name ───────────────────────────────────────────────────────────

def test_find_mk_by_name_hit():
    with patch.object(knesset_odata, "_knesset_fetch",
                      return_value={"value": [_PERSON]}):
        result = knesset_odata.find_mk_by_name("נתניהו")
    assert result is not None
    assert result["PersonID"] == 877
    assert result["LastName"] == "נתניהו"
    assert result["FirstName"] == "בנימין"


def test_find_mk_by_name_disambiguates_by_first_name():
    rows = [
        {**_PERSON, "PersonID": 999, "FirstName": "שרה"},
        _PERSON,   # the one we actually want
    ]
    with patch.object(knesset_odata, "_knesset_fetch",
                      return_value={"value": rows}):
        result = knesset_odata.find_mk_by_name("בנימין נתניהו")
    assert result["PersonID"] == 877


def test_find_mk_by_name_miss():
    with patch.object(knesset_odata, "_knesset_fetch",
                      return_value={"value": []}):
        result = knesset_odata.find_mk_by_name("אין כזה אדם כלל")
    assert result is None


def test_find_mk_by_name_network_error():
    with patch.object(knesset_odata, "_knesset_fetch",
                      side_effect=OSError("connection timeout")):
        result = knesset_odata.find_mk_by_name("נתניהו")
    assert result is None


def test_find_mk_by_name_empty_string():
    # No fetch at all when name is empty
    result = knesset_odata.find_mk_by_name("")
    assert result is None


def test_find_mk_by_name_returns_first_when_no_disambiguation():
    persons = [_PERSON, {**_PERSON, "PersonID": 999, "FirstName": "שרה"}]
    with patch.object(knesset_odata, "_knesset_fetch",
                      return_value={"value": persons}):
        # Only surname given — no first-name hint to disambiguate
        result = knesset_odata.find_mk_by_name("נתניהו")
    assert result["PersonID"] == 877   # first row


# ── fetch_mk_positions ────────────────────────────────────────────────────────

def test_fetch_mk_positions_returns_list():
    with patch.object(knesset_odata, "_knesset_fetch",
                      return_value={"value": _POSITIONS}):
        rows = knesset_odata.fetch_mk_positions(877)
    assert len(rows) == 2
    assert rows[0]["KnessetNum"] == 25


def test_fetch_mk_positions_empty_on_miss():
    with patch.object(knesset_odata, "_knesset_fetch",
                      return_value={"value": []}):
        rows = knesset_odata.fetch_mk_positions(877)
    assert rows == []


def test_fetch_mk_positions_empty_on_error():
    with patch.object(knesset_odata, "_knesset_fetch",
                      side_effect=OSError("timeout")):
        rows = knesset_odata.fetch_mk_positions(877)
    assert rows == []


# ── fetch_faction_name ────────────────────────────────────────────────────────

def test_fetch_faction_name_returns_name():
    with patch.object(knesset_odata, "_knesset_fetch",
                      return_value={"Name": "הליכוד"}):
        name = knesset_odata.fetch_faction_name(23)
    assert name == "הליכוד"


def test_fetch_faction_name_none_on_missing_field():
    with patch.object(knesset_odata, "_knesset_fetch",
                      return_value={}):
        name = knesset_odata.fetch_faction_name(99)
    assert name is None


def test_fetch_faction_name_none_on_error():
    with patch.object(knesset_odata, "_knesset_fetch",
                      side_effect=OSError("404")):
        name = knesset_odata.fetch_faction_name(99)
    assert name is None


# ── format_mk_as_text ─────────────────────────────────────────────────────────

def test_format_mk_as_text_contains_full_name():
    with patch.object(knesset_odata, "_knesset_fetch",
                      return_value={"Name": "הליכוד"}):
        text = knesset_odata.format_mk_as_text(_PERSON, _POSITIONS)
    assert "בנימין נתניהו" in text


def test_format_mk_as_text_contains_gender():
    with patch.object(knesset_odata, "_knesset_fetch",
                      return_value={"Name": "הליכוד"}):
        text = knesset_odata.format_mk_as_text(_PERSON, _POSITIONS)
    assert "זכר" in text


def test_format_mk_as_text_contains_knesset_numbers():
    with patch.object(knesset_odata, "_knesset_fetch",
                      return_value={"Name": "הליכוד"}):
        text = knesset_odata.format_mk_as_text(_PERSON, _POSITIONS)
    assert "25" in text
    assert "24" in text


def test_format_mk_as_text_includes_source_label():
    with patch.object(knesset_odata, "_knesset_fetch",
                      return_value={"Name": "הליכוד"}):
        text = knesset_odata.format_mk_as_text(_PERSON, _POSITIONS)
    assert "כנסת" in text


def test_format_mk_as_text_no_positions():
    with patch.object(knesset_odata, "_knesset_fetch",
                      return_value={"Name": "הליכוד"}):
        text = knesset_odata.format_mk_as_text(_PERSON, [])
    assert "בנימין נתניהו" in text
    assert isinstance(text, str)
    assert len(text) > 20


def test_format_mk_as_text_missing_optional_fields():
    person = {"PersonID": 1, "FirstName": "בנימין", "LastName": "נתניהו"}
    with patch.object(knesset_odata, "_knesset_fetch",
                      return_value={"Name": "הליכוד"}):
        text = knesset_odata.format_mk_as_text(person, [])
    # No crash; optional lines simply omitted
    assert "בנימין נתניהו" in text


# ── inject_knesset_mk_data ────────────────────────────────────────────────────

def test_inject_disabled_by_env(monkeypatch):
    monkeypatch.setenv("KNESSET_ODATA_DISABLED", "1")
    pages: list = []
    seen: set = set()
    knesset_odata.inject_knesset_mk_data("נתניהו", "חברי כנסת", pages, seen)
    assert pages == []
    assert seen == set()


def test_inject_skips_non_mk_entity_type():
    pages: list = []
    seen: set = set()
    knesset_odata.inject_knesset_mk_data("תל אביב", "ראשי ערים", pages, seen)
    assert pages == []


def test_inject_skips_when_no_entity_type():
    pages: list = []
    seen: set = set()
    knesset_odata.inject_knesset_mk_data("נתניהו", None, pages, seen)
    assert pages == []


def test_inject_skips_when_person_not_found():
    with patch.object(knesset_odata, "_knesset_fetch",
                      return_value={"value": []}):
        pages: list = []
        seen: set = set()
        knesset_odata.inject_knesset_mk_data("לא קיים בכלל", "חברי כנסת",
                                              pages, seen)
    assert pages == []


def test_inject_appends_page_on_hit():
    fetch = _make_sequential_mock(
        {"value": [_PERSON]},      # find_mk_by_name
        {"value": _POSITIONS},     # fetch_mk_positions
        {"Name": "הליכוד"},        # fetch_faction_name for FactionID 23
        {"Name": "הליכוד"},        # fetch_faction_name for FactionID 21
    )
    with patch.object(knesset_odata, "_knesset_fetch", side_effect=fetch):
        pages: list = []
        seen: set = set()
        knesset_odata.inject_knesset_mk_data("נתניהו", "חברי כנסת", pages, seen)

    assert len(pages) == 1
    url, content = pages[0]
    assert "knesset.gov.il" in url
    assert "877" in url
    assert len(content) > 50


def test_inject_adds_url_to_seen():
    fetch = _make_sequential_mock(
        {"value": [_PERSON]},
        {"value": []},    # no positions
    )
    with patch.object(knesset_odata, "_knesset_fetch", side_effect=fetch):
        pages: list = []
        seen: set = set()
        knesset_odata.inject_knesset_mk_data("נתניהו", "חברי כנסת", pages, seen)

    assert any("877" in u for u in seen)


def test_inject_skips_duplicate_url():
    existing_url = "https://knesset.gov.il/odata/mk/877"
    with patch.object(knesset_odata, "_knesset_fetch",
                      return_value={"value": [_PERSON]}):
        pages: list = []
        seen = {existing_url}
        knesset_odata.inject_knesset_mk_data("נתניהו", "חברי כנסת", pages, seen)

    assert pages == []   # URL was already in seen


def test_inject_prepends_not_appends():
    """Knesset data should appear first in direct_pages — higher priority."""
    fetch = _make_sequential_mock(
        {"value": [_PERSON]},
        {"value": []},
    )
    with patch.object(knesset_odata, "_knesset_fetch", side_effect=fetch):
        pages = [("https://example.com/page", "some existing content")]
        seen: set = set()
        knesset_odata.inject_knesset_mk_data("נתניהו", "חברי כנסת", pages, seen)

    assert len(pages) == 2
    assert "knesset.gov.il" in pages[0][0]   # Knesset page is first
    assert "example.com" in pages[1][0]


def test_repeated_lookups_hit_cache():
    """Second call with the same name must NOT trigger another HTTP fetch."""
    call_count = 0

    def _counting_fetch(url):
        nonlocal call_count
        call_count += 1
        return {"value": [_PERSON]}

    with patch.object(knesset_odata, "_knesset_fetch", side_effect=_counting_fetch):
        knesset_odata.find_mk_by_name("נתניהו")
        knesset_odata.find_mk_by_name("נתניהו")
        knesset_odata.find_mk_by_name("נתניהו")

    assert call_count == 1


def test_clear_caches_resets_lookup():
    call_count = 0

    def _counting_fetch(url):
        nonlocal call_count
        call_count += 1
        return {"value": [_PERSON]}

    with patch.object(knesset_odata, "_knesset_fetch", side_effect=_counting_fetch):
        knesset_odata.find_mk_by_name("נתניהו")
        knesset_odata.clear_caches()
        knesset_odata.find_mk_by_name("נתניהו")

    assert call_count == 2


def test_inject_network_error_leaves_pages_unchanged():
    with patch.object(knesset_odata, "_knesset_fetch",
                      side_effect=OSError("DNS failure")):
        pages: list = []
        seen: set = set()
        knesset_odata.inject_knesset_mk_data("נתניהו", "חברי כנסת", pages, seen)

    assert pages == []
