"""
End-to-end integration tests for the IL catalog inside the cross-entity
orchestrator. Stubbed search + Claude clients so the test is hermetic.

These prove the canonicalization actually flows through the pipeline:
  - The catalog hit changes what STRING is sent to search_client.search()
  - The user-supplied entity name is unchanged in the returned result keys
  - IL_CATALOG_DISABLED=1 restores byte-identical behavior to the old path
"""
import pytest

from research_agent import extractor, il_catalog
from research_agent.models import ColumnPlan


# ── Stubs (copied from the cross-entity orchestrator test) ───────────────────

class _SpySearch:
    """Records every query string sent + returns canned (URL, content) hits."""
    def __init__(self, urls=None):
        self.calls: list[tuple[str, int]] = []
        self.urls = urls or ["https://example.com/page1"]

    def search(self, query, max_results=5, **kwargs):
        self.calls.append((query, max_results))
        return {"results": [
            {
                "url": u,
                "raw_content": (
                    f"Long-enough mock page mentioning the entity. "
                    f"It includes the answer is HIT_{u} embedded in text "
                    f"so substring-grounding will succeed properly."
                ),
            }
            for u in self.urls[:max_results]
        ]}


class _StubClaude:
    """Returns null extractions — we don't care about extraction in these
    tests, only about which search queries were issued."""
    def __init__(self):
        self.messages = self
    def create(self, **kw):
        import re
        prompt = kw["messages"][0]["content"]
        pair_ids = re.findall(r"pair_id='(p\d+)'", prompt)
        class _B:
            type = "tool_use"
            input = {"extractions": [
                {"pair_id": p, "value": None, "quote_original": None, "confidence": 0.0}
                for p in pair_ids
            ]}
        class _R:
            content = [_B()]
        return _R()


def _mk_field(fid: str) -> ColumnPlan:
    return ColumnPlan(
        id=fid, label_he=fid, label_en=fid, type="free_text",
        search_queries_he=["{entity} מידע"],
        search_queries_en=["{entity} info"],
    )


@pytest.fixture(autouse=True)
def _reset_catalog():
    il_catalog.reset_cache()
    yield
    il_catalog.reset_cache()


# ── Canonical-name normalization flows into the search query ─────────────────

def test_short_form_is_canonicalized_in_search_query():
    """User enters 'תל אביב' — search must be issued with 'תל אביב-יפו' (canonical)."""
    field = _mk_field("f1")
    search = _SpySearch()
    claude = _StubClaude()

    extractor.search_and_extract_cross_entity(
        fields=[field], entities=["תל אביב"],
        resolved_deps_per_entity={"תל אביב": {}},
        search_client=search, claude=claude,
        entity_type="ראשי ערים",
    )

    # Exactly one search; query string must use the canonical form.
    assert len(search.calls) == 1
    query, _n = search.calls[0]
    assert "תל אביב-יפו" in query
    assert query == "תל אביב-יפו מידע"


def test_abbreviation_is_canonicalized():
    """ת״א (gershayim abbreviation) → תל אביב-יפו in the issued query."""
    field = _mk_field("f1")
    search = _SpySearch()
    claude = _StubClaude()

    extractor.search_and_extract_cross_entity(
        fields=[field], entities=["ת״א"],
        resolved_deps_per_entity={"ת״א": {}},
        search_client=search, claude=claude,
        entity_type="ראשי ערים",
    )

    query, _ = search.calls[0]
    assert "תל אביב-יפו" in query


def test_unknown_entity_passes_through_unchanged():
    """No-regression: entity not in catalog → query uses the input verbatim."""
    field = _mk_field("f1")
    search = _SpySearch()
    claude = _StubClaude()

    extractor.search_and_extract_cross_entity(
        fields=[field], entities=["גן יבנה הקטנטונת"],
        resolved_deps_per_entity={"גן יבנה הקטנטונת": {}},
        search_client=search, claude=claude,
        entity_type="ראשי ערים",
    )

    query, _ = search.calls[0]
    assert query == "גן יבנה הקטנטונת מידע"


def test_returned_result_dict_keys_use_original_entity_name():
    """User-visible entity name must NOT be canonicalized — only the search query is."""
    field = _mk_field("f1")
    search = _SpySearch()
    claude = _StubClaude()

    result = extractor.search_and_extract_cross_entity(
        fields=[field], entities=["תל אביב"],
        resolved_deps_per_entity={"תל אביב": {}},
        search_client=search, claude=claude,
        entity_type="ראשי ערים",
    )

    # The dict key is the original input, NOT the canonical form.
    assert "תל אביב" in result
    assert "תל אביב-יפו" not in result


# ── No-regression: kill-switch restores byte-identical old behavior ──────────

def test_kill_switch_makes_search_use_original_entity(monkeypatch):
    """With IL_CATALOG_DISABLED=1, no normalization happens — query uses input as-is."""
    monkeypatch.setenv("IL_CATALOG_DISABLED", "1")
    il_catalog.reset_cache()

    field = _mk_field("f1")
    search = _SpySearch()
    claude = _StubClaude()

    extractor.search_and_extract_cross_entity(
        fields=[field], entities=["תל אביב"],
        resolved_deps_per_entity={"תל אביב": {}},
        search_client=search, claude=claude,
        entity_type="ראשי ערים",
    )

    query, _ = search.calls[0]
    # Original input verbatim — canonicalization is disabled.
    assert query == "תל אביב מידע"
    assert "תל אביב-יפו" not in query


def test_no_entity_type_hint_still_canonicalizes_via_global_lookup():
    """Even without an entity_type hint, a global lookup can find the canonical."""
    field = _mk_field("f1")
    search = _SpySearch()
    claude = _StubClaude()

    extractor.search_and_extract_cross_entity(
        fields=[field], entities=["Tel Aviv"],
        resolved_deps_per_entity={"Tel Aviv": {}},
        search_client=search, claude=claude,
        # entity_type omitted on purpose — global category-less lookup.
    )

    query, _ = search.calls[0]
    assert "תל אביב-יפו" in query
