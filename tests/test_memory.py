"""Tests for research_agent.memory — persistence + retrieval logic."""
import json
import pytest
from research_agent.memory import (
    SuccessMemory,
    format_compiler_examples,
    format_extraction_examples,
    format_extraction_warnings,
    _tokens,
)


@pytest.fixture
def mem(tmp_path):
    """Fresh memory backed by a tmp_path JSON file."""
    return SuccessMemory(tmp_path / "mem.json")


# ── initialization ───────────────────────────────────────────────────────────

def test_loads_empty_when_no_file(tmp_path):
    m = SuccessMemory(tmp_path / "does_not_exist.json")
    assert m.stats() == {
        "compiler_successes": 0,
        "extraction_successes": 0,
        "extraction_failures": 0,
    }


def test_loads_existing_file(tmp_path):
    path = tmp_path / "preloaded.json"
    path.write_text(json.dumps({
        "compiler_successes": [{"id": "x", "research_question": "q"}],
        "extraction_successes": [],
        "extraction_failures": [],
    }), encoding="utf-8")
    m = SuccessMemory(path)
    assert m.stats()["compiler_successes"] == 1


def test_handles_corrupted_file_gracefully(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not valid json", encoding="utf-8")
    # Should not raise — just warn and start empty
    m = SuccessMemory(path)
    assert m.stats()["compiler_successes"] == 0


# ── persistence: round-trip ──────────────────────────────────────────────────

def test_record_persists_to_disk(mem, tmp_path):
    mem.record_extraction_success(
        field_id="mayor", field_label="Mayor", field_type="person_name",
        entity="Tel Aviv", value="John", quote="John was mayor",
        source_url="http://wikipedia.org/x", source_domain="wikipedia.org",
    )
    # Reload from disk in a fresh instance
    m2 = SuccessMemory(mem.path)
    assert m2.stats()["extraction_successes"] == 1
    assert m2._data["extraction_successes"][0]["value"] == "John"


def test_record_assigns_unique_ids(mem):
    id1 = mem.record_extraction_failure(
        field_id="x", field_type="person_name", entity="A",
        claimed_value="V", claimed_quote="Q",
        source_url="u", source_domain="d", reason="r",
    )
    id2 = mem.record_extraction_failure(
        field_id="x", field_type="person_name", entity="B",
        claimed_value="V", claimed_quote="Q",
        source_url="u", source_domain="d", reason="r",
    )
    assert id1 != id2


def test_record_writes_iso_timestamp(mem):
    mem.record_compiler_success("q", "et", {"columns": []})
    record = mem._data["compiler_successes"][0]
    assert "T" in record["validated_at"]   # ISO 8601 includes 'T'


# ── compiler retrieval ───────────────────────────────────────────────────────

def test_compiler_retrieval_prefers_matching_entity_type(mem):
    mem.record_compiler_success("about mayors in 1990", "municipality", {"columns": []})
    mem.record_compiler_success("about company CEOs", "company", {"columns": []})
    mem.record_compiler_success("about mayors elsewhere", "city", {"columns": []})

    results = mem.get_compiler_examples(
        research_question="research mayors",
        entity_type="municipality",
        k=3,
    )
    # entity_type match (+5) should put 'municipality' record first
    assert results[0]["entity_type"] == "municipality"


def test_compiler_retrieval_uses_token_overlap(mem):
    mem.record_compiler_success("Israeli mayors and IDF service", "x", {"columns": []})
    mem.record_compiler_success("totally unrelated cooking recipe", "x", {"columns": []})

    results = mem.get_compiler_examples("Israeli mayors", "y", k=2)
    # First record shares "israeli" + "mayors" tokens; should rank first
    assert "Israeli" in results[0]["research_question"]


def test_compiler_retrieval_respects_k(mem):
    for i in range(10):
        mem.record_compiler_success(f"mayors topic {i}", "city", {"columns": []})
    results = mem.get_compiler_examples("mayors", "city", k=3)
    assert len(results) == 3


def test_compiler_retrieval_returns_empty_when_no_overlap(mem):
    mem.record_compiler_success("totally unrelated text", "alpha", {"columns": []})
    results = mem.get_compiler_examples("nothing in common", "beta", k=3)
    assert results == []


# ── extraction retrieval ─────────────────────────────────────────────────────

def test_extraction_retrieval_filters_by_field_type(mem):
    mem.record_extraction_success(
        field_id="mayor", field_label="Mayor", field_type="person_name",
        entity="A", value="V", quote="Q", source_url="u", source_domain="d",
    )
    mem.record_extraction_success(
        field_id="url", field_label="URL", field_type="url",
        entity="A", value="V", quote="Q", source_url="u", source_domain="d",
    )

    persons = mem.get_extraction_examples("person_name", "Mayor", k=5)
    assert len(persons) == 1
    assert persons[0]["field_type"] == "person_name"


def test_extraction_retrieval_empty_when_no_match(mem):
    mem.record_extraction_success(
        field_id="x", field_label="X", field_type="person_name",
        entity="A", value="V", quote="Q", source_url="u", source_domain="d",
    )
    results = mem.get_extraction_examples("date", "Date", k=3)
    assert results == []


# ── warnings retrieval ───────────────────────────────────────────────────────

def test_warnings_match_by_field_type(mem):
    mem.record_extraction_failure(
        field_id="x", field_type="person_name", entity="E",
        claimed_value="V", claimed_quote="Q",
        source_url="http://a.com/x", source_domain="a.com",
        reason="hallucinated",
    )
    warnings = mem.get_extraction_warnings("person_name", candidate_domain="b.com")
    assert len(warnings) == 1


def test_warnings_match_by_domain_even_different_field_type(mem):
    mem.record_extraction_failure(
        field_id="x", field_type="date", entity="E",
        claimed_value="V", claimed_quote="Q",
        source_url="http://bad.com/x", source_domain="bad.com",
        reason="r",
    )
    warnings = mem.get_extraction_warnings("person_name", candidate_domain="bad.com")
    assert len(warnings) == 1


def test_warnings_capped_at_5(mem):
    for i in range(10):
        mem.record_extraction_failure(
            field_id="x", field_type="person_name", entity=f"E{i}",
            claimed_value="V", claimed_quote="Q",
            source_url="u", source_domain="d", reason="r",
        )
    warnings = mem.get_extraction_warnings("person_name", "d")
    assert len(warnings) == 5


def test_warnings_sorted_recent_first(mem):
    mem.record_extraction_failure(
        field_id="x", field_type="t", entity="OLDEST",
        claimed_value=None, claimed_quote=None,
        source_url="u", source_domain="d", reason="r",
    )
    mem.record_extraction_failure(
        field_id="x", field_type="t", entity="NEWEST",
        claimed_value=None, claimed_quote=None,
        source_url="u", source_domain="d", reason="r",
    )
    warnings = mem.get_extraction_warnings("t", None)
    # Most recent first
    assert warnings[0]["entity"] == "NEWEST"


# ── token helper ─────────────────────────────────────────────────────────────

def test_tokens_lowercases_and_filters_short():
    assert _tokens("The Quick Brown Fox") == {"the", "quick", "brown", "fox"}


def test_tokens_skips_short_words():
    # Words ≤ 2 chars excluded
    tokens = _tokens("a an of the cat")
    assert "a" not in tokens
    assert "an" not in tokens
    assert "of" not in tokens
    assert "the" in tokens
    assert "cat" in tokens


def test_tokens_handles_hebrew():
    tokens = _tokens("שלום עולם")
    assert "שלום" in tokens
    assert "עולם" in tokens


def test_tokens_empty():
    assert _tokens("") == set()
    assert _tokens(None) == set()


# ── formatters ───────────────────────────────────────────────────────────────

def test_format_compiler_examples_empty():
    assert format_compiler_examples([]) == ""


def test_format_compiler_examples_includes_question_and_domains():
    examples = [{
        "research_question": "Who was mayor in 1990?",
        "entity_type": "municipality",
        "plan": {
            "columns": [
                {
                    "id": "mayor",
                    "type": "person_name",
                    "search_queries_he": ["ראש עיריית {entity}"],
                    "preferred_source_domains": ["he.wikipedia.org"],
                }
            ]
        },
    }]
    rendered = format_compiler_examples(examples)
    assert "Who was mayor in 1990?" in rendered
    assert "he.wikipedia.org" in rendered
    assert "ראש עיריית" in rendered


def test_format_extraction_examples_empty():
    assert format_extraction_examples([]) == ""


def test_format_extraction_warnings_empty():
    assert format_extraction_warnings([]) == ""


def test_format_extraction_warnings_includes_avoid_signal():
    warnings = [{
        "source_domain": "bad.com",
        "field_type": "person_name",
        "claimed_value": "Wrong Person",
        "reason": "wrong entity",
    }]
    rendered = format_extraction_warnings(warnings)
    assert "AVOID" in rendered
    assert "bad.com" in rendered
    assert "Wrong Person" in rendered
