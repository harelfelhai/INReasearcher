"""Tests for research_agent.hebrew_utils — pure-function unit tests."""
from research_agent.hebrew_utils import (
    strip_niqqud,
    normalize_hebrew,
    is_grounded,
    _edit_distance,
    _fuzzy_find,
    _extract_window,
)


# ── strip_niqqud ─────────────────────────────────────────────────────────────

def test_strip_niqqud_removes_vowel_marks():
    # "שָׁלוֹם" with niqqud → "שלום"
    assert strip_niqqud("שָׁלוֹם") == "שלום"


def test_strip_niqqud_no_niqqud_unchanged():
    assert strip_niqqud("שלום") == "שלום"


def test_strip_niqqud_preserves_english_and_punctuation():
    assert strip_niqqud("hello, world!") == "hello, world!"


def test_strip_niqqud_empty():
    assert strip_niqqud("") == ""


# ── normalize_hebrew ─────────────────────────────────────────────────────────

def test_normalize_unifies_final_forms():
    # ך/כ, ם/מ, ן/נ, ף/פ, ץ/צ should all be folded
    assert normalize_hebrew("דויךם") == normalize_hebrew("דויכמ")


def test_normalize_strips_rtl_marks():
    text_with_rtl = "שלום‏ עולם"
    assert "‏" not in normalize_hebrew(text_with_rtl)


def test_normalize_collapses_whitespace():
    # normalize folds final-forms (ם→מ), so expected output uses non-final forms
    assert normalize_hebrew("שלום    עולם") == "שלומ עולמ"
    assert normalize_hebrew("  שלום  \n\n  עולם  ") == "שלומ עולמ"
    # The key invariant: multiple internal whitespace becomes single space
    assert "  " not in normalize_hebrew("a    b")


def test_normalize_empty_input():
    assert normalize_hebrew("") == ""


def test_normalize_idempotent():
    text = "שָׁלוֹם עוֹלָם"
    assert normalize_hebrew(normalize_hebrew(text)) == normalize_hebrew(text)


# ── is_grounded: positive cases ──────────────────────────────────────────────

def test_grounded_exact_substring_hebrew():
    source = "ראש העיר היה שלמה להט בשנת 1990"
    grounded, snippet = is_grounded("שלמה להט", source)
    assert grounded is True
    assert "שלמה להט" in snippet


def test_grounded_exact_substring_english():
    source = "The mayor in 1990 was John Smith, who served until 1995."
    grounded, snippet = is_grounded("John Smith", source)
    assert grounded is True
    assert "John Smith" in snippet


def test_grounded_with_niqqud_in_source():
    source = "שָׁלוֹם עוֹלָם זה טקסט עם ניקוד"
    grounded, _ = is_grounded("שלום עולם", source)
    assert grounded is True


def test_grounded_with_final_form_mismatch():
    # Source uses final form, value uses non-final → still match
    source = "ראש העיר דוד כהן"
    grounded, _ = is_grounded("דוד כהנ", source)   # ן→נ in value
    assert grounded is True


# ── is_grounded: negative cases ──────────────────────────────────────────────

def test_not_grounded_when_value_absent():
    source = "ראש העיר היה דוד כהן"
    grounded, snippet = is_grounded("שלמה להט", source)
    assert grounded is False
    assert snippet == ""


def test_not_grounded_empty_value():
    grounded, _ = is_grounded("", "some source text")
    assert grounded is False


def test_not_grounded_empty_source():
    grounded, _ = is_grounded("anything", "")
    assert grounded is False


# ── is_grounded: fuzzy match for short strings ───────────────────────────────

def test_fuzzy_match_single_char_typo():
    # 1-char edit distance should match for short strings (≤ 20 chars)
    source = "The mayor was Jonn Smith in 1990"   # Jonn vs John
    grounded, _ = is_grounded("John Smith", source)
    assert grounded is True


def test_fuzzy_match_does_not_apply_to_long_strings():
    # Long strings don't get fuzzy matching (would create false positives)
    source = "this is a completely unrelated long sentence about cats and dogs"
    long_value = "this is a slightly different long sentence about cats and dogs"
    grounded, _ = is_grounded(long_value, source)
    assert grounded is False


def test_fuzzy_does_not_match_too_short_strings():
    # Strings shorter than 3 chars should not fuzzy match (too noisy)
    assert _fuzzy_find("ab", "xyz", max_edits=1) == -1


# ── _edit_distance ───────────────────────────────────────────────────────────

def test_edit_distance_identical():
    assert _edit_distance("hello", "hello") == 0


def test_edit_distance_one_substitution():
    assert _edit_distance("hello", "hallo") == 1


def test_edit_distance_one_insertion():
    assert _edit_distance("hello", "helloo") == 1


def test_edit_distance_completely_different():
    assert _edit_distance("abc", "xyz") == 3


# ── _extract_window ──────────────────────────────────────────────────────────

def test_extract_window_basic():
    text = "abcdefghijklmnopqrstuvwxyz"
    snippet = _extract_window(text, idx=10, match_len=3, window=10)
    # idx 10 = 'k', match 'klm', window 10 → 5 chars each side
    assert "klm" in snippet


def test_extract_window_at_start_no_leading_ellipsis():
    text = "first words of source"
    snippet = _extract_window(text, idx=0, match_len=5, window=20)
    assert snippet.startswith("first")
    assert not snippet.startswith("...")


def test_extract_window_at_end_no_trailing_ellipsis():
    text = "short text"
    snippet = _extract_window(text, idx=6, match_len=4, window=20)
    assert snippet.endswith("text")
    assert not snippet.endswith("...")


def test_extract_window_middle_has_ellipsis():
    text = "a" * 100 + "MATCH" + "b" * 100
    snippet = _extract_window(text, idx=100, match_len=5, window=20)
    assert snippet.startswith("...")
    assert snippet.endswith("...")
    assert "MATCH" in snippet
