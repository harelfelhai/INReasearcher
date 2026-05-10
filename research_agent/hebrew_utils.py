"""
Hebrew text normalization for grounded extraction verification.

The core problem: two sources might spell the same name differently
(with/without niqqud, final vs non-final forms, RTL marks). Naive
substring matching would reject valid extractions. This module normalizes
before comparison while preserving original text for display.
"""

import re

# Unicode range for Hebrew niqqud (vowel points + cantillation marks)
_NIQQUD = set(range(0x05B0, 0x05C8)) | {0xFB1E}

# Bidirectional control characters that appear in copy-pasted Hebrew text
_BIDI_MARKS = {'‏', '‎', '‪', '‫', '‬', '‭', '‮', '⁦', '⁧', '⁨', '⁩'}

# Final letter forms → standard forms (for comparison only)
_FINAL_FORMS = str.maketrans('ךםןףץ', 'כמנפצ')


def strip_niqqud(text: str) -> str:
    return ''.join(c for c in text if ord(c) not in _NIQQUD)


def normalize_hebrew(text: str) -> str:
    """
    Normalize Hebrew text for fuzzy comparison.
    Do NOT use this for display — it destroys final forms.
    """
    # Remove niqqud and bidi marks
    text = ''.join(c for c in text if ord(c) not in _NIQQUD and c not in _BIDI_MARKS)
    # Unify final letter forms (ך→כ, etc.)
    text = text.translate(_FINAL_FORMS)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def is_grounded(value: str, source_text: str) -> tuple[bool, str]:
    """
    Verify that `value` is genuinely present in `source_text`.

    Returns (is_grounded, matched_snippet).
    The snippet is a ~150-char window from the original source text
    centered on the match — use this as the provenance quote.

    Strategy:
    1. Exact substring match after normalization (fast path)
    2. Single-character edit-distance match for short strings (handles
       common OCR errors and niqqud-stripping artifacts in names)
    """
    if not value or not source_text:
        return False, ''

    norm_value = normalize_hebrew(value.strip())
    norm_source = normalize_hebrew(source_text)

    if not norm_value:
        return False, ''

    # Fast path: exact substring
    idx = norm_source.find(norm_value)
    if idx != -1:
        snippet = _extract_window(source_text, idx, len(norm_value))
        return True, snippet

    # Fuzzy path: only for short strings (names, short terms ≤ 20 chars)
    if len(norm_value) <= 20:
        fuzzy_idx = _fuzzy_find(norm_value, norm_source)
        if fuzzy_idx != -1:
            snippet = _extract_window(source_text, fuzzy_idx, len(norm_value))
            return True, snippet

    return False, ''


def _extract_window(text: str, idx: int, match_len: int, window: int = 150) -> str:
    """Extract a context window from text around position idx."""
    half = window // 2
    start = max(0, idx - half)
    end = min(len(text), idx + match_len + half)
    snippet = text[start:end].strip()
    # Add ellipsis if truncated
    if start > 0:
        snippet = '...' + snippet
    if end < len(text):
        snippet = snippet + '...'
    return snippet


def _fuzzy_find(short: str, long_text: str, max_edits: int = 1) -> int:
    """
    Sliding-window edit-distance search. Returns index of first match
    within max_edits, or -1. Only called for short strings (≤ 20 chars).
    """
    n = len(short)
    if n < 3:
        # Too short for fuzzy — would create too many false positives
        return -1
    for i in range(len(long_text) - n + 1):
        if _edit_distance(short, long_text[i:i + n]) <= max_edits:
            return i
    return -1


def _edit_distance(s1: str, s2: str) -> int:
    """Wagner-Fischer edit distance. Only called on strings ≤ 20 chars."""
    m, n = len(s1), len(s2)
    # Single-row optimization
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return prev[n]
