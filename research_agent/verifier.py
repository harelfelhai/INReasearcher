"""
Verification: multi-source consensus → confidence score

Key design decisions:
  - Votes are counted per DOMAIN, not per URL. Two mirrors of the same article
    count as one source. This prevents inflated corroboration counts.
  - Normalization before comparison: "יצחק רבין" and "יצחק רבּין" (with dagesh)
    are treated as the same value.
  - Authoritative domain boost: a single gov.il or he.wikipedia.org source can
    elevate MEDIUM → HIGH when the claim type warrants it.
  - Conflict detection is non-blocking: we surface it as a flag, not an error.
    Researchers need to see "sources disagree" more than they need silence.
"""

from collections import Counter
from .models import ColumnPlan, ExtractionResult, VerifiedCell
from .hebrew_utils import normalize_hebrew

# Domains considered authoritative for Israeli research
_AUTHORITATIVE = frozenset({
    "he.wikipedia.org",
    "knesset.gov.il",
    "data.gov.il",
    "gov.il",
    "nevo.co.il",
    "takdin.co.il",
    "archive.org",
})


def verify_field(
    field: ColumnPlan,
    extractions: list[ExtractionResult],
) -> VerifiedCell:
    """
    Apply consensus logic over a list of extractions for one field.

    Confidence levels:
      HIGH      ≥ min_corroborations independent domains agree
                OR 1 authoritative domain + min_corroborations == 1
      MEDIUM    below min_corroborations but ≥ 1 grounded result
      LOW       only 1 result, low extractor_confidence, or minority value
      NOT_FOUND no grounded extractions at all
    """
    flags: list[str] = []

    grounded = [e for e in extractions if e.is_grounded and e.value]

    if not grounded:
        return VerifiedCell(
            field_id=field.id,
            label_he=field.label_he,
            value=None,
            confidence="NOT_FOUND",
            corroboration_count=0,
            flags=["no_grounded_sources"],
        )

    # One vote per domain — prevents mirror inflation
    domain_to_norm_value: dict[str, str] = {}
    domain_to_extraction: dict[str, ExtractionResult] = {}

    for e in grounded:
        if e.source_domain not in domain_to_norm_value:
            domain_to_norm_value[e.source_domain] = normalize_hebrew(e.value)
            domain_to_extraction[e.source_domain] = e

    norm_value_counts: Counter = Counter(domain_to_norm_value.values())

    # Most-voted normalized value and how many domains support it
    top_norm_value, top_count = norm_value_counts.most_common(1)[0]

    # Pick the best extraction that matches the winning value
    # Prefer authoritative domains, then highest extractor_confidence
    winning_extractions = [
        e for d, e in domain_to_extraction.items()
        if domain_to_norm_value[d] == top_norm_value
    ]
    winning_extractions.sort(
        key=lambda e: (e.source_domain in _AUTHORITATIVE, e.extractor_confidence),
        reverse=True,
    )
    best = winning_extractions[0]

    # Flag conflicts
    if len(norm_value_counts) > 1:
        flags.append(f"conflict:{len(norm_value_counts)}_distinct_values")

    # Determine confidence
    has_authoritative = any(
        d in _AUTHORITATIVE
        for d, v in domain_to_norm_value.items()
        if v == top_norm_value
    )

    if top_count >= field.min_corroborations and (top_count >= 2 or has_authoritative):
        confidence = "HIGH"
    elif top_count >= field.min_corroborations:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
        flags.append(f"below_min_corroborations:got_{top_count}_need_{field.min_corroborations}")

    # Authoritative boost: single strong source can lift MEDIUM → HIGH
    if confidence == "MEDIUM" and has_authoritative:
        confidence = "HIGH"
        flags.append("authoritative_boost")

    primary_source = {
        "url": best.source_url,
        "domain": best.source_domain,
        "quote": best.quote_original,
        "llm_confidence": best.extractor_confidence,
    }

    all_sources = [
        {
            "url": e.source_url,
            "domain": e.source_domain,
            "quote": e.quote_original,
            "llm_confidence": e.extractor_confidence,
        }
        for e in grounded
    ]

    return VerifiedCell(
        field_id=field.id,
        label_he=field.label_he,
        value=best.value,
        confidence=confidence,
        corroboration_count=top_count,
        primary_source=primary_source,
        all_sources=all_sources,
        flags=flags,
    )
