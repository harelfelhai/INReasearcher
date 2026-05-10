"""
Verification: multi-source consensus → confidence score

Key design decisions:
  - Votes are counted per DOMAIN, not per URL. Two mirrors of the same article
    count as one source. This prevents inflated corroboration counts.
  - Normalization before comparison: "יצחק רבין" and "יצחק רבּין" (with dagesh)
    are treated as the same value.
  - Source-agnostic authoritativeness: there is NO global allowlist of "good"
    domains. The compiler decides per-field which domains are authoritative
    for THIS query (via ColumnPlan.preferred_source_domains) and the verifier
    uses that list. A legal-research query trusts court databases; a corporate-
    research query trusts SEC filings; the system never assumes which.
  - Conflict detection is non-blocking: we surface it as a flag, not an error.
    Researchers need to see "sources disagree" more than they need silence.
"""

from collections import Counter
from .models import ColumnPlan, ExtractionResult, VerifiedCell
from .hebrew_utils import normalize_hebrew


def _is_preferred(domain: str, preferred: list[str]) -> bool:
    """
    Match a result domain against the compiler's preferred-domain list
    for this field. Supports suffix matching so 'gov.il' matches
    'foo.muni.gov.il' and 'sec.gov' matches 'www.sec.gov'.
    """
    if not preferred:
        return False
    domain = domain.lower().lstrip("www.")
    for pref in preferred:
        pref = pref.lower().lstrip("www.")
        if domain == pref or domain.endswith("." + pref):
            return True
    return False


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
    preferred = field.preferred_source_domains or []

    winning_extractions.sort(
        key=lambda e: (_is_preferred(e.source_domain, preferred), e.extractor_confidence),
        reverse=True,
    )
    best = winning_extractions[0]

    # Flag conflicts
    if len(norm_value_counts) > 1:
        flags.append(f"conflict:{len(norm_value_counts)}_distinct_values")

    # Per-field authoritativeness: domains the COMPILER chose for this query
    has_preferred = any(
        _is_preferred(d, preferred)
        for d, v in domain_to_norm_value.items()
        if v == top_norm_value
    )

    if top_count >= field.min_corroborations and (top_count >= 2 or has_preferred):
        confidence = "HIGH"
    elif top_count >= field.min_corroborations:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
        flags.append(f"below_min_corroborations:got_{top_count}_need_{field.min_corroborations}")

    # Preferred-source boost: single match against the compiler's chosen
    # authoritative domains for THIS query can lift MEDIUM → HIGH
    if confidence == "MEDIUM" and has_preferred:
        confidence = "HIGH"
        flags.append("preferred_source_boost")

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
