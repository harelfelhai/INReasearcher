"""
Israeli political-domain entity catalog.

Loads curated JSON files from research_agent/data/il_entities/ and exposes a
narrow lookup API for the rest of the pipeline. Designed to be:

  * additive — every miss falls through to existing behavior
  * observable — every lookup logs hit/miss with rationale
  * killable — IL_CATALOG_DISABLED=1 in the environment disables it entirely

The catalog covers two stable categories:
  - municipality (top ~35 cities, with Hebrew/English variants and abbreviations)
  - ministry    (~27 government ministries)

Used by:
  - extractor._gather_ranked_pages_for_field → normalize entity name to its
    canonical form before building search queries and fetching Wikipedia.
"""
from __future__ import annotations

import json
import os
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


# ── Models ───────────────────────────────────────────────────────────────────

class CatalogEntity(BaseModel):
    """One canonical entity with its alias spellings."""
    canonical_he: str
    canonical_en: str
    aliases_he: list[str] = Field(default_factory=list)
    aliases_en: list[str] = Field(default_factory=list)
    wikipedia_he: Optional[str] = None     # canonical Hebrew Wikipedia title (post-redirect)
    category: str                          # filled in at load time
    metadata: dict = Field(default_factory=dict)

    @property
    def all_names(self) -> list[str]:
        """Every spelling under which this entity might be referenced."""
        out: list[str] = [self.canonical_he, self.canonical_en]
        out.extend(self.aliases_he)
        out.extend(self.aliases_en)
        return [n for n in out if n]


class CatalogCategory(BaseModel):
    """One catalog category (e.g. 'municipality') + the text labels that match it."""
    name: str                              # e.g. "municipality"
    version: str
    labels_he: list[str]                   # how the user might describe this entity_type
    labels_en: list[str]
    entities: list[CatalogEntity]


class Catalog(BaseModel):
    """Top-level catalog object — collection of categories."""
    categories: dict[str, CatalogCategory] = Field(default_factory=dict)


# ── Loading ──────────────────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).parent / "data" / "il_entities"

_CATEGORY_FILES = {
    "municipality": "municipalities.json",
    "ministry":     "ministries.json",
}


def _is_disabled() -> bool:
    """Kill-switch for A/B testing or emergency rollback."""
    return os.environ.get("IL_CATALOG_DISABLED", "").strip().lower() in ("1", "true", "yes")


def _log(event: str, **fields) -> None:
    """
    Structured log line for catalog operations.

    Every catalog touchpoint should call this. Format:
      [il_catalog] event=lookup name='ת״א' category='municipality' result=HIT canonical='תל אביב-יפו' via='alias_he'
    """
    parts = [f"event={event}"]
    for k, v in fields.items():
        if isinstance(v, str):
            parts.append(f"{k}={v!r}")
        else:
            parts.append(f"{k}={v}")
    print(f"[il_catalog] " + " ".join(parts), file=sys.stderr)


@lru_cache(maxsize=1)
def load_catalog() -> Catalog:
    """
    Load the full catalog from JSON. Cached for the process lifetime.

    Returns an empty catalog if disabled via IL_CATALOG_DISABLED — this lets
    every call site fall through to existing behavior with zero branching.
    """
    if _is_disabled():
        _log("load", disabled=True)
        return Catalog(categories={})

    cats: dict[str, CatalogCategory] = {}
    total_entities = 0
    for cat_name, filename in _CATEGORY_FILES.items():
        path = _DATA_DIR / filename
        if not path.exists():
            _log("load_warning", category=cat_name, missing_file=str(path))
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            _log("load_error", category=cat_name, file=str(path), error=str(exc))
            continue

        entities = []
        for e in raw.get("entities", []):
            entities.append(CatalogEntity(
                canonical_he=e["canonical_he"],
                canonical_en=e["canonical_en"],
                aliases_he=e.get("aliases_he", []),
                aliases_en=e.get("aliases_en", []),
                wikipedia_he=e.get("wikipedia_he"),
                category=cat_name,
                metadata=e.get("metadata", {}),
            ))
        cats[cat_name] = CatalogCategory(
            name=cat_name,
            version=raw.get("version", "0.0.0"),
            labels_he=raw.get("category_labels_he", []),
            labels_en=raw.get("category_labels_en", []),
            entities=entities,
        )
        total_entities += len(entities)

    catalog = Catalog(categories=cats)
    _log("load",
         disabled=False,
         categories=list(cats.keys()),
         total_entities=total_entities,
         versions={c.name: c.version for c in cats.values()})
    return catalog


def reset_cache() -> None:
    """Drop the lru_cache so the next load_catalog() re-reads from disk.
    Useful in tests; you probably don't want this in production code."""
    load_catalog.cache_clear()


# ── Normalization helpers ────────────────────────────────────────────────────

def _normalize_for_match(s: str) -> str:
    """
    Normalize a name for matching: NFC-fold, strip whitespace, casefold.
    Keeps Hebrew punctuation distinctions (gershayim variants ״ vs "), so the
    catalog must spell them explicitly — by design, we'd rather miss than
    silently merge characters.
    """
    if not s:
        return ""
    return unicodedata.normalize("NFC", s).strip().casefold()


# ── Public API ───────────────────────────────────────────────────────────────

def find_entity(
    name: str,
    category: Optional[str] = None,
) -> Optional[CatalogEntity]:
    """
    Look up an entity by any of its spellings.

    Args:
      name: the entity name as the user typed / system discovered it.
      category: optional category hint ('municipality' | 'ministry'). When
        given, restricts the search to that category — eliminates false
        positives where a name collides across categories.

    Returns:
      CatalogEntity if found, None otherwise (catalog miss = fall through).

    Logs every call: the hit/miss outcome, which alias matched, and which
    category — these lines are how we debug "why did/didn't this normalize".
    """
    catalog = load_catalog()
    if not catalog.categories:
        _log("lookup", name=name, category=category, result="MISS", reason="catalog_empty")
        return None

    needle = _normalize_for_match(name)
    if not needle:
        _log("lookup", name=name, category=category, result="MISS", reason="empty_input")
        return None

    cats_to_search = (
        [catalog.categories[category]]
        if category and category in catalog.categories
        else list(catalog.categories.values())
    )

    for cat in cats_to_search:
        for entity in cat.entities:
            # Field-by-field check so we can log WHICH field matched.
            if _normalize_for_match(entity.canonical_he) == needle:
                _log("lookup", name=name, category=cat.name, result="HIT",
                     canonical=entity.canonical_he, via="canonical_he")
                return entity
            if _normalize_for_match(entity.canonical_en) == needle:
                _log("lookup", name=name, category=cat.name, result="HIT",
                     canonical=entity.canonical_he, via="canonical_en")
                return entity
            for alias in entity.aliases_he:
                if _normalize_for_match(alias) == needle:
                    _log("lookup", name=name, category=cat.name, result="HIT",
                         canonical=entity.canonical_he, via="alias_he", matched=alias)
                    return entity
            for alias in entity.aliases_en:
                if _normalize_for_match(alias) == needle:
                    _log("lookup", name=name, category=cat.name, result="HIT",
                         canonical=entity.canonical_he, via="alias_en", matched=alias)
                    return entity

    _log("lookup", name=name, category=category, result="MISS",
         reason="no_alias_match",
         categories_searched=[c.name for c in cats_to_search])
    return None


def list_entities(category: str) -> list[CatalogEntity]:
    """Return all entities in a category. Empty list if unknown category."""
    catalog = load_catalog()
    cat = catalog.categories.get(category)
    return list(cat.entities) if cat else []


def match_category(entity_type_text: str) -> Optional[str]:
    """
    Fuzzy-match a free-text entity_type string (e.g. 'ראשי ערים', 'cities',
    'municipalities') to a catalog category name. Substring-based — the user
    might phrase it many ways and we want to catch all reasonable variants.

    Returns the category name (e.g. 'municipality') or None on miss.
    """
    catalog = load_catalog()
    if not catalog.categories:
        _log("match_category", text=entity_type_text, result="MISS", reason="catalog_empty")
        return None

    needle = _normalize_for_match(entity_type_text)
    if not needle:
        _log("match_category", text=entity_type_text, result="MISS", reason="empty_input")
        return None

    for cat in catalog.categories.values():
        for label in cat.labels_he + cat.labels_en:
            label_n = _normalize_for_match(label)
            # Match either direction: user said "ראשי ערים", label "ראשי ערים"
            # match; user said "ראשי עיר", label "ראש עיר" partially matches.
            if label_n in needle or needle in label_n:
                _log("match_category", text=entity_type_text, result="HIT",
                     category=cat.name, via_label=label)
                return cat.name

    _log("match_category", text=entity_type_text, result="MISS",
         reason="no_label_match")
    return None


def normalize_entity_for_search(
    name: str,
    entity_type: Optional[str] = None,
) -> tuple[str, list[str]]:
    """
    Return (canonical_name, alias_list) for a given entity name.

    On catalog hit:
      canonical_name = the catalog's canonical Hebrew name (most likely to
        match Wikipedia + Wikidata cleanly)
      alias_list    = every other spelling (used by callers that want to
        try multiple search variants)

    On catalog miss:
      (name, []) — caller should proceed exactly as if the catalog didn't
      exist. THIS IS THE NO-REGRESSION GUARANTEE.

    `entity_type` is an optional hint to narrow the search and avoid
    cross-category collisions; if it doesn't match any category, we still
    search globally (graceful degradation).
    """
    category = match_category(entity_type) if entity_type else None
    entity = find_entity(name, category=category)
    if entity is None:
        return name, []

    aliases = [n for n in entity.all_names if n != entity.canonical_he]
    return entity.canonical_he, aliases
