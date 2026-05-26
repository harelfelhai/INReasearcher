"""
Knesset OData API adapter.

Queries the Knesset public OData endpoint for structured data about
Members of Knesset (MKs), factions, and positions.

Integration point: inject_knesset_mk_data() is called from
_gather_ranked_pages_for_field() before the generic web search, adding a
high-precision direct page that Claude processes at highest priority.
Works identically to _inject_wikidata but targets Knesset-specific
structured data — skipping the web search for well-typed MK entities.

Kill switch: KNESSET_ODATA_DISABLED=1 disables all API calls.
Base URL override: KNESSET_ODATA_BASE_URL (useful in tests).

Observability: all events emit [knesset_odata] lines to stderr with
structured key=value fields — grep for [knesset_odata] to trace every
lookup, hit, miss, and injection.

No-regression guarantee: every miss or failure leaves direct_pages
unchanged — the caller proceeds to web search exactly as before.
"""
from __future__ import annotations

import json
import os
import sys
import unicodedata
import urllib.parse
import urllib.request
from typing import Optional

# ── Configuration ─────────────────────────────────────────────────────────────

_DEFAULT_BASE = "https://knesset.gov.il/Odata/ParliamentInfo.svc"

# entity_type keywords that indicate we're researching Members of Knesset.
# Checked via substring — the user might write "חברי כנסת ה-25", "ח״כ", etc.
_MK_KEYWORDS: frozenset[str] = frozenset([
    "חבר כנסת", "חברת כנסת", "חברי כנסת",
    'ח"כ', "ח׳כ", "ח'כ", "חכ",
    "member of knesset", "members of knesset",
    "knesset member", "knesset members",
    "mk", "mks",
])


# ── Internal helpers ──────────────────────────────────────────────────────────

def _is_disabled() -> bool:
    return os.environ.get("KNESSET_ODATA_DISABLED", "").strip() == "1"


def _log(event: str, **fields) -> None:
    """Structured stderr line for every OData touchpoint."""
    parts = [f"event={event}"]
    for k, v in fields.items():
        if isinstance(v, str):
            parts.append(f"{k}={v!r}")
        else:
            parts.append(f"{k}={v}")
    print("[knesset_odata] " + " ".join(parts), file=sys.stderr, flush=True)


def _base_url() -> str:
    return os.environ.get("KNESSET_ODATA_BASE_URL", _DEFAULT_BASE).rstrip("/")


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip()


def _knesset_fetch(url: str) -> dict:
    """
    Issue a single GET to the Knesset OData endpoint and return parsed JSON.
    Raises on any network / HTTP error — callers handle exceptions.
    """
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "INResearcher/1.0 (academic research tool)",
        },
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ── Public helpers ────────────────────────────────────────────────────────────

def is_mk_context(entity_type: Optional[str]) -> bool:
    """
    True if entity_type suggests we're researching Members of Knesset.

    Checks every configured keyword as a substring of entity_type (case-folded,
    NFC-normalized), so "חברי כנסת ה-25", "ח\"כ", "Knesset MKs" all match.
    """
    if not entity_type:
        return False
    et = _norm(entity_type).lower()
    return any(kw in et for kw in _MK_KEYWORDS)


def _build_odata_url(entity_set: str, **odata_params: str) -> str:
    """
    Build an OData v3 URL with LITERAL '$' characters in option names.

    Knesset's OData server rejects requests where '$filter', '$format', etc.
    are percent-encoded as '%24filter' (returns 400 Bad Request). The values,
    however, MUST be URL-encoded — including Hebrew text and quote characters.
    """
    parts = []
    for k, v in odata_params.items():
        encoded_value = urllib.parse.quote(str(v), safe="")
        parts.append(f"${k}={encoded_value}")
    return f"{_base_url()}/{entity_set}?{'&'.join(parts)}"


def _full_name(person: dict) -> str:
    """Compose 'FirstName LastName' (Hebrew) from a KNS_Person row."""
    first = _norm(person.get("FirstName") or "")
    last  = _norm(person.get("LastName")  or "")
    return f"{first} {last}".strip()


def find_mk_by_name(name: str) -> Optional[dict]:
    """
    Query KNS_Person for an MK whose LastName contains `name`.

    Strategy:
      1. Use only the LAST WORD of the name (likely surname) as the search
         token — KNS_Person stores LastName and FirstName as separate fields.
      2. Filter substringof() against the LastName field — this is the actual
         field name in the Knesset OData schema (verified against the live
         service document). There is NO 'Name' field.
      3. If multiple rows return and FirstName is supplied, prefer the row
         whose FirstName matches; otherwise fall back to the first hit.
      4. Build URL with literal $ characters (server requirement).
    """
    name_n = _norm(name)
    if not name_n:
        return None

    tokens = name_n.split()
    needle = tokens[-1] if tokens else name_n
    first_hint = tokens[0] if len(tokens) > 1 else ""

    filter_expr = f"substringof('{needle}',LastName) eq true"
    url = _build_odata_url(
        "KNS_Person",
        filter=filter_expr,
        format="json",
        top="10",
    )

    try:
        data = _knesset_fetch(url)
    except Exception as exc:
        _log("mk_lookup", name=name_n, needle=needle, result="ERROR",
             error=str(exc)[:120], url=url[:200])
        return None

    persons = data.get("value", [])
    if not persons:
        _log("mk_lookup", name=name_n, needle=needle,
             result="MISS", reason="no_matches")
        return None

    # Prefer the row whose FirstName matches the hint (disambiguates surname
    # collisions like נתניהו בנימין vs נתניהו שרה).
    if first_hint:
        for p in persons:
            if _norm(p.get("FirstName") or "") == first_hint:
                _log("mk_lookup", name=name_n, needle=needle, result="HIT",
                     person_id=p.get("PersonID"),
                     matched_name=_full_name(p),
                     total_found=len(persons),
                     disambiguated_by="first_name")
                return p

    hit = persons[0]
    _log("mk_lookup", name=name_n, needle=needle, result="HIT",
         person_id=hit.get("PersonID"),
         matched_name=_full_name(hit),
         total_found=len(persons))
    return hit


def fetch_mk_positions(person_id: int) -> list[dict]:
    """
    Fetch KNS_PersonToPosition rows for this person, most recent first.

    Each row contains: PositionID, FactionID, KnessetNum, StartDate, EndDate.
    Returns empty list on error (no-regression).
    """
    url = _build_odata_url(
        "KNS_PersonToPosition",
        filter=f"PersonID eq {person_id}",
        format="json",
        orderby="KnessetNum desc",
        top="30",
    )

    try:
        data = _knesset_fetch(url)
        rows = data.get("value", [])
        _log("positions_fetch", person_id=person_id, rows=len(rows))
        return rows
    except Exception as exc:
        _log("positions_fetch", person_id=person_id,
             result="ERROR", error=str(exc)[:120])
        return []


def fetch_faction_name(faction_id: int) -> Optional[str]:
    """Return the Hebrew name of a faction by its ID. None on miss / error."""
    url = f"{_base_url()}/KNS_Faction({faction_id})?$format=json"
    try:
        data = _knesset_fetch(url)
        name = data.get("Name")
        if name:
            _log("faction_fetch", faction_id=faction_id, name=name)
        else:
            _log("faction_fetch", faction_id=faction_id, result="MISS")
        return name or None
    except Exception as exc:
        _log("faction_fetch", faction_id=faction_id,
             result="ERROR", error=str(exc)[:80])
        return None


def format_mk_as_text(person: dict, positions: list[dict]) -> str:
    """
    Format Knesset OData person + position rows as a Hebrew text snippet
    for Claude's extraction pipeline.

    Deliberately mirrors the structure of a Hebrew Wikipedia info-box so
    that existing extraction strategies (regex anchors, keyword windowing)
    work without any changes.
    """
    full        = _full_name(person)
    email       = _norm(person.get("Email") or "")
    gender_desc = _norm(person.get("GenderDesc") or "")
    is_current  = person.get("IsCurrent")

    lines = [
        "מקור: מאגר הכנסת — API פתוח (OData)",
        f"שם מלא: {full}",
    ]
    if gender_desc:
        lines.append(f"מגדר: {gender_desc}")
    if email:
        lines.append(f"דוא\"ל: {email}")
    if is_current is True:
        lines.append("חבר/ת כנסת מכהן/ת כיום: כן")
    elif is_current is False:
        lines.append("חבר/ת כנסת מכהן/ת כיום: לא")

    # Collect distinct (knesset_num, faction_id) pairs — deduplicated
    knesset_factions: dict[int, set[int]] = {}
    for pos in positions:
        knum = pos.get("KnessetNum")
        fid  = pos.get("FactionID")
        if knum and fid:
            knesset_factions.setdefault(knum, set()).add(fid)

    for knum in sorted(knesset_factions, reverse=True):
        for fid in sorted(knesset_factions[knum]):
            fname = fetch_faction_name(fid) or f"סיעה {fid}"
            lines.append(f"כנסת {knum}: חבר/ת סיעת {fname}")

    # Knesset terms served (summary line — useful for temporal anchoring)
    terms = sorted(set(
        p.get("KnessetNum") for p in positions if p.get("KnessetNum")
    ))
    if terms:
        lines.append(f"כנסות שבהן כיהן/ה: {', '.join(str(k) for k in terms)}")

    return "\n".join(lines)


# ── Main injection entry-point ────────────────────────────────────────────────

def inject_knesset_mk_data(
    entity: str,
    entity_type: Optional[str],
    direct_pages: list[tuple[str, str]],
    seen_urls: set[str],
) -> None:
    """
    High-precision Knesset OData injection for Member of Knesset entities.

    Prepends formatted MK data to `direct_pages` (processed before any
    web-search URL) when:
      1. KNESSET_ODATA_DISABLED is not set
      2. entity_type matches MK keywords (is_mk_context)
      3. entity found in KNS_Person

    Any miss or error leaves direct_pages and seen_urls unchanged.
    The injected URL is added to seen_urls to prevent web search from
    re-fetching the same synthetic Knesset page.
    """
    if _is_disabled():
        _log("inject", entity=entity, result="SKIP", reason="disabled")
        return
    if not is_mk_context(entity_type):
        _log("inject", entity=entity, result="SKIP",
             reason="entity_type_not_mk", entity_type=str(entity_type))
        return

    person = find_mk_by_name(entity)
    if not person:
        return   # find_mk_by_name already logged

    person_id = person.get("PersonID")
    positions = fetch_mk_positions(person_id) if person_id else []

    text = format_mk_as_text(person, positions)
    synthetic_url = f"https://knesset.gov.il/odata/mk/{person_id}"

    if synthetic_url in seen_urls:
        _log("inject", entity=entity, result="SKIP",
             reason="url_already_seen", url=synthetic_url)
        return

    seen_urls.add(synthetic_url)
    direct_pages.insert(0, (synthetic_url, text))   # prepend — highest priority
    _log("inject", entity=entity, person_id=person_id,
         url=synthetic_url, chars=len(text))
