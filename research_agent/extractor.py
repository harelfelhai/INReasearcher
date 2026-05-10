"""
Stage 1: Search → Extract with Provenance

For each (entity, field) pair:
  1. Generate targeted search queries (Hebrew primary, English fallback)
  2. Run Tavily search with raw content enabled
  3. For each result page, ask Claude to extract the value — grounded in the text
  4. Run the grounding check (is_grounded) to verify the quote actually exists
  5. Return all ExtractionResult objects to the verifier

Model: claude-sonnet-4-6
  Rationale: extraction requires precise instruction-following in Hebrew.
  Haiku drifts more on the "return null" rule; Sonnet is more disciplined.
  This is where most token cost lives — budget accordingly (~$0.02-0.05/field).

Cost control levers:
  - max_results caps how many pages we extract from per field
  - early_stop_on_high: stops searching once we hit N grounded results
  - Source content is truncated to 4000 chars (covers most useful content)
"""

import re
import anthropic
from tavily import TavilyClient

from .models import ColumnPlan, ExtractionResult
from .hebrew_utils import is_grounded
from .memory import (
    SuccessMemory,
    format_extraction_examples,
    format_extraction_warnings,
)

# ── Extractor tool (structured output enforcement) ────────────────────────────

_EXTRACTOR_TOOL = {
    "name": "extract_field",
    "description": "Extract a specific data field from provided source text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "value": {
                "type": ["string", "null"],
                "description": (
                    "The extracted value in the language of the source. "
                    "NULL if the information is not clearly present in this text."
                )
            },
            "quote_original": {
                "type": ["string", "null"],
                "description": (
                    "The exact sentence or short phrase from the source text that "
                    "proves this value. DIRECT COPY-PASTE — no paraphrasing. "
                    "NULL if value is null."
                )
            },
            "extractor_confidence": {
                "type": "number",
                "description": "0.0–1.0 confidence that value is correct and fully grounded.",
                "minimum": 0.0,
                "maximum": 1.0
            },
            "not_found_reason": {
                "type": ["string", "null"],
                "description": (
                    "If value is null, briefly explain why: "
                    "'wrong time period', 'entity not mentioned', 'ambiguous entity', "
                    "'information not in text', etc."
                )
            }
        },
        "required": ["value", "quote_original", "extractor_confidence"]
    }
}

_EXTRACTOR_SYSTEM = """\
You are a precision data extractor. You read arbitrary source text — from any
website, in any domain (legal, corporate, academic, governmental, news,
encyclopedic) — and extract specific facts. Nothing more, nothing less.

You are SOURCE-AGNOSTIC. The text may come from a court ruling, a Wikipedia
article, a SEC filing, a Hebrew news site, a PDF excerpt, or a personal blog.
Your job is identical regardless of source structure or domain.

ABSOLUTE RULES (violating them destroys data integrity):

1. GROUNDING ONLY
   You may return a value ONLY if it is explicitly stated in the provided text.
   No background knowledge. No inference. No "it's probably X because...".

2. VERBATIM QUOTE
   The quote_original field must be a direct copy-paste from the text — exactly
   as written, in the original language. Do NOT translate, summarize, or paraphrase.

3. NULL IS CORRECT
   If the information is not clearly in the text, return null for both value
   and quote_original. A confident null beats a hallucinated value every time.

4. ENTITY SPECIFICITY
   Confirm the text refers to the exact entity named in the question.
   If the text mentions a different person/place/case/company with a similar
   name, return null.

5. TEMPORAL SPECIFICITY
   When a year or period is specified, the extracted fact must refer to that
   period. A source saying "served 1995-2000" does NOT answer "served in 1990".

6. SOURCE QUALITY IS NOT YOUR JOB
   Do not refuse to extract based on perceived source quality. Extract what
   the text says; the verifier weighs source authority across multiple results."""


# ── Smart content windowing ───────────────────────────────────────────────────

# Type-specific keywords used to locate relevant passages within long articles.
# Multilingual: Hebrew is primary for Israeli sources, English for global.
_TYPE_KEYWORDS: dict[str, list[str]] = {
    "person_name":   ["ראש העיר", "ראש עיר", "כיהן", "שימש", "נבחר", "מונה",
                      "mayor", "elected", "served", "appointed", "tenure"],
    "url":           ["www.", "http", "אתר", "אתר רשמי", ".gov", ".muni",
                      ".org", ".com", ".il", "website", "official site"],
    "organization":  ["עיריה", "עיריית", "מועצה", "municipality", "council",
                      "city hall"],
    "date":          ["שנת", "בשנת", "תאריך", "year", "date", "established"],
    "number":        ["אוכלוסייה", "תושבים", "population", "total"],
}

_INTRO_CHARS   = 1500    # always include the article intro for context
_WINDOW_RADIUS = 700     # chars on each side of a keyword hit
_BUDGET        = 15000   # total chars sent to Claude per extraction
_ELLIPSIS      = "\n\n[...]\n\n"


def _select_relevant_text(content: str, field, entity: str) -> str:
    """
    For short content (≤ _BUDGET chars) → returns as-is.
    For long content → returns intro + concatenated windows around every
    occurrence of field-relevant keywords (entity name, temporal anchor,
    type-specific terms). Caps at _BUDGET total chars.

    This is what makes the system work on long Wikipedia articles: the
    Tel Aviv article is 73K chars, but the mayor section is only a few
    hundred chars hidden deep inside. Sending the first 4K chars missed
    it entirely; sending all 73K is wasteful. Smart windowing finds it.
    """
    if not content:
        return ""
    if len(content) <= _BUDGET:
        return content

    # 1. Build the keyword set for this field
    keywords: set[str] = {entity}
    if entity and len(entity) > 2:
        keywords.add(entity)
    for word in field.label_he.split() + field.label_en.split():
        if len(word) > 3:
            keywords.add(word.lower())
    if field.temporal_anchor:
        keywords.add(field.temporal_anchor)
        # Also include adjacent years — useful for date-range coverage
        try:
            yr = int(field.temporal_anchor)
            for delta in (-3, -2, -1, 1, 2, 3):
                keywords.add(str(yr + delta))
        except ValueError:
            pass
    keywords.update(_TYPE_KEYWORDS.get(field.type, []))

    # 2. Locate all keyword positions in the content
    content_lower = content.lower()
    positions: list[int] = []
    for kw in keywords:
        kw_lower = kw.lower()
        if len(kw_lower) < 2:
            continue
        start = 0
        # Cap matches per keyword to prevent one common word dominating
        matches_for_kw = 0
        while matches_for_kw < 20:
            idx = content_lower.find(kw_lower, start)
            if idx == -1:
                break
            positions.append(idx)
            start = idx + len(kw_lower)
            matches_for_kw += 1

    if not positions:
        # No keyword matches at all → just send the intro
        return content[:_BUDGET]

    # 3. Build windows around each match position
    raw_windows = sorted(
        (max(0, p - _WINDOW_RADIUS), min(len(content), p + _WINDOW_RADIUS))
        for p in positions
    )

    # 4. Merge overlapping/adjacent windows
    merged: list[tuple[int, int]] = [raw_windows[0]]
    for s, e in raw_windows[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e + 50:                    # within 50 chars → merge
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))

    # 5. Assemble: intro + merged windows, up to budget
    intro = content[:_INTRO_CHARS]
    pieces: list[str] = [intro]
    used = len(intro)

    for s, e in merged:
        if e <= _INTRO_CHARS:                    # already covered by intro
            continue
        if s < _INTRO_CHARS:
            s = _INTRO_CHARS                     # avoid duplicating intro text
        section = content[s:e]
        if used + len(section) + len(_ELLIPSIS) > _BUDGET:
            remaining = _BUDGET - used - len(_ELLIPSIS)
            if remaining > 200:
                pieces.append(_ELLIPSIS)
                pieces.append(section[:remaining])
            break
        pieces.append(_ELLIPSIS)
        pieces.append(section)
        used += len(section) + len(_ELLIPSIS)

    return "".join(pieces)


def extract_from_source(
    field: ColumnPlan,
    entity: str,
    source_url: str,
    source_content: str,
    resolved_deps: dict,
    client: anthropic.Anthropic,
    memory: SuccessMemory | None = None,
) -> ExtractionResult:
    """
    Ask Claude to extract one field from one source page.
    Applies the grounding check before accepting the result.

    If `memory` is provided, injects:
      - up to 2 past validated extractions for this field type (positive few-shot)
      - up to 5 past hallucinations as AVOID warnings (negative few-shot)
    """
    dep_context = ""
    if field.depends_on and field.depends_on in resolved_deps:
        dep_val = resolved_deps[field.depends_on]
        dep_context = (
            f"\nContext: The '{field.depends_on}' for this entity has been "
            f"identified as: {dep_val}. Use this to disambiguate if needed."
        )

    temporal_note = (
        f"\nTEMPORAL CONSTRAINT: The answer must be valid for the year/period "
        f"'{field.temporal_anchor}'. A source stating a date RANGE that includes "
        f"'{field.temporal_anchor}' (e.g. 'served 1974–1993' covers 1990) IS a valid "
        f"answer — extract the value. Only return null if the source's dates clearly "
        f"EXCLUDE '{field.temporal_anchor}', or if no relevant date information exists."
        if field.temporal_anchor else ""
    )

    examples_block = ""
    warnings_block = ""
    if memory is not None:
        good = memory.get_extraction_examples(field.type, field.label_en, k=2)
        examples_block = format_extraction_examples(good)
        bad = memory.get_extraction_warnings(field.type, _domain(source_url))
        warnings_block = format_extraction_warnings(bad)

    user_prompt = (
        f"Entity: {entity}\n"
        f"Field to extract: {field.label_en} / {field.label_he}\n"
        f"Field type: {field.type}"
        f"{temporal_note}"
        f"{dep_context}"
        f"{examples_block}"
        f"{warnings_block}\n\n"
        f"Source URL: {source_url}\n"
        f"Source text:\n---\n{_select_relevant_text(source_content, field, entity)}\n---\n\n"
        f"Extract '{field.label_en}' for entity '{entity}' from the text above. "
        f"Follow all grounding rules strictly. Return null if not clearly present."
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=_EXTRACTOR_SYSTEM,
        tools=[_EXTRACTOR_TOOL],
        tool_choice={"type": "any"},
        messages=[{"role": "user", "content": user_prompt}]
    )

    tool_block = next(b for b in response.content if b.type == "tool_use")
    data = tool_block.input

    raw_value = data.get("value")
    raw_quote = data.get("quote_original")
    llm_confidence = float(data.get("extractor_confidence", 0.0))

    # Grounding gate: reject if the quote or value can't be found in source
    grounded = False
    verified_quote = None
    if raw_value and raw_quote:
        grounded, verified_quote = is_grounded(raw_quote, source_content)
        if not grounded:
            # Secondary: maybe the quote has formatting artifacts — check value directly
            grounded, verified_quote = is_grounded(raw_value, source_content)
    elif raw_value:
        grounded, verified_quote = is_grounded(raw_value, source_content)

    return ExtractionResult(
        field_id=field.id,
        value=raw_value if grounded else None,
        quote_original=verified_quote if grounded else None,
        source_url=source_url,
        source_domain=_domain(source_url),
        is_grounded=grounded,
        extractor_confidence=llm_confidence if grounded else 0.0,
    )


def search_and_extract(
    field: ColumnPlan,
    entity: str,
    resolved_deps: dict,
    tavily: TavilyClient,
    claude: anthropic.Anthropic,
    memory: SuccessMemory | None = None,
    max_results: int = 6,
    early_stop_on_high: int = 3,
) -> list[ExtractionResult]:
    """
    Run search queries for a field, extract from each result.

    Query order: Hebrew queries first (better for Israeli sources),
    then English. Stops early if we accumulate enough grounded results.
    """
    # Substitute entity name into query templates
    queries_he = [q.replace("{entity}", entity) for q in field.search_queries_he]
    queries_en = [q.replace("{entity}", entity) for q in field.search_queries_en[:2]]

    # If depends_on resolved to a person name, also search by that name
    if field.depends_on and field.depends_on in resolved_deps:
        dep_val = resolved_deps[field.depends_on]
        if dep_val:
            bonus = [q.replace("{entity}", dep_val) for q in field.search_queries_he[:2]]
            queries_he = queries_he + bonus

    all_queries = queries_he + queries_en

    results: list[ExtractionResult] = []
    seen_urls: set[str] = set()

    is_wikipedia = isinstance(tavily, WikipediaSearchClient)
    query_cap = 3 if is_wikipedia else 6

    # For Wikipedia: fetch the entity's own article first (most reliable source).
    # This gets e.g. the full "תל אביב-יפו" article (with redirect) which
    # contains the mayor list, official website, and municipality name —
    # avoiding the problem of search returning tangentially related articles.
    if is_wikipedia:
        direct = tavily.fetch_entity_article(entity)
        for hit in direct.get("results", []):
            url = hit.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                content = hit.get("raw_content") or hit.get("content", "")
                if content and len(content) >= 80:
                    extraction = extract_from_source(
                        field=field, entity=entity,
                        source_url=url, source_content=content,
                        resolved_deps=resolved_deps, client=claude, memory=memory,
                    )
                    if extraction.value:
                        results.append(extraction)

    for query in all_queries[:query_cap]:
        try:
            response = tavily.search(
                query=query,
                max_results=3,
                include_raw_content=True,
            )
        except Exception as exc:
            print(f"    [search error] {query[:60]!r}: {exc}")
            continue

        hits = response.get("results", [])
        if hits:
            print(f"    [search] '{query[:55]}' → {len(hits)} result(s): "
                  + ", ".join(h.get("title", h.get("url","?"))[:30] for h in hits[:3]))
        for hit in hits:
            url = hit.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            content = hit.get("raw_content") or hit.get("content", "")
            if not content or len(content) < 80:
                continue

            extraction = extract_from_source(
                field=field,
                entity=entity,
                source_url=url,
                source_content=content,
                resolved_deps=resolved_deps,
                client=claude,
                memory=memory,
            )

            if extraction.is_grounded and extraction.value:
                results.append(extraction)
                if len(results) >= early_stop_on_high:
                    return results   # enough corroborations, stop burning tokens

    return results


def _domain(url: str) -> str:
    m = re.search(r'https?://([^/]+)', url)
    return m.group(1) if m else url


# ── Mock search layer (for testing without a live Tavily key) ─────────────────

_MOCK_CONTENT_TEL_AVIV = """\
תל אביב-יפו היא עיר בישראל, הגדולה בישראל מבחינת אוכלוסייה עירונית.
שלמה להט (צ'יץ') כיהן כראש עיריית תל אביב-יפו בין השנים 1974 ל-1993.
בשנת 1990 כיהן שלמה להט בתפקיד ראש העיר.
האתר הרשמי של עיריית תל אביב הוא www.tel-aviv.gov.il.
תל אביב הוכרזה כעיר בשנת 1950 לאחר איחודה עם יפו.
"""

_MOCK_CONTENT_HAIFA = """\
חיפה היא עיר בצפון ישראל הממוקמת על הר הכרמל.
עריאל שרון ואחרים שימשו בתפקידים שונים בחיפה.
גוריון ביינארט כיהן כראש עיריית חיפה בתחילת שנות התשעים.
אריה גוראל היה ראש עיריית חיפה בין השנים 1983 ל-1993.
בשנת 1990 כיהן אריה גוראל כראש עיריית חיפה.
האתר הרשמי של עיריית חיפה הוא www.haifa.muni.il.
"""

_MOCK_RESULTS = {
    "תל אביב": [
        {
            "url": "https://he.wikipedia.org/wiki/תל_אביב-יפו",
            "title": "תל אביב-יפו — ויקיפדיה",
            "content": _MOCK_CONTENT_TEL_AVIV,
            "raw_content": _MOCK_CONTENT_TEL_AVIV,
        },
        {
            "url": "https://www.tel-aviv.gov.il/about",
            "title": "אודות עיריית תל אביב-יפו",
            "content": "שלמה להט כיהן כראש עיריית תל אביב בשנת 1990. האתר הרשמי: www.tel-aviv.gov.il",
            "raw_content": "שלמה להט כיהן כראש עיריית תל אביב בשנת 1990. האתר הרשמי: www.tel-aviv.gov.il",
        },
    ],
    "tel aviv": [
        {
            "url": "https://en.wikipedia.org/wiki/Tel_Aviv",
            "title": "Tel Aviv — Wikipedia",
            "content": "Shlomo Lahat (Chich) served as mayor of Tel Aviv from 1974 to 1993. In 1990 the mayor was Shlomo Lahat. Official site: www.tel-aviv.gov.il",
            "raw_content": "Shlomo Lahat (Chich) served as mayor of Tel Aviv from 1974 to 1993. In 1990 the mayor was Shlomo Lahat. Official site: www.tel-aviv.gov.il",
        }
    ],
    "חיפה": [
        {
            "url": "https://he.wikipedia.org/wiki/חיפה",
            "title": "חיפה — ויקיפדיה",
            "content": _MOCK_CONTENT_HAIFA,
            "raw_content": _MOCK_CONTENT_HAIFA,
        }
    ],
    "default": [
        {
            "url": "https://he.wikipedia.org/wiki/רשימת_ראשי_עיר_בישראל",
            "title": "רשימת ראשי עיר בישראל — ויקיפדיה",
            "content": "רשימת ראשי עיר ומועצות מקומיות בישראל לפי שנים. בשנת 1990 כיהנו ראשי עיר שונים ברחבי הארץ.",
            "raw_content": "רשימת ראשי עיר ומועצות מקומיות בישראל לפי שנים. בשנת 1990 כיהנו ראשי עיר שונים ברחבי הארץ.",
        }
    ],
}


class MockTavilyClient:
    """Drop-in replacement for TavilyClient that returns canned results."""

    def __init__(self, results: dict | None = None):
        self._results = results or _MOCK_RESULTS

    def search(self, query: str, **kwargs) -> dict:
        # Return domain-specific mock if available, else default
        for key, hits in self._results.items():
            if key != "default" and key in query.lower():
                return {"results": hits}
        return {"results": self._results.get("default", [])}


# ── DuckDuckGo search client (no API key required) ────────────────────────────

class WikipediaSearchClient:
    """
    Tavily-compatible client that searches and fetches full article content
    from Wikipedia (Hebrew-first, English fallback).

    Why Wikipedia instead of DuckDuckGo/Google:
      - No API key, no rate limits, no IP blocks
      - Returns full structured article text — ideal for grounding
      - Hebrew Wikipedia (he.wikipedia.org) has good coverage of Israeli
        municipalities, politicians, and public figures
      - The MediaWiki API is stable and officially supported

    Strategy per query:
      1. Detect language (Hebrew chars → he.wikipedia.org, else en)
      2. Call the Wikipedia search API to find matching article titles
      3. Fetch each article's full plaintext via the extracts API
    """

    _UA = "INResearcher/1.0 (autonomous research agent; github.com/harelfelhai/inreasearcher)"
    _MAX_CONTENT = 6000
    _RATE_DELAY  = 2.0   # seconds between every Wikipedia API call

    def search(self, query: str, max_results: int = 3, **kwargs) -> dict:
        import urllib.request, urllib.parse, json as _json

        he_chars = sum(1 for c in query if 'א' <= c <= 'ת')
        lang = "he" if he_chars > 2 else "en"
        base = f"https://{lang}.wikipedia.org/w/api.php"

        search_params = urllib.parse.urlencode({
            "action": "query", "list": "search",
            "srsearch": query, "srlimit": max_results,
            "format": "json", "utf8": 1,
        })
        import time

        titles = []
        try:
            titles = self._api_get(base, search_params)["query"]["search"]
            titles = [r["title"] for r in titles]
        except Exception as exc:
            print(f"    [wikipedia search error] {exc}")
            return {"results": []}

        if not titles:
            return {"results": []}

        extract_params = urllib.parse.urlencode({
            "action": "query", "prop": "extracts",
            "titles": "|".join(titles),
            "explaintext": 1, "exsectionformat": "plain",
            "format": "json", "utf8": 1,
        })
        results = []
        try:
            pages = self._api_get(base, extract_params)["query"]["pages"]
            for page in pages.values():
                title   = page.get("title", "")
                extract = page.get("extract", "")
                if not extract:
                    continue
                url = (
                    f"https://{lang}.wikipedia.org/wiki/"
                    + urllib.parse.quote(title.replace(" ", "_"))
                )
                content = extract[: self._MAX_CONTENT]
                results.append({
                    "url": url, "title": title,
                    "content": content[:400],
                    "raw_content": content,
                })
        except Exception as exc:
            print(f"    [wikipedia fetch error] {exc}")

        return {"results": results}

    def fetch_entity_article(self, entity: str) -> dict:
        """
        Directly fetch the Wikipedia article for `entity` by title.
        Follows redirects (e.g. 'תל אביב' → 'תל אביב-יפו').
        Returns a Tavily-compatible results dict.
        """
        import urllib.parse

        he_chars = sum(1 for c in entity if 'א' <= c <= 'ת')
        lang = "he" if he_chars > 0 else "en"
        base = f"https://{lang}.wikipedia.org/w/api.php"

        params = urllib.parse.urlencode({
            "action": "query", "prop": "extracts",
            "titles": entity,
            "explaintext": 1, "exsectionformat": "plain",
            "redirects": 1,          # תל אביב → תל אביב-יפו automatically
            "format": "json", "utf8": 1,
        })
        try:
            data = self._api_get(base, params)
            pages = data.get("query", {}).get("pages", {})
            results = []
            for page in pages.values():
                if page.get("missing") is not None:
                    continue
                title   = page.get("title", entity)
                extract = page.get("extract", "")
                if not extract:
                    continue
                url = (
                    f"https://{lang}.wikipedia.org/wiki/"
                    + urllib.parse.quote(title.replace(" ", "_"))
                )
                content = extract[: self._MAX_CONTENT]
                results.append({
                    "url": url, "title": title,
                    "content": content[:400],
                    "raw_content": content,
                })
                print(f"    [direct fetch] '{title}' ({len(extract):,} chars)")
            return {"results": results}
        except Exception as exc:
            print(f"    [direct fetch error] {exc}")
            return {"results": []}

    def _api_get(self, base: str, params: str) -> dict:
        """
        Make one Wikipedia API call with rate-limiting and a single 429 retry.
        Sleeps _RATE_DELAY seconds BEFORE and AFTER the call so successive
        queries within a field don't hammer the endpoint.
        """
        import time, urllib.request, json as _json

        time.sleep(self._RATE_DELAY)
        req = urllib.request.Request(
            f"{base}?{params}",
            headers={"User-Agent": self._UA}
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = _json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            # Retry once after a longer back-off on 429
            if "429" in str(exc):
                print(f"    [wikipedia 429 — waiting 10s before retry]")
                time.sleep(10)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = _json.loads(resp.read().decode("utf-8"))
            else:
                raise
        time.sleep(self._RATE_DELAY)
        return data


class DuckDuckGoClient:
    """
    Tavily-compatible client backed by DuckDuckGo (ddgs package).
    Fetches full page content for better grounding. Falls back to snippet.
    Note: may rate-limit in server environments — prefer WikipediaSearchClient
    for research tasks, use DuckDuckGo only when broader web coverage is needed.
    """

    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
        "Accept-Language": "he,en;q=0.9",
    }
    _FETCH_TIMEOUT = 8
    _MAX_CONTENT = 6000

    def search(self, query: str, max_results: int = 5, **kwargs) -> dict:
        import urllib.request, time

        raw_hits: list[dict] = []
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                raw_hits = list(ddgs.text(query, max_results=max_results))
        except ImportError:
            try:
                from duckduckgo_search import DDGS
                with DDGS() as ddgs:
                    raw_hits = list(ddgs.text(query, max_results=max_results))
            except Exception as exc:
                print(f"    [ddg error] {exc}")
                return {"results": []}
        except Exception as exc:
            print(f"    [ddg search error] {exc}")
            return {"results": []}

        results = []
        for hit in raw_hits:
            url     = hit.get("href", "")
            snippet = hit.get("body", "")
            title   = hit.get("title", "")
            raw_content = self._fetch(url) or snippet
            results.append({"url": url, "title": title,
                             "content": snippet, "raw_content": raw_content})
            time.sleep(0.5)   # gentle rate-limit buffer

        return {"results": results}

    def _fetch(self, url: str) -> str | None:
        import urllib.request
        if not url.startswith("http"):
            return None
        try:
            req = urllib.request.Request(url, headers=self._HEADERS)
            with urllib.request.urlopen(req, timeout=self._FETCH_TIMEOUT) as resp:
                raw = resp.read()
                charset = resp.headers.get_content_charset() or "utf-8"
                try:
                    html = raw.decode(charset)
                except (UnicodeDecodeError, LookupError):
                    try:
                        html = raw.decode("windows-1255")
                    except Exception:
                        html = raw.decode("utf-8", errors="replace")
                return _strip_html(html)[: self._MAX_CONTENT]
        except Exception:
            return None


def _strip_html(html: str) -> str:
    """
    Extract readable text from HTML using stdlib html.parser.
    Strips tags, decodes entities, collapses whitespace.
    """
    from html.parser import HTMLParser

    class _Stripper(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self._buf: list[str] = []
            self._skip = False

        def handle_starttag(self, tag, attrs):
            if tag in {"script", "style", "nav", "footer", "header"}:
                self._skip = True

        def handle_endtag(self, tag):
            if tag in {"script", "style", "nav", "footer", "header"}:
                self._skip = False

        def handle_data(self, data):
            if not self._skip:
                stripped = data.strip()
                if stripped:
                    self._buf.append(stripped)

        def get_text(self) -> str:
            return "\n".join(self._buf)

    parser = _Stripper()
    parser.feed(html)
    text = parser.get_text()
    # Collapse runs of blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()