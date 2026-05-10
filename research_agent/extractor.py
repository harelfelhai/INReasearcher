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
        f"\nTEMPORAL CONSTRAINT: Extract information specifically about "
        f"the year/period '{field.temporal_anchor}'. Ignore data from other periods."
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
        f"Source text:\n---\n{source_content[:4000]}\n---\n\n"
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

    # Wikipedia returns 3 articles per call (each a full page), so 3 queries
    # is enough for solid coverage.  Tavily/DDG return snippets so allow more.
    is_wikipedia = isinstance(tavily, WikipediaSearchClient)
    query_cap = 3 if is_wikipedia else 6

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

        for hit in response.get("results", []):
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