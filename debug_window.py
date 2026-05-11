"""
Diagnostic: windowing + Wikidata structured data for Tel Aviv.

Run:  py debug_window.py
"""
import sys
sys.path.insert(0, ".")

from research_agent.extractor import _select_relevant_text, WikipediaSearchClient, _inject_wikidata
from research_agent.models import ColumnPlan, ExtractionResult


def check(label, text, term):
    if term in text:
        pos = text.find(term)
        snippet = text[max(0, pos - 120): pos + 250]
        print(f"  ✓ '{term}' נמצא (מיקום {pos:,})")
        print(f"    {snippet!r}")
    else:
        print(f"  ✗ '{term}' לא נמצא")


def main():
    print("=== שלב 1: הורדת מאמר תל אביב ===")
    client = WikipediaSearchClient()
    result = client.fetch_entity_article("תל אביב")
    hits = result.get("results", [])
    if not hits:
        print("ERROR: לא הצלחנו להוריד את המאמר")
        sys.exit(1)

    content = hits[0]["raw_content"]
    print(f"אורך המאמר: {len(content):,} תווים\n")

    print("=== שלב 2: חיפוש במאמר המלא ===")
    for term in ["להט", "שלמה להט", "1990", "ראש העיר", "tel-aviv.gov.il", "www."]:
        check("מאמר מלא", content, term)

    print("\n=== שלב 3: חלון mayor_1990 (מאמר ראשי) ===")
    field = ColumnPlan(
        id="mayor_1990", label_he="ראש העיר בשנת 1990", label_en="Mayor in 1990",
        type="person_name", temporal_anchor="1990",
        search_queries_he=[], search_queries_en=[],
        preferred_source_domains=[], min_corroborations=2,
    )
    windowed = _select_relevant_text(content, field, "תל אביב")
    print(f"אורך החלון: {len(windowed):,} תווים\n")

    for term in ["להט", "שלמה להט", "1990", "ראש העיר"]:
        check("חלון", windowed, term)

    print("\n=== שלב 3b: מאמר ראשי עיר תל אביב-יפו (ייעודי) ===")
    result2 = client.fetch_entity_article("תל אביב", extra_titles=["ראשי עיר תל אביב-יפו"])
    mayors_article = next((h["raw_content"] for h in result2.get("results", [])
                           if "ראשי עיר" in h.get("title", "")), None)
    if mayors_article:
        print(f"אורך מאמר ראשי עיר: {len(mayors_article):,} תווים")
        for term in ["להט", "שלמה להט", "1990", "1974", "1993"]:
            check("מאמר ראשי עיר", mayors_article, term)
    else:
        print("  מאמר 'ראשי עיר תל אביב-יפו' לא נמצא בוויקיפדיה")

    print("\n=== שלב 4: חלון official_website ===")
    field_url = ColumnPlan(
        id="official_website", label_he="אתר רשמי", label_en="Official Website URL",
        type="url", temporal_anchor=None,
        search_queries_he=[], search_queries_en=[],
        preferred_source_domains=[], min_corroborations=1,
    )
    windowed_url = _select_relevant_text(content, field_url, "תל אביב")
    print(f"אורך החלון: {len(windowed_url):,} תווים\n")

    for term in ["tel-aviv.gov.il", "www.", "http", "אתר"]:
        check("חלון URL", windowed_url, term)

    print("\n=== שלב 5: Wikidata — P6 (ראש עיר) ===")
    wd_results_mayor: list[ExtractionResult] = []
    field_mayor = ColumnPlan(
        id="mayor_1990", label_he="ראש העיר בשנת 1990", label_en="Mayor in 1990",
        type="person_name", temporal_anchor="1990",
        search_queries_he=[], search_queries_en=[],
        preferred_source_domains=[], min_corroborations=2,
    )
    _inject_wikidata("תל אביב", field_mayor, wd_results_mayor, set())
    if wd_results_mayor:
        for r in wd_results_mayor:
            print(f"  ✓ value={r.value!r}  quote={r.quote_original!r}")
    else:
        print("  ✗ Wikidata P6 returned no results for year 1990")

    print("\n=== שלב 6: Wikidata — P856 (אתר רשמי) ===")
    wd_results_url: list[ExtractionResult] = []
    field_url = ColumnPlan(
        id="official_website", label_he="אתר רשמי", label_en="Official Website URL",
        type="url", temporal_anchor=None,
        search_queries_he=[], search_queries_en=[],
        preferred_source_domains=[], min_corroborations=1,
    )
    _inject_wikidata("תל אביב", field_url, wd_results_url, set())
    if wd_results_url:
        for r in wd_results_url:
            print(f"  ✓ value={r.value!r}")
    else:
        print("  ✗ Wikidata P856 returned no results")


def debug_wikidata_raw():
    """Print raw Wikidata API response to diagnose lookup failures."""
    import urllib.request, urllib.parse, json as _json, time

    UA = WikipediaSearchClient._UA

    for title in ["תל אביב", "תל אביב-יפו"]:
        params = urllib.parse.urlencode({
            "action": "wbgetentities", "sites": "hewiki", "titles": title,
            "props": "claims", "format": "json", "utf8": 1,
        })
        time.sleep(1)
        req = urllib.request.Request(
            f"https://www.wikidata.org/w/api.php?{params}",
            headers={"User-Agent": UA},
        )
        print(f"\n=== Wikidata raw: '{title}' ===")
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = _json.loads(resp.read().decode("utf-8"))
            for qid, ent in data.get("entities", {}).items():
                print(f"  QID: {qid}  missing={ent.get('missing', False)}")
                claims = ent.get("claims", {})
                print(f"  P6  claims: {len(claims.get('P6', []))}")
                print(f"  P856 claims: {len(claims.get('P856', []))}")
                # Show first P6 claim raw
                for i, c in enumerate(claims.get("P6", [])[:2]):
                    print(f"  P6[{i}] mainsnak: {c.get('mainsnak',{}).get('datavalue')}")
                    print(f"  P6[{i}] qualifiers: {list(c.get('qualifiers',{}).keys())}")
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
    debug_wikidata_raw()
