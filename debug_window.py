"""
Diagnostic: does _select_relevant_text find the mayor section in the real
Tel Aviv Wikipedia article?

Run:  py debug_window.py
"""
import sys
sys.path.insert(0, ".")

from research_agent.extractor import _select_relevant_text, WikipediaSearchClient
from research_agent.models import ColumnPlan


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

    print("\n=== שלב 3: חלון mayor_1990 ===")
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

    print("\n=== סיכום ===")
    print(f"להט במאמר המלא:  {'כן' if 'להט' in content else 'לא'}")
    print(f"להט בחלון mayor: {'כן' if 'להט' in windowed else 'לא'}")
    print(f"gov.il בחלון URL: {'כן' if 'gov.il' in windowed_url else 'לא'}")

    if "להט" in content and "להט" not in windowed:
        print("\n⚠  WINDOWING BUG: המאמר מכיל את המידע אבל החלון מפספס אותו")
        # Show where in the article להט appears vs what the window covers
        pos = content.find("להט")
        print(f"   'להט' נמצא בתו {pos:,} מתוך {len(content):,}")
        print("   החלון כנראה לא מכסה את האזור הזה")
    elif "להט" in windowed:
        print("\n✓ WINDOWING OK: החלון מכיל את המידע — הבעיה היא בחילוץ של Claude")


if __name__ == "__main__":
    main()
