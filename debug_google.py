"""
Diagnostic: test Google Custom Search API directly.
Run: py debug_google.py
"""
import os, urllib.request, urllib.parse, json
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY", "")
cse_id  = os.getenv("GOOGLE_CSE_ID", "")

print(f"API key: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else '(too short)'}")
print(f"CSE ID:  {cse_id!r}")
print()

# Test 1: simple ASCII query (eliminates Hebrew encoding as a variable)
for query in ["tel aviv mayor", "תל אביב ראש עיר"]:
    params = urllib.parse.urlencode({
        "key": api_key,
        "cx":  cse_id,
        "q":   query,
        "num": 3,
    })
    url = f"https://www.googleapis.com/customsearch/v1?{params}"
    print(f"Testing query: {query!r}")
    print(f"URL (key masked): {url.replace(api_key, 'KEY_HIDDEN')}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "test/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            items = data.get("items", [])
            print(f"  SUCCESS — {len(items)} results")
            for item in items[:2]:
                print(f"    • {item.get('title','?')[:60]}")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"  HTTP {e.code} error:")
        try:
            err = json.loads(body)
            print(f"  {json.dumps(err.get('error', err), indent=2, ensure_ascii=False)}")
        except Exception:
            print(f"  Raw: {body[:500]}")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()
