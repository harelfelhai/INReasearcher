# Nature Reserves — End-to-End Test

Goal: stress-test the full pipeline on a non-person, non-municipality domain
with 6 entities × 5 fields, including obscure entities. Also exercise the
guided-prompting (clarification) loop by starting with a deliberately vague
question.

## Step 1 — Vague query (expected: preflight fails, clarifying questions)

Run this first to see the clarification mechanism kick in:

```cmd
chcp 65001
python main.py ^
  -q "אני רוצה ללמוד על שמורות טבע וגנים לאומיים בישראל" ^
  -f test_runs\nature_reserves\entities.txt ^
  --plan-only
```

Expected: the compiler should refuse to produce a plan because the question
has no fields, no temporal anchor, and no table-shape intent. It should
print clarifying questions and a suggested prompt template.

## Step 2 — Refined query, plan only (sanity-check schema before running)

```cmd
python main.py ^
  -q "עבור כל שמורת טבע או גן לאומי בישראל, מצא: (א) שנת ההכרזה הרשמית, (ב) השטח בדונמים, (ג) המחוז הגיאוגרפי (צפון/חיפה/מרכז/ירושלים/דרום), (ד) הנחל או מקור המים המרכזי באתר, (ה) שם היישוב הקרוב ביותר." ^
  -f test_runs\nature_reserves\entities.txt ^
  --plan-only
```

Look for:
- 5 columns generated with sensible `preferred_source_domains` (parks.org.il, gov.il, wikipedia)
- `entity_type` inferred as something like "שמורת טבע" / "גן לאומי"
- Audit report shows no critical issues

## Step 3 — Full run with timing

```cmd
python -c "import time; t=time.time(); import subprocess; subprocess.run(['python','main.py','-q','עבור כל שמורת טבע או גן לאומי בישראל, מצא: (א) שנת ההכרזה הרשמית, (ב) השטח בדונמים, (ג) המחוז הגיאוגרפי (צפון/חיפה/מרכז/ירושלים/דרום), (ד) הנחל או מקור המים המרכזי באתר, (ה) שם היישוב הקרוב ביותר.','-f','test_runs\\nature_reserves\\entities.txt','--output-json','test_runs\\nature_reserves\\out.json','--output-csv','test_runs\\nature_reserves\\out.csv','--verbose'], check=True); print(f'\\nTOTAL WALL TIME: {time.time()-t:.1f}s')"
```

Or simpler with PowerShell:

```powershell
Measure-Command {
  python main.py `
    -q "עבור כל שמורת טבע או גן לאומי בישראל, מצא: (א) שנת ההכרזה הרשמית, (ב) השטח בדונמים, (ג) המחוז הגיאוגרפי (צפון/חיפה/מרכז/ירושלים/דרום), (ד) הנחל או מקור המים המרכזי באתר, (ה) שם היישוב הקרוב ביותר." `
    -f test_runs\nature_reserves\entities.txt `
    --output-json test_runs\nature_reserves\out.json `
    --output-csv  test_runs\nature_reserves\out.csv `
    --verbose
} | Select-Object TotalSeconds
```

## What to look at afterward

1. **Coverage** — out of 30 cells (6×5), how many are HIGH / MEDIUM / LOW / NOT_FOUND?
2. **Long-tail accuracy** — does נחל קזיב get answers? If everything is HIGH except for that entity, the system is doing the right thing on a genuinely hard case.
3. **Sources** — are the LOW cells low because the data really isn't available, or because the search engine missed an obvious page?
4. **Timing breakdown** — `--verbose` should show per-field search & extract times. Tally:
   - Schema compile + audit
   - Per-entity total (look at the slowest one)
   - Probe-verify overhead

## Ground-truth answers (for your own scoring)

Approximate, from public sources — use to spot-check, not as authoritative:

| Entity | שנה | שטח (דונם) | מחוז | מים | יישוב סמוך |
|---|---|---|---|---|---|
| עין גדי | 1971/1972 | ~14,000 | דרום | נחל דוד / נחל ערוגות | קיבוץ עין גדי |
| מצדה | 1966 (גן לאומי) | ~8,400 | דרום | — (אין נחל קבוע) | ערד / עין גדי |
| גן השלושה | 1971 | ~800 | צפון | נחל אסי / נחל חרוד | בית שאן |
| תל דן | 1974 | ~480 | צפון | נחל דן | קיבוץ דן |
| חורשת טל | 1964 | ~750 | צפון | נחל דן / נחל שניר | קיבוץ חורשת טל / כפר בלום |
| נחל קזיב | 1965 | ~10,000 | צפון/חיפה | נחל כזיב | מעלות-תרשיחא / שתולה |
