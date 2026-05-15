# Run Verification Checklist

Use this after the first serious run to systematically check what worked
and what didn't. Each item has WHAT, HOW, and WHAT-TO-LOOK-FOR.

Trace file lives at `logs/run_{session_id}.jsonl`. The CLI reader is
`python scripts/read_trace.py <file>` (see `--help` for filters).

---

## 1. Pipeline correctness — did each step run?

### 1.1 Did the compiler classify field volatility correctly?
- **How:** Open the plan in the UI's compile-schema step, or check
  `plan_json` in the session row. For each column, look at `volatility`.
- **Look for:**
  - Historical / time-anchored fields (with `temporal_anchor`) → `stable`
  - Current officeholders, contact info, prices, hours → `volatile`
  - URLs → `volatile` (sites change)
- **Failure signal:** Everything is `stable` (compiler ignored the flag),
  or current data marked stable (compiler doesn't distinguish).

### 1.2 Were search queries reasonable?
- **How:** `python scripts/read_trace.py logs/run_X.jsonl --event field_search`
- **Look for:** Queries that a human researcher would write. Hebrew queries
  for Israeli entities. Year included when temporal anchor exists.
- **Failure signal:** Generic queries ("Tel Aviv information"), missing
  temporal anchor, English queries for Hebrew-only sources.

### 1.3 Did search return content?
- **How:** `python scripts/read_trace.py logs/run_X.jsonl --event search_hit`
- **Look for:** `content_length` > 0 for most accepted hits. `accepted: true`
  for the top results.
- **Failure signal:** Many hits with `content_length: 0` and `accepted: false`.
  Indicates JS-rendered sites or extraction failure.
  - If `pdf_fallback: true` appears with content_length > 0, the PDF
    recovery worked.

### 1.4 Did publication dates get extracted?
- **How:** Same `search_hit` events — check the `publication_date` field.
- **Look for:** Date strings on news sites, Wikipedia, dated articles.
- **Failure signal:** All dates are `null`. Means our 3 extraction strategies
  (Tavily metadata / URL pattern / in-text regex) all failed.

### 1.5 Did the extractor produce grounded values?
- **How:** `python scripts/read_trace.py logs/run_X.jsonl --event extraction -v`
- **Look for:** `is_grounded: true` for results that have values. Quote
  snippets should look like real text from the source.
- **Failure signal:** Many `is_grounded: false` with non-null `raw_value` —
  means the LLM is hallucinating values that aren't in the source.

### 1.6 Did the verifier make sensible decisions?
- **How:** `python scripts/read_trace.py logs/run_X.jsonl --event verification`
- **Look for:** `HIGH` confidence when ≥ 2 grounded sources from different
  domains agree. `MEDIUM` for single grounded source. `NOT_FOUND` only
  when zero grounded sources.
- **Failure signal:** `HIGH` with `corroboration_count: 1` and no preferred
  domain (broken consensus logic), or `NOT_FOUND` despite `extraction`
  events showing grounded results (verifier rejected good extractions).

---

## 2. New feature checks (built in this session)

### 2.1 Memory feedback loop
- **How:** After the run completes, in the Results screen, open the
  "סקירת תוצאות לשיפור עתידי" panel. Mark some cells ✓ and some ✗.
  Submit. Then check `memory.json`.
- **Look for:** `extraction_successes` list grew. `extraction_failures` list
  grew. If any ✓: `compiler_successes` list grew by 1.
- **Failure signal:** memory.json unchanged after submit, or 500 error
  in the network tab.

### 2.2 Memory feedback influences future runs
- **How:** Run the same question (or similar) AGAIN after submitting feedback.
  Check the second run's trace for `field_search` events.
- **Look for:** Better-targeted queries on fields that had ✓ feedback.
  Compare search query quality between run 1 and run 2.
- **Failure signal:** Identical queries to run 1 — memory examples aren't
  reaching the compiler/extractor prompts.

### 2.3 Admin memory seeding
- **How:** Log in as admin → admin dashboard → "זיכרון המערכת" → seed a
  test example. Check memory.json.
- **Look for:** New entry appears in the right list (success or failure).
- **Failure signal:** Form submits but memory.json unchanged.

### 2.4 PDF fallback extraction
- **How:** If any search hit was a `.pdf` URL, check the trace for
  `pdf_fallback: true` and stderr for `[pdf] extracted N chars`.
- **Look for:** PDFs that yielded usable content >= 80 chars got accepted.
- **Failure signal:** All PDF hits show `accepted: false` despite the
  fallback firing (indicates pdfplumber couldn't read them — likely scans).

### 2.5 Source domain visible in UI
- **How:** Look at the results table.
- **Look for:** Each cell with a value shows the domain (e.g.
  `[HIGH] · he.wikipedia.org · 2024-03`) — NOT the generic word "מקור".
- **Failure signal:** Generic "מקור" link or no link at all.

### 2.6 Stale source warnings
- **How:** Look at any volatile field's cells in the results table.
- **Look for:** If the source date is older than 2 years, an amber
  `⚠ ישן` indicator appears next to the value.
- **Failure signal:** Old dates shown but no warning (volatility flag
  not being read), or warnings on stable fields (volatility wrongly assigned).

---

## 3. Result quality — is the output actually useful?

### 3.1 Spot-check the top-confidence results
- **How:** Pick 5 cells marked `HIGH` confidence. Click the source link.
- **Look for:** The quote shown in the cell appears verbatim on the
  linked page, AND it actually supports the extracted value.
- **Failure signal:** Quote not on the page (broken grounding), or quote
  present but doesn't say what the value claims (extractor mis-reading).

### 3.2 Spot-check the LOW-confidence results
- **How:** Pick 3 cells marked `LOW`. Read the flags.
- **Look for:** Flags explain why confidence is low (`below_min_corroborations`,
  `preferred_source_boost`, `primary_source_stale`).
- **Failure signal:** `LOW` with no flags — verifier downgraded for unclear
  reasons.

### 3.3 Investigate every NOT_FOUND
- **How:** `python scripts/read_trace.py logs/run_X.jsonl --not-found`
  then for each one trace back: were queries sent? Did search return pages?
  Did extraction run? What did it say?
- **Categorise each NOT_FOUND into one of:**
  - (a) Search returned nothing relevant → query problem
  - (b) Search returned pages but content was empty → JS-rendered or PDF-fail
  - (c) Pages had content but extractor returned null → genuinely not in text
  - (d) Pages had content AND text → extractor failed (check `not_found_reason`)
- **This is the most important diagnostic.** The category distribution tells
  you which layer needs work next.

---

## 4. Operational checks

### 4.1 Cost
- **How:** Final `done` SSE event has `cost_used`.
- **Look for:** Cost roughly proportional to entities × fields. Order of
  magnitude: $0.02–0.05 per field per entity.
- **Failure signal:** Massively higher than expected — could indicate
  retries, oversized prompts, or duplicate extractions per page.

### 4.2 Speed
- **How:** Trace `run_start` and `run_done` timestamps.
- **Look for:** Roughly entity_count × 30–60s for typical fields.
  Concurrent entity processing should help — check that `entity_done`
  events are NOT strictly sequential.
- **Failure signal:** Linear timing implies parallelism broke.

### 4.3 No mid-run errors
- **How:** Check stderr logs and trace for any error-shaped events.
- **Look for:** No `[error]`, no traceback printouts, no `[search error]`
  spam.

---

## 5. What to write down for next sprint

After running through this list, for each failure mode found, write:
- Which step failed (compiler / search / extraction / verifier / UI)
- For how many cells
- One concrete example (entity + field + what went wrong)

This becomes the input for the next iteration. Without this list, you'll
re-debug the same issues next run.
