#!/usr/bin/env python3
"""
Autonomous Research Agent — CLI entry point

Usage (single entity):
    python main.py -q "מי היה ראש עיריית תל אביב בשנת 1990?" -e "תל אביב"

Usage (entity file):
    python main.py -q "..." -f municipalities.txt --output-csv results.csv

Usage (preview plan only, no search):
    python main.py -q "..." -e "תל אביב" --plan-only

The CLI implements the Guided Prompting loop:
  1. Run preflight check on the user's question.
  2. If not executable → print clarifying questions and a prompt template,
     then ask the user to refine (interactive) or exit (non-interactive).
  3. If executable → compile plan, show it, proceed to research.
"""

import os
import sys
import json
import argparse

from dotenv import load_dotenv
import anthropic
from tavily import TavilyClient

from research_agent.compiler import (
    compile_schema,
    audit_schema,
    generate_mock_rows,
    enrich_with_queries,
)
from research_agent.extractor import (
    search_and_extract, probe_field_list,
    MockTavilyClient, DuckDuckGoClient,
    WikipediaSearchClient, GoogleSearchClient, SerpApiClient,
)
from research_agent.verifier import verify_field
from research_agent.output import write_csv, write_json, print_summary
from research_agent.memory import SuccessMemory
from research_agent.models import (
    EntityResult,
    ClarificationRequest,
    ExecutableResearchPlan,
    ResearchPlan,
    FieldAuditReport,
    MockRow,
)

load_dotenv()

# ── Guided prompting helpers ──────────────────────────────────────────────────

_PROMPT_GUIDE = """\
╔══════════════════════════════════════════════════════════════════════╗
║          Autonomous Research Agent — Prompt Guide                   ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  A good question = a clearly-defined TABLE. The agent will:          ║
║    • Treat each entity as a row                                      ║
║    • Treat each requested data point as a column                     ║
║    • Find one verifiable value per cell, with a source URL + quote   ║
║                                                                      ║
║  Every column needs:                                                 ║
║    1. A SINGLE answer (not a list, not a paragraph)                  ║
║    2. A TIMEFRAME if historical (e.g. "in 1990", not "historically") ║
║    3. A CANONICAL source someone could plausibly look at             ║
║    4. An OBJECTIVE answer (not "best", "most important")             ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  ✓ GOOD examples                                                    ║
║                                                                      ║
║  "For each Israeli municipality, find: (a) who served as mayor in    ║
║   1990 (full name), (b) the official municipality website URL."      ║
║   → bounded fields, temporal anchor, clear sources.                  ║
║                                                                      ║
║  "For each member of the 25th Knesset, find: party, faction at       ║
║   election, year first elected."                                     ║
║   → single values, all available from knesset.gov.il.                ║
║                                                                      ║
║  ✗ BAD examples                                                     ║
║                                                                      ║
║  "Tell me about Israeli mayors."                                     ║
║   → no fields, no scope, no timeframe.                               ║
║                                                                      ║
║  "For each city, describe the mayor's career path."                  ║
║   → "career path" is unbounded (every job ever held?).               ║
║                                                                      ║
║  "Who was the best mayor of each city?"                              ║
║   → "best" is subjective — two researchers would disagree.           ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""


def _print_clarification(cr: ClarificationRequest, interactive: bool) -> str | None:
    """
    Display the clarification request and either prompt the user
    interactively or exit with instructions.
    Returns the refined question if interactive, None otherwise.
    """
    print("\n⚠  The research question needs clarification before it can run.\n")
    print(f"   Reason: {cr.reason}\n")

    if cr.questions:
        print("   Please answer the following:\n")
        for i, q in enumerate(cr.questions, 1):
            print(f"   {i}. [{q.field}] {q.question_he}")
            print(f"      {q.question_en}")
            print(f"      Example answer: {q.example}\n")

    if cr.prompt_template:
        print("─" * 66)
        print("   Suggested prompt template (fill in the [PLACEHOLDERS]):\n")
        print("   " + cr.prompt_template.replace("\n", "\n   "))
        print("─" * 66)

    if interactive:
        print("\n   Enter your refined research question below")
        print("   (or press Ctrl+C to exit):\n")
        try:
            refined = input("   > ").strip()
            return refined if refined else None
        except (KeyboardInterrupt, EOFError):
            return None
    else:
        print("\n   Re-run with a more specific --question and try again.")
        return None


def _print_audit(report: FieldAuditReport) -> None:
    print("\n⚠  Field-clarity audit found issues:\n")
    for issue in report.issues:
        print(f"  • [{issue.field_id}]  ({issue.issue_kind})")
        print(f"    {issue.explanation_he}")
        print(f"    {issue.explanation_en}")
        print(f"    Suggested fix: {issue.suggested_fix_en}\n")


def _render_table(headers: list[str], rows: list[list[str]], max_col: int = 32) -> str:
    """Simple ASCII table that handles Hebrew (no padding tricks for RTL)."""
    def trunc(s: str) -> str:
        s = (s or "").replace("\n", " ")
        return s if len(s) <= max_col else s[: max_col - 1] + "…"

    cols = [trunc(h) for h in headers]
    body = [[trunc(c) for c in r] for r in rows]
    widths = [max(len(c) for c in [cols[i]] + [r[i] for r in body])
              for i in range(len(cols))]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    def fmt(cells): return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    out = [sep, fmt(cols), sep]
    out += [fmt(r) for r in body]
    out.append(sep)
    return "\n".join(out)


def _print_mock_preview(plan: ResearchPlan, rows: list[MockRow]) -> None:
    print("\n┌──────────────────────────────────────────────────────────────────┐")
    print("│  Mock preview — fake data showing your table's SHAPE             │")
    print("│  (values are illustrative; nothing has been searched yet)        │")
    print("└──────────────────────────────────────────────────────────────────┘")
    headers = ["entity"] + [c.id for c in plan.columns]
    body = [
        [r.entity_name] + [r.values.get(c.id, "—") for c in plan.columns]
        for r in rows
    ]
    print(_render_table(headers, body))


def _print_real_preview(results: list[EntityResult], plan: ResearchPlan) -> None:
    print("\n┌──────────────────────────────────────────────────────────────────┐")
    print("│  Real preview — actual values for the first entities             │")
    print("└──────────────────────────────────────────────────────────────────┘")
    headers = ["entity"] + [c.id for c in plan.columns] + ["flags"]
    body = []
    for r in results:
        row = [r.entity_name]
        for c in plan.columns:
            cell = r.cells.get(c.id)
            if cell is None or cell.value is None:
                row.append("✗ NOT_FOUND")
            else:
                row.append(f"{cell.value} [{cell.confidence}]")
        row.append(",".join(r.row_flags))
        body.append(row)
    print(_render_table(headers, body, max_col=36))


def _ask(prompt: str, choices: list[str], default: str | None = None) -> str:
    """Prompt the user; returns the chosen single-letter code (lower)."""
    suffix = f"[{'/'.join(choices)}]"
    if default:
        suffix = suffix.replace(default, default.upper())
    while True:
        try:
            ans = input(f"  {prompt} {suffix} > ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return "q"
        if not ans and default:
            return default
        if ans in choices:
            return ans


def _show_plan(plan: ResearchPlan) -> None:
    print(f"\n[plan] Entity type: {plan.entity_type}")
    print(f"[plan] {len(plan.columns)} fields to extract:\n")
    for col in plan.columns:
        dep = f" (depends on: {col.depends_on})" if col.depends_on else ""
        anchor = f" [{col.temporal_anchor}]" if col.temporal_anchor else ""
        print(f"  • {col.label_he} / {col.label_en}{anchor}{dep}")
        print(f"    min_corroborations: {col.min_corroborations}  type: {col.type}")
        print(f"    search_he[0]: {col.search_queries_he[0] if col.search_queries_he else '—'}")
    print()


# ── Per-entity research pipeline ──────────────────────────────────────────────

def research_entity(
    entity: str,
    plan: ResearchPlan,
    tavily: TavilyClient,
    claude: anthropic.Anthropic,
    memory: SuccessMemory | None = None,
    verbose: bool = False,
    probe_results: dict | None = None,   # field_id → {entity → ExtractionResult}
) -> EntityResult:
    print(f"\n[entity] {entity}", file=sys.stderr)

    resolved_deps: dict[str, str] = {}   # {field_id: resolved_value}
    cells = {}
    deferred = []                          # fields whose depends_on isn't resolved yet

    def _process_field(field):
        nonlocal resolved_deps

        if field.depends_on and field.depends_on not in resolved_deps:
            deferred.append(field)
            if verbose:
                print(f"  [defer] {field.id} waiting on {field.depends_on}", file=sys.stderr)
            return

        # Use probe result if available for this (field, entity) pair
        if probe_results and field.id in probe_results:
            probe_hit = probe_results[field.id].get(entity)
            if probe_hit and probe_hit.value:
                cell = verify_field(field, [probe_hit])
                cells[field.id] = cell
                if cell.value:
                    resolved_deps[field.id] = cell.value
                    print(f"  [field] {field.id} → ✓ {cell.value[:60]} [probe/{cell.confidence}]",
                          file=sys.stderr)
                return

        print(f"  [field] {field.id} ({field.label_he})", file=sys.stderr)
        extractions = search_and_extract(
            field=field,
            entity=entity,
            resolved_deps=resolved_deps,
            tavily=tavily,
            claude=claude,
            memory=memory,
        )
        cell = verify_field(field, extractions)
        cells[field.id] = cell

        if cell.value:
            resolved_deps[field.id] = cell.value
            status = f"✓ {cell.value[:60]} [{cell.confidence}]"
        else:
            status = "✗ NOT_FOUND"

        print(f"    → {status}", file=sys.stderr)
        if verbose and cell.flags:
            print(f"    flags: {', '.join(cell.flags)}", file=sys.stderr)

    # First pass
    for field in plan.columns:
        _process_field(field)

    # Second pass for deferred fields.
    # If the dependency field came back NOT_FOUND, fall back to the entity
    # name so dependent fields can still attempt their own searches.
    for field in list(deferred):
        if field.depends_on and field.depends_on not in resolved_deps:
            resolved_deps[field.depends_on] = entity
        _process_field(field)

    row_flags = []
    not_found = sum(1 for c in cells.values() if c.confidence == "NOT_FOUND")
    if cells and not_found > len(cells) // 2:
        row_flags.append("majority_not_found")

    return EntityResult(entity_name=entity, cells=cells, row_flags=row_flags)


# ── Interactive Review (Success Memory feedback loop) ────────────────────────

_REVIEW_HELP = """\
Mark each cell to teach the system:
  [y] correct        → stored as success (used as few-shot for similar fields)
  [n] hallucinated   → stored as failure (used as AVOID warning for same domain+type)
  [s] skip           → no signal recorded
  [q] quit review    → stop reviewing further cells
"""


def review_results(
    results: list[EntityResult],
    plan: ResearchPlan,
    memory: SuccessMemory,
    research_question: str,
) -> None:
    """
    Interactive feedback loop. Walks every cell with a value and lets
    the user validate or flag it. Validated cells are stored in memory
    and become few-shot examples for future runs. Flagged cells become
    AVOID warnings for the same field_type + source_domain.
    """
    print("\n" + "═" * 66)
    print("  REVIEW MODE  —  teach the system from your judgement")
    print("═" * 66)
    print(_REVIEW_HELP)

    plan_dict = plan.model_dump()
    field_by_id = {c.id: c for c in plan.columns}
    plan_validated_for_session = False
    n_success = n_failure = n_skip = 0

    try:
        for result in results:
            for field_id, cell in result.cells.items():
                if cell.value is None:
                    continue   # nothing to validate on NOT_FOUND cells

                field = field_by_id[field_id]
                src = cell.primary_source or {}

                print("─" * 66)
                print(f"  Entity:     {result.entity_name}")
                print(f"  Field:      {field.label_he} / {field.label_en}  [{cell.confidence}]")
                print(f"  Value:      {cell.value}")
                print(f"  Source:     {src.get('url', '')}")
                print(f"  Quote:      {(src.get('quote') or '')[:200]}")
                if cell.flags:
                    print(f"  Flags:      {', '.join(cell.flags)}")

                choice = input("  [y/n/s/q] > ").strip().lower()

                if choice == "q":
                    raise KeyboardInterrupt
                if choice == "y":
                    memory.record_extraction_success(
                        field_id=field.id,
                        field_label=field.label_en,
                        field_type=field.type,
                        entity=result.entity_name,
                        value=cell.value,
                        quote=src.get("quote", "") or "",
                        source_url=src.get("url", "") or "",
                        source_domain=src.get("domain", "") or "",
                    )
                    n_success += 1
                elif choice == "n":
                    reason = input("  Brief reason (e.g. 'wrong period', 'wrong person'): ").strip()
                    memory.record_extraction_failure(
                        field_id=field.id,
                        field_type=field.type,
                        entity=result.entity_name,
                        claimed_value=cell.value,
                        claimed_quote=src.get("quote"),
                        source_url=src.get("url", "") or "",
                        source_domain=src.get("domain", "") or "",
                        reason=reason or "marked_as_hallucination",
                    )
                    n_failure += 1
                else:
                    n_skip += 1

                # Once any cell is validated, also record the plan as a success.
                # The plan is what produced these correct answers.
                if choice == "y" and not plan_validated_for_session:
                    memory.record_compiler_success(
                        research_question=research_question,
                        entity_type=plan.entity_type,
                        plan=plan_dict,
                    )
                    plan_validated_for_session = True
    except (KeyboardInterrupt, EOFError):
        print("\n  Review interrupted.")

    print("\n" + "═" * 66)
    print(f"  Recorded: {n_success} success(es), {n_failure} failure(s), {n_skip} skipped")
    print(f"  Memory now contains: {memory.stats()}")
    print("═" * 66)


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Autonomous Research Agent for Israeli academic research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--question", "-q", help="Research question (Hebrew or English)")
    p.add_argument("--entity-type", "-t", default="",
                   help="Entity type hint (e.g. 'עיריה', 'חבר כנסת')")
    p.add_argument("--entity", "-e", help="Single entity name to research")
    p.add_argument("--entities-file", "-f",
                   help="Path to a UTF-8 file with one entity name per line")
    p.add_argument("--output-csv", default="output.csv")
    p.add_argument("--output-json", default="output.json")
    p.add_argument("--plan-only", action="store_true",
                   help="Compile and print the research plan, then exit")
    p.add_argument("--guide", action="store_true",
                   help="Print prompt guide and exit")
    p.add_argument("--memory-path", default="memory.json",
                   help="Path to the success-memory JSON file")
    p.add_argument("--no-memory", action="store_true",
                   help="Disable memory injection (no few-shot examples)")
    p.add_argument("--review", action="store_true",
                   help="After research completes, interactively review every cell to teach the system")
    p.add_argument("--mock-search", action="store_true",
                   help="Use mock search responses (no API key needed) to test the full pipeline")
    p.add_argument("--search-engine",
                   choices=["tavily", "duckduckgo", "wikipedia", "google", "serpapi"],
                   default="wikipedia",
                   help="Search backend (default: wikipedia — no API key needed)")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.guide:
        print(_PROMPT_GUIDE)
        return

    if not args.question:
        print(_PROMPT_GUIDE)
        parser.error("--question / -q is required")

    if not args.entity and not args.entities_file and not args.plan_only:
        parser.error("Provide --entity or --entities-file (or --plan-only to preview the plan)")

    # Validate API keys
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if not anthropic_key:
        sys.exit("ERROR: ANTHROPIC_API_KEY not set. Copy .env.example → .env and add your key.")

    claude = anthropic.Anthropic(api_key=anthropic_key)

    if args.mock_search:
        search = MockTavilyClient()
        print("[search] Using MOCK search — no API key required.", file=sys.stderr)
    elif args.search_engine == "wikipedia":
        search = WikipediaSearchClient()
        print("[search] Using Wikipedia API (Hebrew-first, no key required).", file=sys.stderr)
    elif args.search_engine == "duckduckgo":
        search = DuckDuckGoClient()
        print("[search] Using DuckDuckGo (no API key required).", file=sys.stderr)
    elif args.search_engine == "serpapi":
        serpapi_key = os.getenv("SERPAPI_KEY", "")
        if not serpapi_key:
            sys.exit(
                "ERROR: SERPAPI_KEY not set in .env\n"
                "  1. Sign up free (no card) at serpapi.com\n"
                "  2. Copy your API key from the dashboard\n"
                "  3. Add to .env:  SERPAPI_KEY=your_key_here"
            )
        search = SerpApiClient(api_key=serpapi_key)
        print("[search] Using SerpAPI (Google Search).", file=sys.stderr)
    elif args.search_engine == "google":
        google_key = os.getenv("GOOGLE_API_KEY", "")
        google_cse  = os.getenv("GOOGLE_CSE_ID", "")
        if not google_key or not google_cse:
            sys.exit(
                "ERROR: GOOGLE_API_KEY and GOOGLE_CSE_ID must be set in .env\n"
                "  1. console.cloud.google.com → Enable 'Custom Search API' → Create API key\n"
                "  2. programmablesearchengine.google.com → New engine → copy cx value\n"
                "  3. Add both to .env:  GOOGLE_API_KEY=...  GOOGLE_CSE_ID=..."
            )
        search = GoogleSearchClient(api_key=google_key, cse_id=google_cse)
        print("[search] Using Google Custom Search API.", file=sys.stderr)
    else:
        if not tavily_key and not args.plan_only:
            sys.exit("ERROR: TAVILY_API_KEY not set. Use --search-engine wikipedia or add your Tavily key.")
        search = TavilyClient(api_key=tavily_key)
        print("[search] Using Tavily.", file=sys.stderr)
    tavily = search   # rest of code uses `tavily` variable name

    # ── Memory ────────────────────────────────────────────────────────────────
    memory = None if args.no_memory else SuccessMemory(args.memory_path)
    if memory:
        stats = memory.stats()
        print(f"[memory] Loaded {args.memory_path}: {stats}", file=sys.stderr)

    # ── Guided Prompting Loop ─────────────────────────────────────────────────
    question = args.question
    entity_type = args.entity_type
    interactive = sys.stdin.isatty()
    max_rounds = 3

    if interactive:
        print(_PROMPT_GUIDE)

    plan: ResearchPlan | None = None

    for attempt in range(1, max_rounds + 1):
        print(f"\n[compiler] Building schema (attempt {attempt})…", file=sys.stderr)

        # Phase A + B1: preflight & bare schema
        schema_result = compile_schema(question, entity_type, claude, memory=memory)

        if isinstance(schema_result, ClarificationRequest):
            refined = _print_clarification(schema_result, interactive)
            if refined:
                question = refined
                continue
            sys.exit(1)

        schema_plan = schema_result.plan
        _show_plan(schema_plan)

        # Phase C: field-clarity audit
        print("\n[compiler] Auditing field clarity…", file=sys.stderr)
        audit = audit_schema(schema_plan, claude)

        if not audit.all_clear:
            _print_audit(audit)
            if interactive:
                choice = _ask(
                    "Refine the question, or proceed anyway?",
                    ["r", "p", "q"],
                    default="r",
                )
                if choice == "q":
                    sys.exit(0)
                if choice == "r":
                    try:
                        refined = input("  Enter refined question:\n  > ").strip()
                    except (KeyboardInterrupt, EOFError):
                        sys.exit(1)
                    if refined:
                        question = refined
                        continue
                # 'p' falls through to mock preview, marking issues with LOW later
                print("  → Proceeding with flagged fields (will be marked LOW).")
        else:
            print("  ✓ All fields look clear.", file=sys.stderr)

        # Phase D: mock-data preview
        print("\n[compiler] Generating mock-data preview…", file=sys.stderr)
        try:
            mock_rows = generate_mock_rows(schema_plan, claude)
        except Exception as exc:
            print(f"  (mock preview failed: {exc} — skipping)", file=sys.stderr)
            mock_rows = []

        if mock_rows:
            _print_mock_preview(schema_plan, mock_rows)

        if not interactive:
            plan = schema_plan
            break

        choice = _ask(
            "Does this table match what you want?",
            ["y", "r", "q"],
            default="y",
        )
        if choice == "q":
            sys.exit(0)
        if choice == "r":
            try:
                refined = input("  Enter refined question:\n  > ").strip()
            except (KeyboardInterrupt, EOFError):
                sys.exit(1)
            if refined:
                question = refined
                continue

        plan = schema_plan
        break
    else:
        sys.exit("Could not produce an approved schema after clarification attempts.")

    # Phase B2: enrich with search queries (only NOW, after schema is approved)
    print("\n[compiler] Generating search queries for approved schema…", file=sys.stderr)
    plan = enrich_with_queries(plan, claude)
    _show_plan(plan)

    if args.plan_only:
        print(json.dumps(plan.model_dump(), ensure_ascii=False, indent=2))
        return

    # ── Load entities ─────────────────────────────────────────────────────────
    entities: list[str] = []
    if args.entity:
        entities = [args.entity]
    elif args.entities_file:
        with open(args.entities_file, encoding="utf-8") as fh:
            entities = [ln.strip() for ln in fh if ln.strip()]

    print(f"\n[start] {len(entities)} entities × {len(plan.columns)} fields", file=sys.stderr)

    def _run_probes(current_plan: ResearchPlan) -> dict:
        """Run field-list probes for all fields, return probe_results dict."""
        probes: dict = {}
        probe_fields = [f for f in current_plan.columns if f.directory_probe_query_he]
        if probe_fields:
            print(f"\n[probe] Checking {len(probe_fields)} field(s) for directory pages…",
                  file=sys.stderr)
            for field in probe_fields:
                result = probe_field_list(field, entities, tavily, claude)
                if result:
                    found_n = sum(1 for r in result.values() if r.value)
                    print(f"  [probe] field={field.id!r}: directory page covers "
                          f"{found_n}/{len(entities)} entities", file=sys.stderr)
                    probes[field.id] = result
        return probes

    # ── Research, with a real-data checkpoint after the first 2 entities ─────
    results: list[EntityResult] = []
    preview_n = min(2, len(entities))
    probe_data = _run_probes(plan)

    for entity in entities[:preview_n]:
        results.append(research_entity(entity, plan, tavily, claude, memory, args.verbose,
                                       probe_results=probe_data))

    if interactive and len(entities) > preview_n:
        _print_real_preview(results, plan)
        print("\n  How does this look?")
        print("    [y] continue with the remaining entities")
        print("    [r] stop and re-plan search queries (keeps schema)")
        print("    [q] stop here, save what we have")
        choice = _ask("", ["y", "r", "q"], default="y")
        if choice == "q":
            pass  # fall through to output
        elif choice == "r":
            print("\n[compiler] Re-generating search queries…", file=sys.stderr)
            plan = enrich_with_queries(plan, claude)
            _show_plan(plan)
            probe_data = _run_probes(plan)
            results = []
            for entity in entities[:preview_n]:
                results.append(research_entity(entity, plan, tavily, claude, memory, args.verbose,
                                               probe_results=probe_data))
            for entity in entities[preview_n:]:
                results.append(research_entity(entity, plan, tavily, claude, memory, args.verbose,
                                               probe_results=probe_data))
        else:  # 'y'
            for entity in entities[preview_n:]:
                results.append(research_entity(entity, plan, tavily, claude, memory, args.verbose,
                                               probe_results=probe_data))
    else:
        for entity in entities[preview_n:]:
            results.append(research_entity(entity, plan, tavily, claude, memory, args.verbose,
                                           probe_results=probe_data))

    # ── Output ────────────────────────────────────────────────────────────────
    write_csv(results, args.output_csv, plan.columns)
    write_json(results, args.output_json)
    print_summary(results)

    # ── Optional interactive review (Success Memory feedback loop) ───────────
    if args.review and memory is not None:
        if not interactive:
            print("[review] --review requires an interactive TTY; skipped.", file=sys.stderr)
        else:
            review_results(results, plan, memory, question)


if __name__ == "__main__":
    main()
