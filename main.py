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

from research_agent.compiler import compile_research_plan
from research_agent.extractor import (
    search_and_extract, MockTavilyClient, DuckDuckGoClient,
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
)

load_dotenv()

# ── Guided prompting helpers ──────────────────────────────────────────────────

_PROMPT_GUIDE = """\
╔══════════════════════════════════════════════════════════════════╗
║          Autonomous Research Agent — Prompt Guide               ║
╠══════════════════════════════════════════════════════════════════╣
║  A good research question must specify:                         ║
║                                                                  ║
║  1. ENTITY TYPE   What kind of thing? (municipality, politician) ║
║  2. FIELDS        What specific data points do you need?         ║
║  3. TIMEFRAME     Historical or current? Which year?             ║
║  4. SCOPE         Israel only? Global? Hebrew sources?           ║
║                                                                  ║
║  Example (weak):   "Tell me about Israeli mayors"               ║
║  Example (strong): "For each Israeli municipality, find who     ║
║                    served as mayor in 1990, their IDF unit,     ║
║                    and a link to the official municipal record." ║
╚══════════════════════════════════════════════════════════════════╝
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
    max_clarification_rounds = 3

    plan: ResearchPlan | None = None

    for attempt in range(1, max_clarification_rounds + 1):
        print(f"\n[compiler] Evaluating research question (attempt {attempt})…", file=sys.stderr)

        result = compile_research_plan(question, entity_type, claude, memory=memory)

        if isinstance(result, ClarificationRequest):
            refined = _print_clarification(result, interactive)
            if refined:
                question = refined
                continue
            else:
                sys.exit(1)
        else:
            plan = result.plan
            break
    else:
        sys.exit("Could not produce an executable plan after clarification attempts.")

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

    # ── Research ──────────────────────────────────────────────────────────────
    results: list[EntityResult] = []
    for entity in entities:
        row = research_entity(entity, plan, tavily, claude, memory, args.verbose)
        results.append(row)

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
