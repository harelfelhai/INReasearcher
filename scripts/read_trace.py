#!/usr/bin/env python3
"""
Read and pretty-print a run trace file.

Usage:
  python scripts/read_trace.py logs/run_<session_id>.jsonl
  python scripts/read_trace.py logs/run_<session_id>.jsonl --entity "Tel Aviv"
  python scripts/read_trace.py logs/run_<session_id>.jsonl --field mayor_1990
  python scripts/read_trace.py logs/run_<session_id>.jsonl --event verification
  python scripts/read_trace.py logs/run_<session_id>.jsonl --not-found
"""

import argparse
import json
import sys
from pathlib import Path


RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
DIM    = "\033[2m"


def _conf_color(conf: str) -> str:
    return {
        "HIGH": GREEN, "MEDIUM": YELLOW, "LOW": RED, "NOT_FOUND": RED,
    }.get(conf, RESET)


def render(rec: dict, verbose: bool = False) -> str:
    ev = rec.get("event", "?")
    ts = rec.get("ts", "")[:19]

    if ev == "run_start":
        return (
            f"{BOLD}{CYAN}▶ RUN START{RESET}  {ts}\n"
            f"  question : {rec.get('question', '')}\n"
            f"  entities : {rec.get('entity_count')} ({', '.join((rec.get('entities') or [])[:5])}…)\n"
            f"  fields   : {', '.join(c['id'] for c in (rec.get('columns') or []))}\n"
            f"  engine   : {rec.get('search_engine')}"
        )

    if ev == "field_search":
        qs = rec.get("queries", [])
        return (
            f"{CYAN}  🔍 search{RESET}  entity={rec.get('entity')!r}  "
            f"field={rec.get('field_id')!r}\n"
            + "\n".join(f"       query: {q}" for q in qs)
        )

    if ev == "search_hit":
        accepted = rec.get("accepted", False)
        icon = "✓" if accepted else "✗"
        col = GREEN if accepted else DIM
        return (
            f"{col}    {icon} {rec.get('url', '')[:80]}  "
            f"({rec.get('content_length', 0):,} chars){RESET}"
        )

    if ev == "extraction":
        grounded = rec.get("is_grounded", False)
        icon = "✓" if grounded else "✗"
        col = GREEN if grounded else RED
        val = rec.get("raw_value") or "(null)"
        nfr = rec.get("not_found_reason")
        line = (
            f"{col}    {icon} extract  field={rec.get('field_id')!r}  "
            f"url={rec.get('url', '')[:55]}\n"
            f"        value={val!r}  conf={rec.get('llm_confidence', 0):.2f}"
        )
        if verbose and rec.get("quote_snippet"):
            line += f"\n        quote: {rec['quote_snippet']!r}"
        if nfr:
            line += f"\n        {DIM}reason: {nfr}{RESET}"
        return line + RESET

    if ev == "batch_extraction":
        lines = [
            f"    ⚡ batch  url={rec.get('url', '')[:55]}  "
            f"entity={rec.get('entity')!r}"
        ]
        for r in rec.get("field_results", []):
            g = r.get("is_grounded", False)
            lines.append(
                f"      {'✓' if g else '✗'} {r['field_id']}: "
                f"{r.get('value') or '(null)'}"
            )
        return "\n".join(lines)

    if ev == "verification":
        conf = rec.get("confidence", "?")
        col = _conf_color(conf)
        val = rec.get("winner_value") or "(not found)"
        flags = rec.get("flags") or []
        lane = rec.get("lane", "")
        lane_tag = f"[{lane}] " if lane else ""
        line = (
            f"\n  {col}{BOLD}▸ {lane_tag}{rec.get('field_id')}{RESET}"
            f"  entity={rec.get('entity')!r}\n"
            f"    value      : {val}\n"
            f"    confidence : {col}{conf}{RESET}"
            f"  corroborations={rec.get('corroboration_count', 0)}\n"
            f"    candidates : {rec.get('candidates', 0)} total  "
            f"{rec.get('grounded', 0)} grounded"
        )
        if flags:
            line += f"\n    flags      : {', '.join(flags)}"
        return line

    if ev == "entity_done":
        nf = rec.get("not_found_fields") or []
        return (
            f"\n{BOLD}  ✓ entity_done  {rec.get('entity')!r}{RESET}  "
            f"elapsed={rec.get('elapsed_sec')}s  "
            f"found={rec.get('found')}  "
            + (f"missing={nf}" if nf else "")
        )

    if ev == "run_done":
        return (
            f"\n{BOLD}{GREEN}▶ RUN DONE{RESET}  "
            f"entities={rec.get('entity_count')}  "
            f"cost=${rec.get('cost_usd', 0):.4f}  "
            f"tokens_in={rec.get('tokens_in', 0):,}  "
            f"tokens_out={rec.get('tokens_out', 0):,}\n"
            f"  trace → {rec.get('trace_path')}"
        )

    # Fallback — show raw JSON
    return f"{DIM}  [{ev}] {json.dumps(rec, ensure_ascii=False)[:120]}{RESET}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretty-print a run trace JSONL file.")
    parser.add_argument("trace_file", help="Path to the .jsonl trace file")
    parser.add_argument("--entity", help="Filter: only show records for this entity")
    parser.add_argument("--field",  help="Filter: only show records for this field_id")
    parser.add_argument("--event",  help="Filter: only show this event type")
    parser.add_argument("--not-found", action="store_true",
                        help="Show only verification events where value was not found")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show quotes in extraction events")
    args = parser.parse_args()

    path = Path(args.trace_file)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    with path.open(encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]

    # Apply filters
    filtered = records
    if args.entity:
        filtered = [r for r in filtered if r.get("entity") == args.entity]
    if args.field:
        filtered = [r for r in filtered if r.get("field_id") == args.field]
    if args.event:
        filtered = [r for r in filtered if r.get("event") == args.event]
    if args.not_found:
        filtered = [
            r for r in filtered
            if r.get("event") == "verification" and r.get("winner_value") is None
        ]

    print(f"\n{BOLD}Trace: {path}  ({len(records)} events, showing {len(filtered)}){RESET}\n")
    for rec in filtered:
        print(render(rec, verbose=args.verbose))
    print()


if __name__ == "__main__":
    main()
