"""
Output writers: CSV (Excel-safe Hebrew) and full-provenance JSON.

CSV notes:
  - utf-8-sig (UTF-8 with BOM) is mandatory for Hebrew Excel compatibility.
    Without BOM, Israeli Excel installations render Hebrew as mojibake.
  - Each data column expands into 4 sub-columns:
      [value] [confidence] [primary source URL] [supporting quote]
  - NOT_FOUND cells write empty strings, not "null" — cleaner for analysis.

JSON notes:
  - Full provenance: all sources, all flags, all quotes.
  - ensure_ascii=False to keep Hebrew characters readable.
"""

import csv
import json
import sys
from .models import EntityResult, VerifiedCell, ColumnPlan


def write_xlsx(
    results: list[EntityResult],
    output_path: str,
    plan_columns: list[ColumnPlan],
) -> None:
    """Write results to a native .xlsx file (no BOM hacks, Hebrew-safe)."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment

    col_ids = [c.id for c in plan_columns]
    labels = {c.id: c.label_he for c in plan_columns}

    wb = Workbook()
    ws = wb.active
    ws.title = "Results"
    ws.sheet_view.rightToLeft = True

    headers = ["ישות"]
    for col_id in col_ids:
        lbl = labels[col_id]
        headers += [lbl, f"{lbl} · ביטחון", f"{lbl} · מקור", f"{lbl} · ציטוט"]
    ws.append(headers)
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="2563EB")
    for cell in ws[1]:
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")

    for result in results:
        row = [result.entity_name]
        for col_id in col_ids:
            cell = result.cells.get(col_id)
            if cell is None or cell.confidence == "NOT_FOUND":
                row += ["", "NOT_FOUND", "", ""]
                continue
            src_url = cell.primary_source.get("url", "") if cell.primary_source else ""
            quote = cell.primary_source.get("quote", "") if cell.primary_source else ""
            row += [cell.value or "", cell.confidence, src_url, quote or ""]
        ws.append(row)

    for column_cells in ws.columns:
        max_len = max((len(str(c.value)) if c.value else 0) for c in column_cells)
        ws.column_dimensions[column_cells[0].column_letter].width = min(max_len + 4, 60)

    wb.save(output_path)
    _log(f"XLSX → {output_path} ({len(results)} rows)")

_CONF_EMOJI = {
    "HIGH": "✓",
    "MEDIUM": "~",
    "LOW": "⚠",
    "NOT_FOUND": "✗",
}


def write_csv(
    results: list[EntityResult],
    output_path: str,
    plan_columns: list[ColumnPlan],
) -> None:
    """Write results to a UTF-8-with-BOM CSV file."""
    col_ids = [c.id for c in plan_columns]
    labels = {c.id: c.label_he for c in plan_columns}

    # Header: entity + 4 sub-columns per field
    headers = ["ישות"]
    for col_id in col_ids:
        lbl = labels[col_id]
        headers += [lbl, f"{lbl}__ביטחון", f"{lbl}__מקור", f"{lbl}__ציטוט"]

    with open(output_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)

        for result in results:
            row = [result.entity_name]
            for col_id in col_ids:
                cell: VerifiedCell | None = result.cells.get(col_id)
                if cell is None or cell.confidence == "NOT_FOUND":
                    row += ["", "NOT_FOUND", "", ""]
                    continue
                src_url = cell.primary_source.get("url", "") if cell.primary_source else ""
                quote = cell.primary_source.get("quote", "") if cell.primary_source else ""
                row += [cell.value or "", cell.confidence, src_url, quote or ""]
            writer.writerow(row)

    _log(f"CSV → {output_path} ({len(results)} rows)")


def write_json(results: list[EntityResult], output_path: str) -> None:
    """Write full provenance JSON (all sources, all flags)."""
    data = [r.model_dump() for r in results]
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    _log(f"JSON → {output_path}")


def print_summary(results: list[EntityResult]) -> None:
    all_cells = [c for r in results for c in r.cells.values()]
    total = len(all_cells)
    if not total:
        print("No results.")
        return

    counts = {lvl: sum(1 for c in all_cells if c.confidence == lvl)
              for lvl in ("HIGH", "MEDIUM", "LOW", "NOT_FOUND")}

    print(f"\n{'─' * 52}")
    print(f"  {len(results)} entities  ·  {total} cells total")
    for lvl, emoji in _CONF_EMOJI.items():
        n = counts[lvl]
        pct = 100 * n // total if total else 0
        bar = "█" * (pct // 5)
        print(f"  {emoji} {lvl:<12} {n:>4}  {pct:>3}%  {bar}")
    print(f"{'─' * 52}\n")


def _log(msg: str) -> None:
    print(f"[output] {msg}", file=sys.stderr)
