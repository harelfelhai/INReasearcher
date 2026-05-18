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


_CONF_HE = {
    "HIGH": "גבוה",
    "MEDIUM": "בינוני",
    "LOW": "נמוך",
    "NOT_FOUND": "לא נמצא",
}

_CONF_COLOR = {
    "HIGH": "16A34A",    # green-600
    "MEDIUM": "D97706",  # amber-600
    "LOW": "EA580C",     # orange-600
    "NOT_FOUND": "6B7280",  # gray-500
}

_MAX_SOURCES = 3  # per-field source columns written to xlsx


def _build_xlsx_workbook(results: list[EntityResult], plan_columns: list[ColumnPlan]):
    """Build and return an openpyxl Workbook for the given results."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment

    col_ids = [c.id for c in plan_columns]
    labels = {c.id: c.label_he for c in plan_columns}

    wb = Workbook()
    ws = wb.active
    ws.title = "תוצאות"
    ws.sheet_view.rightToLeft = True

    # ── Header row ────────────────────────────────────────────────────────────
    headers = ["ישות"]
    for col_id in col_ids:
        lbl = labels[col_id]
        headers += [lbl, f"{lbl} · ביטחון", f"{lbl} · אימותים"]
        for i in range(1, _MAX_SOURCES + 1):
            headers += [f"{lbl} · מקור {i}", f"{lbl} · ציטוט {i}"]

    ws.append(headers)
    header_font = Font(bold=True, color="FFFFFF", name="Arial")
    header_fill = PatternFill("solid", fgColor="2563EB")
    header_align = Alignment(horizontal="center", vertical="center",
                             wrap_text=True, readingOrder=2)
    for cell in ws[1]:
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_align

    # ── Data rows ─────────────────────────────────────────────────────────────
    rtl_align = Alignment(horizontal="right", vertical="top",
                          wrap_text=True, readingOrder=2)
    default_font = Font(name="Arial")

    for result in results:
        row_data: list = [result.entity_name]
        for col_id in col_ids:
            cell_obj = result.cells.get(col_id)
            if cell_obj is None or cell_obj.confidence == "NOT_FOUND":
                row_data += ["", _CONF_HE["NOT_FOUND"], ""]
                row_data += ["", ""] * _MAX_SOURCES
                continue

            conf_label = _CONF_HE.get(cell_obj.confidence, cell_obj.confidence)
            corr = getattr(cell_obj, "corroboration_count", 0) or 0
            row_data += [cell_obj.value or "", conf_label, str(corr)]

            # Collect up to _MAX_SOURCES sources (prefer all_sources, fall back to primary)
            sources = list(getattr(cell_obj, "all_sources", None) or [])
            if not sources and cell_obj.primary_source:
                sources = [cell_obj.primary_source]
            sources = sources[:_MAX_SOURCES]

            for src in sources:
                url = src.get("url", "") if isinstance(src, dict) else getattr(src, "url", "") or ""
                quote = src.get("quote", "") if isinstance(src, dict) else getattr(src, "quote", "") or ""
                row_data += [url or "", quote or ""]
            # Pad missing source slots with empty pairs
            for _ in range(_MAX_SOURCES - len(sources)):
                row_data += ["", ""]

        ws.append(row_data)
        # Style the last appended row
        row_idx = ws.max_row
        for col_idx, cell in enumerate(ws[row_idx], start=1):
            cell.font = default_font
            cell.alignment = rtl_align

    # ── Confidence column coloring ─────────────────────────────────────────────
    # Per-field block layout: value(0), confidence(1), corr_count(2), src1_url(3), src1_q(4), ...
    field_block_size = 3 + 2 * _MAX_SOURCES
    for f_idx in range(len(col_ids)):
        # 1-based: entity(col 1), then blocks of field_block_size; confidence is offset 2 in block
        conf_col = 1 + f_idx * field_block_size + 2
        for row_idx in range(2, ws.max_row + 1):
            cell = ws.cell(row=row_idx, column=conf_col)
            conf_key = next(
                (k for k, v in _CONF_HE.items() if v == cell.value), None
            )
            if conf_key and conf_key in _CONF_COLOR:
                cell.font = Font(name="Arial", bold=True, color=_CONF_COLOR[conf_key])

    # ── Column widths ──────────────────────────────────────────────────────────
    for col_cells in ws.columns:
        max_len = max(
            (len(str(c.value).split("\n")[0]) if c.value else 0) for c in col_cells
        )
        ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 4, 55)

    return wb


def build_xlsx_bytes(results: list[EntityResult], plan_columns: list[ColumnPlan]) -> bytes:
    """Build an xlsx workbook and return its raw bytes (for streaming HTTP responses)."""
    import io as _io
    wb = _build_xlsx_workbook(results, plan_columns)
    buf = _io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def write_xlsx(
    results: list[EntityResult],
    output_path: str,
    plan_columns: list[ColumnPlan],
) -> None:
    """Write results to a native .xlsx file (no BOM hacks, Hebrew-safe)."""
    wb = _build_xlsx_workbook(results, plan_columns)
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
