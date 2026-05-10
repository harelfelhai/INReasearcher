"""
Success Memory — a JSON-backed store of validated and rejected results,
used for few-shot prompting of the compiler and extractor.

Three record types:
  - compiler_successes:    research questions whose plans the user validated
  - extraction_successes:  individual cells the user marked as correct
  - extraction_failures:   cells the user marked as hallucinations
                           (used to inject AVOID warnings into the extractor)

Retrieval is intentionally simple (keyword + type matching, no embeddings)
to keep the MVP dependency-free. Swap in a vector store later if needed.
"""

from __future__ import annotations
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokens(text: str) -> set[str]:
    """Lowercase token set for keyword overlap scoring."""
    if not text:
        return set()
    # Split on whitespace + punctuation; keep Hebrew, Latin, digits
    tokens = re.findall(r'\w+', text.lower(), flags=re.UNICODE)
    return {t for t in tokens if len(t) > 2}


class SuccessMemory:
    """JSON-file-backed memory store. Atomic writes via tmp + rename."""

    def __init__(self, path: str | Path = "memory.json"):
        self.path = Path(path)
        self._data: dict[str, list[dict[str, Any]]] = {
            "compiler_successes": [],
            "extraction_successes": [],
            "extraction_failures": [],
        }
        self._load()

    # ── persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open(encoding="utf-8") as fh:
                loaded = json.load(fh)
            for key in self._data:
                if key in loaded and isinstance(loaded[key], list):
                    self._data[key] = loaded[key]
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[memory] WARN: could not load {self.path}: {exc}")

    def _save(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(self._data, fh, ensure_ascii=False, indent=2)
        tmp.replace(self.path)

    # ── recording ─────────────────────────────────────────────────────────────

    def record_compiler_success(
        self,
        research_question: str,
        entity_type: str,
        plan: dict,
    ) -> str:
        """Store a validated research plan. Returns the record id."""
        record = {
            "id": str(uuid.uuid4()),
            "validated_at": _now_iso(),
            "research_question": research_question,
            "entity_type": entity_type,
            "plan": plan,
        }
        self._data["compiler_successes"].append(record)
        self._save()
        return record["id"]

    def record_extraction_success(
        self,
        field_id: str,
        field_label: str,
        field_type: str,
        entity: str,
        value: str,
        quote: str,
        source_url: str,
        source_domain: str,
    ) -> str:
        record = {
            "id": str(uuid.uuid4()),
            "validated_at": _now_iso(),
            "field_id": field_id,
            "field_label": field_label,
            "field_type": field_type,
            "entity": entity,
            "value": value,
            "quote": quote,
            "source_url": source_url,
            "source_domain": source_domain,
        }
        self._data["extraction_successes"].append(record)
        self._save()
        return record["id"]

    def record_extraction_failure(
        self,
        field_id: str,
        field_type: str,
        entity: str,
        claimed_value: str | None,
        claimed_quote: str | None,
        source_url: str,
        source_domain: str,
        reason: str,
    ) -> str:
        """
        Mark an extraction as hallucinated / wrong. Used to generate
        AVOID warnings for future extractions on the same domain+field_type.
        """
        record = {
            "id": str(uuid.uuid4()),
            "marked_at": _now_iso(),
            "field_id": field_id,
            "field_type": field_type,
            "entity": entity,
            "claimed_value": claimed_value,
            "claimed_quote": claimed_quote,
            "source_url": source_url,
            "source_domain": source_domain,
            "reason": reason,
        }
        self._data["extraction_failures"].append(record)
        self._save()
        return record["id"]

    # ── retrieval (few-shot selection) ────────────────────────────────────────

    def get_compiler_examples(
        self,
        research_question: str,
        entity_type: str,
        k: int = 3,
    ) -> list[dict]:
        """
        Return up to k past validated plans similar to the current question.
        Scoring: exact entity_type match (+5) + keyword overlap.
        """
        q_tokens = _tokens(research_question)
        scored = []
        for record in self._data["compiler_successes"]:
            score = 0
            if entity_type and record.get("entity_type") == entity_type:
                score += 5
            score += len(q_tokens & _tokens(record.get("research_question", "")))
            if score > 0:
                scored.append((score, record))
        scored.sort(key=lambda x: -x[0])
        return [r for _, r in scored[:k]]

    def get_extraction_examples(
        self,
        field_type: str,
        field_label: str,
        k: int = 2,
    ) -> list[dict]:
        """Return past validated extractions for the same field_type."""
        label_tokens = _tokens(field_label)
        scored = []
        for record in self._data["extraction_successes"]:
            if record.get("field_type") != field_type:
                continue
            score = 1  # base for matching type
            score += len(label_tokens & _tokens(record.get("field_label", "")))
            scored.append((score, record))
        scored.sort(key=lambda x: -x[0])
        return [r for _, r in scored[:k]]

    def get_extraction_warnings(
        self,
        field_type: str,
        candidate_domain: str | None = None,
    ) -> list[dict]:
        """
        Return past failures relevant to current extraction.
        Includes:
          - All failures for this field_type (general lesson)
          - Specific failures from candidate_domain if provided
        Capped at 5 most recent to control prompt length.
        """
        relevant = [
            r for r in self._data["extraction_failures"]
            if r.get("field_type") == field_type
            or (candidate_domain and r.get("source_domain") == candidate_domain)
        ]
        relevant.sort(key=lambda r: r.get("marked_at", ""), reverse=True)
        return relevant[:5]

    # ── stats / introspection ─────────────────────────────────────────────────

    def stats(self) -> dict[str, int]:
        return {k: len(v) for k, v in self._data.items()}


# ── few-shot prompt formatters ────────────────────────────────────────────────

def format_compiler_examples(examples: list[dict]) -> str:
    """Render compiler examples for injection into the user message."""
    if not examples:
        return ""
    lines = ["", "═══ Past validated research plans (use as STYLE GUIDES, not content) ═══"]
    for i, ex in enumerate(examples, 1):
        plan = ex.get("plan", {})
        cols = plan.get("columns", [])
        lines.append(f"\nExample {i}:")
        lines.append(f"  Question:    {ex.get('research_question', '')[:200]}")
        lines.append(f"  Entity type: {ex.get('entity_type', '')}")
        lines.append(f"  Fields:")
        for col in cols[:4]:
            sq = (col.get("search_queries_he") or col.get("search_queries_en") or [""])[0]
            lines.append(f"    • {col.get('id')} ({col.get('type')}) — sample query: {sq[:80]}")
        psd = list({d for col in cols for d in (col.get("preferred_source_domains") or [])})[:5]
        if psd:
            lines.append(f"  Domains chosen: {', '.join(psd)}")
    lines.append("\n═══ End of examples. Adapt to the CURRENT question. ═══")
    return "\n".join(lines)


def format_extraction_examples(examples: list[dict]) -> str:
    """Render extraction examples (good cases) for the extractor prompt."""
    if not examples:
        return ""
    lines = ["", "═══ Past validated extractions for this field type ═══"]
    for i, ex in enumerate(examples, 1):
        lines.append(f"\nExample {i}:")
        lines.append(f"  Entity:  {ex.get('entity', '')}")
        lines.append(f"  Field:   {ex.get('field_label', '')} ({ex.get('field_type')})")
        lines.append(f"  Source:  {ex.get('source_domain', '')}")
        lines.append(f"  Value:   {ex.get('value', '')}")
        quote = (ex.get("quote") or "")[:160]
        lines.append(f"  Quote:   {quote}")
    lines.append("\n═══ Note the precision of value-quote alignment in these examples. ═══")
    return "\n".join(lines)


def format_extraction_warnings(warnings: list[dict]) -> str:
    """Render past hallucinations as AVOID warnings."""
    if not warnings:
        return ""
    lines = ["", "⚠ AVOID — past hallucinations on this field type / domain ⚠"]
    for w in warnings:
        lines.append(
            f"  • Source {w.get('source_domain')} previously produced a wrong "
            f"'{w.get('field_type')}' value: \"{(w.get('claimed_value') or '')[:60]}\""
        )
        if w.get("reason"):
            lines.append(f"    Reason: {w['reason']}")
    lines.append("  → Be EXTRA strict on grounding and entity specificity below.")
    return "\n".join(lines)
