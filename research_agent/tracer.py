"""
Per-run structured trace writer.

Every pipeline step emits a JSON line to logs/run_{session_id}.jsonl so that
bad results can be fully autopsied after the fact without re-running.

Trace event taxonomy
--------------------
run_start       — question, plan columns, entity list
field_search    — queries sent to search engine per (entity, field)
search_hit      — each URL returned by search (content_length, is_new)
extraction      — one LLM extraction call (value, quote, is_grounded)
batch_extraction— multi-field LLM call result per URL (entity scope)
probe_query     — directory-probe search query
probe_hit       — probe page coverage score
probe_result    — probe accepted/rejected
verification    — verifier decision for (entity, field)
entity_done     — per-entity summary (found/missing, elapsed)
run_done        — final cost, token totals, elapsed
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunTracer:
    """Thread-safe JSONL trace writer for one research run."""

    def __init__(self, session_id: str, logs_dir: Path | str = "logs"):
        logs_dir = Path(logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.path = logs_dir / f"run_{session_id}.jsonl"
        self._lock = threading.Lock()
        self.session_id = session_id

    def emit(self, event: str, **kwargs) -> None:
        record = {"ts": _now_iso(), "session_id": self.session_id, "event": event, **kwargs}
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


class NullTracer:
    """Drop-all tracer used in tests and when no session is attached."""

    path = None
    session_id = None

    def emit(self, event: str, **kwargs) -> None:  # noqa: D401
        pass


# Module-level singleton — avoids passing a tracer through every call stack
# that doesn't need observability (e.g. compiler, verifier).
# Overwritten by api/main.py per request; safe because entity processing is
# already thread-isolated (one thread per entity, tracer is thread-safe).
_null = NullTracer()
