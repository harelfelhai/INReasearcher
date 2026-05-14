"""OpenAPI snapshot test.

Locks the public API surface (paths, request/response shapes, model schemas)
into a snapshot file. If the snapshot drifts, the test fails — that's
the cue to:

  1. Update the corresponding frontend types in web/src/types.ts and api.ts.
  2. Regenerate the snapshot:  UPDATE_OPENAPI_SNAPSHOT=1 pytest tests/test_openapi_snapshot.py

The fail message lists the exact added / removed / changed paths so the
diff is actionable instead of "schema changed somewhere".
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# The auth + api imports trigger module-level engine creation; use an
# isolated DB so importing app.openapi() during tests does not touch the
# real data/app.db.
_TEST_DB = Path(__file__).parent / "_test_openapi.db"
if _TEST_DB.exists():
    _TEST_DB.unlink()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TEST_DB}")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")

from api.main import app  # noqa: E402

SNAPSHOT = Path(__file__).parent / "snapshots" / "openapi.json"


def _normalize(schema: dict) -> dict:
    """Drop fields that aren't part of the contract (FastAPI version etc.)."""
    s = dict(schema)
    # 'info' contains the FastAPI app version, irrelevant to the contract
    s.pop("info", None)
    return s


def _diff_keys(a: dict, b: dict, path: str = "") -> list[str]:
    """Recursively report keys that differ between two dicts. Best-effort."""
    diffs: list[str] = []
    only_in_a = set(a) - set(b)
    only_in_b = set(b) - set(a)
    for k in sorted(only_in_a):
        diffs.append(f"  - removed: {path}.{k}".lstrip("."))
    for k in sorted(only_in_b):
        diffs.append(f"  + added:   {path}.{k}".lstrip("."))
    for k in sorted(set(a) & set(b)):
        av, bv = a[k], b[k]
        if isinstance(av, dict) and isinstance(bv, dict):
            diffs.extend(_diff_keys(av, bv, f"{path}.{k}".lstrip(".")))
        elif av != bv:
            diffs.append(f"  ~ changed: {path}.{k}".lstrip("."))
    return diffs


def test_openapi_schema_matches_snapshot():
    current = _normalize(app.openapi())

    if os.environ.get("UPDATE_OPENAPI_SNAPSHOT"):
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(json.dumps(current, indent=2, sort_keys=True,
                                       ensure_ascii=False))
        pytest.skip(f"Regenerated snapshot at {SNAPSHOT}")

    if not SNAPSHOT.exists():
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(json.dumps(current, indent=2, sort_keys=True,
                                       ensure_ascii=False))
        pytest.skip(
            f"Bootstrapped snapshot at {SNAPSHOT}. "
            "Inspect it and commit; future runs will detect drift."
        )

    saved = json.loads(SNAPSHOT.read_text())
    if current != saved:
        diffs = _diff_keys(saved, current) or ["  (values differ — see snapshot diff)"]
        msg = (
            "OpenAPI schema drifted from snapshot. Path differences:\n"
            + "\n".join(diffs[:40])
            + "\n\nIf intentional: bump web/src/types.ts + web/src/api.ts to match, "
            "then regenerate with:\n"
            "  UPDATE_OPENAPI_SNAPSHOT=1 python -m pytest tests/test_openapi_snapshot.py"
        )
        pytest.fail(msg)


def test_critical_paths_remain_in_schema():
    """Belt-and-braces: even if someone regenerates the snapshot carelessly,
    these well-known endpoints must always exist."""
    paths = set(app.openapi()["paths"].keys())
    for required in (
        "/api/health",
        "/api/auth/login",
        "/api/auth/me",
        "/api/compile-schema",
        "/api/audit",
        "/api/enrich",
        "/api/run",
        "/api/discover-entities/plan",
        "/api/discover-entities/run",
        "/api/admin/users",
        "/api/admin/users/{user_id}/budget",
        "/api/user/sessions",
        "/api/user/exports/{export_id}",
    ):
        assert required in paths, f"missing endpoint: {required}"
