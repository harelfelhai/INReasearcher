"""Track Anthropic token usage and estimate $ cost per session.

Wraps an anthropic.Anthropic client so every messages.create() call accumulates
input/output tokens. The dollar estimate is best-effort based on hard-coded
per-million-token prices.
"""
from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import anthropic


# Approximate USD per 1M tokens. Update when Anthropic pricing changes.
_PRICING = {
    # Sonnet 4.6 (used by the extractor)
    "claude-sonnet-4-6":         {"in": 3.00, "out": 15.00},
    # Haiku 4.5 (used by the compiler)
    "claude-haiku-4-5-20251001": {"in": 1.00, "out":  5.00},
    # Generic fallbacks
    "default":                   {"in": 3.00, "out": 15.00},
}


def _price_for(model: str) -> dict:
    return _PRICING.get(model, _PRICING["default"])


@dataclass
class UsageTotals:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    per_model: dict = field(default_factory=dict)

    def __post_init__(self):
        self._lock = threading.Lock()

    def add(self, model: str, in_tok: int, out_tok: int) -> None:
        with self._lock:
            self.input_tokens += in_tok
            self.output_tokens += out_tok
            self.calls += 1
            p = _price_for(model)
            cost = (in_tok / 1_000_000.0) * p["in"] + (out_tok / 1_000_000.0) * p["out"]
            self.cost_usd += cost
            m = self.per_model.setdefault(model, {"in": 0, "out": 0, "cost": 0.0, "calls": 0})
            m["in"] += in_tok
            m["out"] += out_tok
            m["cost"] += cost
            m["calls"] += 1


class _TrackedMessages:
    def __init__(self, inner: Any, totals: UsageTotals):
        self._inner = inner
        self._totals = totals

    def create(self, **kwargs):
        # Retry on transient rate-limit / overload errors. Per-minute token
        # limits clear within 60s, so a few backed-off retries reliably get
        # us through. Surface the error after _MAX_RETRIES so the caller
        # can still fail loudly on persistent quota exhaustion.
        _MAX_RETRIES = 5
        attempt = 0
        while True:
            try:
                resp = self._inner.create(**kwargs)
                break
            except (anthropic.RateLimitError, anthropic.APIStatusError) as exc:
                status = getattr(exc, "status_code", None)
                is_retryable = isinstance(exc, anthropic.RateLimitError) or status in (429, 529, 503)
                if not is_retryable or attempt >= _MAX_RETRIES:
                    raise
                # Honor Retry-After header when Anthropic provides one.
                retry_after = None
                resp_hdrs = getattr(getattr(exc, "response", None), "headers", None)
                if resp_hdrs:
                    try:
                        retry_after = float(resp_hdrs.get("retry-after") or 0) or None
                    except (TypeError, ValueError):
                        retry_after = None
                # Exponential backoff with jitter, capped at 60s, with a
                # 15s floor on the first retry since rate-limit windows
                # are minute-based.
                base = retry_after if retry_after else min(60.0, 15.0 * (2 ** attempt))
                wait = base + random.uniform(0, 2.0)
                print(
                    f"    [claude-retry] attempt {attempt + 1}/{_MAX_RETRIES} "
                    f"after {type(exc).__name__} — sleeping {wait:.1f}s"
                )
                time.sleep(wait)
                attempt += 1
        model = kwargs.get("model", "default")
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self._totals.add(
                model,
                int(getattr(usage, "input_tokens", 0) or 0),
                int(getattr(usage, "output_tokens", 0) or 0),
            )
        return resp

    def __getattr__(self, name):
        return getattr(self._inner, name)


class TrackedAnthropic:
    """Drop-in proxy for anthropic.Anthropic that accumulates token usage."""

    def __init__(self, client: Any):
        self._client = client
        self.totals = UsageTotals()
        self._messages = _TrackedMessages(client.messages, self.totals)

    @property
    def messages(self):
        return self._messages

    def __getattr__(self, name):
        return getattr(self._client, name)
