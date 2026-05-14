"""Test doubles for the Anthropic and search clients.

MockClaude:
  A drop-in replacement for `anthropic.Anthropic` whose `messages.create`
  returns a canned `tool_use` block keyed by tool name. Every call is recorded
  so tests can assert how many times each tool was invoked and what context
  was passed (the prompt body, the system prompt, etc.).

StubSearch:
  An in-memory search client that returns a programmable list of (url, content)
  hits per query. Implements the small interface the extractor relies on:
  `.search(query, max_results, include_raw_content)`.

Together these let us test the orchestration end-to-end with zero network
calls, zero LLM cost, and complete determinism.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


# ── MockClaude ───────────────────────────────────────────────────────────────

@dataclass
class _RecordedCall:
    tool: str
    system: str
    user: str
    model: str
    full_kwargs: dict


class _ToolBlock:
    """Mimics anthropic.types.ToolUseBlock enough for our parsing code."""
    type = "tool_use"

    def __init__(self, name: str, payload: dict):
        self.name = name
        self.input = payload


class _Usage:
    """Mimics anthropic.types.Usage."""
    def __init__(self, input_tokens: int = 100, output_tokens: int = 50):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _Response:
    def __init__(self, tool_name: str, payload: dict):
        self.content = [_ToolBlock(tool_name, payload)]
        self.usage = _Usage()


class _MockMessages:
    def __init__(self, parent: "MockClaude"):
        self._parent = parent

    def create(self, **kwargs):
        tool_name = _infer_tool_name(kwargs)
        user = ""
        for m in kwargs.get("messages", []):
            if m.get("role") == "user":
                c = m.get("content", "")
                user = c if isinstance(c, str) else str(c)
        call = _RecordedCall(
            tool=tool_name,
            system=kwargs.get("system", ""),
            user=user,
            model=kwargs.get("model", ""),
            full_kwargs=kwargs,
        )
        self._parent.calls.append(call)

        handler = self._parent.responses.get(tool_name)
        if handler is None:
            raise RuntimeError(
                f"MockClaude: no response registered for tool {tool_name!r}. "
                f"Registered: {list(self._parent.responses)}"
            )

        if callable(handler):
            payload = handler(call)
        elif isinstance(handler, list):
            if not handler:
                raise RuntimeError(
                    f"MockClaude: response queue for tool {tool_name!r} exhausted"
                )
            payload = handler.pop(0)
        else:
            payload = handler

        return _Response(tool_name, payload)


def _infer_tool_name(kwargs: dict) -> str:
    tc = kwargs.get("tool_choice") or {}
    if isinstance(tc, dict) and tc.get("type") == "tool" and tc.get("name"):
        return tc["name"]
    tools = kwargs.get("tools") or []
    if tools:
        return tools[0].get("name", "")
    return ""


class MockClaude:
    """Programmable fake Anthropic client. Use `.on(tool, response)` to register
    a canned response for a tool, then pass this instance anywhere a real
    `anthropic.Anthropic` is expected."""

    def __init__(self):
        self.responses: Dict[str, Any] = {}
        self.calls: List[_RecordedCall] = []
        self._messages = _MockMessages(self)

    def on(self, tool_name: str, response: Any) -> "MockClaude":
        """Register a canned response for `tool_name`.

        `response` may be:
          - dict       — returned for every call to this tool
          - list[dict] — popped FIFO (queue) — exhausting raises
          - callable   — called with the _RecordedCall, must return a dict
        """
        self.responses[tool_name] = response
        return self

    @property
    def messages(self):
        return self._messages

    def calls_for(self, tool_name: str) -> List[_RecordedCall]:
        return [c for c in self.calls if c.tool == tool_name]


# ── StubSearch ───────────────────────────────────────────────────────────────

@dataclass
class _Hit:
    url: str
    content: str
    title: str = ""


class StubSearch:
    """In-memory search client. Register hits per query substring or wildcard."""

    def __init__(self):
        # Each entry: (substring | None, list[hit dicts]). None matches anything
        # not matched by a more specific entry first.
        self._rules: List[tuple] = []
        self.search_log: List[str] = []

    def add(self, query_substring: str | None, hits: List[Dict[str, str]]):
        """Register hits to return when query contains `query_substring`.

        Pass `None` for a wildcard fallback. Rules are evaluated in insertion
        order; the first matching one wins.
        """
        self._rules.append((query_substring, hits))
        return self

    def search(self, query: str, max_results: int = 5, **kwargs):
        self.search_log.append(query)
        for needle, hits in self._rules:
            if needle is None or needle in query:
                return {"results": hits[:max_results]}
        return {"results": []}
