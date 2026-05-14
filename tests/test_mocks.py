"""Sanity checks for the MockClaude / StubSearch test doubles."""
from __future__ import annotations

import pytest

from tests.mocks import MockClaude, StubSearch


def test_mock_claude_returns_registered_payload():
    claude = MockClaude().on("my_tool", {"value": 42})
    resp = claude.messages.create(
        model="claude-x", max_tokens=100,
        system="you are a tester",
        tools=[{"name": "my_tool", "input_schema": {}}],
        tool_choice={"type": "tool", "name": "my_tool"},
        messages=[{"role": "user", "content": "hi"}],
    )
    assert resp.content[0].type == "tool_use"
    assert resp.content[0].input == {"value": 42}
    assert resp.usage.input_tokens == 100
    assert len(claude.calls) == 1
    assert claude.calls[0].tool == "my_tool"
    assert claude.calls[0].user == "hi"


def test_mock_claude_queue_pops_responses_in_order():
    claude = MockClaude().on("t", [{"i": 1}, {"i": 2}])
    a = claude.messages.create(model="m", max_tokens=1, tools=[{"name": "t"}],
                               tool_choice={"type": "tool", "name": "t"},
                               messages=[])
    b = claude.messages.create(model="m", max_tokens=1, tools=[{"name": "t"}],
                               tool_choice={"type": "tool", "name": "t"},
                               messages=[])
    assert a.content[0].input == {"i": 1}
    assert b.content[0].input == {"i": 2}
    with pytest.raises(RuntimeError, match="exhausted"):
        claude.messages.create(model="m", max_tokens=1, tools=[{"name": "t"}],
                               tool_choice={"type": "tool", "name": "t"},
                               messages=[])


def test_mock_claude_callable_response_sees_call():
    claude = MockClaude().on("echo", lambda call: {"saw_user": call.user[:10]})
    resp = claude.messages.create(
        model="m", max_tokens=1, tools=[{"name": "echo"}],
        tool_choice={"type": "tool", "name": "echo"},
        messages=[{"role": "user", "content": "hello world long"}],
    )
    assert resp.content[0].input == {"saw_user": "hello worl"}


def test_mock_claude_unregistered_tool_raises():
    claude = MockClaude()
    with pytest.raises(RuntimeError, match="no response registered"):
        claude.messages.create(model="m", max_tokens=1, tools=[{"name": "x"}],
                               tool_choice={"type": "tool", "name": "x"},
                               messages=[])


def test_stub_search_matches_substring_then_falls_through():
    s = StubSearch()
    s.add("mayor", [{"url": "u1", "content": "...", "raw_content": "long mayor page"}])
    s.add(None, [{"url": "u2", "content": "...", "raw_content": "fallback page"}])
    assert s.search("who is the mayor of TLV")["results"][0]["url"] == "u1"
    assert s.search("something else")["results"][0]["url"] == "u2"
    assert s.search_log == ["who is the mayor of TLV", "something else"]
