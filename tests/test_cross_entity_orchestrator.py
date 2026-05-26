"""
Unit tests for the cross-entity search+extract orchestrator.

We stub the search client and the Claude client so the test is hermetic.
The goal is to verify:
  1. ONE search runs per (entity, field) pair.
  2. Each search asks for 6 results.
  3. Tier 1 processes only the top-3 URLs across all pairs, deduplicated.
  4. Tier 2 fallback runs only for (entity, field) pairs without grounded values.
  5. Shared URLs across entities trigger ONE Claude call covering all pairs.
"""
from research_agent import extractor
from research_agent.models import ColumnPlan, ExtractionResult


class _FakeSearch:
    """Returns canned ranked URLs per (entity, field) — 6 URLs each."""
    def __init__(self, urls_per_call: dict[str, list[str]]):
        self.urls_per_call = urls_per_call
        self.calls: list[tuple[str, int]] = []  # (query, max_results)

    def search(self, query, max_results=5, **kwargs):
        self.calls.append((query, max_results))
        urls = self.urls_per_call.get(query, [])[:max_results]
        return {"results": [
            {
                "url": u,
                "raw_content": (
                    f"This is a long enough fake page about {u}. "
                    f"It contains the answer is HIT_{u} embedded in some text "
                    f"so the substring grounding check can succeed reliably."
                ),
            }
            for u in urls
        ]}


class _FakeClaude:
    """
    Stub of anthropic.Anthropic. Returns grounded extractions for URLs whose
    name starts with "good", null for others. Records every Claude call.
    """
    def __init__(self):
        self.messages = self
        self.calls: list[dict] = []

    def create(self, *, model, max_tokens, system, tools, tool_choice, messages):
        prompt = messages[0]["content"]
        # Extract pair_ids and source URL from the prompt.
        import re
        url_match = re.search(r"Source URL: (\S+)", prompt)
        url = url_match.group(1) if url_match else ""
        pair_ids = re.findall(r"pair_id='(p\d+)'", prompt)
        self.calls.append({"url": url, "pair_ids": pair_ids})

        extractions = []
        good = "good" in url
        for pid in pair_ids:
            if good:
                extractions.append({
                    "pair_id": pid,
                    "value": f"HIT_{url}",
                    "quote_original": f"answer is HIT_{url}",
                    "confidence": 0.9,
                })
            else:
                extractions.append({
                    "pair_id": pid, "value": None,
                    "quote_original": None, "confidence": 0.0,
                })

        class _Block:
            type = "tool_use"
            input = {"extractions": extractions}

        class _Resp:
            content = [_Block()]
        return _Resp()


def _mk_field(fid: str) -> ColumnPlan:
    return ColumnPlan(
        id=fid, label_he=fid, label_en=fid, type="free_text",
        search_queries_he=[f"query_{fid}_{{entity}}"],
        search_queries_en=[],
    )


def test_one_search_per_pair_and_six_results():
    f1 = _mk_field("f1")
    entities = ["E1", "E2"]
    urls = {
        "query_f1_E1": [f"https://bad{i}.com/e1f1" for i in range(6)],
        "query_f1_E2": [f"https://bad{i}.com/e2f1" for i in range(6)],
    }
    search = _FakeSearch(urls)
    claude = _FakeClaude()

    extractor.search_and_extract_cross_entity(
        fields=[f1], entities=entities,
        resolved_deps_per_entity={"E1": {}, "E2": {}},
        search_client=search, claude=claude,
    )

    # Exactly 2 searches: one per (entity, field) pair.
    assert len(search.calls) == 2
    # Each search asked for 6 results.
    for _, n in search.calls:
        assert n == 6


def test_tier_1_uses_only_top_3():
    """If top-3 URLs already answer everything, tier-2 should not run."""
    f1 = _mk_field("f1")
    # 6 URLs, but only the first is 'good' — top-3 contains a good URL.
    urls = {
        "query_f1_E1": [
            "https://good1.com/e1",  # rank 1 — answers it
            "https://bad-2.com/e1", "https://bad-3.com/e1",
            "https://bad-4.com/e1", "https://bad-5.com/e1", "https://bad-6.com/e1",
        ],
    }
    search = _FakeSearch(urls)
    claude = _FakeClaude()

    extractor.search_and_extract_cross_entity(
        fields=[f1], entities=["E1"],
        resolved_deps_per_entity={"E1": {}},
        search_client=search, claude=claude,
    )

    # Tier 1 = top-3 URLs → 3 Claude calls. Tier 2 should NOT run because
    # the pair already has a grounded value from good1.com.
    tier1_urls = {c["url"] for c in claude.calls}
    assert "https://good1.com/e1" in tier1_urls
    # Tier 2 URLs (rank 4-6) should not appear in Claude calls.
    assert not any("bad-4" in u or "bad-5" in u or "bad-6" in u for u in tier1_urls)


def test_tier_2_runs_when_tier_1_finds_nothing():
    f1 = _mk_field("f1")
    urls = {
        "query_f1_E1": [
            "https://bad-1.com/e1", "https://bad-2.com/e1", "https://bad-3.com/e1",
            "https://good4.com/e1",  # rank 4 — only good URL is in tier 2
            "https://bad-5.com/e1", "https://bad-6.com/e1",
        ],
    }
    search = _FakeSearch(urls)
    claude = _FakeClaude()

    extractor.search_and_extract_cross_entity(
        fields=[f1], entities=["E1"],
        resolved_deps_per_entity={"E1": {}},
        search_client=search, claude=claude,
    )

    called_urls = [c["url"] for c in claude.calls]
    # Tier 1 saw the 3 bad URLs.
    assert "https://bad-1.com/e1" in called_urls
    # Tier 2 was triggered and processed the good rank-4 URL.
    assert "https://good4.com/e1" in called_urls


def test_shared_url_one_call_covering_all_pairs():
    """If both entities' searches surface the same URL, one Claude call covers both."""
    f1 = _mk_field("f1")
    shared_url = "https://good-shared.com/list"
    urls = {
        "query_f1_E1": [shared_url, "https://other-e1.com/x", "https://x-e1.com/y",
                        "https://x2-e1.com/y", "https://x3-e1.com/y", "https://x4-e1.com/y"],
        "query_f1_E2": [shared_url, "https://other-e2.com/x", "https://x-e2.com/y",
                        "https://x2-e2.com/y", "https://x3-e2.com/y", "https://x4-e2.com/y"],
    }
    search = _FakeSearch(urls)
    claude = _FakeClaude()

    extractor.search_and_extract_cross_entity(
        fields=[f1], entities=["E1", "E2"],
        resolved_deps_per_entity={"E1": {}, "E2": {}},
        search_client=search, claude=claude,
    )

    # Exactly one Claude call for the shared URL, covering 2 pair_ids.
    shared_calls = [c for c in claude.calls if c["url"] == shared_url]
    assert len(shared_calls) == 1
    assert len(shared_calls[0]["pair_ids"]) == 2
