"""Pure pieces: settings, reducers, memory, cache, tools, cost, evaluation, rendering."""

from datetime import timedelta

import pytest
from fakes import KEYS, Clock, FakeLLM, page

from research_analyst.cache import ToolCache
from research_analyst.cli import format_report
from research_analyst.config import DEFAULT_STRONG_MODEL, ConfigError, load_settings
from research_analyst.evaluation import (
    CaseResult,
    ClaimCheck,
    CoverageReview,
    GoldenCase,
    GroundednessReview,
    PointCheck,
    find_regressions,
    groundedness_score,
    load_golden,
    run_case,
    summarize,
)
from research_analyst.llm import estimate_cost_usd
from research_analyst.memory import KnowledgeStore, canonical_entity
from research_analyst.schemas import (
    Citation,
    Finding,
    ResearchAnswer,
    Source,
    add_usage,
    merge_findings,
    merge_sources,
)
from research_analyst.tools import build_fetch_tool, build_search_tool

PRICES = {"m": (1.0, 5.0)}


class FakeTavily:
    """Stands in for TavilyClient with canned responses, counting calls."""

    def __init__(self, search_response: dict, extract_response: dict):
        """Store the responses each method will return."""
        self.search_response = search_response
        self.extract_response = extract_response
        self.calls = 0

    def search(self, query: str, max_results: int) -> dict:
        """Return the canned search response."""
        self.calls += 1
        return self.search_response

    def extract(self, urls: list[str]) -> dict:
        """Return the canned extract response."""
        self.calls += 1
        return self.extract_response


def call(tool, args: dict):
    """Invoke a tool the way the graph does, returning its ToolMessage."""
    return tool.invoke({"name": tool.name, "args": args, "id": "1", "type": "tool_call"})


def finding(statement: str, entities: list[str]) -> Finding:
    """A finding citing the canned page 'a'."""
    return Finding(
        statement=statement,
        source_url=page("a").url,
        quote="q",
        entities=entities,
        sub_question="sq",
        from_memory=False,
    )


def make_cache(tmp_path, clock: Clock) -> ToolCache:
    """A cache with a one-hour TTL."""
    return ToolCache(tmp_path / "cache.db", timedelta(hours=1), clock)


def test_settings_use_defaults_when_only_keys_are_set():
    settings = load_settings(KEYS)

    assert settings.strong_model == DEFAULT_STRONG_MODEL
    assert settings.judge_model == DEFAULT_STRONG_MODEL
    assert settings.cheap_effort is None


def test_settings_name_the_missing_credential():
    with pytest.raises(ConfigError, match="TAVILY_API_KEY"):
        load_settings({"ANTHROPIC_API_KEY": "a"})


def test_settings_reject_non_numeric_limit():
    with pytest.raises(ConfigError, match="RESEARCH_MAX_TOOL_ROUNDS"):
        load_settings({**KEYS, "RESEARCH_MAX_TOOL_ROUNDS": "many"})


def test_settings_reject_unknown_routing_mode():
    with pytest.raises(ConfigError, match="RESEARCH_ROUTING"):
        load_settings({**KEYS, "RESEARCH_ROUTING": "fastest"})


def test_merge_sources_keeps_position_and_prefers_longer_content():
    snippet = Source(url="u1", title="A", content="short")
    other = Source(url="u2", title="B", content="b")
    full = Source(url="u1", title="A", content="the full page text")

    assert merge_sources([snippet, other], [full]) == [full, other]


def test_merge_findings_treats_case_and_spacing_variants_as_one():
    first = finding("Acme was founded in 1999", ["Acme"])
    variant = finding("acme  was founded in 1999", ["Acme"])

    assert merge_findings([first], [variant]) == [first]


def test_add_usage_sums_per_model():
    total = add_usage(
        {"m": {"input_tokens": 1, "output_tokens": 2}},
        {
            "m": {"input_tokens": 10, "output_tokens": 20},
            "n": {"input_tokens": 5, "output_tokens": 5},
        },
    )

    assert total["m"] == {"input_tokens": 11, "output_tokens": 22}
    assert total["n"] == {"input_tokens": 5, "output_tokens": 5}


def test_cost_uses_per_model_prices():
    usage = {"m": {"input_tokens": 1_000_000, "output_tokens": 200_000}}

    assert estimate_cost_usd(usage, PRICES) == pytest.approx(2.0)


def test_cost_is_unknown_for_an_unpriced_model():
    assert estimate_cost_usd({"x": {"input_tokens": 1, "output_tokens": 1}}, PRICES) is None


def test_entity_names_differing_in_case_article_and_punctuation_are_one_entity():
    assert canonical_entity("The EU AI Act") == canonical_entity("eu-ai-act") == "eu ai act"


def test_store_deduplicates_entities_and_recalls_by_any_variant(tmp_path):
    store = KnowledgeStore(tmp_path / "memory.db")
    clock = Clock()
    store.save([finding("one", ["The EU AI Act"]), finding("two", ["EU AI act"])], {}, clock())

    assert store.entity_names(10) == ["The EU AI Act"]
    assert [r.finding.statement for r in store.recall(["eu ai act"])] == ["one", "two"]


def test_store_recall_of_unknown_entity_is_empty(tmp_path):
    assert KnowledgeStore(tmp_path / "memory.db").recall(["nobody"]) == []


def test_saving_a_finding_again_refreshes_its_timestamp(tmp_path):
    store = KnowledgeStore(tmp_path / "memory.db")
    clock = Clock()
    store.save([finding("one", ["Acme"])], {}, clock())
    clock.now += timedelta(days=40)
    store.save([finding("one", ["Acme"])], {}, clock())
    records = store.recall(["Acme"])

    assert [record.recorded_at for record in records] == [clock.now]


def test_cache_entry_expires_after_ttl(tmp_path):
    clock = Clock()
    cache = make_cache(tmp_path, clock)
    cache.put("search", {"query": "q"}, [1])

    assert cache.get("search", {"query": "q"}) == [1]
    clock.now += timedelta(hours=2)
    assert cache.get("search", {"query": "q"}) is None


def test_repeated_search_is_served_from_cache(tmp_path):
    hit = {"url": "u1", "title": "A", "content": "snippet"}
    client = FakeTavily({"results": [hit]}, {})
    tool = build_search_tool(client, 5, make_cache(tmp_path, Clock()))
    first, second = call(tool, {"query": "q"}), call(tool, {"query": "q"})

    assert client.calls == 1
    assert first.artifact == second.artifact == [Source(url="u1", title="A", content="snippet")]


def test_search_with_no_hits_says_so(tmp_path):
    tool = build_search_tool(FakeTavily({"results": []}, {}), 5, make_cache(tmp_path, Clock()))

    assert "No results found" in call(tool, {"query": "q"}).content


def test_fetch_marks_truncated_pages(tmp_path):
    client = FakeTavily({}, {"results": [{"url": "u1", "raw_content": "x" * 50}]})
    tool = build_fetch_tool(client, 10, make_cache(tmp_path, Clock()))
    message = call(tool, {"url": "u1"})

    assert message.content.startswith("x" * 10 + "\n")
    assert "10-character fetch limit" in message.content


def test_failed_fetch_yields_no_source_and_is_not_cached(tmp_path):
    client = FakeTavily({}, {"results": []})
    tool = build_fetch_tool(client, 10, make_cache(tmp_path, Clock()))
    first = call(tool, {"url": "u1"})
    call(tool, {"url": "u1"})

    assert first.artifact == []
    assert "Could not fetch u1" in first.content
    assert client.calls == 2


def test_report_lists_only_cited_sources_and_the_cost():
    state = {
        "answer": ResearchAnswer(
            answer="Claim [2].", citations=[Citation(source_number=2, quote="b")]
        ),
        "answer_sources": [page("a"), page("b")],
        "usage": {"claude-haiku-4-5": {"input_tokens": 1_000_000, "output_tokens": 0}},
    }
    report = format_report(state)

    assert "[2] Page b - https://example.com/b" in report
    assert "example.com/a" not in report
    assert "Cost: $1.0000" in report


def test_groundedness_is_share_of_supported_claims():
    review = GroundednessReview(
        claims=[ClaimCheck(claim="a", supported=True), ClaimCheck(claim="b", supported=False)]
    )

    assert groundedness_score(review) == 0.5
    assert groundedness_score(GroundednessReview(claims=[])) == 1.0


def eval_llm() -> FakeLLM:
    """Judges that find one unsupported claim of two and cover point 1 of two."""
    return FakeLLM(
        {},
        {
            GroundednessReview: lambda _prompt: GroundednessReview(
                claims=[
                    ClaimCheck(claim="good", supported=True),
                    ClaimCheck(claim="bad", supported=False),
                ]
            ),
            CoverageReview: lambda _prompt: CoverageReview(
                points=[PointCheck(point_number=1, covered=True)]
            ),
        },
    )


CASE = GoldenCase(id="c1", question="q", expected_points=["first", "second"])


def test_run_case_scores_answer_and_reports_what_was_missed():
    state = {
        "answer": ResearchAnswer(answer="text", citations=[]),
        "answer_sources": [page("a")],
        "usage": {"claude-haiku-4-5": {"input_tokens": 1_000_000, "output_tokens": 0}},
    }
    result = run_case(CASE, lambda _question: state, eval_llm())

    assert (result.groundedness, result.coverage) == (0.5, 0.5)
    assert result.unsupported_claims == ["bad"]
    assert result.missed_points == ["second"]
    assert result.cost_usd == pytest.approx(1.0)


def test_run_case_records_a_failure_as_an_error_not_a_score():
    def fail(_question: str):
        raise ValueError("response cut off")

    result = run_case(CASE, fail, eval_llm())

    assert result.error == "ValueError: response cut off"
    assert result.groundedness is None


def test_summary_averages_only_scored_cases():
    results = [
        CaseResult(case_id="a", groundedness=1.0, coverage=0.5),
        CaseResult(case_id="b", error="boom"),
    ]
    metrics = summarize(results)

    assert metrics["groundedness_mean"] == 1.0
    assert metrics["coverage_mean"] == 0.5
    assert metrics["error_count"] == 1


def test_regression_is_flagged_only_beyond_tolerance():
    previous = {"groundedness_mean": 0.9, "coverage_mean": 0.8}
    current = {"groundedness_mean": 0.8, "coverage_mean": 0.78}

    assert find_regressions(current, previous, 0.05) == [
        "groundedness_mean fell from 0.900 to 0.800"
    ]
    assert find_regressions(current, {}, 0.05) == []


def test_bundled_golden_set_is_valid():
    cases = load_golden(__import__("pathlib").Path("evals/golden.json"))

    assert len({case.id for case in cases}) == len(cases) >= 5
