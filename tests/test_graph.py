"""Run-graph behavior across the stages, driven by scripted models and tools."""

from datetime import timedelta

from fakes import Clock, FakeLLM, fake_search, make_settings, page, search_call
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from research_analyst.config import RoutingMode
from research_analyst.graph import build_graph, initial_state
from research_analyst.llm import ModelTier
from research_analyst.memory import KnowledgeStore
from research_analyst.prompts import FOLLOW_UP_TEMPLATE
from research_analyst.researcher import build_researcher
from research_analyst.schemas import (
    Citation,
    Critique,
    Difficulty,
    EntityMatches,
    FindingsReport,
    ReportedFinding,
    ResearchAnswer,
    ResearchPlan,
    ReviewedFinding,
    SubQuestion,
)

QUESTION = "What is known about Acme?"
ANSWER = ResearchAnswer(answer="Answer [1].", citations=[Citation(source_number=1, quote="q")])
WEAK_REASON = "the source does not say this"


def plan_of(**difficulty_by_question: Difficulty) -> ResearchPlan:
    """A plan with one sub-question per keyword argument."""
    return ResearchPlan(
        sub_questions=[
            SubQuestion(question=question, difficulty=difficulty)
            for question, difficulty in difficulty_by_question.items()
        ]
    )


def one_search_then_notes(*questions: str) -> dict[str, list[AIMessage]]:
    """Researcher scripts: each question is searched once, then notes are written."""
    return {q: [search_call(q), AIMessage(f"notes on {q}")] for q in questions}


def report_one_fact(prompt: str) -> FindingsReport:
    """Report a single finding named after the sub-question in the prompt."""
    question = prompt.split("<question>\n")[1].split("\n</question>")[0]
    finding = ReportedFinding(
        statement=f"fact about {question}", source_number=1, quote="q", entities=["Acme"]
    )
    return FindingsReport(findings=[finding])


def approve_all(prompt: str) -> Critique:
    """Mark every finding in the prompt as supported."""
    count = prompt.count("<finding number=")
    return Critique(
        reviews=[
            ReviewedFinding(finding_number=n, supported=True, reason="ok")
            for n in range(1, count + 1)
        ]
    )


def reject_second_once():
    """A critic that rejects finding 2 on its first call and approves everything after."""
    calls = iter([True])

    def critic(prompt: str) -> Critique:
        critique = approve_all(prompt)
        if next(calls, False):
            critique.reviews[1] = ReviewedFinding(
                finding_number=2, supported=False, reason=WEAK_REASON
            )
        return critique

    return critic


def make_llm(plan: ResearchPlan, chat_turns: dict, critic=approve_all, entities=()) -> FakeLLM:
    """A scripted LLM with the given plan, researcher turns, critic, and memory matches."""
    return FakeLLM(
        chat_turns,
        {
            EntityMatches: lambda _prompt: EntityMatches(entities=list(entities)),
            ResearchPlan: lambda _prompt: plan,
            FindingsReport: report_one_fact,
            Critique: critic,
            ResearchAnswer: lambda _prompt: ANSWER,
        },
    )


def run(tmp_path, llm: FakeLLM, clock: Clock, wants_review: bool = False, **overrides):
    """Build the graph over a shared memory file and run the question once."""
    settings = make_settings(tmp_path, **overrides)
    graph = build_graph(
        llm,
        build_researcher(llm, [fake_search], settings),
        KnowledgeStore(tmp_path / "memory.db"),
        settings,
        clock,
        InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "t"}}
    return graph, config, graph.invoke(initial_state(QUESTION, wants_review), config)


FOLLOW_UP_B = FOLLOW_UP_TEMPLATE.format(statement="fact about b", reason=WEAK_REASON)


def test_plan_fans_out_and_merges_findings_from_all_researchers(tmp_path):
    llm = make_llm(plan_of(a=Difficulty.EASY, b=Difficulty.HARD), one_search_then_notes("a", "b"))
    _, _, state = run(tmp_path, llm, Clock())

    assert [f.statement for f in state["findings"]] == ["fact about a", "fact about b"]
    assert state["answer_sources"] == [page("a"), page("b")]
    assert state["answer"] == ANSWER


def test_auto_routing_sends_easy_to_cheap_and_hard_to_strong(tmp_path):
    llm = make_llm(plan_of(a=Difficulty.EASY, b=Difficulty.HARD), one_search_then_notes("a", "b"))
    run(tmp_path, llm, Clock())

    assert llm.chat_tiers == {"a": ModelTier.CHEAP, "b": ModelTier.STRONG}


def test_cheap_routing_overrides_difficulty(tmp_path):
    llm = make_llm(plan_of(b=Difficulty.HARD), one_search_then_notes("b"))
    run(tmp_path, llm, Clock(), routing=RoutingMode.CHEAP)

    assert llm.chat_tiers == {"b": ModelTier.CHEAP}


def test_usage_is_summed_across_nodes_and_researchers(tmp_path):
    llm = make_llm(plan_of(a=Difficulty.EASY), one_search_then_notes("a"))
    _, _, state = run(tmp_path, llm, Clock())

    # Cheap: two researcher turns and the report. Strong: plan, critique, synthesis.
    assert state["usage"]["fake-cheap"]["input_tokens"] == 30
    assert state["usage"]["fake-strong"]["input_tokens"] == 30


def test_critic_sends_weak_finding_back_to_the_strong_model(tmp_path):
    llm = make_llm(
        plan_of(a=Difficulty.EASY, b=Difficulty.EASY),
        one_search_then_notes("a", "b", FOLLOW_UP_B),
        critic=reject_second_once(),
    )
    _, _, state = run(tmp_path, llm, Clock())
    synthesis_prompt = llm.prompts[ResearchAnswer][0]

    assert state["retry_rounds"] == 1
    assert llm.chat_tiers[FOLLOW_UP_B] == ModelTier.STRONG
    assert "Claim: fact about a\n" in synthesis_prompt
    assert f"Claim: fact about {FOLLOW_UP_B}\n" in synthesis_prompt
    assert "Claim: fact about b\n" not in synthesis_prompt


def test_weak_finding_is_dropped_when_no_retry_rounds_remain(tmp_path):
    llm = make_llm(
        plan_of(a=Difficulty.EASY, b=Difficulty.EASY),
        one_search_then_notes("a", "b"),
        critic=reject_second_once(),
    )
    _, _, state = run(tmp_path, llm, Clock(), max_retry_rounds=0)

    assert state["retry_rounds"] == 0
    assert state["answer_sources"] == [page("a")]


def test_weak_finding_is_dropped_when_token_budget_is_spent(tmp_path):
    llm = make_llm(
        plan_of(a=Difficulty.EASY, b=Difficulty.EASY),
        one_search_then_notes("a", "b"),
        critic=reject_second_once(),
    )
    _, _, state = run(tmp_path, llm, Clock(), token_budget=1)

    assert state["retry_rounds"] == 0
    assert state["answer_sources"] == [page("a")]


def test_second_run_answers_from_memory_without_researching(tmp_path):
    clock = Clock()
    run(tmp_path, make_llm(plan_of(a=Difficulty.EASY), one_search_then_notes("a")), clock)
    llm = make_llm(plan_of(), {}, entities=["Acme"])
    _, _, state = run(tmp_path, llm, clock)

    assert "- fact about a" in llm.prompts[ResearchPlan][0]
    assert llm.chat_tiers == {}
    assert [f.from_memory for f in state["findings"]] == [True]
    assert [source.url for source in state["answer_sources"]] == [page("a").url]


def test_stale_memory_is_offered_as_a_lead_but_not_used_as_evidence(tmp_path):
    clock = Clock()
    run(tmp_path, make_llm(plan_of(a=Difficulty.EASY), one_search_then_notes("a")), clock)
    clock.now += timedelta(days=31)
    llm = make_llm(plan_of(), {}, entities=["Acme"])
    _, _, state = run(tmp_path, llm, clock)

    assert state["stale_leads"] == ["fact about a"]
    assert state["findings"] == []


def test_review_pauses_and_resumes_with_the_edited_plan(tmp_path):
    llm = make_llm(plan_of(a=Difficulty.EASY), one_search_then_notes("a", "c"))
    graph, config, paused = run(tmp_path, llm, Clock(), wants_review=True)
    edited = plan_of(c=Difficulty.HARD).model_dump()

    assert paused["__interrupt__"][0].value["plan"] == plan_of(a=Difficulty.EASY).model_dump(
        mode="json"
    )
    assert llm.chat_tiers == {}

    state = graph.invoke(Command(resume=edited), config)

    assert list(llm.chat_tiers) == ["c"]
    assert state["answer"] == ANSWER
