"""The research run: recall, plan, fan out to researchers, critique, synthesize, remember.

START -> recall_memory -> make_plan -> review -+-> researcher (xN) -> critique -+
                                               |        ^                       |
                                               |        +--- weak findings -----+
                                               +-----------------> synthesize <-+
                                                                       |
                                                                   remember -> END
"""

import logging
from collections.abc import Callable
from datetime import datetime
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send, interrupt

from research_analyst.config import Settings
from research_analyst.llm import LLM, ModelTier, total_tokens
from research_analyst.memory import KnowledgeStore, is_stale
from research_analyst.prompts import (
    CRITIC_SYSTEM_PROMPT,
    CRITIC_USER_TEMPLATE,
    FOLLOW_UP_TEMPLATE,
    PLAN_SYSTEM_PROMPT,
    PLAN_USER_TEMPLATE,
    RECALL_SYSTEM_PROMPT,
    RECALL_USER_TEMPLATE,
    SYNTHESIS_SYSTEM_PROMPT,
    SYNTHESIS_USER_TEMPLATE,
    bullet_list,
    format_evidence,
    format_findings,
    format_source_list,
)
from research_analyst.researcher import researcher_input
from research_analyst.schemas import (
    Critique,
    Difficulty,
    EntityMatches,
    Finding,
    ResearchAnswer,
    ResearchPlan,
    RunState,
    Source,
    SubQuestion,
    Verdict,
    add_usage,
)

logger = logging.getLogger(__name__)

# Entity names offered to the model when matching a question against memory.
MAX_KNOWN_ENTITIES = 200
RECALLED_REASON = "Verified in an earlier run and recalled from memory."
NO_VERDICT_REASON = "The critic gave no verdict for this finding."


def initial_state(question: str, wants_review: bool) -> RunState:
    """Build the starting state for a research question."""
    return RunState(
        question=question,
        wants_review=wants_review,
        stale_leads=[],
        plan=None,
        pending=[],
        findings=[],
        verdicts={},
        sources=[],
        retry_rounds=0,
        usage={},
        answer=None,
        answer_sources=[],
    )


def sources_cited_by(findings: list[Finding], sources: list[Source]) -> list[Source]:
    """Return the sources the findings rely on, in their original order."""
    cited_urls = {finding.source_url for finding in findings}
    return [source for source in sources if source.url in cited_urls]


def verdicts_for(findings: list[Finding], critique: Critique) -> dict[str, Verdict]:
    """Match the critic's numbered reviews to findings; an unreviewed finding is unsupported."""
    reviews = {review.finding_number: review for review in critique.reviews}
    verdicts = {}
    for number, finding in enumerate(findings, start=1):
        review = reviews.get(number)
        verdicts[finding.key] = (
            Verdict(supported=review.supported, reason=review.reason)
            if review
            else Verdict(supported=False, reason=NO_VERDICT_REASON)
        )
    return verdicts


def follow_ups(findings: list[Finding], verdicts: dict[str, Verdict]) -> list[SubQuestion]:
    """Turn each unsupported finding into a sub-question asking for better evidence.

    Follow-ups are marked hard: the first attempt already failed, so they go to
    the strong model.
    """
    return [
        SubQuestion(
            question=FOLLOW_UP_TEMPLATE.format(
                statement=finding.statement, reason=verdicts[finding.key].reason
            ),
            difficulty=Difficulty.HARD,
        )
        for finding in findings
        if not verdicts[finding.key].supported
    ]


def keep_known_citations(answer: ResearchAnswer, source_count: int) -> ResearchAnswer:
    """Drop citations that point at a source number that does not exist."""
    known = [c for c in answer.citations if 1 <= c.source_number <= source_count]
    dropped = len(answer.citations) - len(known)
    if dropped:
        logger.warning("dropped %d citation(s) pointing at unknown sources", dropped)
    return answer.model_copy(update={"citations": known})


def build_graph(
    llm: LLM,
    researcher: CompiledStateGraph,
    store: KnowledgeStore,
    settings: Settings,
    clock: Callable[[], datetime],
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph:
    """Compile the research run graph.

    Args:
        llm: Model provider for planning, critique, and synthesis.
        researcher: Compiled researcher subgraph, run once per sub-question.
        store: Knowledge graph queried before research and updated after it.
        settings: Supplies the round, budget, and staleness limits.
        clock: Returns the current time; injected so staleness is testable.
        checkpointer: Persists state after each step so a run can pause and resume.
    """

    def recall_memory(state: RunState) -> dict:
        """Look up what earlier runs established about the entities in the question."""
        known = store.entity_names(MAX_KNOWN_ENTITIES)
        if not known:
            return {}
        prompt = RECALL_USER_TEMPLATE.format(
            question=state["question"], entities=bullet_list(known)
        )
        messages = [SystemMessage(RECALL_SYSTEM_PROMPT), HumanMessage(prompt)]
        matches, usage = llm.structured(ModelTier.CHEAP, EntityMatches, messages)
        records = store.recall([name for name in matches.entities if name in known])
        now = clock()
        fresh = [r for r in records if not is_stale(r, now, settings.memory_max_age)]
        stale = [r for r in records if is_stale(r, now, settings.memory_max_age)]
        return {
            "findings": [record.finding for record in fresh],
            "verdicts": {
                record.finding.key: Verdict(supported=True, reason=RECALLED_REASON)
                for record in fresh
            },
            "sources": [
                Source(
                    url=record.finding.source_url,
                    title=record.source_title,
                    content=record.finding.quote,
                )
                for record in fresh
            ],
            "stale_leads": [record.finding.statement for record in stale],
            "usage": usage,
        }

    def make_plan(state: RunState) -> dict:
        """Split the question into sub-questions, skipping what memory already covers."""
        prompt = PLAN_USER_TEMPLATE.format(
            question=state["question"],
            known_findings=bullet_list([finding.statement for finding in state["findings"]]),
            stale_leads=bullet_list(state["stale_leads"]),
            max_sub_questions=settings.max_sub_questions,
        )
        messages = [SystemMessage(PLAN_SYSTEM_PROMPT), HumanMessage(prompt)]
        plan, usage = llm.structured(ModelTier.STRONG, ResearchPlan, messages)
        return {"plan": plan, "usage": usage}

    def review(state: RunState) -> dict:
        """Pause for a human to edit the plan when asked to, then queue it for research.

        Raises:
            pydantic.ValidationError: If the edited plan does not match ResearchPlan.
        """
        plan = state["plan"]
        if state["wants_review"]:
            edited = interrupt(
                {"question": state["question"], "plan": plan.model_dump(mode="json")}
            )
            plan = ResearchPlan.model_validate(edited)
        return {"plan": plan, "pending": plan.sub_questions[: settings.max_sub_questions]}

    def dispatch(state: RunState) -> list[Send] | Literal["synthesize"]:
        """Fan out one researcher per pending sub-question, or move on if there are none."""
        if not state["pending"]:
            return "synthesize"
        return [Send("researcher", researcher_input(sub)) for sub in state["pending"]]

    def critique(state: RunState) -> dict:
        """Check new findings against their sources and send weak ones back while budget lasts."""
        unreviewed = [f for f in state["findings"] if f.key not in state["verdicts"]]
        if not unreviewed:
            return {"pending": []}
        cited = sources_cited_by(unreviewed, state["sources"])
        prompt = CRITIC_USER_TEMPLATE.format(
            sources=format_evidence(cited), findings=format_findings(unreviewed, cited)
        )
        messages = [SystemMessage(CRITIC_SYSTEM_PROMPT), HumanMessage(prompt)]
        critique_result, usage = llm.structured(ModelTier.STRONG, Critique, messages)
        verdicts = verdicts_for(unreviewed, critique_result)

        has_rounds = state["retry_rounds"] < settings.max_retry_rounds
        has_tokens = total_tokens(add_usage(state["usage"], usage)) < settings.token_budget
        retries = follow_ups(unreviewed, verdicts) if has_rounds and has_tokens else []
        return {
            "verdicts": verdicts,
            "pending": retries[: settings.max_sub_questions],
            "retry_rounds": state["retry_rounds"] + bool(retries),
            "usage": usage,
        }

    def synthesize(state: RunState) -> dict:
        """Write the structured, cited answer from the findings the critic accepted."""
        supported = [f for f in state["findings"] if state["verdicts"][f.key].supported]
        cited = sources_cited_by(supported, state["sources"])
        prompt = SYNTHESIS_USER_TEMPLATE.format(
            question=state["question"],
            sources=format_source_list(cited),
            findings=format_findings(supported, cited),
        )
        messages = [SystemMessage(SYNTHESIS_SYSTEM_PROMPT), HumanMessage(prompt)]
        answer, usage = llm.structured(ModelTier.STRONG, ResearchAnswer, messages)
        return {
            "answer": keep_known_citations(answer, len(cited)),
            "answer_sources": cited,
            "usage": usage,
        }

    def remember(state: RunState) -> dict:
        """Store the newly verified findings in the knowledge graph for later runs."""
        verified = [
            finding
            for finding in state["findings"]
            if state["verdicts"][finding.key].supported and not finding.from_memory
        ]
        titles = {source.url: source.title for source in state["sources"]}
        store.save(verified, titles, clock())
        return {}

    graph = StateGraph(RunState)
    graph.add_node("recall_memory", recall_memory)
    graph.add_node("make_plan", make_plan)
    graph.add_node("review", review)
    graph.add_node("researcher", researcher)
    graph.add_node("critique", critique)
    graph.add_node("synthesize", synthesize)
    graph.add_node("remember", remember)
    graph.add_edge(START, "recall_memory")
    graph.add_edge("recall_memory", "make_plan")
    graph.add_edge("make_plan", "review")
    graph.add_conditional_edges("review", dispatch, ["researcher", "synthesize"])
    graph.add_edge("researcher", "critique")
    graph.add_conditional_edges("critique", dispatch, ["researcher", "synthesize"])
    graph.add_edge("synthesize", "remember")
    graph.add_edge("remember", END)
    return graph.compile(checkpointer=checkpointer)
