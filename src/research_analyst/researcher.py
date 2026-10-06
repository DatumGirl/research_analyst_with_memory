"""The researcher subgraph: one worker investigating one sub-question.

    START -> research -> tools -> record_sources -> research -> ... -> report -> END

The run fans out to one of these per sub-question. Each keeps its own
conversation and hands back only findings, sources, and token usage.
"""

from typing import Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from tavily.errors import BadRequestError
from tavily.errors import TimeoutError as TavilyTimeoutError

from research_analyst.config import Settings
from research_analyst.llm import LLM, tier_for
from research_analyst.prompts import (
    REPORT_SYSTEM_PROMPT,
    REPORT_USER_TEMPLATE,
    RESEARCH_SYSTEM_PROMPT,
    format_evidence,
)
from research_analyst.schemas import (
    Finding,
    FindingsReport,
    ResearcherOutput,
    ResearcherState,
    Source,
    SubQuestion,
)

# Failures the model can route around by rephrasing or picking another URL.
# Anything else (bad API key, exhausted quota) should stop the run.
RECOVERABLE_TOOL_ERRORS = (BadRequestError, TavilyTimeoutError)


def researcher_input(sub_question: SubQuestion) -> dict:
    """Build the starting state for a researcher assigned a sub-question."""
    return {
        "sub_question": sub_question,
        "messages": [HumanMessage(sub_question.question)],
        "sources": [],
        "tool_rounds": 0,
    }


def collect_new_sources(messages: list[AnyMessage]) -> list[Source]:
    """Gather the sources attached to the most recent round of tool results."""
    sources: list[Source] = []
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            break
        # Artifact is None when the tool call failed and ToolNode reported the error.
        sources = list(message.artifact or []) + sources
    return sources


def latest_notes(messages: list[AnyMessage]) -> str:
    """Return the text of the researcher's last turn, which holds its working notes."""
    last = messages[-1]
    return last.text if isinstance(last, AIMessage) else ""


def to_findings(report: FindingsReport, sources: list[Source], question: str) -> list[Finding]:
    """Resolve reported source numbers to URLs, dropping findings that cite no real source."""
    return [
        Finding(
            statement=reported.statement,
            source_url=sources[reported.source_number - 1].url,
            quote=reported.quote,
            entities=reported.entities,
            sub_question=question,
            from_memory=False,
        )
        for reported in report.findings
        if 1 <= reported.source_number <= len(sources)
    ]


def build_researcher(llm: LLM, tools: list[BaseTool], settings: Settings) -> CompiledStateGraph:
    """Compile the researcher subgraph.

    Args:
        llm: Model provider; the tier used depends on the sub-question's difficulty.
        tools: The tools the researcher may call.
        settings: Supplies the routing mode and the tool round limit.
    """

    def research(state: ResearcherState) -> dict:
        """Let the researcher decide its next step: more tool calls, or working notes."""
        tier = tier_for(state["sub_question"].difficulty, settings.routing)
        prompt = [SystemMessage(RESEARCH_SYSTEM_PROMPT), *state["messages"]]
        response, usage = llm.chat(tier, prompt, tools)
        return {"messages": [response], "usage": usage}

    def route_after_research(state: ResearcherState) -> Literal["tools", "report"]:
        """Run requested tools while budget remains; otherwise move on to the report."""
        wants_tools = bool(state["messages"][-1].tool_calls)
        has_budget = state["tool_rounds"] < settings.max_tool_rounds
        return "tools" if wants_tools and has_budget else "report"

    def record_sources(state: ResearcherState) -> dict:
        """Add the round's tool artifacts to the evidence and count the round."""
        return {
            "sources": collect_new_sources(state["messages"]),
            "tool_rounds": state["tool_rounds"] + 1,
        }

    def report(state: ResearcherState) -> dict:
        """Turn the evidence and notes into findings, each tied to a source and a quote."""
        question = state["sub_question"].question
        prompt = REPORT_USER_TEMPLATE.format(
            question=question,
            evidence=format_evidence(state["sources"]),
            notes=latest_notes(state["messages"]),
        )
        tier = tier_for(state["sub_question"].difficulty, settings.routing)
        messages = [SystemMessage(REPORT_SYSTEM_PROMPT), HumanMessage(prompt)]
        reported, usage = llm.structured(tier, FindingsReport, messages)
        return {"findings": to_findings(reported, state["sources"], question), "usage": usage}

    graph = StateGraph(ResearcherState, output_schema=ResearcherOutput)
    graph.add_node("research", research)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=RECOVERABLE_TOOL_ERRORS))
    graph.add_node("record_sources", record_sources)
    graph.add_node("report", report)
    graph.add_edge(START, "research")
    graph.add_conditional_edges("research", route_after_research)
    graph.add_edge("tools", "record_sources")
    graph.add_edge("record_sources", "research")
    graph.add_edge("report", END)
    return graph.compile()
