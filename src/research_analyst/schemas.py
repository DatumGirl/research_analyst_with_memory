"""Data contracts: plans, evidence, findings, verdicts, answers, and graph state."""

import hashlib
from enum import StrEnum
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

FINDING_KEY_LENGTH = 16


class Difficulty(StrEnum):
    """How demanding a sub-question is; drives model routing."""

    EASY = "easy"
    HARD = "hard"


class SubQuestion(BaseModel):
    """One independently researchable piece of the main question."""

    question: str = Field(description="A self-contained question a researcher can answer alone.")
    difficulty: Difficulty = Field(
        description=(
            "easy: a lookup of a well-documented fact. hard: needs comparison of sources, "
            "synthesis, or judgment about contested or technical material."
        )
    )


class ResearchPlan(BaseModel):
    """The planner's breakdown of the main question."""

    sub_questions: list[SubQuestion]


class Source(BaseModel):
    """One piece of evidence gathered by a tool or recalled from memory."""

    url: str
    title: str
    content: str


class ReportedFinding(BaseModel):
    """A claim as a researcher reports it, citing its own numbered evidence."""

    statement: str = Field(description="One factual claim, understandable on its own.")
    source_number: int = Field(description="The number of the source that supports the claim.")
    quote: str = Field(description="A short verbatim passage from that source backing the claim.")
    entities: list[str] = Field(
        description="The people, organizations, products, or concepts the claim is about, "
        "each by its most common full name."
    )


class FindingsReport(BaseModel):
    """Everything one researcher established for its sub-question."""

    findings: list[ReportedFinding]


class Finding(BaseModel):
    """A claim tied to the evidence that supports it."""

    statement: str
    source_url: str
    quote: str
    entities: list[str]
    sub_question: str
    from_memory: bool

    @property
    def key(self) -> str:
        """Stable identity for the claim-and-source pair, used to deduplicate."""
        text = f"{' '.join(self.statement.casefold().split())}|{self.source_url}"
        return hashlib.sha256(text.encode()).hexdigest()[:FINDING_KEY_LENGTH]


class Verdict(BaseModel):
    """The critic's judgment of one finding."""

    supported: bool
    reason: str


class ReviewedFinding(BaseModel):
    """The critic's judgment of one numbered finding, as it reports it."""

    finding_number: int = Field(description="The number of the finding being judged.")
    supported: bool = Field(
        description="True only if the source text itself states or directly implies the claim."
    )
    reason: str = Field(description="One sentence explaining the judgment.")


class Critique(BaseModel):
    """The critic's judgments for a batch of findings."""

    reviews: list[ReviewedFinding]


class EntityMatches(BaseModel):
    """Known entities relevant to a question."""

    entities: list[str] = Field(
        description="Names copied exactly from the known-entities list. Empty if none apply."
    )


class Citation(BaseModel):
    """A link from the answer to the evidence that supports it."""

    source_number: int = Field(description="The number of the source, as listed in the evidence.")
    quote: str = Field(description="A short verbatim passage from that source backing the claim.")


class ResearchAnswer(BaseModel):
    """The final structured answer to a research question."""

    answer: str = Field(
        description="The answer in Markdown. Every factual claim ends with a [n] source marker."
    )
    citations: list[Citation] = Field(description="One entry per source the answer relies on.")


# Every custom type that can appear in RunState; the checkpoint serializer may restore these.
CHECKPOINTED_TYPES = (
    Difficulty,
    SubQuestion,
    ResearchPlan,
    Source,
    Finding,
    Verdict,
    Citation,
    ResearchAnswer,
)


class TokenUsage(TypedDict):
    """Tokens consumed by one model."""

    input_tokens: int
    output_tokens: int


UsageByModel = dict[str, TokenUsage]


def merge_sources(existing: list[Source], incoming: list[Source]) -> list[Source]:
    """Reducer that appends new sources and deduplicates by URL.

    A URL keeps its first position, so source order stays stable across rounds.
    When the same URL arrives again, the longer content wins: a fetched page
    replaces the search snippet that led to it.
    """
    merged = {source.url: source for source in existing}
    for source in incoming:
        known = merged.get(source.url)
        if known is None or len(source.content) > len(known.content):
            merged[source.url] = source
    return list(merged.values())


def merge_findings(existing: list[Finding], incoming: list[Finding]) -> list[Finding]:
    """Reducer that appends findings from parallel researchers, keeping the first of each key."""
    merged = {finding.key: finding for finding in existing}
    for finding in incoming:
        merged.setdefault(finding.key, finding)
    return list(merged.values())


def merge_verdicts(
    existing: dict[str, Verdict], incoming: dict[str, Verdict]
) -> dict[str, Verdict]:
    """Reducer that adds new verdicts, keyed by finding key."""
    return {**existing, **incoming}


def add_usage(existing: UsageByModel, incoming: UsageByModel) -> UsageByModel:
    """Reducer that sums token usage per model across nodes and parallel researchers."""
    total = dict(existing)
    for model, used in incoming.items():
        before = total.get(model, TokenUsage(input_tokens=0, output_tokens=0))
        total[model] = TokenUsage(
            input_tokens=before["input_tokens"] + used["input_tokens"],
            output_tokens=before["output_tokens"] + used["output_tokens"],
        )
    return total


class RunState(TypedDict):
    """State of one research run.

    Attributes:
        question: The research question being answered.
        wants_review: Whether to pause for a human to edit the plan.
        stale_leads: Remembered claims too old to trust, offered to the planner to re-verify.
        plan: The planner's sub-questions, after any human edits.
        pending: Sub-questions waiting to be researched in the next round.
        findings: Claims gathered so far, from researchers and from memory.
        verdicts: The critic's judgment per finding key.
        sources: Evidence gathered so far.
        retry_rounds: Times the critic has sent findings back for more research.
        usage: Tokens consumed per model.
        answer: The structured answer, set by the synthesis node.
        answer_sources: The sources the answer's citation numbers refer to, in order.
    """

    question: str
    wants_review: bool
    stale_leads: list[str]
    plan: ResearchPlan | None
    pending: list[SubQuestion]
    findings: Annotated[list[Finding], merge_findings]
    verdicts: Annotated[dict[str, Verdict], merge_verdicts]
    sources: Annotated[list[Source], merge_sources]
    retry_rounds: int
    usage: Annotated[UsageByModel, add_usage]
    answer: ResearchAnswer | None
    answer_sources: list[Source]


class ResearcherState(TypedDict):
    """State of one researcher working on one sub-question.

    Attributes:
        sub_question: The sub-question assigned to this researcher.
        messages: The researcher's conversation, including tool calls and results.
        sources: Evidence this researcher gathered, numbered by position (1-based).
        tool_rounds: How many rounds of tool calls have completed.
        findings: The claims this researcher reports.
        usage: Tokens this researcher consumed per model.
    """

    sub_question: SubQuestion
    messages: Annotated[list[AnyMessage], add_messages]
    sources: Annotated[list[Source], merge_sources]
    tool_rounds: int
    findings: list[Finding]
    usage: Annotated[UsageByModel, add_usage]


class ResearcherOutput(TypedDict):
    """What a researcher hands back to the run; its conversation stays private."""

    findings: list[Finding]
    sources: list[Source]
    usage: UsageByModel
