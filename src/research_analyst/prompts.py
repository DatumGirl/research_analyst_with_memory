"""Prompt text for every model call, and the helpers that fill it in."""

from research_analyst.schemas import Finding, Source

NONE_PLACEHOLDER = "None."

RECALL_SYSTEM_PROMPT = """\
You connect a research question to a knowledge base. Given the question and the \
names of entities the knowledge base already holds findings about, pick the \
entities the question is about. Copy names exactly as listed. Pick none if no \
listed entity is relevant.\
"""

RECALL_USER_TEMPLATE = """\
<question>
{question}
</question>

<known_entities>
{entities}
</known_entities>\
"""

PLAN_SYSTEM_PROMPT = """\
You plan research. Break the user's question into sub-questions that separate \
researchers will investigate in parallel, each with web search. Researchers do \
not see each other's work or the original question, so each sub-question must \
be self-contained: name the specific subject rather than saying "it" or "the company".

Known findings come from earlier research and are already verified, so do not \
plan sub-questions for what they establish. Stale leads are earlier findings \
that are now too old to trust; plan to re-verify the ones that matter to this \
question. If the known findings fully answer the question, return no sub-questions.

Label each sub-question easy or hard. Easy sub-questions go to a faster, \
cheaper model, so reserve hard for work that needs comparison of sources, \
synthesis, or judgment about contested or technical material.\
"""

PLAN_USER_TEMPLATE = """\
<question>
{question}
</question>

<known_findings>
{known_findings}
</known_findings>

<stale_leads>
{stale_leads}
</stale_leads>

Plan at most {max_sub_questions} sub-questions.\
"""

RESEARCH_SYSTEM_PROMPT = """\
You are a research analyst. Your job is to gather evidence that answers the \
user's question, using the search and fetch tools.

Search snippets are previews and are often out of context, so fetch the pages \
you intend to rely on. Prefer primary and authoritative sources, and look for a \
second source when a claim is surprising or contested. Independent tool calls \
can be issued together in one turn.

When you have enough evidence, stop calling tools and write short working notes: \
what you found, which sources support it, and anything that remains uncertain. \
A separate step turns your notes and the evidence into reported findings.\
"""

REPORT_SYSTEM_PROMPT = """\
You report what a research session established. From the numbered evidence and \
the analyst's notes, list the findings that answer the question.

Each finding is one factual claim that stands on its own, tied to the single \
source that best supports it and a short passage quoted verbatim from that \
source. A reviewer will check every claim against its source text, so report \
only what the evidence actually says. If the evidence does not answer the \
question, return no findings.\
"""

REPORT_USER_TEMPLATE = """\
<question>
{question}
</question>

<evidence>
{evidence}
</evidence>

<analyst_notes>
{notes}
</analyst_notes>\
"""

CRITIC_SYSTEM_PROMPT = """\
You review research findings before they are used in an answer. For each \
numbered finding, decide whether its source supports the claim.

A finding is supported only when the source text states the claim or directly \
implies it. It is not supported when the source is silent on the claim, says \
something weaker or different, contradicts it, or when the quote does not \
appear in the source. Judge only from the source text provided, not from what \
you know to be true. Give a verdict for every finding.\
"""

CRITIC_USER_TEMPLATE = """\
<sources>
{sources}
</sources>

<findings>
{findings}
</findings>\
"""

FOLLOW_UP_TEMPLATE = (
    "Find reliable evidence that confirms or refutes this claim: {statement} "
    "(Earlier evidence was rejected because: {reason})"
)

SYNTHESIS_SYSTEM_PROMPT = """\
You write the final answer to a research question from verified findings.

Use only the findings provided. Each factual claim in the answer ends with a \
marker such as [2] naming the source that supports it, and each source you rely \
on gets a citation entry with the quote that backs the claim. If the findings \
do not answer part of the question, say so plainly rather than filling the gap \
from memory.\
"""

SYNTHESIS_USER_TEMPLATE = """\
<question>
{question}
</question>

<sources>
{sources}
</sources>

<findings>
{findings}
</findings>

Write the final answer to the question.\
"""


def bullet_list(items: list[str]) -> str:
    """Render items as a dashed list, or a placeholder when there are none."""
    return "\n".join(f"- {item}" for item in items) or NONE_PLACEHOLDER


def format_evidence(sources: list[Source]) -> str:
    """Render sources with their full text as numbered blocks."""
    return (
        "\n".join(
            f'<source number="{number}" url="{source.url}" title="{source.title}">\n'
            f"{source.content}\n</source>"
            for number, source in enumerate(sources, start=1)
        )
        or NONE_PLACEHOLDER
    )


def format_source_list(sources: list[Source]) -> str:
    """Render sources as a numbered list of titles and URLs, without their text."""
    return (
        "\n".join(
            f"[{number}] {source.title} ({source.url})"
            for number, source in enumerate(sources, start=1)
        )
        or NONE_PLACEHOLDER
    )


def format_findings(findings: list[Finding], sources: list[Source]) -> str:
    """Render findings as numbered blocks, each naming its source by number."""
    number_by_url = {source.url: number for number, source in enumerate(sources, start=1)}
    return (
        "\n".join(
            f'<finding number="{number}" source_number="{number_by_url[finding.source_url]}">\n'
            f"Claim: {finding.statement}\nQuote: {finding.quote}\n</finding>"
            for number, finding in enumerate(findings, start=1)
        )
        or NONE_PLACEHOLDER
    )
