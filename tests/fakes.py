"""Scripted stand-ins for Claude and Tavily, shared by the tests."""

import dataclasses
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel

from research_analyst.config import Settings, load_settings
from research_analyst.llm import ModelTier
from research_analyst.schemas import Source, TokenUsage, UsageByModel

KEYS = {"ANTHROPIC_API_KEY": "a", "TAVILY_API_KEY": "t"}
START_TIME = datetime(2026, 1, 1, tzinfo=UTC)
TOKENS_PER_CALL = TokenUsage(input_tokens=10, output_tokens=5)


def make_settings(data_dir: Path, **overrides) -> Settings:
    """Build Settings rooted at a temporary directory, with selected fields replaced."""
    return dataclasses.replace(load_settings(KEYS), data_dir=data_dir, **overrides)


class Clock:
    """A clock the test moves by hand."""

    def __init__(self):
        """Start at a fixed moment."""
        self.now = START_TIME

    def __call__(self) -> datetime:
        """Return the current fake time."""
        return self.now


def page(name: str) -> Source:
    """A canned web page whose URL, title, and text derive from name."""
    return Source(url=f"https://example.com/{name}", title=f"Page {name}", content=f"text {name}")


@tool("search", response_format="content_and_artifact")
def fake_search(query: str) -> tuple[str, list[Source]]:
    """Return one canned page named after the query."""
    source = page(query)
    return source.content, [source]


def search_call(query: str) -> AIMessage:
    """A researcher turn that requests one search."""
    return AIMessage("", tool_calls=[{"name": "search", "args": {"query": query}, "id": query}])


class FakeLLM:
    """An LLM that replays scripted chat turns and computes structured replies.

    Attributes:
        chat_tiers: The tier used for each researcher, keyed by its sub-question.
        prompts: The final prompt text of each structured call, keyed by schema.
    """

    def __init__(
        self,
        chat_turns: dict[str, list[AIMessage]],
        replies: dict[type, Callable[[str], BaseModel]],
    ):
        """Store the script.

        Args:
            chat_turns: Researcher turns to replay, keyed by sub-question text.
            replies: For each schema, a function from the prompt text to the reply.
        """
        self.chat_turns = {question: iter(turns) for question, turns in chat_turns.items()}
        self.replies = replies
        self.chat_tiers: dict[str, ModelTier] = {}
        self.prompts: dict[type, list[str]] = {}

    def chat(
        self, tier: ModelTier, messages: list[AnyMessage], tools: list[BaseTool]
    ) -> tuple[AIMessage, UsageByModel]:
        """Replay the next scripted turn for the sub-question in the conversation."""
        question = next(m.content for m in messages if isinstance(m, HumanMessage))
        self.chat_tiers[question] = tier
        return next(self.chat_turns[question]), {f"fake-{tier}": TOKENS_PER_CALL}

    def structured(
        self, tier: ModelTier, schema: type, messages: list[AnyMessage]
    ) -> tuple[BaseModel, UsageByModel]:
        """Record the prompt and return the scripted reply for the schema."""
        prompt = messages[-1].content
        self.prompts.setdefault(schema, []).append(prompt)
        return self.replies[schema](prompt), {f"fake-{tier}": TOKENS_PER_CALL}
