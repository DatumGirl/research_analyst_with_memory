"""Model access: tiers, the LLM interface the graph depends on, and cost accounting."""

from enum import StrEnum
from typing import Protocol, TypeVar

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from research_analyst.config import RoutingMode, Settings
from research_analyst.schemas import Difficulty, TokenUsage, UsageByModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)

# Current Claude models reject forced tool choice, which LangChain's default
# structured-output method relies on; json_schema uses native structured outputs.
STRUCTURED_OUTPUT_METHOD = "json_schema"
TOKENS_PER_MTOK = 1_000_000

# (input, output) USD per million tokens, Anthropic first-party rates as of 2026-09.
PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


class ModelTier(StrEnum):
    """The roles a model can play in a run."""

    CHEAP = "cheap"
    STRONG = "strong"
    JUDGE = "judge"


class LLM(Protocol):
    """What the graph needs from a language model provider."""

    def chat(
        self, tier: ModelTier, messages: list[AnyMessage], tools: list[BaseTool]
    ) -> tuple[AIMessage, UsageByModel]:
        """Get the next assistant turn, which may request the given tools."""
        ...

    def structured(
        self, tier: ModelTier, schema: type[SchemaT], messages: list[AnyMessage]
    ) -> tuple[SchemaT, UsageByModel]:
        """Get a response parsed into schema."""
        ...


def usage_of(model_id: str, message: AIMessage) -> UsageByModel:
    """Read the token usage the API reported for one response."""
    reported = message.usage_metadata or {}
    return {
        model_id: TokenUsage(
            input_tokens=reported.get("input_tokens", 0),
            output_tokens=reported.get("output_tokens", 0),
        )
    }


class ClaudeLLM:
    """LLM backed by one Claude chat model per tier."""

    def __init__(self, models: dict[ModelTier, ChatAnthropic]):
        """Store the chat model to use for each tier."""
        self.models = models

    def chat(
        self, tier: ModelTier, messages: list[AnyMessage], tools: list[BaseTool]
    ) -> tuple[AIMessage, UsageByModel]:
        """Get the next assistant turn from the tier's model with tools bound."""
        model = self.models[tier]
        response = model.bind_tools(tools).invoke(messages)
        return response, usage_of(model.model, response)

    def structured(
        self, tier: ModelTier, schema: type[SchemaT], messages: list[AnyMessage]
    ) -> tuple[SchemaT, UsageByModel]:
        """Get a schema-constrained response from the tier's model.

        Raises:
            ValueError: If the response could not be parsed, for example because
                it was cut off at max_tokens or the model declined the request.
        """
        model = self.models[tier]
        constrained = model.with_structured_output(
            schema, method=STRUCTURED_OUTPUT_METHOD, include_raw=True
        )
        result = constrained.invoke(messages)
        raw = result["raw"]
        if result["parsed"] is None:
            stop_reason = raw.response_metadata.get("stop_reason")
            raise ValueError(
                f"{model.model} returned no parseable {schema.__name__} "
                f"(stop_reason={stop_reason}): {result['parsing_error']}"
            )
        return result["parsed"], usage_of(model.model, raw)


def build_chat_model(
    model_id: str, effort: str | None, api_key: str, max_tokens: int
) -> ChatAnthropic:
    """Create a Claude chat model, sending an effort level only when one is given.

    Some models (Claude Haiku 4.5) reject the effort parameter, so it is optional.
    """
    options = {"output_config": {"effort": effort}} if effort else {}
    return ChatAnthropic(model=model_id, api_key=api_key, max_tokens=max_tokens, **options)


def build_llm(settings: Settings) -> ClaudeLLM:
    """Create the Claude-backed LLM with one model per tier."""
    key, limit = settings.anthropic_api_key, settings.max_tokens
    return ClaudeLLM(
        {
            ModelTier.CHEAP: build_chat_model(
                settings.cheap_model, settings.cheap_effort, key, limit
            ),
            ModelTier.STRONG: build_chat_model(
                settings.strong_model, settings.strong_effort, key, limit
            ),
            ModelTier.JUDGE: build_chat_model(
                settings.judge_model, settings.strong_effort, key, limit
            ),
        }
    )


def tier_for(difficulty: Difficulty, routing: RoutingMode) -> ModelTier:
    """Choose the model tier that researches a sub-question of the given difficulty."""
    if routing is RoutingMode.CHEAP:
        return ModelTier.CHEAP
    if routing is RoutingMode.STRONG:
        return ModelTier.STRONG
    return ModelTier.CHEAP if difficulty is Difficulty.EASY else ModelTier.STRONG


def total_tokens(usage: UsageByModel) -> int:
    """Sum input and output tokens across all models."""
    return sum(used["input_tokens"] + used["output_tokens"] for used in usage.values())


def estimate_cost_usd(usage: UsageByModel, prices: dict[str, tuple[float, float]]) -> float | None:
    """Price the usage, or return None if any model in it has no known price."""
    if any(model not in prices for model in usage):
        return None
    return (
        sum(
            used["input_tokens"] * prices[model][0] + used["output_tokens"] * prices[model][1]
            for model, used in usage.items()
        )
        / TOKENS_PER_MTOK
    )
