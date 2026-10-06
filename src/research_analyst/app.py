"""Wiring: turn Settings into a runnable research graph backed by real services."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph
from tavily import TavilyClient

from research_analyst.cache import ToolCache
from research_analyst.config import Settings
from research_analyst.graph import build_graph
from research_analyst.llm import LLM
from research_analyst.memory import KnowledgeStore
from research_analyst.researcher import build_researcher
from research_analyst.schemas import CHECKPOINTED_TYPES
from research_analyst.tools import build_fetch_tool, build_search_tool

MEMORY_DB_NAME = "memory.db"
CACHE_DB_NAME = "cache.db"
CHECKPOINT_DB_NAME = "checkpoints.db"

# A researcher round is three steps (research, tools, record_sources), plus a
# final research step and the report. LangGraph counts steps per graph, and the
# run graph needs far fewer than a researcher, so the limit is sized for the latter.
STEPS_PER_TOOL_ROUND = 3
RESEARCHER_CLOSING_STEPS = 2
RUN_STEPS_PER_RETRY_ROUND = 2
RUN_FIXED_STEPS = 7


def utc_now() -> datetime:
    """Return the current time in UTC."""
    return datetime.now(UTC)


def default_memory_path(settings: Settings) -> Path:
    """Return where the knowledge graph lives for normal runs."""
    return settings.data_dir / MEMORY_DB_NAME


def recursion_limit(settings: Settings) -> int:
    """Compute the step limit that lets every configured round complete."""
    researcher_steps = settings.max_tool_rounds * STEPS_PER_TOOL_ROUND + RESEARCHER_CLOSING_STEPS
    run_steps = RUN_FIXED_STEPS + settings.max_retry_rounds * RUN_STEPS_PER_RETRY_ROUND
    return max(researcher_steps, run_steps)


def run_config(settings: Settings, thread_id: str) -> dict:
    """Build the LangGraph config identifying a run's checkpoint thread."""
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit(settings)}


def checkpoint_serializer() -> JsonPlusSerializer:
    """Create a serializer that is allowed to restore this project's state types."""
    allowed = [(kind.__module__, kind.__name__) for kind in CHECKPOINTED_TYPES]
    return JsonPlusSerializer(allowed_msgpack_modules=allowed)


def build_app(settings: Settings, llm: LLM, memory_path: Path) -> CompiledStateGraph:
    """Assemble the research graph with Tavily tools, cache, memory, and checkpoints.

    Args:
        settings: Run configuration.
        llm: Model provider.
        memory_path: Knowledge graph database; the evaluation harness passes a
            throwaway path so cases cannot learn from each other.
    """
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    tavily = TavilyClient(api_key=settings.tavily_api_key)
    cache = ToolCache(settings.data_dir / CACHE_DB_NAME, settings.cache_ttl, utc_now)
    tools = [
        build_search_tool(tavily, settings.max_search_results, cache),
        build_fetch_tool(tavily, settings.max_fetch_chars, cache),
    ]
    # Researchers checkpoint from worker threads, so the connection must be shareable.
    connection = sqlite3.connect(settings.data_dir / CHECKPOINT_DB_NAME, check_same_thread=False)
    return build_graph(
        llm,
        build_researcher(llm, tools, settings),
        KnowledgeStore(memory_path),
        settings,
        utc_now,
        SqliteSaver(connection, serde=checkpoint_serializer()),
    )
