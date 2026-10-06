"""Runtime settings, read from the environment in one place."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from pathlib import Path

ANTHROPIC_API_KEY_VAR = "ANTHROPIC_API_KEY"
TAVILY_API_KEY_VAR = "TAVILY_API_KEY"
STRONG_MODEL_VAR = "RESEARCH_STRONG_MODEL"
STRONG_EFFORT_VAR = "RESEARCH_STRONG_EFFORT"
CHEAP_MODEL_VAR = "RESEARCH_CHEAP_MODEL"
CHEAP_EFFORT_VAR = "RESEARCH_CHEAP_EFFORT"
JUDGE_MODEL_VAR = "RESEARCH_JUDGE_MODEL"
ROUTING_VAR = "RESEARCH_ROUTING"
MAX_TOKENS_VAR = "RESEARCH_MAX_TOKENS"
MAX_SEARCH_RESULTS_VAR = "RESEARCH_MAX_SEARCH_RESULTS"
MAX_FETCH_CHARS_VAR = "RESEARCH_MAX_FETCH_CHARS"
MAX_TOOL_ROUNDS_VAR = "RESEARCH_MAX_TOOL_ROUNDS"
MAX_SUB_QUESTIONS_VAR = "RESEARCH_MAX_SUB_QUESTIONS"
MAX_RETRY_ROUNDS_VAR = "RESEARCH_MAX_RETRY_ROUNDS"
TOKEN_BUDGET_VAR = "RESEARCH_TOKEN_BUDGET"
MEMORY_MAX_AGE_DAYS_VAR = "RESEARCH_MEMORY_MAX_AGE_DAYS"
CACHE_TTL_HOURS_VAR = "RESEARCH_CACHE_TTL_HOURS"
DATA_DIR_VAR = "RESEARCH_DATA_DIR"
MLFLOW_TRACKING_URI_VAR = "MLFLOW_TRACKING_URI"
MLFLOW_EXPERIMENT_VAR = "RESEARCH_MLFLOW_EXPERIMENT"
REGRESSION_TOLERANCE_VAR = "RESEARCH_REGRESSION_TOLERANCE"

DEFAULT_STRONG_MODEL = "claude-opus-5-5"
DEFAULT_STRONG_EFFORT = "medium"
DEFAULT_CHEAP_MODEL = "claude-haiku-4-5"
DEFAULT_MAX_TOKENS = 16000
DEFAULT_MAX_SEARCH_RESULTS = 5
DEFAULT_MAX_FETCH_CHARS = 20000
DEFAULT_MAX_TOOL_ROUNDS = 6
DEFAULT_MAX_SUB_QUESTIONS = 4
DEFAULT_MAX_RETRY_ROUNDS = 1
DEFAULT_TOKEN_BUDGET = 2_000_000
DEFAULT_MEMORY_MAX_AGE_DAYS = 30
DEFAULT_CACHE_TTL_HOURS = 24
DEFAULT_DATA_DIR = ".research"
DEFAULT_MLFLOW_EXPERIMENT = "research-analyst"
DEFAULT_REGRESSION_TOLERANCE = 0.05
MLFLOW_DB_NAME = "mlflow.db"


class ConfigError(Exception):
    """Raised when required configuration is missing or malformed."""


class RoutingMode(StrEnum):
    """How sub-questions are assigned to models."""

    AUTO = "auto"  # easy sub-questions go to the cheap model, hard ones to the strong model
    CHEAP = "cheap"  # every sub-question goes to the cheap model
    STRONG = "strong"  # every sub-question goes to the strong model


@dataclass(frozen=True)
class Settings:
    """Everything tunable about a research run.

    Attributes:
        anthropic_api_key: Credential for the Claude API.
        tavily_api_key: Credential for the Tavily search and extract APIs.
        strong_model: Model for planning, critique, synthesis, and hard sub-questions.
        strong_effort: Effort level for the strong model.
        cheap_model: Model for easy sub-questions and memory lookups.
        cheap_effort: Effort level for the cheap model; None when it takes no effort setting.
        judge_model: Model that grades answers in the evaluation harness.
        routing: How sub-questions are assigned to the cheap and strong models.
        max_tokens: Per-response output token ceiling.
        max_search_results: Results returned by one search call.
        max_fetch_chars: Page text kept from one fetch before it is cut off.
        max_tool_rounds: Rounds of tool calls one researcher may make.
        max_sub_questions: Sub-questions researched in parallel per round.
        max_retry_rounds: Times the critic may send weak findings back for more research.
        token_budget: Total tokens after which the critic stops requesting more research.
        memory_max_age: Age after which a remembered finding is stale.
        cache_ttl: Age after which a cached search or fetch result is discarded.
        data_dir: Directory holding the memory, cache, and checkpoint databases.
        mlflow_tracking_uri: Where evaluation runs are recorded.
        mlflow_experiment: MLflow experiment name for evaluation runs.
        regression_tolerance: Score drop between evaluation runs that counts as a regression.
    """

    anthropic_api_key: str
    tavily_api_key: str
    strong_model: str
    strong_effort: str
    cheap_model: str
    cheap_effort: str | None
    judge_model: str
    routing: RoutingMode
    max_tokens: int
    max_search_results: int
    max_fetch_chars: int
    max_tool_rounds: int
    max_sub_questions: int
    max_retry_rounds: int
    token_budget: int
    memory_max_age: timedelta
    cache_ttl: timedelta
    data_dir: Path
    mlflow_tracking_uri: str
    mlflow_experiment: str
    regression_tolerance: float


def require(env: Mapping[str, str], name: str) -> str:
    """Return a mandatory environment value.

    Raises:
        ConfigError: If the variable is unset or empty.
    """
    value = env.get(name, "").strip()
    if not value:
        raise ConfigError(f"{name} is not set; add it to your environment or .env file")
    return value


def read_text(env: Mapping[str, str], name: str, default: str) -> str:
    """Return an optional text setting, falling back to its default when unset or blank."""
    return env.get(name, "").strip() or default


def read_int(env: Mapping[str, str], name: str, default: int, minimum: int) -> int:
    """Return an optional integer setting, falling back to its default.

    Raises:
        ConfigError: If the variable is not an integer or is below minimum.
    """
    raw = env.get(name, "").strip()
    if not raw:
        return default
    if not raw.isdigit() or int(raw) < minimum:
        raise ConfigError(f"{name} must be an integer of at least {minimum}, got {raw!r}")
    return int(raw)


def read_fraction(env: Mapping[str, str], name: str, default: float) -> float:
    """Return an optional setting between 0 and 1, falling back to its default.

    Raises:
        ConfigError: If the variable is not a number between 0 and 1.
    """
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a number between 0 and 1, got {raw!r}") from None
    if not 0 <= value <= 1:
        raise ConfigError(f"{name} must be a number between 0 and 1, got {raw!r}")
    return value


def read_routing(env: Mapping[str, str]) -> RoutingMode:
    """Return the routing mode, defaulting to automatic routing.

    Raises:
        ConfigError: If the variable names an unknown mode.
    """
    raw = read_text(env, ROUTING_VAR, RoutingMode.AUTO)
    try:
        return RoutingMode(raw)
    except ValueError:
        modes = ", ".join(RoutingMode)
        raise ConfigError(f"{ROUTING_VAR} must be one of {modes}, got {raw!r}") from None


def load_settings(env: Mapping[str, str]) -> Settings:
    """Build Settings from an environment mapping.

    Raises:
        ConfigError: If a credential is missing or a setting is malformed.
    """
    data_dir = Path(read_text(env, DATA_DIR_VAR, DEFAULT_DATA_DIR))
    strong_model = read_text(env, STRONG_MODEL_VAR, DEFAULT_STRONG_MODEL)
    return Settings(
        anthropic_api_key=require(env, ANTHROPIC_API_KEY_VAR),
        tavily_api_key=require(env, TAVILY_API_KEY_VAR),
        strong_model=strong_model,
        strong_effort=read_text(env, STRONG_EFFORT_VAR, DEFAULT_STRONG_EFFORT),
        cheap_model=read_text(env, CHEAP_MODEL_VAR, DEFAULT_CHEAP_MODEL),
        cheap_effort=env.get(CHEAP_EFFORT_VAR, "").strip() or None,
        judge_model=read_text(env, JUDGE_MODEL_VAR, strong_model),
        routing=read_routing(env),
        max_tokens=read_int(env, MAX_TOKENS_VAR, DEFAULT_MAX_TOKENS, 1),
        max_search_results=read_int(env, MAX_SEARCH_RESULTS_VAR, DEFAULT_MAX_SEARCH_RESULTS, 1),
        max_fetch_chars=read_int(env, MAX_FETCH_CHARS_VAR, DEFAULT_MAX_FETCH_CHARS, 1),
        max_tool_rounds=read_int(env, MAX_TOOL_ROUNDS_VAR, DEFAULT_MAX_TOOL_ROUNDS, 1),
        max_sub_questions=read_int(env, MAX_SUB_QUESTIONS_VAR, DEFAULT_MAX_SUB_QUESTIONS, 1),
        max_retry_rounds=read_int(env, MAX_RETRY_ROUNDS_VAR, DEFAULT_MAX_RETRY_ROUNDS, 0),
        token_budget=read_int(env, TOKEN_BUDGET_VAR, DEFAULT_TOKEN_BUDGET, 1),
        memory_max_age=timedelta(
            days=read_int(env, MEMORY_MAX_AGE_DAYS_VAR, DEFAULT_MEMORY_MAX_AGE_DAYS, 0)
        ),
        cache_ttl=timedelta(hours=read_int(env, CACHE_TTL_HOURS_VAR, DEFAULT_CACHE_TTL_HOURS, 0)),
        data_dir=data_dir,
        mlflow_tracking_uri=read_text(
            env, MLFLOW_TRACKING_URI_VAR, f"sqlite:///{data_dir / MLFLOW_DB_NAME}"
        ),
        mlflow_experiment=read_text(env, MLFLOW_EXPERIMENT_VAR, DEFAULT_MLFLOW_EXPERIMENT),
        regression_tolerance=read_fraction(
            env, REGRESSION_TOLERANCE_VAR, DEFAULT_REGRESSION_TOLERANCE
        ),
    )
