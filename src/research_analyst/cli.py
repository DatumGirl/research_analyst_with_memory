"""Command-line entry point: ask a question, resume a paused run, or run the evaluation."""

import argparse
import dataclasses
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langgraph.types import Command
from pydantic import ValidationError

from research_analyst.app import build_app, default_memory_path, run_config
from research_analyst.config import ConfigError, RoutingMode, Settings, load_settings
from research_analyst.evaluation import (
    CaseResult,
    find_regressions,
    golden_fingerprint,
    load_golden,
    log_run,
    previous_metrics,
    run_case,
    summarize,
)
from research_analyst.graph import initial_state
from research_analyst.llm import PRICES_USD_PER_MTOK, build_llm, estimate_cost_usd
from research_analyst.schemas import ResearchPlan, RunState

DEFAULT_GOLDEN_PATH = "evals/golden.json"
PLANS_DIR_NAME = "plans"
THREAD_ID_LENGTH = 8
INTERRUPT_KEY = "__interrupt__"
EXIT_REGRESSION = 1


def new_thread_id() -> str:
    """Generate a short identifier for a run's checkpoint thread."""
    return uuid.uuid4().hex[:THREAD_ID_LENGTH]


def plan_path(settings: Settings, thread_id: str) -> Path:
    """Return where a paused run's editable plan is written."""
    return settings.data_dir / PLANS_DIR_NAME / f"{thread_id}.json"


def format_report(state: RunState) -> str:
    """Render the answer, the sources it cites with their quotes, and the run's cost."""
    answer, sources = state["answer"], state["answer_sources"]
    lines = [answer.answer, "", "Sources"]
    for citation in sorted(answer.citations, key=lambda c: c.source_number):
        source = sources[citation.source_number - 1]
        lines.append(f"[{citation.source_number}] {source.title} - {source.url}")
        lines.append(f'    "{citation.quote}"')
    cost = estimate_cost_usd(state["usage"], PRICES_USD_PER_MTOK)
    tokens = ", ".join(
        f"{model}: {used['input_tokens']} in / {used['output_tokens']} out"
        for model, used in sorted(state["usage"].items())
    )
    cost_text = f"${cost:.4f}" if cost is not None else "unknown (no price for a model used)"
    lines += ["", f"Cost: {cost_text} ({tokens})"]
    return "\n".join(lines)


def format_eval_summary(
    results: list[CaseResult], metrics: dict[str, float], regressions: list[str]
) -> str:
    """Render per-case scores, the averages, and any regressions against the last run."""
    lines = []
    for result in results:
        if result.error:
            lines.append(f"{result.case_id}: ERROR {result.error}")
            continue
        lines.append(
            f"{result.case_id}: groundedness {result.groundedness:.2f}, "
            f"coverage {result.coverage:.2f}"
        )
    lines.append("")
    lines += [f"{name}: {value:.4f}" for name, value in sorted(metrics.items())]
    lines.append("")
    lines += [f"REGRESSION: {text}" for text in regressions] or ["No regressions."]
    return "\n".join(lines)


def pause_for_review(settings: Settings, thread_id: str, state: dict) -> str:
    """Write the interrupted run's plan to disk and return instructions for resuming."""
    path = plan_path(settings, thread_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state[INTERRUPT_KEY][0].value["plan"], indent=2))
    return (
        f"Run {thread_id} is paused for plan review.\n"
        f"Edit {path}, then continue with:\n"
        f"  research-analyst resume {thread_id}"
    )


def ask(settings: Settings, args: argparse.Namespace) -> None:
    """Research a question; with --review, stop after planning so the plan can be edited."""
    graph = build_app(settings, build_llm(settings), default_memory_path(settings))
    thread_id = new_thread_id()
    state = graph.invoke(initial_state(args.question, args.review), run_config(settings, thread_id))
    if INTERRUPT_KEY in state:
        print(pause_for_review(settings, thread_id, state))
        return
    print(format_report(state))


def resume(settings: Settings, args: argparse.Namespace) -> None:
    """Continue a paused run with the plan as edited on disk."""
    path = Path(args.plan) if args.plan else plan_path(settings, args.thread_id)
    if not path.exists():
        sys.exit(f"No plan file at {path}; pass --plan or check the run id.")
    try:
        plan = ResearchPlan.model_validate_json(path.read_text())
    except ValidationError as error:
        sys.exit(f"The plan in {path} is not valid:\n{error}")

    graph = build_app(settings, build_llm(settings), default_memory_path(settings))
    config = run_config(settings, args.thread_id)
    if not graph.get_state(config).next:
        sys.exit(f"Run {args.thread_id} has nothing to resume.")
    print(format_report(graph.invoke(Command(resume=plan.model_dump()), config)))


def evaluate(settings: Settings, args: argparse.Namespace) -> None:
    """Run the golden set, log the results to MLflow, and exit non-zero on a regression."""
    golden_path = Path(args.golden)
    cases = load_golden(golden_path)
    llm = build_llm(settings)
    comparison_params = {
        "routing": settings.routing.value,
        "golden_set": golden_fingerprint(golden_path),
    }
    params = {
        **comparison_params,
        "strong_model": settings.strong_model,
        "cheap_model": settings.cheap_model,
        "judge_model": settings.judge_model,
    }
    previous = previous_metrics(settings, comparison_params)

    results = []
    # Each case gets an empty knowledge graph so it cannot reuse another case's findings.
    with tempfile.TemporaryDirectory() as scratch:
        for case in cases:
            graph = build_app(settings, llm, Path(scratch) / f"{case.id}.db")
            config = run_config(settings, new_thread_id())
            results.append(
                run_case(
                    case,
                    lambda q, graph=graph, config=config: graph.invoke(
                        initial_state(q, False), config
                    ),
                    llm,
                )
            )

    metrics = summarize(results)
    regressions = find_regressions(metrics, previous, settings.regression_tolerance)
    log_run(settings, params, metrics, results, regressions)
    print(format_eval_summary(results, metrics, regressions))
    if regressions:
        sys.exit(EXIT_REGRESSION)


def build_parser() -> argparse.ArgumentParser:
    """Define the ask, resume, and eval commands."""
    parser = argparse.ArgumentParser(description="Research questions and answer with citations.")
    parser.add_argument(
        "--routing",
        choices=list(RoutingMode),
        type=RoutingMode,
        help="Override how sub-questions are assigned to the cheap and strong models.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ask_parser = commands.add_parser("ask", help="Research a question.")
    ask_parser.add_argument("question", help="The question to research.")
    ask_parser.add_argument(
        "--review", action="store_true", help="Pause after planning so the plan can be edited."
    )
    ask_parser.set_defaults(handler=ask)

    resume_parser = commands.add_parser("resume", help="Continue a run paused for plan review.")
    resume_parser.add_argument("thread_id", help="The run id printed when the run paused.")
    resume_parser.add_argument("--plan", help="Plan file to use instead of the saved one.")
    resume_parser.set_defaults(handler=resume)

    eval_parser = commands.add_parser("eval", help="Run the golden set and track regressions.")
    eval_parser.add_argument("--golden", default=DEFAULT_GOLDEN_PATH, help="Golden set file.")
    eval_parser.set_defaults(handler=evaluate)
    return parser


def main() -> None:
    """Parse arguments, load settings, and run the chosen command."""
    args = build_parser().parse_args()
    load_dotenv()
    try:
        settings = load_settings(os.environ)
    except ConfigError as error:
        sys.exit(f"Configuration error: {error}")
    if args.routing:
        settings = dataclasses.replace(settings, routing=args.routing)
    args.handler(settings, args)


if __name__ == "__main__":
    main()
