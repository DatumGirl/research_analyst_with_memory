# Research analyst with memory

A research agent built with LangGraph and Claude. It plans a question into
sub-questions, researches them in parallel, checks every claim against its
source, answers with citations, and remembers what it verified for next time.

## Setup

```sh
uv sync
cp .env.example .env   # then fill in ANTHROPIC_API_KEY and TAVILY_API_KEY
```

## Use

```sh
# Research a question
uv run research-analyst ask "What changed in the EU AI Act's obligations for general-purpose models in 2025?"

# Pause after planning, edit the plan, and resume
uv run research-analyst ask --review "Compare the CAP theorem with the PACELC theorem"
#   -> writes .research/plans/<run-id>.json and prints the run id
uv run research-analyst resume <run-id>

# Run the evaluation and record it in MLflow
uv run research-analyst eval
uv run research-analyst --routing cheap eval     # same set, cheap model only
uv run mlflow ui --backend-store-uri sqlite:///.research/mlflow.db
```

`ask` prints the answer with `[n]` markers, each cited source with its
supporting quote, and the run's token usage and cost.

## How a run works

```
recall_memory -> make_plan -> review -+-> researcher (xN) -> critique -+
                                      |        ^                       |
                                      |        +--- weak findings -----+
                                      +-----------------> synthesize <-+
                                                              |
                                                          remember
```

| Step | What it does |
|---|---|
| `recall_memory` | Matches the question to entities in the knowledge graph and loads the findings recorded about them. Fresh findings become evidence; stale ones become leads to re-verify. |
| `make_plan` | Splits the question into self-contained sub-questions, each labelled easy or hard, skipping what memory already establishes. |
| `review` | With `--review`, interrupts the run so the plan can be edited. The run's state is checkpointed, so it resumes in a later process. |
| `researcher` | One subgraph per sub-question, run in parallel. Each loops over `search` and `fetch`, then reports findings: a claim, its source, and a verbatim quote. |
| `critique` | Checks each new finding against its source text. Unsupported findings become follow-up sub-questions and go back to the researchers. |
| `synthesize` | Writes the cited answer from the findings the critic accepted. |
| `remember` | Stores the newly verified findings in the knowledge graph. |

## The seven stages

| Stage | Idea | Where to look |
|---|---|---|
| 1. Single agent with tools | Tool design, state schemas, structured output | `tools.py`, `researcher.py`, `schemas.py` |
| 2. Planner and parallel researchers | Subgraphs, `Send` fan-out, reducers that merge parallel results | `make_plan` and `dispatch` in `graph.py`; reducers in `schemas.py` |
| 3. Critic loop | Cycles, stop conditions, budgets | `critique` in `graph.py` |
| 4. Persistent memory | Knowledge graph, entity deduplication, staleness | `memory.py`; `recall_memory` and `remember` in `graph.py` |
| 5. Human-in-the-loop | Checkpointing and interrupts | `review` in `graph.py`; `resume` in `cli.py` |
| 6. Evaluation harness | Golden set, LLM judges, regression tracking | `evaluation.py`, `evals/golden.json` |
| 7. Cost-aware routing | Model tiers, caching, cost accounting | `llm.py`, `cache.py` |

### Stop conditions and budgets

The critic loop ends when any of these holds:

- every finding is supported;
- `RESEARCH_MAX_RETRY_ROUNDS` rounds of follow-up research have run;
- the run has used `RESEARCH_TOKEN_BUDGET` tokens.

Findings still unsupported at that point are left out of the answer. Each
researcher is separately limited to `RESEARCH_MAX_TOOL_ROUNDS` rounds of tool calls.

### Memory

The knowledge graph is a SQLite file with entities and findings as nodes and
"mentions" as edges.

- **Deduplication.** Entity names that differ only in case, punctuation, or a
  leading "the" share one node. Identical claims from the same source are stored once.
- **Staleness.** A finding older than `RESEARCH_MEMORY_MAX_AGE_DAYS` is not used
  as evidence. It is handed to the planner as a lead, and saving it again after
  re-verification refreshes its timestamp.

### Routing and caching

With `auto` routing, easy sub-questions go to the cheap model and hard ones to
the strong model. Follow-ups from the critic always count as hard. Planning,
critique, and synthesis always use the strong model. Search and fetch results
are cached for `RESEARCH_CACHE_TTL_HOURS`.

To measure the trade-off, run `eval` once per routing mode and compare
`groundedness_mean`, `coverage_mean`, and `cost_usd_mean` in MLflow.

### Evaluation

Each golden case is a question with the points a good answer must make. Two
judges score each answer:

- **groundedness**: the share of the answer's factual claims its cited sources support;
- **coverage**: the share of the expected points the answer conveys.

A case that fails to run is recorded as an error and left out of the averages.
Each case runs with an empty knowledge graph, so cases cannot learn from each
other. A run is compared with the latest earlier run that used the same golden
set and routing mode; a drop of more than `RESEARCH_REGRESSION_TOLERANCE` in
either score is reported and makes the command exit non-zero.

`evals/golden.json` is a five-case starter set. Its limits:

- four cases are well-known facts a model could answer from memory, so coverage
  on them says little about the research itself;
- five cases at one run each is too few to tell small differences from noise;
- the judge defaults to the same model that writes the answers.

Replace it with questions from your own domain before relying on the numbers.

## Data

Everything the agent stores lives under `.research/` (git-ignored):

| File | Contents |
|---|---|
| `memory.db` | The knowledge graph |
| `cache.db` | Cached search and fetch results |
| `checkpoints.db` | Run state, for pause and resume |
| `plans/` | Plans awaiting review |
| `mlflow.db` | Evaluation runs |

MLflow writes per-case result files to `mlruns/`, also git-ignored.

## Tests

```sh
uv run pytest
```

Tests use scripted stand-ins for Claude and Tavily, so they need no API keys
and make no network calls.
