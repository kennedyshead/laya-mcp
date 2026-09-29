# Laya MCP

Local, advisory routing decisions for coding agents, powered by
[Laya](https://huggingface.co/convaiinnovations/laya) without generating text or
sending task data to a cloud model. The server chooses a local MLX, PyTorch, or
ONNX Runtime adapter without changing its MCP contract.

The server starts and serves `catalog` without importing any model runtime. The
first routing request lazily loads the selected backend and, when needed,
downloads its checkpoint. Later requests reuse the loaded local model.

## Backends and Installation

`LAYA_MCP_BACKEND` selects the inference runtime. Its default is `auto`.

| Host | `auto` selects | Default model |
| --- | --- | --- |
| Apple Silicon macOS | `mlx` | `aac6fef/laya-multilingual-mlx` |
| Linux | `torch` | `convaiinnovations/laya` |

Install only the backend the host needs:

```shell
# Apple Silicon macOS
uv sync --extra mlx

# Linux CPU
uv sync --extra torch

# Linux CUDA, when a self-hosted runner or deployment needs a CUDA wheel
uv sync --extra torch-cuda
```

For a published package, use `pip install "laya-mcp[mlx]"` or
`pip install "laya-mcp[torch]"`. For a CPU-only Linux pip environment, install
PyTorch from `https://download.pytorch.org/whl/cpu` first, then install the
Laya MCP Torch extra. Missing optional dependencies do not stop the MCP server
from starting; the first inference instead reports the matching `uv sync --extra
...` command.

Set `LAYA_MCP_BACKEND` to `auto`, `mlx`, `torch`, or `onnx` to choose a runtime
explicitly:

```shell
LAYA_MCP_BACKEND=torch uv run laya-mcp
```

| Variable | Purpose |
| --- | --- |
| `LAYA_MCP_BACKEND` | `auto`, `mlx`, `torch`, or `onnx`; `auto` uses the platform defaults above. |
| `LAYA_MCP_MODEL` | Overrides the MLX or Torch model ID. It is also the ONNX original-model fallback. |
| `LAYA_MCP_ONNX_MODEL` | Original upstream Laya checkpoint used by ONNX for tokenizer and configuration. Defaults to the Torch model ID. |
| `LAYA_MCP_ONNX_ARTIFACT_PATH` | Required path to an exported `.onnx` artifact when `LAYA_MCP_BACKEND=onnx`. |

The ONNX adapter uses upstream Laya's `ONNXAgent` and requires both the original
checkpoint and an explicit graph path. It never converts a checkpoint on demand:

```shell
uv sync --extra onnx
export LAYA_MCP_BACKEND=onnx
export LAYA_MCP_ONNX_MODEL=convaiinnovations/laya
export LAYA_MCP_ONNX_ARTIFACT_PATH=/models/laya.onnx
uv run laya-mcp
```

Generate or obtain the graph separately with upstream Laya's
`scripts/export_onnx.py`, using the same original checkpoint. This keeps graph
generation out of MCP startup and makes the deployed artifact explicit.

### Downloads and Hardware

MLX downloads the converted MLX checkpoint on its first prediction. Torch
downloads the original upstream checkpoint on its first prediction. ONNX uses
the supplied graph and may download only the original checkpoint's tokenizer and
configuration the first time. Hugging Face caches downloaded artifacts, so later
starts reuse them while the cache remains available.

`laya` is a 422M-parameter English model. Reserve several hundred MB for
model artifacts and several GB of working memory; exact use depends on backend,
precision, batch size, and device. MLX is the best default for Apple Silicon.
Torch is the portable Linux default and works on CPU, but cold loading and single
request latency are materially slower than Apple GPU or CUDA inference. A
CUDA-enabled PyTorch installation is automatically usable by the upstream Torch
runtime; use `uv sync --extra torch-cuda` for the project-managed CUDA wheel.
ONNX is intended for deployments that manage a CPU-optimized exported artifact
and want its path under explicit operational control.

`catalog` exposes the configured backend before a model is loaded:

```json
{
  "backend": {
    "requested": "auto",
    "active": "torch",
    "model": "convaiinnovations/laya",
    "loaded": false
  }
}
```

## What It Decides

| Need | Tool | Result |
| --- | --- | --- |
| Start a consequential coding task | `recommend` | Workflow, model, agent, verification, and applicable MCPs in one batch |
| Choose one exclusive option | `route` | One selected candidate with probabilities |
| Make several exclusive choices | `route_many` | One selected candidate per named route set in one batch |
| Select zero or more applicable capabilities | `screen` | Per-candidate probabilities and every selection above a threshold |
| Inspect defaults without loading Laya | `catalog` | Configured candidates, tool semantics, and documentation resources |

Every tool returns a compact JSON string. The raw Laya result is retained under
`result` so agents can inspect confidence and probabilities instead of treating
a selected label as unquestionable.

## Agent Quick Start

Use `recommend` once when a task has a real routing decision. It evaluates the
configured workflow, model, agent, verification, and MCP candidates in one
local inference batch.

```json
{
  "task": "Investigate a TypeScript regression, implement the smallest fix, and run relevant tests"
}
```

The result has this shape:

```json
{
  "recommendations": {
    "workflow": "implement",
    "model": "openai/gpt-5.6-terra",
    "agent": "direct",
    "verification": "targeted"
  },
  "tool_recommendations": ["jcodemunch", "docs-mcp-server"],
  "tool_threshold": 0.5,
  "decisions": {},
  "tool_scores": {},
  "result": {}
}
```

The selected labels above are illustrative. Treat the actual response as
advice, not permission to skip user instructions, repository rules, required
inspection, confirmation, or validation.

Do not route trivial work that already names the required tool or action. For
example, use the requested tool directly when the user says to run a known test
command or inspect a named file.

## Tool Reference

### `recommend`

The primary entry point for a consequential coding task. It returns one choice
for each configured `workflow`, `model`, `agent`, and `verification` route. For
MCPs, explicit Jira, Bitbucket, and Grafana identifiers are matched before MCP
screening. Otherwise it returns only the model's top MCP when it clears the
minimum score and leads the runner-up by the configured margin.

Set `tool_threshold` (default `0.4`) and `tool_margin` (default `0.1`) from `0`
through `1` to control the minimum top score and required lead. All model
probabilities remain in `tool_scores`.

### `route`

Choose one option from a configured route type or from custom candidates.
Configured types are `model`, `mcp`, `workflow`, `agent`, and `verification`.

```json
{
  "task": "Pick the best way to investigate a flaky CI failure",
  "route_type": "agent"
}
```

```json
{
  "task": "Choose an implementation strategy for a backwards-compatible migration",
  "route_type": "custom",
  "candidates": {
    "incremental": "Add the new path while preserving the old behavior.",
    "cutover": "Replace the old path in one coordinated change."
  }
}
```

Custom candidate labels and descriptions must be nonempty, and there must be at
least two candidates.

### `route_many`

Evaluate several independent exclusive decisions in one Laya batch. Each key
names a decision and maps to at least two candidates.

```json
{
  "task": "Prepare a production bug fix",
  "route_sets": {
    "implementation": {
      "minimal": "Change only the faulty condition.",
      "refactor": "Restructure the affected subsystem first."
    },
    "validation": {
      "targeted": "Run focused tests and linting.",
      "full": "Run the full project verification suite."
    }
  }
}
```

### `screen`

Use `screen` when more than one capability can be appropriate. It asks an
independent true-or-false question for each candidate instead of forcing a
single winner.

```json
{
  "task": "Diagnose a new dependency API error",
  "candidates": {
    "source": "Inspect the repository source and dependency usage.",
    "docs": "Read current third-party documentation.",
    "ci": "Inspect CI logs and recent failed runs."
  },
  "threshold": 0.6
}
```

`selected` includes candidates at or above `threshold`; `scores` retains the
probability for every candidate, including those not selected.

### `catalog` and Resources

`catalog` returns the active candidate catalog, selected backend, and model
without loading Laya. MCP clients that support resources can instead read:

| Resource | Purpose |
| --- | --- |
| `laya://guide` | Agent usage rules and result interpretation |
| `laya://routes` | JSON route catalog and tool semantics |

The server's MCP instructions also direct agents to these resources.

## Configured Defaults

The built-in routes cover:

| Route | Candidates |
| --- | --- |
| `model` | Sol, Luna, Terra |
| `workflow` | Answer, explore, plan, implement, review |
| `agent` | Direct, explore, general, scout, log-digger |
| `verification` | None, targeted, full |
| `mcp` | jcodemunch, docs-mcp-server, gitea, asuswrt |

### Customize MCP Candidates

Set `LAYA_MCP_MCP_SERVERS` to a JSON object to replace the built-in `mcp`
candidates without editing source. Keys are the MCP server names returned to an
agent; values should explain when each server is useful to Laya's router.

```shell
export LAYA_MCP_MCP_SERVERS='{
  "jcodemunch": "Indexed source code structure, symbols, and references.",
  "linear": "Project issues, priorities, and engineering work tracking.",
  "docs": "Current third-party library documentation."
}'
uv run laya-mcp
```

The override applies consistently to `recommend`, `route` with
`route_type="mcp"`, and `catalog`. It must be a nonempty JSON object with
nonempty names and descriptions. Call `catalog` rather than relying on this
table when an environment override is active.

## Run It

After installing the platform backend, configure an MCP client to run this
command from the repository root:

```shell
uv run laya-mcp
```

For an absolute client command, use the repository path with uv's directory
option:

```shell
uv --directory /path/to/laya-mcp run laya-mcp
```

The server communicates over stdio.

## Development

```shell
# Apple Silicon development
uv sync --extra dev --extra mlx

# Linux development
uv sync --extra dev --extra torch

uv run ruff check .
uv run pytest
uv run laya-mcp
```

The default test run uses backend doubles and does not download a model. Run a
real contract check for one or more installed runtimes when needed:

```shell
LAYA_MCP_RUN_MODEL_TESTS=1 \
LAYA_MCP_INTEGRATION_BACKENDS=torch \
LAYA_MCP_BACKEND=torch \
USE_TF=0 \
uv run pytest -m integration
```

GitHub Actions runs unit tests and this Torch CPU integration check on Linux.
The optional CUDA contract job only runs on a labeled self-hosted GPU runner,
either when manually requested or when `LAYA_MCP_ENABLE_CUDA=true` enables its
weekly scheduled run.

### Building a Routing Corpus

OpenCode session history can provide candidate routing examples without putting
private prompts in this repository. Extract locally observed MCP calls into an
ignored review file, then assign one expected MCP or `null` before adding only
the reviewed, sanitized examples to model integration tests:

```shell
uv run python scripts/extract_opencode_mcp_corpus.py \
  --output corpus/opencode-mcp-candidates.jsonl
```

Observed calls are weak labels, not training truth: one task may have used an
unnecessary MCP or several valid MCPs. The extractor never copies tool output,
only the first user task and the one MCP used in that session.

#### Training

Review the candidate file by setting `label_status` to `reviewed` and `label` to
one configured MCP name or `null`. Save the active MCP mapping separately; it
is the exact set of capability descriptions used during training.

Create and validate a balanced, privacy-safe seed corpus when no reviewed
production labels exist yet:

```shell
uv run python scripts/build_mcp_seed_corpus.py --output corpus/mcp-reviewed.jsonl
uv run python scripts/validate_mcp_corpus.py \
  --corpus corpus/mcp-reviewed.jsonl --servers corpus/servers.json
```

```shell
uv sync --extra training --extra torch-cuda
cat > corpus/servers.json <<'EOF'
{"jcodemunch":"Indexed source code structure, symbols, references, and change impact.","docs-mcp-server":"Current third-party library documentation and API guidance.","gitea":"Repository issues, pull requests, reviews, and project work tracking."}
EOF
uv run laya-mcp-train \
  --corpus corpus/opencode-mcp-reviewed.jsonl \
  --servers corpus/servers.json \
  --output models/mcp-router
```

Training requires a CUDA or Apple Metal GPU and saves a complete Laya checkpoint
locally under the ignored `models/` directory. Each reviewed task becomes one binary `noul`
decision per MCP, matching `recommend`'s independent MCP-screening questions.
The deterministic holdout report is evidence of fit, not a release gate; keep a
separate untouched regression corpus for deployment decisions.

Test a trained checkpoint by setting `LAYA_MCP_MODEL=models/mcp-router` before
starting the server. Do not replace the production default until its held-out
and regression results improve on the current checkpoint.

For an 8 GiB GPU, full encoder training needs factorized optimizer state and
activation checkpointing:

```shell
uv run laya-mcp-train ... --optimizer adafactor --gradient-checkpointing --batch-size 1
```

Fit the no-ulterior-loss screening temperature on the deterministic held-out
tasks before using a checkpoint's confidence values:

```shell
uv run python scripts/calibrate_mcp_checkpoint.py \
  --corpus corpus/mcp-reviewed.jsonl --servers corpus/servers.json \
  --model models/mcp-router
```

For an explicitly experimental local run, `--allow-weak-labels` uses the one
observed MCP recorded by the extractor as its label. This is not a substitute
for review; its `training_report.json` records `weak_labels: true`.
