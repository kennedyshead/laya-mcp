# Laya MCP

Local, advisory routing decisions for coding agents, powered by
[Laya](https://huggingface.co/convaiinnovations/laya) without generating text or
sending task data to a cloud model. The server chooses a local MLX, PyTorch, or
ONNX Runtime adapter without changing its MCP contract.

The server starts and serves `catalog` without importing any model runtime. The
first routing request lazily loads the selected backend and, when needed,
downloads its checkpoint. Later requests reuse the loaded local model.

## Documentation

[Documentation discovery](docs/documentation.md) maps setup, source instructions,
experiment results and cross-project search. Search Docs MCP with **`laya-mcp`** /
**`committed`**, verify provenance/hook status, and read local drafts separately.
Uncommitted experimental changes are not deployed behavior.

The policy-training and specialist sections below describe local experimental
work whose implementation is not yet committed or published. This documentation
update does not release those capabilities; use them as design/experiment notes
until the implementation is separately reviewed and published.

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

The project `opencode.jsonc` starts the calibrated remote checkpoint over SSH.
Its machine-specific settings are in `.envrc`: `LAYA_MCP_REMOTE_HOST`,
`LAYA_MCP_REMOTE_GPU`, `LAYA_MCP_REMOTE_MODEL`, and
`LAYA_MCP_REMOTE_EXECUTABLE`. The launcher loads this file and honors values
already exported in the environment. Restart OpenCode after changing this
configuration.

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

### Private Normalization And Joint Policy Training

The joint pipeline separates canonical private history from training-approved
examples. It never infers correct models or tools from historical usage.

```shell
.venv/bin/python -m laya_mcp.corpus \
  --input ~/.local/state/laya/corpus/opencode \
  --output ~/.local/state/laya/corpus/normalized/<new-run>
.venv/bin/python -m laya_mcp.policy_corpus \
  --output corpus/<new-policy-run>
```

Normalization verifies each immutable batch checksum, keeps host namespaces,
retains message/part source representations and roles, prefers V2 for matching
message IDs, and groups child/fork sessions. Its database and manifest are
private (0700 directory, 0600 files). Different-ID semantic duplicates and
cross-host copies are not silently merged. The normalized database is **not
training-approved**. Do not commit or upload it.

The synthetic authoring command generates 108 training, 36 validation and 36
test rows, representing 36/12/12 distinct wording groups. Prefix variants stay
in the same split. Intent categories are shared across splits, so this is a
small policy test, not evidence of unseen-domain generalization. Labels encode
capability suitability and provisional benchmark-informed model/variant policy;
they do not claim that those coding models outperform alternatives. No raw
conversation text appears in these examples.

The joint trainer updates decision heads with the encoder frozen, using binary
MCP questions (including multiple/no tools) and an exclusive model/variant
question. Choice option order is shuffled during encoding; binary MCP options
retain upstream's fixed `[false, true]` semantic order. Validation selects the
checkpoint; final test rows are checked for split leakage but are not trained on.

```shell
USE_TF=0 HIP_VISIBLE_DEVICES=<verified-device> \
  python -m laya_mcp.joint_training evaluate \
  --model <existing-checkpoint> --candidates corpus/<run>/candidates.json \
  --test corpus/<run>/test.jsonl --output <run>/before.json
USE_TF=0 HIP_VISIBLE_DEVICES=<verified-device> \
  python -m laya_mcp.joint_training train \
  --model <existing-checkpoint> --candidates corpus/<run>/candidates.json \
  --train corpus/<run>/train.jsonl --validation corpus/<run>/validation.jsonl \
  --test corpus/<run>/test.jsonl --output <new-checkpoint>
USE_TF=0 HIP_VISIBLE_DEVICES=<verified-device> \
  python -m laya_mcp.joint_training evaluate \
  --model <new-checkpoint> --candidates corpus/<run>/candidates.json \
  --test corpus/<run>/test.jsonl --output <run>/after.json
```

These commands require the declared training dependencies and the existing
Torch GPU environment; they do not install a GPU stack. Discover the GPU again
before each run, check competing workloads, and do not interrupt VR or services.
Use versioned output directories, preserve old checkpoints, and allow long
training commands enough time. An incomplete output is not an installable model.

Compare model-policy accuracy/macro accuracy, MCP exact-set accuracy and F1,
Brier score, warm latency and peak GPU memory on the same cases/candidates and
hardware. The report retains per-case predictions and dataset fingerprints.
The Mac MLX checkpoint and a remote Torch checkpoint are distinct baselines:
do not present a cross-hardware comparison as a controlled latency improvement.
Synthetic improvements alone do not satisfy a real-workload promotion gate.

Serving supports `LAYA_MCP_MODELS` to restrict model/variant candidates to an
eligible JSON mapping. Explicit user pins and authorization must be applied by
the caller before requesting advice. The current default model mapping is the
provisional joint-policy mapping, not a measured leaderboard ranking.
`LAYA_MCP_TOOL_SELECTION=independent` enables multiple MCP recommendations;
`top-1-margin` remains the compatibility default. Use `tool_threshold=0.5`
for comparison with the joint evaluation. Neither mode executes a recommended
tool or automatically changes an OpenCode session model.

`sh scripts/run_policy_candidate.sh` is an explicitly gated, versioned remote
candidate launcher, not the active MCP configuration. It refuses to start unless
the run has a `candidate-approved.json` promotion receipt. When approved, it uses
CPU inference (`LAYA_MCP_DEVICE=cpu`) to avoid competing for the VR/Ollama GPUs,
and keeps stdin available for MCP traffic. Version-1 experiment results did not
pass promotion: no approval receipt exists and the active checkpoint is unchanged.

### Dataset Revision 2

```shell
.venv/bin/python -m laya_mcp.policy_corpus_v2 \
  --database ~/.local/state/laya/corpus/normalized/20261008-routing-v3/normalized.sqlite3 \
  --output corpus/<new-v2-run>
```

This revision has 120 distinct authored intents and 240 cases: 170 training /
30 validation / 40 test rows in 85 / 15 / 20 groups. Each contrast changes scope,
resource availability, or reporting requirements; there are no prefix duplicates.
All siblings stay together. Final test intents are absent from the authoring
pool, and every split must cover positive and negative examples of every MCP.

Corpus is queried locally, read-only, for conservative intent counts. Only these
aggregate cues inform authoring; original wording is not copied, transferred or
trained on. Rows are explicitly tagged `corpus-shaped-authored` or synthetic,
and raw tasks have **not** been individually adjudicated. This is a policy
benchmark, not an independently reviewed real-task success benchmark.

Rows can specify `acceptable_models`, `available_models`, and `available_mcps`.
The trainer maximizes probability mass over acceptable model choices instead of
forcing arbitrary winner labels. Research cases allow Sonnet medium or Sol
medium when both are eligible; explicit OpenAI-only cases exclude Claude.
Unavailable tools are removed from the questions, not learned as permissions.
Serving callers still own availability filtering and user-pin enforcement.

Loss weighting allocates half the mass to model choice and half to MCP questions,
balances tools equally, and balances each tool's positive/negative class.
Validation selects by mean acceptable-model accuracy and MCP F1, rather than
letting abundant binary negatives dominate. Reports retain preferred-model and
acceptable-set accuracy separately, plus availability-constrained case counts.

Before training, compare both checkpoints and the fixed, transparent lexical
baseline on identical cases:

```shell
.venv/bin/python -m laya_mcp.policy_baseline \
  --candidates corpus/<run>/candidates.json \
  --test corpus/<run>/test.jsonl --output corpus/<run>/heuristic.json
```

The baseline reads task text and eligible candidates, never expected labels.
Do not tune it after viewing final test results. Version-1's exposed test remains
a regression check; it is not the new release holdout. Improved authored-policy
scores still do not establish coding competence or real-workload deployment safety.

### Validation-Only Iteration

`scripts/iterate_policy.py` runs a bounded sequence on the training server:
tiny-set fit diagnostics, accumulated-gradient head optimization, and adaptation
of the last two or four ModernBERT encoder layers. It reads only `train.jsonl`,
`validation.jsonl`, and `candidates.json`; no final test is required or opened.
All attempts start from the same base, save separate artifacts, and use the
serving API for validation. It stops on validation accuracy >=75%, MCP exact-set
>=75%, and F1 >=0.80, or when the bounded attempts are exhausted. Failure to fit
the tiny set stops the full experiments for further diagnosis.

```shell
USE_TF=0 HIP_VISIBLE_DEVICES=<verified-device> OMP_NUM_THREADS=2 \
  /path/to/training/venv/bin/python scripts/iterate_policy.py \
  --run /path/to/new-iteration --base /path/to/preserved-checkpoint \
  --data corpus/<run>
```

The trainer also accepts `--accumulation`, `--encoder-layers`, and
`--train-metrics`; `--test` is optional for training and required for evaluation.
Encoder adaptation is opt-in. Validation checkpoint selection averages model
acceptable-set accuracy, tool F1, and tool exact-set accuracy; tool-free rows
count as empty-set decisions, not omitted observations. A validation pass is
**not** a deployment approval. Inspect class-specific errors, freeze a fresh
reviewed real-task holdout, and evaluate the chosen artifact once before promotion.

For diagnostic comparisons, `--objective model` or `--objective tools` restricts
optimization and checkpoint selection to that objective; the default is `joint`.
`python -m laya_mcp.policy_composition` evaluates separate model-choice and tool
checkpoints through the serving API. This is explicitly a two-checkpoint
architecture, not a claim that one model meets both gates. It preserves candidate
availability and tool thresholds, but can require roughly twice the memory and
two inference calls. It does not change the active server or its launcher.

### One Endpoint, Specialized Routing Models

The server supports independently selected **Laya checkpoints** behind one MCP
endpoint. These are routing models, not the coding LLMs they recommend.
`LAYA_MCP_SPECIALISTS` is an opt-in startup JSON mapping:

```shell
export LAYA_MCP_SPECIALISTS='{"model":{"backend":"torch","model":"/path/to/model-specialist"},"tools":{"backend":"torch","model":"/path/to/tool-specialist"}}'
export LAYA_MCP_TOOL_SELECTION=independent
export LAYA_MCP_TOOL_THRESHOLD=0.5
```

Keep `LAYA_MCP_BACKEND` / `LAYA_MCP_MODEL` for the default checkpoint. Workflow,
agent, verification, custom choices, and exclusive `route_type="mcp"` decisions
remain on that backend. `route_type="model"` and a `route_many` set named `model`
use the model specialist; `screen` and `recommend`'s independent MCP checks use
the tools specialist. Supply only one specialist if only that dimension needs
specialization. Each definition needs explicit `backend` and `model`; ONNX also
needs `artifact_path`. Backend runtimes and weights are lazily loaded, not loaded
by `catalog`. Torch specialists honor the existing `LAYA_MCP_DEVICE` setting.

Requests retain their existing answers and candidate availability. Raw results
include `routing` metadata and `specialist_results` for component provenance;
specialist failures propagate rather than silently substituting the default.
Inference is serialized within the endpoint. Model-only or tool-only requests
do not load the other checkpoints; a complete `recommend` can load three
checkpoints and run three batches, so measure memory and latency before deploying.
Generic custom candidates and the default dimensions have not gained validation
coverage just because the five-model/five-MCP benchmark passes.

`python -m laya_mcp.policy_composition --mcp-router ...` measures the actual server
dispatch path. `scripts/specialize_policy.py` first pairs completed checkpoints
using validation scores, then trains isolated objectives only if necessary. It
waits for prior GPU work through a Linux process-exit event (not status polling),
checks for VR and available eGPU memory, and never opens exposed final tests.
Passing validation writes a status report, **not** a deployment approval receipt.
