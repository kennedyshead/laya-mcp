# Laya MCP Agent Instructions

## Find Documentation

- Read `README.md` for installation, API/backend behavior and specialist setup;
  `docs/documentation.md` defines local sources and indexing boundaries.
- Search Docs MCP with `laya-mcp` / `committed`. Verify provenance and shared
  `project_docs.py --status`; use local Markdown for drafts or unavailable search.
- Local `~/dotfiles/DOCUMENTATION.md` maps all project libraries and maintenance
  guides. The private routing plan is at
  `~/dotfiles/docs/plans/model-routing-optimization.md`, in the separate `dotfiles`
  committed projection after a verified docs commit.
- Use jcodemunch for code. Preserve uncommitted work; documentation or benchmark
  scores do not authorize commits, indexing private artifacts or deployment.

## Purpose

Laya MCP supplies local, typed routing advice for coding work. It does not read
the repository, perform work, bypass approvals, or replace agent judgment.

## Default Use

Call `recommend` once at the start of a consequential task when the best
workflow, model, delegation approach, validation scope, or MCPs are unclear.
It batches all configured default decisions into one local inference request.

Call `catalog` or read `laya://routes` before assuming which configured labels
are available. Read `laya://guide` for the MCP-discoverable version of these
instructions.

## Select The Right Tool

| Situation | Use |
| --- | --- |
| One exclusive decision | `route` |
| Several exclusive decisions for one task | `route_many` |
| Several capabilities may all apply | `screen` |
| Broad initial task triage | `recommend` |
| Discover configured choices without model loading | `catalog` |

Use custom `candidates` with `route` for any mutually exclusive choice not in
the configured catalog. Use `screen` rather than `route` when the correct
answer can contain multiple candidates.

## Interpret Results

Treat every response as advisory evidence. Keep user instructions, repository
policies, security constraints, and required verification ahead of a routing
result. Inspect `probabilities`, `decisions`, and `tool_scores` when a decision
is consequential or close.

Do not call Laya for a trivial request that already gives the required action,
tool, or command. Do not use a recommendation to skip code inspection before a
change, confirmation before destructive work, or tests required by the project.

## Result Contracts

`route` returns one `selected` label and its probabilities.

`route_many` returns a `routes` mapping with one decision per route set.

`screen` returns all above-threshold labels in `selected` and every score in
`scores`.

`recommend` returns `recommendations` for workflow, model, agent, and
verification. It returns independently selected MCPs in `tool_recommendations`.

All tool results include the raw Laya response under `result` for auditing.
