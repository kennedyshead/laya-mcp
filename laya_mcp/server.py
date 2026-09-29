"""MCP server that routes coding tasks with local Laya inference."""

import asyncio
import json
import os
import re
from collections.abc import Mapping
from typing import Any

from mcp.server.fastmcp import FastMCP

from .backends import (
    Backend,
    BackendConfigurationError,
    backend_metadata,
    create_backend,
)

SERVER_NAME = "laya-mcp"
MCP_SERVERS_ENV = "LAYA_MCP_MCP_SERVERS"
DEFAULT_TOOL_THRESHOLD = 0.4
DEFAULT_TOOL_MARGIN = 0.1
MCP_KEYWORDS: dict[str, tuple[str, ...]] = {
    "jira": (r"\bESAPI-\d+\b",),
    "bitbucket": (r"\b(?:PR|pull request)\s*#?\d+\b", r"\bpipeline\b"),
    "grafana": (r"\bmeta24\b", r"\btest\.api\b", r"\bloki\b"),
}

ROUTES: dict[str, dict[str, str]] = {
    "model": {
        "openai/gpt-5.6-sol": (
            "Fast small model for titles, simple questions, and isolated edits."
        ),
        "openai/gpt-5.6-luna": (
            "Fast subagent for bounded codebase exploration and research."
        ),
        "openai/gpt-5.6-terra": (
            "Primary model for complex implementation, debugging, and review."
        ),
    },
    "mcp": {
        "jcodemunch": "Indexed codebase structure, symbols, references, and impact.",
        "docs-mcp-server": "Current third-party library and framework documentation.",
        "gitea": "Repositories, issues, pull requests, releases, and CI on Gitea.",
        "asuswrt": "Router status and administration.",
    },
    "workflow": {
        "answer": "Answer directly without changing files or using tools.",
        "explore": "Inspect the codebase or external facts before deciding.",
        "plan": "Clarify requirements and make a multi-step implementation plan.",
        "implement": "Make a focused code or configuration change and verify it.",
        "review": "Review changes for bugs, regressions, and missing tests.",
    },
    "agent": {
        "direct": "Handle the task in the current agent without delegation.",
        "explore": "Delegate bounded codebase exploration or factual research.",
        "general": "Delegate an independent multi-step implementation task.",
        "scout": "Delegate fast definition, reference, and file discovery.",
        "log-digger": "Delegate extraction of failures from verbose CI or test logs.",
    },
    "verification": {
        "none": "No executable check applies because the task is informational only.",
        "targeted": "Run focused tests, linting, or checks for the changed area.",
        "full": "Run the broad project checks because the change is high-risk or wide.",
    },
}

RECOMMENDATION_ROUTE_TYPES = ("workflow", "model", "agent", "verification")

AGENT_GUIDE = """# Laya MCP Agent Guide

Laya MCP makes fast, local, advisory routing decisions. It does not inspect a
repository, perform actions, override user instructions, or replace required
validation.

## Start Here

Use `recommend` once at the start of a consequential coding task when you need
advice on workflow, model, delegation, validation scope, and applicable MCPs.
It evaluates those dimensions in one Laya inference batch.

Use `route` for one mutually exclusive decision. Pass a configured
`route_type` (`model`, `mcp`, `workflow`, `agent`, or `verification`) or pass
your own `candidates` for any custom decision.

Use `route_many` when several mutually exclusive decisions share the same task.
It receives named candidate sets and evaluates all of them in one batch.

Use `screen` when zero, one, or many capabilities may apply. It returns a
probability for every candidate and selects those at or above `threshold`.

Use `catalog` or the `laya://routes` resource to inspect configured defaults.
Set `LAYA_MCP_MCP_SERVERS` to a JSON object of MCP server names and descriptions
to replace the built-in MCP candidates.

## Interpret Results

Treat every result as evidence, not authority. Preserve user instructions,
repository rules, safety constraints, and required checks even when a route
disagrees. Inspect the returned probabilities when the choice is close or the
task has material risk. Do not route trivial requests that already name the
required tool, workflow, or model.
"""


class DecisionRouter:
    """Load one selected Laya backend lazily for bounded routing choices."""

    def __init__(
        self,
        model: str | None = None,
        backend: Backend | None = None,
    ) -> None:
        self._model = model
        self._backend = backend
        self._agent: Any | None = None
        self._lock = asyncio.Lock()

    async def route(
        self,
        task: str,
        candidates: Mapping[str, str],
    ) -> dict[str, Any]:
        """Choose one route and retain Laya's probability evidence."""
        assessment = await self.assess(task, {"route": candidates})
        decision = assessment["choices"]["route"]
        return {
            "selected": decision["selected"],
            "candidates": decision["candidates"],
            "result": assessment["result"],
        }

    async def route_many(
        self,
        task: str,
        route_sets: Mapping[str, Mapping[str, str]],
    ) -> dict[str, Any]:
        """Choose one candidate in each named route set in a single inference."""
        assessment = await self.assess(task, route_sets)
        return {
            "routes": assessment["choices"],
            "result": assessment["result"],
        }

    async def screen(
        self,
        task: str,
        candidates: Mapping[str, str],
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Score every independent candidate and select the applicable ones."""
        threshold = _validate_threshold(threshold)
        assessment = await self.assess(task, {}, candidates)
        scores = assessment["checks"]
        return {
            "selected": [
                name
                for name, score in scores.items()
                if score["probability"] >= threshold
            ],
            "threshold": threshold,
            "candidates": dict(candidates),
            "scores": scores,
            "result": assessment["result"],
        }

    async def assess(
        self,
        task: str,
        route_sets: Mapping[str, Mapping[str, str]],
        checks: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Run exclusive choices and independent applicability checks together."""
        task = _validate_task(task)
        route_sets = _normalize_route_sets(route_sets)
        checks = _normalize_candidates(
            checks or {},
            minimum=0,
            label="screen candidate",
        )
        if not route_sets and not checks:
            raise ValueError("at least one route set or screen candidate is required")

        questions: dict[str, dict[str, Any]] = {}
        for name, candidates in route_sets.items():
            questions[f"choice:{name}"] = {
                "type": "choice",
                "instructions": (
                    "Choose the best route for the coding task. Select the "
                    "candidate that most directly and efficiently satisfies it."
                ),
                "criteria": candidates,
            }
        for name, description in checks.items():
            questions[f"check:{name}"] = {
                "type": "noul",
                "instructions": (
                    "Decide whether this capability materially helps complete the "
                    "coding task. Answer true only when using it is appropriate."
                ),
                "criteria": {
                    "false": "This capability does not materially help this task.",
                    "true": description,
                },
            }

        result = await self._predict(task, questions)
        answers = result["answers"]
        choices = {
            name: _choice_decision(answers[f"choice:{name}"], candidates)
            for name, candidates in route_sets.items()
        }
        scored_checks = {
            name: _check_decision(answers[f"check:{name}"]) for name in checks
        }
        return {"choices": choices, "checks": scored_checks, "result": result}

    async def _predict(
        self,
        task: str,
        questions: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        agent = await self._get_agent()
        result = await asyncio.to_thread(agent.predict, {"task": task}, questions)
        if not isinstance(result, Mapping):
            raise RuntimeError("Laya returned an invalid prediction result")
        answers = result.get("answers")
        if not isinstance(answers, Mapping):
            raise RuntimeError("Laya prediction result is missing answers")
        return dict(result)

    async def _get_agent(self) -> Any:
        async with self._lock:
            if self._agent is None:
                backend = self._backend
                if backend is None:
                    backend = create_backend(model_override=self._model)
                    self._backend = backend
                self._agent = await asyncio.to_thread(backend.load)
            return self._agent

    def backend_info(self) -> dict[str, Any]:
        """Describe the configured backend without loading its optional runtime."""
        if self._backend is not None:
            config = getattr(self._backend, "config", None)
            details: dict[str, Any] = {
                "requested": getattr(config, "requested", self._backend.name),
                "active": self._backend.name,
                "model": self._backend.model,
            }
            artifact_path = getattr(config, "onnx_artifact_path", None)
            if artifact_path is not None:
                details["artifact_path"] = artifact_path
        else:
            try:
                details = backend_metadata(model_override=self._model)
            except BackendConfigurationError as error:
                details = {"error": str(error)}
        details["loaded"] = self._agent is not None
        return details


def _validate_task(task: str) -> str:
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must not be empty")
    return task.strip()


def _normalize_candidates(
    candidates: Mapping[str, str],
    *,
    minimum: int,
    label: str,
) -> dict[str, str]:
    if not isinstance(candidates, Mapping) or len(candidates) < minimum:
        if minimum == 2:
            raise ValueError("at least two candidates are required")
        raise ValueError(f"at least {minimum} {label}s are required")

    normalized: dict[str, str] = {}
    for name, description in candidates.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{label} names must be nonempty strings")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"{label} descriptions must be nonempty strings")
        normalized[name] = description
    return normalized


def _normalize_route_sets(
    route_sets: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    if not isinstance(route_sets, Mapping):
        raise ValueError("route_sets must be a mapping of named candidate sets")

    normalized: dict[str, dict[str, str]] = {}
    for name, candidates in route_sets.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("route set names must be nonempty strings")
        normalized[name] = _normalize_candidates(
            candidates,
            minimum=2,
            label="candidate",
        )
    return normalized


def _validate_threshold(threshold: float) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("threshold must be a number from 0 to 1")
    threshold = float(threshold)
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be a number from 0 to 1")
    return threshold


def _keyword_mcp_recommendations(
    task: str,
    candidates: Mapping[str, str],
) -> list[str]:
    """Return configured MCPs identified by unambiguous task identifiers."""
    selected: list[str] = []
    for name in candidates:
        patterns = MCP_KEYWORDS.get(name.lower())
        matches_keyword = patterns and any(
            re.search(pattern, task, re.IGNORECASE) for pattern in patterns
        )
        if matches_keyword:
            selected.append(name)
    return selected


def _top_mcp_recommendation(
    scores: Mapping[str, Mapping[str, float]],
    threshold: float,
    margin: float,
) -> list[str]:
    """Select one clear model winner instead of thresholding noisy scores."""
    ranked = sorted(
        scores.items(), key=lambda item: item[1]["probability"], reverse=True
    )
    if not ranked or ranked[0][1]["probability"] < threshold:
        return []
    if len(ranked) > 1 and (
        ranked[0][1]["probability"] - ranked[1][1]["probability"] < margin
    ):
        return []
    return [ranked[0][0]]


def _choice_decision(answer: Any, candidates: Mapping[str, str]) -> dict[str, Any]:
    if not isinstance(answer, Mapping):
        raise RuntimeError("Laya returned an invalid choice answer")
    selected = answer.get("choice")
    if not isinstance(selected, str) or selected not in candidates:
        raise RuntimeError("Laya selected a candidate outside the supplied route")
    return {
        "selected": selected,
        "candidates": dict(candidates),
        "confidence": answer.get("confidence"),
        "probabilities": answer.get("probabilities"),
    }


def _check_decision(answer: Any) -> dict[str, float]:
    if not isinstance(answer, Mapping):
        raise RuntimeError("Laya returned an invalid screening answer")
    probability = answer.get("noul")
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        raise RuntimeError("Laya screening answer is missing a probability")
    probability = float(probability)
    if not 0 <= probability <= 1:
        raise RuntimeError("Laya screening probability must be from 0 to 1")
    confidence = answer.get("confidence")
    return {
        "probability": probability,
        "confidence": float(confidence)
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
        else probability,
    }


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _configured_mcp_servers() -> dict[str, str]:
    """Read an optional replacement MCP candidate mapping from the environment."""
    configured = os.getenv(MCP_SERVERS_ENV)
    if configured is None:
        return dict(ROUTES["mcp"])
    try:
        candidates = json.loads(configured)
    except json.JSONDecodeError as error:
        raise ValueError(f"{MCP_SERVERS_ENV} must be a JSON object") from error
    if not isinstance(candidates, Mapping):
        raise ValueError(f"{MCP_SERVERS_ENV} must be a JSON object")
    return _normalize_candidates(candidates, minimum=1, label="MCP server")


def _configured_routes() -> dict[str, dict[str, str]]:
    """Return route defaults with the environment-selected MCP candidates."""
    return {**ROUTES, "mcp": _configured_mcp_servers()}


def _catalog() -> dict[str, Any]:
    return {
        "backend": router.backend_info(),
        "routes": _configured_routes(),
        "tools": {
            "recommend": (
                "One-call defaults for workflow, model, agent, validation, and MCPs."
            ),
            "route": (
                "One mutually exclusive route, including caller-supplied candidates."
            ),
            "route_many": (
                "Several mutually exclusive route sets in one inference batch."
            ),
            "screen": (
                "Independent capability screening with per-candidate probabilities."
            ),
            "catalog": "Configured routes and tool semantics without loading Laya.",
        },
        "resources": ["laya://guide", "laya://routes"],
    }


router = DecisionRouter()
mcp = FastMCP(
    SERVER_NAME,
    instructions=(
        "Use recommend for a complete local routing recommendation. Use route for "
        "one exclusive choice, route_many for several choices, and screen when "
        "multiple capabilities may apply. Read laya://guide or call catalog for "
        "usage details. Treat every result as advisory, not authority."
    ),
)


@mcp.tool()
async def route(
    task: str,
    route_type: str = "workflow",
    candidates: dict[str, str] | None = None,
) -> str:
    """Choose one model, MCP, workflow, agent, or caller-supplied route.

    Use a configured `route_type` or pass at least two named `candidates` for
    any custom mutually exclusive decision. The response includes probabilities.
    """
    if candidates is None:
        candidates = _configured_routes().get(route_type)
    if candidates is None:
        valid = ", ".join([*ROUTES, "custom"])
        raise ValueError(
            f"route_type must be one of {valid}, or candidates must be supplied"
        )
    decision = await router.route(task, candidates)
    return _json(decision)


@mcp.tool()
async def route_many(
    task: str,
    route_sets: dict[str, dict[str, str]],
) -> str:
    """Choose one candidate in each named route set in one Laya inference batch.

    Each route set must contain at least two candidates. Use this for independent
    exclusive decisions such as implementation approach and validation scope.
    """
    decision = await router.route_many(task, route_sets)
    return _json(decision)


@mcp.tool()
async def screen(
    task: str,
    candidates: dict[str, str],
    threshold: float = 0.5,
) -> str:
    """Select every applicable capability instead of forcing a single choice.

    Each candidate receives an independent probability. Candidates at or above
    `threshold` (0 to 1) appear in `selected`; all scores remain in the result.
    """
    decision = await router.screen(task, candidates, threshold)
    return _json(decision)


@mcp.tool()
async def recommend(
    task: str,
    tool_threshold: float = DEFAULT_TOOL_THRESHOLD,
    tool_margin: float = DEFAULT_TOOL_MARGIN,
) -> str:
    """Recommend workflow, model, agent, verification, and applicable MCPs.

    This is the primary entry point for a consequential coding task. It evaluates
    the configured defaults in one local inference batch and returns evidence.
    """
    tool_threshold = _validate_threshold(tool_threshold)
    tool_margin = _validate_threshold(tool_margin)
    routes = _configured_routes()
    keyword_recommendations = _keyword_mcp_recommendations(task, routes["mcp"])
    assessment = await router.assess(
        task,
        {name: routes[name] for name in RECOMMENDATION_ROUTE_TYPES},
        {} if keyword_recommendations else routes["mcp"],
    )
    tool_scores = assessment["checks"]
    tool_recommendations = keyword_recommendations or _top_mcp_recommendation(
        tool_scores, tool_threshold, tool_margin
    )
    decision = {
        "recommendations": {
            name: choice["selected"] for name, choice in assessment["choices"].items()
        },
        "tool_recommendations": tool_recommendations,
        "tool_threshold": tool_threshold,
        "tool_margin": tool_margin,
        "tool_selection": "keyword" if keyword_recommendations else "top-1-margin",
        "decisions": assessment["choices"],
        "tool_scores": tool_scores,
        "result": assessment["result"],
    }
    return _json(decision)


@mcp.tool()
def catalog() -> str:
    """Return configured route candidates and tool semantics without loading Laya."""
    return _json(_catalog())


@mcp.resource(
    "laya://guide",
    name="Laya MCP Agent Guide",
    description="How agents should choose and interpret Laya MCP tools.",
)
def agent_guide() -> str:
    """Provide concise operational guidance for agents using this MCP server."""
    return AGENT_GUIDE


@mcp.resource(
    "laya://routes",
    name="Laya MCP Route Catalog",
    description="Configured route candidates and Laya MCP tool semantics.",
)
def route_catalog() -> str:
    """Provide the current route catalog as JSON without loading the model."""
    return _json(_catalog())


def main() -> None:
    """Run the routing server over MCP stdio."""
    mcp.run()
