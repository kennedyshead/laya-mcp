"""MCP server that routes coding tasks with local Laya MLX inference."""

import asyncio
import json
from collections.abc import Mapping
from typing import Any

import laya_mlx as laya
from mcp.server.fastmcp import FastMCP

SERVER_NAME = "laya-mcp"
DEFAULT_MODEL = "aac6fef/laya-multilingual-mlx"

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
        "laya-mcp": "Fast local typed routing decisions.",
    },
    "workflow": {
        "answer": "Answer directly without changing files or using tools.",
        "explore": "Inspect the codebase or external facts before deciding.",
        "plan": "Clarify requirements and make a multi-step implementation plan.",
        "implement": "Make a focused code or configuration change and verify it.",
        "review": "Review changes for bugs, regressions, and missing tests.",
    },
}


class DecisionRouter:
    """Load one Laya model lazily and use it for bounded routing choices."""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self._model = model
        self._agent: Any | None = None
        self._lock = asyncio.Lock()

    async def route(
        self,
        task: str,
        candidates: Mapping[str, str],
    ) -> dict[str, Any]:
        """Choose one route and retain Laya's probability evidence."""
        if not task.strip():
            raise ValueError("task must not be empty")
        if len(candidates) < 2:
            raise ValueError("at least two candidates are required")

        agent = await self._get_agent()
        result = await asyncio.to_thread(
            agent.predict,
            {"task": task},
            {
                "route": {
                    "type": "choice",
                    "instructions": (
                        "Choose the best route for the coding task. Select the "
                        "candidate that most directly and efficiently satisfies it."
                    ),
                    "criteria": dict(candidates),
                }
            },
        )
        answer = result["answers"]["route"]
        selected = answer.get("choice") if isinstance(answer, Mapping) else answer
        return {
            "selected": selected,
            "candidates": dict(candidates),
            "result": result,
        }

    async def _get_agent(self) -> Any:
        async with self._lock:
            if self._agent is None:
                self._agent = await asyncio.to_thread(laya.load, self._model)
            return self._agent


router = DecisionRouter()
mcp = FastMCP(
    SERVER_NAME,
    instructions=(
        "Use route for a fast local recommendation among model, MCP, workflow, "
        "or caller-supplied routes. Treat it as a recommendation, not authority."
    ),
)


@mcp.tool()
async def route(
    task: str,
    route_type: str = "workflow",
    candidates: dict[str, str] | None = None,
) -> str:
    """Recommend the best model, MCP, workflow, or supplied route for a task.

    Use route_type `model`, `mcp`, or `workflow` for configured defaults. Use
    `custom` with candidates to decide among any two or more named options.
    """
    if candidates is None:
        candidates = ROUTES.get(route_type)
    if candidates is None:
        valid = ", ".join([*ROUTES, "custom"])
        raise ValueError(
            f"route_type must be one of {valid}, or candidates must be supplied"
        )
    decision = await router.route(task, candidates)
    return json.dumps(decision, ensure_ascii=True, separators=(",", ":"))


def main() -> None:
    """Run the routing server over MCP stdio."""
    mcp.run()
