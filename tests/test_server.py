"""Tests for Laya MCP routing behavior."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from laya_mcp import backends, server


class FakeAgent:
    """Small Laya test double with deterministic choice and screening results."""

    def __init__(self, tool_probability: float = 0.75) -> None:
        self.tool_probability = tool_probability
        self.calls = 0

    def predict(
        self,
        _state: object,
        questions: dict[str, object],
    ) -> dict[str, object]:
        self.calls += 1
        answers: dict[str, object] = {}
        for question_id, question in questions.items():
            definition = question  # type: ignore[assignment]
            if definition["type"] == "choice":  # type: ignore[index]
                criteria = definition["criteria"]  # type: ignore[index]
                names = list(criteria)  # type: ignore[arg-type]
                answers[question_id] = {
                    "choice": names[0],
                    "confidence": 0.9,
                    "probabilities": {
                        name: 1.0 if index == 0 else 0.0
                        for index, name in enumerate(names)
                    },
                }
            else:
                answers[question_id] = {
                    "noul": self.tool_probability,
                    "confidence": self.tool_probability,
                }
        return {
            "answers": answers,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }


class FakeBackend:
    """Backend double that records whether the router had to load it."""

    def __init__(self, agent: FakeAgent, name: str, model: str) -> None:
        self.agent = agent
        self.name = name
        self.model = model
        self.loads = 0

    def load(self) -> FakeAgent:
        self.loads += 1
        return self.agent


@pytest.mark.asyncio
async def test_router_uses_laya_choice_result() -> None:
    """The selected label and original Laya result are retained."""
    router = server.DecisionRouter()
    router._agent = FakeAgent()

    result = await router.route(
        "Fix a production regression",
        {"terra": "strong", "sol": "fast"},
    )

    assert result["selected"] == "terra"
    assert result["candidates"] == {"terra": "strong", "sol": "fast"}


@pytest.mark.asyncio
async def test_router_loads_the_selected_backend_only_for_its_first_request() -> None:
    """Creating the MCP router never imports or loads a model eagerly."""
    agent = FakeAgent()
    backend = FakeBackend(agent, name="torch", model="test/laya")
    router = server.DecisionRouter(backend=backend)

    assert backend.loads == 0
    assert router.backend_info() == {
        "requested": "torch",
        "active": "torch",
        "model": "test/laya",
        "loaded": False,
    }

    candidates = {"terra": "strong", "sol": "fast"}
    await router.route("Fix a production regression", candidates)
    await router.route("Fix a production regression", candidates)

    assert backend.loads == 1
    assert agent.calls == 2


@pytest.mark.asyncio
async def test_router_rejects_invalid_input() -> None:
    """Routing needs a task and a genuine choice."""
    router = server.DecisionRouter()

    with pytest.raises(ValueError, match="must not be empty"):
        await router.route(" ", {"one": "only"})
    with pytest.raises(ValueError, match="at least two"):
        await router.route("task", {"one": "only"})


@pytest.mark.asyncio
async def test_route_uses_default_workflow_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The MCP tool exposes the configured workflow routes by default."""
    captured: dict[str, object] = {}

    async def fake_route(task: str, candidates: dict[str, str]) -> dict[str, object]:
        captured["task"] = task
        captured["candidates"] = candidates
        return {"selected": "implement", "candidates": candidates, "result": {}}

    monkeypatch.setattr(server.router, "route", fake_route)
    result = await server.route("Add an endpoint")

    assert '"selected":"implement"' in result
    assert captured["candidates"] == server.ROUTES["workflow"]


@pytest.mark.asyncio
async def test_route_requires_known_type_or_candidates() -> None:
    """A misspelled route type never silently picks an unrelated route."""
    with pytest.raises(ValueError, match="route_type must be one of"):
        await server.route("Task", route_type="missing")


@pytest.mark.asyncio
async def test_router_batches_multiple_route_sets() -> None:
    """Several exclusive decisions use one Laya prediction."""
    router = server.DecisionRouter()
    agent = FakeAgent()
    router._agent = agent

    result = await router.route_many(
        "Add a production endpoint",
        {
            "implementation": {"small": "minimal edit", "large": "broad redesign"},
            "verification": {"targeted": "focused checks", "full": "all checks"},
        },
    )

    assert agent.calls == 1
    assert result["routes"]["implementation"]["selected"] == "small"
    assert result["routes"]["verification"]["selected"] == "targeted"


@pytest.mark.asyncio
async def test_router_screens_multiple_applicable_capabilities() -> None:
    """Independent candidates retain scores instead of forcing one winner."""
    router = server.DecisionRouter()
    router._agent = FakeAgent(tool_probability=0.8)

    result = await router.screen(
        "Investigate a library regression",
        {"code": "Inspect source", "docs": "Read current documentation"},
        threshold=0.75,
    )

    assert result["selected"] == ["code", "docs"]
    assert result["scores"]["code"]["probability"] == 0.8


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_name", ("mlx", "torch", "onnx"))
async def test_backend_contract_is_stable_across_runtime_adapters(
    backend_name: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """All adapters expose valid choice and noul answers without numeric parity."""
    agent = FakeAgent(tool_probability=0.8)
    artifact_path: str | None = None
    if backend_name == "mlx":
        monkeypatch.setattr(backends.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(backends.platform, "machine", lambda: "arm64")
    if backend_name == "onnx":
        artifact = tmp_path / "laya.onnx"
        artifact.touch()
        artifact_path = str(artifact)

    def import_module(name: str) -> SimpleNamespace:
        if backend_name == "onnx":
            assert name == "laya.onnx_agent"
            return SimpleNamespace(ONNXAgent=lambda _model, _artifact: agent)
        expected = "laya_mlx" if backend_name == "mlx" else "laya"
        assert name == expected
        return SimpleNamespace(load=lambda _model: agent)

    monkeypatch.setattr(backends.importlib, "import_module", import_module)
    router = server.DecisionRouter(
        backend=backends.create_backend(
            backends.BackendConfig(
                requested=backend_name,
                name=backend_name,  # type: ignore[arg-type]
                model="test/laya",
                onnx_artifact_path=artifact_path,
            )
        )
    )

    result = await router.assess(
        "Inspect the source before deciding how to fix the issue.",
        {
            "workflow": {
                "explore": "Inspect source and facts before making a change.",
                "implement": "Make a code change immediately.",
            }
        },
        {
            "inspection_requested": (
                "The task explicitly asks to inspect source before changing code."
            )
        },
    )

    choice = result["choices"]["workflow"]
    probabilities = choice["probabilities"]
    assert choice["selected"] == "explore"
    assert set(probabilities) == {"explore", "implement"}
    assert all(0 <= probability <= 1 for probability in probabilities.values())
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert result["checks"]["inspection_requested"]["probability"] == 0.8


@pytest.mark.asyncio
async def test_recommend_returns_all_default_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The primary tool batches the configured defaults and MCP screening."""
    agent = FakeAgent()
    monkeypatch.setattr(server.router, "_agent", agent)

    result = json.loads(await server.recommend("Implement a new feature"))

    assert agent.calls == 1
    assert set(result["recommendations"]) == set(server.RECOMMENDATION_ROUTE_TYPES)
    assert result["tool_recommendations"] == []
    assert result["tool_selection"] == "top-1-margin"
    assert result["recommendations"]["workflow"] == "answer"


@pytest.mark.asyncio
async def test_mcp_servers_can_be_replaced_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The MCP candidates used by every public surface share one override."""
    candidates = {
        "source": "Inspect repository source code and symbols.",
        "tracker": "Read project issues and work tracking.",
    }
    monkeypatch.setenv(server.MCP_SERVERS_ENV, json.dumps(candidates))
    agent = FakeAgent()
    monkeypatch.setattr(server.router, "_agent", agent)

    recommendation = json.loads(await server.recommend("Investigate a regression"))
    route = json.loads(await server.route("Inspect the issue", route_type="mcp"))
    catalog = json.loads(server.catalog())

    assert recommendation["tool_recommendations"] == []
    assert route["candidates"] == candidates
    assert catalog["routes"]["mcp"] == candidates


@pytest.mark.asyncio
async def test_recommend_uses_a_clear_top_model_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A leading score selects one MCP instead of admitting every high score."""
    async def fake_assess(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "choices": {
                name: {"selected": next(iter(candidates))}
                for name, candidates in server.ROUTES.items()
                if name in server.RECOMMENDATION_ROUTE_TYPES
            },
            "checks": {
                "jcodemunch": {"probability": 0.62, "confidence": 0.62},
                "docs-mcp-server": {"probability": 0.45, "confidence": 0.45},
                "gitea": {"probability": 0.1, "confidence": 0.1},
                "asuswrt": {"probability": 0.1, "confidence": 0.1},
            },
            "result": {},
        }

    monkeypatch.setattr(server.router, "assess", fake_assess)

    result = json.loads(await server.recommend("Inspect the repository"))

    assert result["tool_recommendations"] == ["jcodemunch"]


@pytest.mark.asyncio
async def test_recommend_short_circuits_explicit_mcp_keywords(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit service identifiers do not depend on a model score."""
    candidates = {
        "jira": "Issue tracker",
        "bitbucket": "Pull requests and pipelines",
        "grafana": "Observability",
    }
    monkeypatch.setenv(server.MCP_SERVERS_ENV, json.dumps(candidates))
    captured: dict[str, object] = {}

    async def fake_assess(
        _task: str,
        route_sets: dict[str, dict[str, str]],
        checks: dict[str, str],
    ) -> dict[str, object]:
        captured["checks"] = checks
        return {
            "choices": {
                name: {"selected": next(iter(options))}
                for name, options in route_sets.items()
            },
            "checks": {},
            "result": {},
        }

    monkeypatch.setattr(server.router, "assess", fake_assess)

    result = json.loads(await server.recommend("Investigate ESAPI-123"))

    assert result["tool_recommendations"] == ["jira"]
    assert result["tool_selection"] == "keyword"
    assert captured["checks"] == {}


def test_mcp_server_override_requires_a_json_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed override fails with configuration guidance, not model output."""
    monkeypatch.setenv(server.MCP_SERVERS_ENV, "[\"source\"]")

    with pytest.raises(ValueError, match="LAYA_MCP_MCP_SERVERS must be a JSON object"):
        server.catalog()


def test_catalog_and_agent_resources_are_available_without_laya() -> None:
    """Agents can discover routes and operating guidance without model loading."""
    catalog = json.loads(server.catalog())

    assert {"requested", "active", "model", "loaded"} <= set(catalog["backend"])
    assert set(catalog["routes"]) == set(server.ROUTES)
    assert {"recommend", "route_many", "screen"} <= set(catalog["tools"])
    assert "Use `recommend`" in server.agent_guide()
    assert json.loads(server.route_catalog())["resources"] == [
        "laya://guide",
        "laya://routes",
    ]


@pytest.mark.asyncio
async def test_mcp_registers_routing_tools_and_documentation_resources() -> None:
    """Clients discover every agent-facing capability through the MCP protocol."""
    tools = {tool.name for tool in await server.mcp.list_tools()}
    resources = {str(resource.uri) for resource in await server.mcp.list_resources()}

    assert {"route", "route_many", "screen", "recommend", "catalog"} <= tools
    assert resources == {"laya://guide", "laya://routes"}


@pytest.mark.asyncio
async def test_router_rejects_empty_batch_and_invalid_threshold() -> None:
    """Batch routing and screening reject requests that cannot be meaningful."""
    router = server.DecisionRouter()

    with pytest.raises(ValueError, match="at least one route set"):
        await router.route_many("Task", {})
    with pytest.raises(ValueError, match="threshold must be a number from 0 to 1"):
        await router.screen("Task", {"docs": "Read docs"}, threshold=1.1)
