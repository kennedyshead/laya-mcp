"""Tests for Laya MCP routing behavior."""

import pytest

from laya_mcp import server


class FakeAgent:
    """Small Laya test double that chooses the first supplied candidate."""

    def predict(
        self,
        _state: object,
        questions: dict[str, object],
    ) -> dict[str, object]:
        criteria = questions["route"]["criteria"]  # type: ignore[index]
        return {
            "answers": {
                "route": {
                    "choice": next(iter(criteria)),
                    "probabilities": {},
                }
            }
        }


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
