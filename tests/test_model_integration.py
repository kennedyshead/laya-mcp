"""Optional real-model contract checks for every configured Laya runtime."""

import asyncio
import os

import pytest

from laya_mcp import backends

RUN_MODEL_TESTS = os.environ.get("LAYA_MCP_RUN_MODEL_TESTS") == "1"
INTEGRATION_BACKENDS = tuple(
    name.strip()
    for name in os.environ.get("LAYA_MCP_INTEGRATION_BACKENDS", "torch").split(",")
    if name.strip()
)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(
    not RUN_MODEL_TESTS,
    reason="set LAYA_MCP_RUN_MODEL_TESTS=1 to run model-backed integration tests",
)
@pytest.mark.parametrize("backend_name", INTEGRATION_BACKENDS)
async def test_model_backend_contract(backend_name: str) -> None:
    """Choice labels and noul probability shape stay stable across backends."""
    if os.environ.get("LAYA_MCP_EXPECT_CUDA") == "1":
        import torch

        assert torch.cuda.is_available(), (
            "the CUDA contract job requires a CUDA torch build"
        )

    environ = dict(os.environ)
    environ[backends.BACKEND_ENV] = backend_name
    config = backends.resolve_backend_config(environ=environ)
    backend = backends.create_backend(config)
    agent = await asyncio.to_thread(backend.load)
    result = await asyncio.to_thread(
        agent.predict,
        "I was charged twice for invoice 4411. Please refund the duplicate.",
        {
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this request?",
                "criteria": {
                    "billing": "invoices, payments, refunds",
                    "technical": "bugs and outages",
                    "sales": "pricing and new purchases",
                },
            },
            "refund_requested": {
                "type": "noul",
                "instructions": "Does the customer ask for a refund?",
            },
        },
    )

    choice = result["answers"]["department"]
    probabilities = choice["probabilities"]
    assert choice["choice"] == "billing"
    assert set(probabilities) == {"billing", "technical", "sales"}
    assert all(0 <= probability <= 1 for probability in probabilities.values())
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=0.001)

    noul = result["answers"]["refund_requested"]
    assert 0 <= noul["noul"] <= 1
    assert 0 <= noul["confidence"] <= 1
