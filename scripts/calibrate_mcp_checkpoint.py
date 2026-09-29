"""Fit a binary MCP-screening temperature on the trainer's held-out tasks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from laya_mcp.training import _load_examples, _split_examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--servers", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()

    import torch
    from laya import Agent

    servers = json.loads(arguments.servers.read_text(encoding="utf-8"))
    examples = _load_examples(arguments.corpus, set(servers))
    holdout, _ = _split_examples(examples, arguments.holdout_fraction, arguments.seed)
    agent = Agent(str(arguments.model), device=arguments.device)
    logits: list[float] = []
    targets: list[float] = []
    for example in holdout:
        for name, description in servers.items():
            result = agent.predict(
                {"task": example.task},
                {
                    name: {
                        "type": "noul",
                        "instructions": (
                            "Decide whether this capability materially helps complete "
                            "the coding task. Answer true only when using it is "
                            "appropriate."
                        ),
                        "criteria": {
                            "false": (
                                "This capability does not materially help this task."
                            ),
                            "true": description,
                        },
                    }
                },
            )
            probability = result["answers"][name]["noul"]
            probability = min(max(float(probability), 1e-6), 1 - 1e-6)
            logits.append(math.log(probability / (1 - probability)))
            targets.append(float(example.label == name))

    raw_logits = torch.tensor(logits)
    target = torch.tensor(targets)
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            raw_logits / log_temperature.exp(), target
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    adjustment = float(torch.clamp(log_temperature.exp(), 0.5, 5.0).item())

    config_path = arguments.model / "rl_agent_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    temperatures = list(config.get("temperature", [1.0, 1.0, 1.0]))
    temperatures[2] = min(max(float(temperatures[2]) * adjustment, 0.5), 5.0)
    config["temperature"] = temperatures
    config.pop("temperature_by_options", None)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "holdout_decisions": len(targets),
                "noul_temperature": temperatures[2],
                "adjustment": adjustment,
            }
        )
    )


if __name__ == "__main__":
    main()
