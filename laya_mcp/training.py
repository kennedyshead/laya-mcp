"""Fine-tune Laya's decision head from a reviewed MCP-routing corpus."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Example:
    task: str
    label: str | None


def _load_examples(
    path: Path, servers: set[str], *, allow_weak_labels: bool = False
) -> list[Example]:
    examples: list[Example] = []
    with path.open(encoding="utf-8") as corpus:
        for line_number, line in enumerate(corpus, start=1):
            row = json.loads(line)
            task = row.get("task")
            label = row.get("label")
            if row.get("label_status") != "reviewed":
                observed = row.get("observed_mcps")
                if (
                    not allow_weak_labels
                    or not isinstance(observed, list)
                    or len(observed) != 1
                ):
                    continue
                label = observed[0]
                if label not in servers:
                    continue
            if not isinstance(task, str) or not task.strip():
                raise ValueError(f"line {line_number}: task must be a nonempty string")
            if label is not None and label not in servers:
                raise ValueError(
                    f"line {line_number}: label must name a configured MCP or null"
                )
            examples.append(Example(task=task.strip(), label=label))
    if len(examples) < 20:
        raise ValueError("at least 20 reviewed examples are required")
    return examples


def _training_item(
    example: Example,
    server: str,
    description: str,
    tokenizer: Any,
    config: dict[str, Any],
    build_sequence: Any,
    qtypes: dict[str, int],
) -> dict[str, Any] | None:
    question = {
        "t": "noul",
        "ins": (
            "Decide whether this capability materially helps complete the coding "
            "task. Answer true only when using it is appropriate."
        ),
        "crit": {
            "false": "This capability does not materially help this task.",
            "true": description,
        },
    }
    ids, markers = build_sequence(
        tokenizer,
        {"task": example.task},
        question,
        config["max_len"],
        config["head_max_len"],
    )
    if len(markers) != 2:
        return None
    positive = float(example.label == server)
    return {
        "ids": ids,
        "markers": markers,
        "qtype": qtypes["noul"],
        "target": [1.0 - positive, positive],
        "label": int(positive),
    }


def _collate(
    items: list[dict[str, Any]], pad_token_id: int, torch: Any
) -> dict[str, Any]:
    length = max(len(item["ids"]) for item in items)
    ids = torch.full((len(items), length), pad_token_id, dtype=torch.long)
    attention = torch.zeros((len(items), length), dtype=torch.long)
    markers = torch.zeros((len(items), 2), dtype=torch.long)
    marker_mask = torch.ones((len(items), 2), dtype=torch.bool)
    targets = torch.zeros((len(items), 2), dtype=torch.float32)
    for index, item in enumerate(items):
        item_length = len(item["ids"])
        ids[index, :item_length] = torch.tensor(item["ids"])
        attention[index, :item_length] = 1
        markers[index] = torch.tensor(item["markers"])
        targets[index] = torch.tensor(item["target"])
    return {
        "input_ids": ids,
        "attention_mask": attention,
        "marker_pos": markers,
        "marker_mask": marker_mask,
        "target": targets,
        "qtype": torch.tensor([item["qtype"] for item in items]),
        "label": torch.tensor([item["label"] for item in items]),
    }


def _evaluate(
    model: Any,
    items: list[dict[str, Any]],
    tokenizer: Any,
    device: Any,
    torch: Any,
) -> float:
    correct = 0
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(items), 16):
            batch = _collate(items[offset : offset + 16], tokenizer.pad_token_id, torch)
            logits, _ = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["marker_pos"].to(device),
                batch["marker_mask"].to(device),
                batch["qtype"].to(device),
            )
            correct += int((logits.argmax(dim=-1).cpu() == batch["label"]).sum())
    return correct / len(items)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--servers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="convaiinnovations/laya")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2.5e-5)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--allow-weak-labels", action="store_true")
    arguments = parser.parse_args()

    if not 0 < arguments.holdout_fraction < 0.5:
        raise ValueError("--holdout-fraction must be greater than 0 and less than 0.5")
    if arguments.epochs < 1 or arguments.batch_size < 1 or arguments.learning_rate <= 0:
        raise ValueError("epochs, batch size, and learning rate must be positive")

    import torch
    from huggingface_hub import snapshot_download
    from laya.agent import _fix_tokenizer_config
    from laya.common import QTYPES, build_model, build_sequence
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer

    servers = json.loads(arguments.servers.read_text(encoding="utf-8"))
    if not isinstance(servers, dict) or not all(
        isinstance(name, str) and isinstance(description, str)
        for name, description in servers.items()
    ):
        raise ValueError("--servers must be a JSON object of MCP names to descriptions")
    examples = _load_examples(
        arguments.corpus, set(servers), allow_weak_labels=arguments.allow_weak_labels
    )

    if arguments.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(arguments.device)
    if device.type == "cpu":
        raise RuntimeError("training requires a CUDA or Metal GPU")

    model_dir = Path(snapshot_download(arguments.model))
    _fix_tokenizer_config(str(model_dir))
    config = json.loads((model_dir / "rl_agent_config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    model = build_model(config, encoder_dir=str(model_dir / "encoder"))
    model.load_state_dict(load_file(model_dir / "model.safetensors"), strict=True)
    model.to(device)

    if arguments.freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable, lr=arguments.learning_rate, weight_decay=0.01
    )

    items = [
        item
        for example in examples
        for name, description in servers.items()
        if (
            item := _training_item(
                example, name, description, tokenizer, config, build_sequence, QTYPES
            )
        )
        is not None
    ]
    random.Random(arguments.seed).shuffle(items)
    holdout_size = max(1, round(len(items) * arguments.holdout_fraction))
    holdout, training = items[:holdout_size], items[holdout_size:]

    for epoch in range(arguments.epochs):
        random.Random(arguments.seed + epoch).shuffle(training)
        model.train()
        loss_total = 0.0
        for offset in range(0, len(training), arguments.batch_size):
            batch = _collate(
                training[offset : offset + arguments.batch_size],
                tokenizer.pad_token_id,
                torch,
            )
            logits, _ = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["marker_pos"].to(device),
                batch["marker_mask"].to(device),
                batch["qtype"].to(device),
            )
            loss = torch.nn.functional.cross_entropy(logits, batch["label"].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            loss_total += loss.detach().item()
        accuracy = _evaluate(model, holdout, tokenizer, device, torch)
        print(
            f"epoch {epoch + 1}/{arguments.epochs}: "
            f"loss={loss_total / max(1, len(training)):.4f} "
            f"holdout_accuracy={accuracy:.3f}"
        )

    arguments.output.mkdir(parents=True, exist_ok=True)
    weights = {
        name: value.half().contiguous().cpu()
        for name, value in model.state_dict().items()
    }
    save_file(weights, arguments.output / "model.safetensors")
    model.encoder.config.save_pretrained(arguments.output / "encoder")
    tokenizer.save_pretrained(arguments.output / "tokenizer")
    config["fine_tuned"] = True
    (arguments.output / "rl_agent_config.json").write_text(json.dumps(config, indent=2))
    (arguments.output / "training_report.json").write_text(
        json.dumps(
            {
                "examples": len(examples),
                "decisions": len(items),
                "holdout_accuracy": _evaluate(model, holdout, tokenizer, device, torch),
                "servers": list(servers),
                "model": arguments.model,
                "device": str(device),
                "weak_labels": arguments.allow_weak_labels,
                "freeze_encoder": arguments.freeze_encoder,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
