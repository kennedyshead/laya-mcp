"""Validate reviewed MCP-routing corpus labels and balance before training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--servers", type=Path, required=True)
    parser.add_argument("--minimum-per-label", type=int, default=20)
    arguments = parser.parse_args()

    servers = json.loads(arguments.servers.read_text(encoding="utf-8"))
    labels = set(servers) | {None}
    counts: Counter[str | None] = Counter()
    tasks: set[str] = set()
    with arguments.corpus.open(encoding="utf-8") as corpus:
        for line_number, line in enumerate(corpus, start=1):
            row = json.loads(line)
            task = row.get("task")
            label = row.get("label")
            if row.get("label_status") != "reviewed":
                raise ValueError(f"line {line_number}: label_status must be reviewed")
            if not isinstance(task, str) or not task.strip() or task in tasks:
                raise ValueError(
                    f"line {line_number}: task must be unique and nonempty"
                )
            if label not in labels:
                raise ValueError(
                    f"line {line_number}: label is not in the serving schema"
                )
            tasks.add(task)
            counts[label] += 1

    if any(counts[label] < arguments.minimum_per_label for label in labels):
        raise ValueError(f"unbalanced corpus: {dict(counts)}")
    print(json.dumps({"examples": len(tasks), "labels": dict(counts)}, default=str))


if __name__ == "__main__":
    main()
