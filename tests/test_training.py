"""Tests for the local reviewed-corpus training entry point."""

import json
from pathlib import Path

import pytest

from laya_mcp import training


def test_load_examples_uses_only_reviewed_labels(tmp_path: Path) -> None:
    """Weak candidates cannot accidentally enter a training run."""
    corpus = tmp_path / "corpus.jsonl"
    rows = [
        {"task": "Inspect source", "label": "jcodemunch", "label_status": "reviewed"},
        {"task": "Ignore this", "label": "gitea", "label_status": "needs_review"},
    ] * 20
    corpus.write_text("".join(json.dumps(row) + "\n" for row in rows))

    examples = training._load_examples(corpus, {"jcodemunch", "gitea"})

    assert len(examples) == 20
    assert examples[0] == training.Example(task="Inspect source", label="jcodemunch")


def test_load_examples_rejects_unknown_labels(tmp_path: Path) -> None:
    """A corpus cannot train against an MCP absent from the serving schema."""
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "".join(
            json.dumps(
                {
                    "task": "Inspect source",
                    "label": "missing",
                    "label_status": "reviewed",
                }
            )
            + "\n"
            for _ in range(20)
        )
    )

    with pytest.raises(ValueError, match="configured MCP"):
        training._load_examples(corpus, {"jcodemunch"})


def test_load_examples_requires_explicit_weak_label_opt_in(tmp_path: Path) -> None:
    """Observed MCP use remains excluded unless an experimental run requests it."""
    corpus = tmp_path / "corpus.jsonl"
    row = {
        "task": "Inspect source",
        "observed_mcps": ["jcodemunch"],
        "label": None,
        "label_status": "needs_review",
    }
    corpus.write_text("".join(json.dumps(row) + "\n" for _ in range(20)))

    with pytest.raises(ValueError, match="at least 20"):
        training._load_examples(corpus, {"jcodemunch"})
    assert len(
        training._load_examples(
            corpus, {"jcodemunch"}, allow_weak_labels=True
        )
    ) == 20


def test_load_examples_skips_weak_labels_outside_the_serving_schema(
    tmp_path: Path,
) -> None:
    """Historical MCPs cannot become labels for a replacement server configuration."""
    corpus = tmp_path / "corpus.jsonl"
    row = {
        "task": "Inspect router status",
        "observed_mcps": ["asuswrt"],
        "label": None,
        "label_status": "needs_review",
    }
    corpus.write_text("".join(json.dumps(row) + "\n" for _ in range(20)))

    with pytest.raises(ValueError, match="at least 20"):
        training._load_examples(
            corpus, {"jcodemunch"}, allow_weak_labels=True
        )
