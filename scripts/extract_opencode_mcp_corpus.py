"""Extract reviewable MCP-routing examples from local OpenCode sessions.

The OpenCode database remains local. This script emits only the first user task
and observed MCP names, which are weak labels requiring human review before use.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_DATABASE = Path.home() / ".local/share/opencode/opencode.db"
MCP_PREFIXES = {
    "jcodemunch_": "jcodemunch",
    "docs-mcp-server_": "docs-mcp-server",
    "gitea_": "gitea",
    "asuswrt_": "asuswrt",
}
PATH_PATTERN = re.compile(r"(?:/Users/|~?/)[^\s'\"]+")


def _mcp_name(tool: object) -> str | None:
    if not isinstance(tool, str):
        return None
    for prefix, name in MCP_PREFIXES.items():
        if tool.startswith(prefix):
            return name
    return None


def _sanitize(task: str) -> str:
    return PATH_PATTERN.sub("<path>", " ".join(task.split()))


def extract(database: Path) -> list[dict[str, Any]]:
    """Return one weakly labelled candidate for each session using one MCP."""
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT p.session_id, p.data
            FROM part AS p
            JOIN message AS m ON m.id = p.message_id
            WHERE json_extract(m.data, '$.role') = 'user'
              AND json_extract(p.data, '$.type') = 'text'
            ORDER BY p.time_created
            """
        )
        tasks: dict[str, str] = {}
        for session_id, data in rows:
            payload = json.loads(data)
            task = payload.get("text")
            if session_id not in tasks and isinstance(task, str) and task.strip():
                tasks[session_id] = _sanitize(task)

        rows = connection.execute(
            "SELECT session_id, data FROM part "
            "WHERE json_extract(data, '$.type') = 'tool'"
        )
        used: dict[str, set[str]] = defaultdict(set)
        for session_id, data in rows:
            payload = json.loads(data)
            name = _mcp_name(payload.get("tool"))
            if name:
                used[session_id].add(name)
    finally:
        connection.close()

    return [
        {
            "task": tasks[session_id],
            "observed_mcps": sorted(names),
            "label": None,
            "label_status": "needs_review",
        }
        for session_id, names in used.items()
        if session_id in tasks and len(names) == 1
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    candidates = extract(arguments.database)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", encoding="utf-8") as output:
        for candidate in candidates:
            output.write(json.dumps(candidate, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
