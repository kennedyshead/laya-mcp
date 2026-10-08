"""Preserve private OpenCode records from an offline SQLite snapshot.

Run on the source host: raw databases can contain credentials and must not be
transferred as corpus data. Batches are immutable, lossless for included rows,
and NOT sanitized or reviewed training examples. Repeat with a fresh snapshot
to extend the corpus; use host/table/key and row hashes for downstream dedup.
"""

from __future__ import annotations

import argparse
import base64
import datetime
import gzip
import hashlib
import json
import os
import sqlite3
import tempfile
import urllib.parse
from pathlib import Path

# Explicit allowlist: newly introduced tables require privacy review.
TABLES = frozenset(
    {
        "event",
        "event_sequence",
        "instruction_blob",
        "instruction_entry",
        "instruction_state",
        "message",
        "part",
        "permission",
        "project",
        "project_directory",
        "session",
        "session_inbox",
        "session_message",
        "session_pending",
        "session_v2",
        "todo",
        "workspace",
        "worktree",
    }
)


def encode(value: object) -> object:
    if isinstance(value, bytes):
        return {"$sqlite_blob_base64": base64.b64encode(value).decode("ascii")}
    return value


def canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_snapshot(snapshot: Path, output: Path, host: str) -> Path:
    """Create one batch; never alter the source database or existing batches."""
    if (
        not host
        or host in {".", ".."}
        or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
            for c in host
        )
    ):
        raise ValueError("Host must be a safe, explicit provenance label")
    snapshot = snapshot.resolve(strict=True)
    metadata = json.loads((snapshot.parent / "source.json").read_text())
    captured = metadata["captured_at"]
    # Trust no metadata-derived path components.
    datetime.datetime.strptime(captured, "%Y%m%dT%H%M%SZ")
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    if output.stat().st_mode & 0o077:
        raise ValueError("Corpus root must have mode 0700")
    parent = output / host
    parent.mkdir(mode=0o700, exist_ok=True)
    if parent.is_symlink() or parent.stat().st_mode & 0o077:
        raise ValueError("Host directory must be private and not a symlink")
    destination = parent / captured
    if destination.exists():
        raise FileExistsError(f"Immutable batch already exists: {destination}")
    uri = "file:" + urllib.parse.quote(str(snapshot)) + "?mode=ro&immutable=1"
    with tempfile.TemporaryDirectory(prefix=".export-", dir=parent) as temporary:
        staging = Path(temporary)
        records = staging / "records.jsonl.gz"
        counts = {}
        schemas = {}
        with sqlite3.connect(uri, uri=True) as database:
            if database.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("Snapshot integrity check failed")
            names = {
                row[0]
                for row in database.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            with gzip.open(records, "wb") as stream:
                for table in sorted(names & TABLES):
                    quoted = '"' + table.replace('"', '""') + '"'
                    columns = list(database.execute(f"PRAGMA table_info({quoted})"))
                    schemas[table] = {
                        "columns": columns,
                        "sql": database.execute(
                            "SELECT sql FROM sqlite_master "
                            "WHERE type='table' AND name=?",
                            (table,),
                        ).fetchone()[0],
                    }
                    keys = [
                        column[1]
                        for column in sorted(columns, key=lambda c: c[5])
                        if column[5]
                    ]
                    counts[table] = 0
                    cursor = database.execute(f"SELECT * FROM {quoted}")
                    fields = [column[0] for column in cursor.description]
                    for row in cursor:
                        data = dict(zip(fields, map(encode, row), strict=True))
                        record = {
                            "format": "opencode-sqlite-row-v1",
                            "host": host,
                            "table": table,
                            "key": {key: data[key] for key in keys},
                            "row_sha256": hashlib.sha256(canonical(data)).hexdigest(),
                            "row": data,
                        }
                        stream.write(canonical(record) + b"\n")
                        counts[table] += 1
        records.chmod(0o600)
        # Parse every emitted record, verifying hashes and row counts before publish.
        verified = {table: 0 for table in counts}
        with gzip.open(records, "rb") as stream:
            for line in stream:
                record = json.loads(line)
                if (
                    hashlib.sha256(canonical(record["row"])).hexdigest()
                    != record["row_sha256"]
                ):
                    raise ValueError("Record hash mismatch")
                verified[record["table"]] += 1
        if verified != counts:
            raise ValueError("Record count mismatch")
        manifest = {
            "format": "opencode-corpus-batch-v1",
            "host": host,
            "captured_at": captured,
            "opencode_version": metadata["version"],
            "active_sessions_at_capture": metadata.get("active_sessions", []),
            "counts": counts,
            "schemas": schemas,
            "excluded_tables": sorted(names - TABLES),
            "records_sha256": sha256(records),
            "privacy": "Private raw session content; not redacted or training-approved",
            "limitations": [
                "Only data persisted in this SQLite snapshot is retained",
                "No external attachment files, PTY buffers or service logs collected",
                "Absent or already-pruned history cannot be reconstructed",
                "Credentials, account stores, generic KV and share tokens excluded",
            ],
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_bytes(canonical(manifest) + b"\n")
        manifest_path.chmod(0o600)
        # mkdir is exclusive: a racing export must never overwrite a batch.
        destination.mkdir(mode=0o700)
        records.rename(destination / records.name)
        manifest_path.rename(destination / manifest_path.name)
    return destination


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--host", required=True)
    parser.add_argument(
        "--output", type=Path, default=Path.home() / ".local/state/laya/corpus/opencode"
    )
    args = parser.parse_args()
    path = export_snapshot(args.snapshot, args.output, args.host)
    manifest = json.loads((path / "manifest.json").read_text())
    print(
        json.dumps(
            {
                "batch": str(path),
                "counts": manifest["counts"],
                "excluded_tables": manifest["excluded_tables"],
            }
        )
    )


if __name__ == "__main__":
    main()
