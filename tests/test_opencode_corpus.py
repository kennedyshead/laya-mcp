"""Synthetic-only tests: never inspect private conversations or run models."""

import base64
import gzip
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "export_opencode_corpus",
    Path(__file__).parents[1] / "scripts/export_opencode_corpus.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def snapshot(tmp_path):
    source = tmp_path / "source.db"
    with sqlite3.connect(source) as database:
        database.executescript("""
            CREATE TABLE session (id TEXT PRIMARY KEY, data TEXT, payload BLOB);
            CREATE TABLE session_message (id TEXT PRIMARY KEY, data TEXT);
            CREATE TABLE credential (secret TEXT);
            CREATE TABLE kv (value TEXT);
            CREATE TABLE unknown_new_table (value TEXT);
            INSERT INTO credential VALUES ('do-not-export');
            INSERT INTO kv VALUES ('do-not-export');
            INSERT INTO unknown_new_table VALUES ('do-not-export');
        """)
        database.execute(
            "INSERT INTO session VALUES (?, ?, ?)",
            ("ses_test", '{"reasoning":"exact \\u2603"}', b"\x00\xff"),
        )
        database.execute(
            "INSERT INTO session_message VALUES (?, ?)",
            ("msg_test", '{"tool":"anything","output":"retained"}'),
        )
    (tmp_path / "source.json").write_text(
        json.dumps(
            {
                "captured_at": "20261008T093828Z",
                "version": "2.0.24",
                "active_sessions": ["ses_test"],
            }
        )
    )
    return source


def test_lossless_private_immutable_export(tmp_path):
    source = snapshot(tmp_path)
    before = source.read_bytes()
    batch = module.export_snapshot(source, tmp_path / "corpus", "test-host")
    manifest = json.loads((batch / "manifest.json").read_text())
    with gzip.open(batch / "records.jsonl.gz", "rt") as stream:
        records = [json.loads(line) for line in stream]
    assert manifest["counts"] == {"session": 1, "session_message": 1}
    assert set(manifest["excluded_tables"]) == {"credential", "kv", "unknown_new_table"}
    assert "do-not-export" not in json.dumps(records)
    assert records[0]["row"]["data"] == '{"reasoning":"exact \\u2603"}'
    assert (
        base64.b64decode(records[0]["row"]["payload"]["$sqlite_blob_base64"])
        == b"\x00\xff"
    )
    assert source.read_bytes() == before
    assert batch.stat().st_mode & 0o077 == 0
    assert (batch / "records.jsonl.gz").stat().st_mode & 0o077 == 0
    with pytest.raises(FileExistsError):
        module.export_snapshot(source, tmp_path / "corpus", "test-host")


def test_reject_public_output_and_bad_host(tmp_path):
    source = snapshot(tmp_path)
    public = tmp_path / "public"
    public.mkdir(mode=0o755)
    public.chmod(0o755)
    with pytest.raises(ValueError, match="0700"):
        module.export_snapshot(source, public, "host")
    with pytest.raises(ValueError, match="Host"):
        module.export_snapshot(source, tmp_path / "corpus", "../host")
