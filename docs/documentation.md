# Documentation discovery

## Authoritative sources

- [README](../README.md): installation, backends, MCP tools, policy training and
  specialist routing configuration. Experimental source is not proof of deployment.
- [Agent instructions](../AGENTS.md): tool selection and advisory boundaries.
- This page: where to look and how to distinguish committed search from drafts.
- The local cross-project map at `~/dotfiles/DOCUMENTATION.md`: exact project
  libraries, source directories and maintenance entry points.
- Local experiment plan at `~/dotfiles/docs/plans/model-routing-optimization.md`:
  measurements, rejected candidates, artifacts and unfinished deployment gates.
  Search it under the `dotfiles` / `committed` library after a verified docs commit;
  it is not part of the Laya projection. Read local drafts separately.

## Docs MCP status

Search project documentation with library **`laya-mcp`**, version **`committed`**.
Do not substitute a similarly named upstream library. Check index provenance and
local hook status for freshness; if unavailable, use the local sources above.

The manifest-scoped committed projection includes reviewed `main` README and regular
Markdown under `docs/` only. Raw Corpus, datasets, checkpoints, logs, private
runtime configuration, credentials, agent instructions and code are not crawled.
Uncommitted drafts and other branches remain out of the committed index.

Project lookup uses:

```json
{"library":"laya-mcp","version":"committed","query":"specialist routing validation gates","limit":5}
```

## Hooks, status and recovery

The shared indexer is supplied by the dotfiles `mcp` Stow package. This repo's
`.docs-index.json` declares the library and exact README/docs roots; private
connections remain outside Git at `~/.config/project-docs/laya-mcp.json`.

```sh
python3 ~/.scripts/docs-maintenance/project_docs.py --repo . --install-hooks
python3 ~/.scripts/docs-maintenance/project_docs.py --repo . --check-staged
python3 ~/.scripts/docs-maintenance/project_docs.py --repo . --status
python3 ~/.scripts/docs-maintenance/project_docs.py --repo . --force
```

Pre-commit checks staged Markdown (including additions/deletions), not arbitrary
working-tree text. Post-commit indexes manifest-scoped committed `main` Git blobs,
then verifies document count and provenance. Existing validators are preserved;
an existing different post-commit hook requires deliberate chaining, not replacement.
The hook never pushes or rolls back a successful commit. Index failures are recorded
in `~/.local/share/project-docs-index/laya-mcp/status.json`; inspect the private log
there and retry without widening roots. A pull may need an explicit retry.

Verify `_index-provenance.md` and a distinctive changed fact after an authorized
documentation commit before calling the new content searchable. Raw conversations,
datasets, models, code and private configuration never enter the project projection.

## Other information

Use jcodemunch for source symbols/references and the repository's declared checks
for implementation verification. Use Docs MCP's separately indexed upstream Laya
or dependency documentation only for those libraries' APIs, with verified snapshot
versions. Neither old project search results nor a training score establishes the
state of the active router: consult the experiment plan and verify live configuration.
