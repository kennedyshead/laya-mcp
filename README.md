# Laya MCP

Local MCP routing decisions for OpenCode, powered by [Laya MLX](https://github.com/mizorewww/laya-mlx).

The `route` tool selects a route for a coding task. It supports configured
`model`, `mcp`, and `workflow` candidates, or caller-supplied `custom`
candidates. The first request downloads the multilingual Laya model; later
requests use local Apple Silicon inference.

```json
{
  "task": "Investigate a TypeScript type error in an unfamiliar repository",
  "route_type": "workflow"
}
```

Routing is advisory. An MCP tool cannot directly change OpenCode's selected
model or invoke another MCP server; agents use the result to make that choice.

## Development

```shell
uv sync --extra dev
uv run pytest
uv run laya-mcp
```
