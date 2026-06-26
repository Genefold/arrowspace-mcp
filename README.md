# ArrowSpace MCP

MCP (Model Context Protocol) server for [ArrowSpace](https://github.com/Mec-iS/arrowspace-rs) spectral vector search.

Exposes ArrowSpace's graph-Laplacian-based spectral search as MCP tools for AI agents.

## Quick start

```bash
uv run arrowspace-mcp
```

This starts a stdio MCP server. Configure it in Claude Desktop, Cursor, or OpenCode:

```json
{
  "mcpServers": {
    "arrowspace": {
      "command": "uvx",
      "args": ["arrowspace-mcp"]
    }
  }
}
```

## Transports

| Transport | Command |
|-----------|---------|
| stdio (default) | `uv run arrowspace-mcp` |
| SSE | `uv run arrowspace-mcp --transport sse --port 8765` |

## Tools

| Tool | Description |
|------|-------------|
| `build_index` | Build an ArrowSpace spectral index from vectors |
| `search` | Query with λτ spectral gating (planned) |
| `lambdas` | Per-item λτ scores (planned) |

## Development

```bash
uv pip install -e .
uv run arrowspace-mcp
```

## License

Apache 2.0
