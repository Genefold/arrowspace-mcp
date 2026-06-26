# ArrowSpace MCP

MCP (Model Context Protocol) server for [ArrowSpace](https://github.com/Mec-iS/arrowspace-rs) spectral vector search.

Exposes ArrowSpace's graph-Laplacian-based spectral search as MCP tools for AI agents.

## Quick start

```bash
# Run with spectral analysis support (scipy, scikit-learn)
uv sync --extra spectral
uv run arrowspace-mcp

# Run with skill-based parameter suggestions
uv sync --extra skilled
uv run arrowspace-mcp

# Run with everything
uv sync --extra all
uv run arrowspace-mcp
```

## Client configuration

### Claude Desktop

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

### Cursor

Add to Cursor MCP configuration:

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

### OpenCode

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

## CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--transport` | `stdio` | Transport protocol (`stdio` or `sse`) |
| `--host` | `127.0.0.1` | Host to bind (SSE only) |
| `--port` | `8765` | Port to bind (SSE only) |
| `--allowed-paths` | `.` | Colon-separated allowed directories for Zarr input paths |
| `--ttl` | `3600` | Idle TTL in seconds; index expires when unused for this duration |
| `--max-indexes` | `100` | Maximum concurrent indexes |

## Tools

| Tool | Description |
|------|-------------|
| `build_index` | Build a spectral index from vectors (inline or Zarr path) |
| `search` | Query with λτ spectral gating |
| `lambdas` | Per-item λτ scores with optional summary stats |
| `lambdas_sorted` | Sorted λτ scores (least to most coherent) |
| `spectral_analysis` | Laplacian eigenvalue spectrum, components, or clusters |
| `suggest_params` | Heuristic graph parameters based on dataset size and dimensionality |
| `delete_index` | Remove an index from memory |

## SSE deployment

```bash
uv run arrowspace-mcp --transport sse --host 0.0.0.0 --port 8765
```

Connect via `http://<host>:8765/sse` with messages at `http://<host>:8765/messages/`.

## Development

```bash
git clone https://github.com/Genefold/arrowspace-mcp
cd arrowspace-mcp
uv sync
uv run arrowspace-mcp
```

### With optional deps for spectral analysis

```bash
uv sync --extra spectral --extra skilled
```

## License

Apache 2.0
