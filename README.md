# ArrowSpace MCP

MCP (Model Context Protocol) server for [ArrowSpace](https://github.com/Mec-iS/arrowspace-rs) spectral vector search.

Exposes ArrowSpace's graph-Laplacian-based spectral search as MCP tools for AI agents.

## Quick start

```bash
uv run arrowspace_mcp
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
| `--allowed-paths` | `.` | Colon-separated allowed directories for `file_path` |
| `--ttl` | `3600` | Index TTL in seconds |
| `--max-indexes` | `100` | Maximum concurrent indexes |

## Tools

| Tool | Description |
|------|-------------|
| `build_index` | Build a spectral index from vectors (inline or file) |
| `search` | Query with λτ spectral gating |
| `lambdas` | Per-item λτ scores with optional summary stats |
| `lambdas_sorted` | Sorted λτ scores (least to most coherent) |
| `spectral_analysis` | Laplacian eigenvalue spectrum, components, or clusters |
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
uv pip install -e .
uv run arrowspace-mcp
```

### With optional deps for spectral analysis

```bash
uv pip install -e ".[spectral]"
```

## License

Apache 2.0
