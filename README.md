# ArrowSpace MCP

MCP (Model Context Protocol) server for [ArrowSpace](https://github.com/Mec-iS/arrowspace-rs) spectral vector search.

Exposes ArrowSpace's graph-Laplacian-based spectral search as MCP tools for AI agents.

## Choose by input format

Use the following workflow when choosing how to process user-provided data:

| Input | Recommended workflow |
|-------|----------------------|
| Zarr files | Use [arro-server](https://github.com/Genefold/arro-server) through its REST API. |
| Already embedded vectors in `.npy` files | Use Python ArrowSpace, following the [pyarrowspace tests](https://github.com/tuned-org-uk/pyarrowspace/tree/main/tests). |
| Raw text | Suggest embedding the text first with MiniLM (for example, `sentence-transformers/all-MiniLM-L6-v2`), `ibm-granite/granite-embedding-30m-english`, or `Snowflake/snowflake-arctic-embed-s`. Save the embeddings as `.npy` and follow the Python ArrowSpace workflow. |

For text retrieval, embed queries with the same model and its prescribed query
preprocessing. Rust and Python ArrowSpace provide Parquet persistence through
the `storage` feature; this is distinct from arro-server's REST/Zarr workflow.

These are workflow recommendations for agents. This MCP server currently accepts
inline vectors or a Zarr path; it does not automatically route requests to
arro-server, load `.npy` files, or embed text.

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if it is not
already available (macOS/Linux):

```bash
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
```

```bash
# Clone with the arrowspace-skills submodule
git clone --recurse-submodules https://github.com/Genefold/arrowspace-mcp
cd arrowspace-mcp
# (or, if already cloned: git submodule update --init)

# Use this environment in each shell where you run the commands below.
export UV_PROJECT_ENVIRONMENT=.venv-genefold
if [ ! -d "$UV_PROJECT_ENVIRONMENT" ]; then
    uv venv "$UV_PROJECT_ENVIRONMENT" --python 3.13
fi

# Run with spectral analysis support (scipy, scikit-learn)
uv sync --extra spectral
uv run arrowspace-mcp

# Run with everything, including the optional embedding models' dependencies
uv sync --extra all
uv run arrowspace-mcp
```

Alternatively, `bash install.sh` installs uv if needed, creates the environment
if absent, initialises submodules and installs spectral dependencies. Use
`bash install.sh --extra embeddings` to include embedding generation. Export
`UV_PROJECT_ENVIRONMENT=.venv-genefold` in your shell afterwards.

## Ingestion and embedding scripts

These standalone commands follow the input routing above. They are also
available as `uv run scripts/embed_text.py` and `uv run scripts/ingest.py`.
Run the Quick start setup first. Successful commands print a JSON summary;
errors exit non-zero. Local output directories must be new: existing data is
never overwritten. A completed bundle contains `manifest.json`.

### Generate text embeddings

```bash
uv sync --extra embeddings
uv run --extra embeddings arrowspace-embed documents.jsonl \
    --output data/embedded --model minilm --batch-size 32
```

Input is UTF-8 `.txt` (one document per non-blank line), `.jsonl` or `.ndjson`.
JSONL records default to `{"id": "doc-1", "text": "Document text"}`; use
`--text-field` and `--id-field` for other top-level fields. Missing IDs default
to source line numbers. Blank lines are skipped; empty or malformed records
are rejected. Pre-split long documents into chunks: model token limits still
apply, and the encoder can truncate text beyond them.

| `--model` alias | Model |
|----------------|-------|
| `minilm` (default) | `sentence-transformers/all-MiniLM-L6-v2` |
| `granite` | `ibm-granite/granite-embedding-30m-english` |
| `snowflake` | `Snowflake/snowflake-arctic-embed-s` |

Full Sentence Transformers model IDs and local model directories are also
accepted. Models download on first use. `--device` defaults to `cpu`; select
`cuda` or `mps` when available. `--revision` can pin a model commit. The seed is
3407; reproducibility across devices and dependency versions is not guaranteed.

The output bundle contains:
- `embeddings.npy`: row-ordered, 2-D float64 vectors, encoded in batches.
- `rows.jsonl`: row index, original ID, source line and text.
- `manifest.json`: model/revision, preprocessing, dimensions and input/output
  hashes. Keep this and the row mapping with the vectors for interpreting hits.

Embeddings are L2-normalised by default. `--no-normalise` disables this;
`--scale` applies an explicit positive multiplier afterwards (default 1.0).
No corpus-specific scaling from the upstream CVE experiments is assumed.
Text records are held in memory; embedding batches are written to a disk-backed
NumPy array. Existing bundles are not reused as an implicit embedding cache.

Encode queries with the **same model, revision, normalisation and scale**:

```bash
uv run --extra embeddings arrowspace-embed queries.txt \
    --output data/queries --model snowflake --role query
```

Snowflake queries automatically use its `query` prompt; documents do not.
MiniLM and Granite need no additional prompt. `--prompt-name` supports named
prompts in custom/local models.

### Ingest embedded NumPy data

```bash
uv run arrowspace-ingest npy data/embedded/embeddings.npy --output data/index
# Optional graph overrides: a JSON object with eps, k, topk, p and/or sigma
uv run arrowspace-ingest npy vectors.npy --output data/custom-index \
    --graph-params graph-params.json
```

The script validates a finite, non-empty real matrix with no zero vectors,
converts it to contiguous float64, and calls
`ArrowSpaceBuilder().with_seed(3407).build_and_store(...)`. It writes named
Parquet artefacts and a manifest containing the graph settings required to
reload them. Index construction loads the matrix into memory.

Defaults are `eps=1.0`, `k=6`, `topk=3`, `p=2.0`, `sigma=None` (uses `eps`).
They are starting points, not dataset-tuned values. Inspect
`near_zero_lambdas` in the summary; a degenerate graph needs different graph
parameters before spectral queries will be useful.

Reload and query with the Python API, as in the pyarrowspace tests:

```bash
uv run python - <<'PY'
import json
from pathlib import Path
import numpy as np
from arrowspace import load_arrowspace

root = Path("data/index")
meta = json.loads((root / "manifest.json").read_text())
aspace, graph = load_arrowspace(
    str(root), meta["dataset_name"], meta["graph_params"], meta["energy"]
)
# Query row 0 as a simple reload check; use matching query embeddings in practice.
query = np.load("data/embedded/embeddings.npy", allow_pickle=False)[0]
print(aspace.search(query, graph, tau=1.0))
PY
```

### Ingest Zarr through REST

```bash
uv run arrowspace-ingest zarr --server http://localhost:8000 \
    --dataset-id main--vectors.zarr --server-path /data/vectors.zarr
```

Run `arro-server` separately with the Zarr array under a configured data root.
`--server-path` is an absolute path on that server, not a client-side upload.
For a remote server, transfer/mount the Zarr data there first. The dataset ID
must match the configured root label and relative array path (use IDs returned
by `/api/datasets`; current arro-server uses `--` separators).

The script calls `POST /api/upload/commit`, then
`POST /api/datasets/{id}/index`. It propagates REST errors and never reads Zarr
locally. `--graph-params graph-params.json` supplies explicit overrides; omitting
it preserves the server's tuned/default settings. `--timeout` defaults to 300
seconds per request. A timeout does not cancel server-side work: check the
server before retrying. Indexing an existing dataset rebuilds its server index.

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
| `search` | Query with blended cosine + λτ spectral scoring |
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
git clone --recurse-submodules https://github.com/Genefold/arrowspace-mcp
cd arrowspace-mcp
git submodule update --init   # if cloned without --recurse-submodules
uv sync
uv run arrowspace-mcp
```

Skill resources (`arrowspace://skills/*`) are served from the
[`arrowspace-skills`](https://github.com/Genefold/arrowspace-skills) submodule.
The submodule is not included in PyPI distributions; clone the repo with
`--recurse-submodules` (or run `git submodule update --init`) to use them.

### With optional deps for spectral analysis

```bash
uv sync --extra spectral
```

## License

Apache 2.0
