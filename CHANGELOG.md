# Changelog

## 0.30.0 - 2026-10-16

- Align the MCP package version with ArrowSpace 0.30.0.
- Add `arrowspace-embed` for batched text/JSONL embeddings using MiniLM,
  Granite, Snowflake or a custom Sentence Transformers model. Save NumPy
  vectors, source-row mappings and model/preprocessing metadata.
- Add `arrowspace-ingest npy` for seeded, reloadable Parquet index builds.
- Add `arrowspace-ingest zarr` for REST registration and index builds through
  arro-server, using server-visible Zarr arrays.
- Add optional embedding dependencies and standalone script wrappers.
- Document uv installation and environment creation; use uv in `install.sh`.
