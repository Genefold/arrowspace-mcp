"""Generate document or query embeddings for the NumPy ingestion workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from arrowspace_mcp.workflow_io import (
    file_digest,
    new_output_directory,
    validate_vectors,
    write_manifest,
)

MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "granite": "ibm-granite/granite-embedding-30m-english",
    "snowflake": "Snowflake/snowflake-arctic-embed-s",
}
SEED = 3407


def read_records(source: Path, text_field: str, id_field: str) -> list[dict]:
    """Read non-blank text lines or JSONL records in their original order."""
    if source.suffix.lower() not in {".txt", ".jsonl", ".ndjson"}:
        raise ValueError("Text input must be .txt, .jsonl or .ndjson.")
    records = []
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = (
                json.loads(line)
                if source.suffix.lower() != ".txt"
                else {text_field: line.strip()}
            )
            text = record.get(text_field) if isinstance(record, dict) else None
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Line {line_number}: '{text_field}' must be non-empty text."
                )
            records.append(
                {
                    "row": len(records),
                    "id": record.get(id_field, str(line_number)),
                    "line": line_number,
                    "text": text,
                }
            )
    if not records:
        raise ValueError("Input contains no text records.")
    return records


def embed_file(
    source: Path,
    output: Path,
    *,
    model: str = "minilm",
    role: str = "document",
    text_field: str = "text",
    id_field: str = "id",
    batch_size: int = 32,
    device: str = "cpu",
    revision: str | None = None,
    normalise: bool = True,
    scale: float = 1.0,
    prompt_name: str | None = None,
    encoder=None,
) -> dict:
    """Write embeddings, source-row mapping and model settings as a new bundle."""
    if role not in {"document", "query"}:
        raise ValueError("role must be document or query.")
    if batch_size < 1 or not np.isfinite(scale) or scale <= 0:
        raise ValueError("batch_size and scale must be positive; scale must be finite.")
    records = read_records(source, text_field, id_field)
    model_id = MODELS.get(model, model)
    if prompt_name is None and role == "query" and model_id == MODELS["snowflake"]:
        prompt_name = "query"
    manifest = {
        "model": model_id,
        "revision": revision,
        "role": role,
        "prompt_name": prompt_name,
        "normalise": normalise,
        "scale": scale,
        "seed": SEED,
        "device": device,
        "batch_size": batch_size,
        "source": str(source.resolve()),
        "input_sha256": file_digest(source),
        "text_field": text_field,
        "id_field": id_field,
        "n_items": len(records),
        "dtype": "float64",
    }
    with new_output_directory(output):
        if encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                from transformers import set_seed
            except ImportError as exc:
                raise RuntimeError(
                    "Install embedding dependencies with uv sync --extra embeddings."
                ) from exc
            set_seed(SEED)
            encoder = SentenceTransformer(model_id, device=device, revision=revision)
        vectors = None
        try:
            for start in range(0, len(records), batch_size):
                texts = [
                    record["text"] for record in records[start : start + batch_size]
                ]
                # Explicit empty prompt suppresses any model default for documents.
                prompt = {"prompt_name": prompt_name} if prompt_name else {"prompt": ""}
                batch = validate_vectors(
                    np.asarray(
                        encoder.encode(
                            texts,
                            batch_size=batch_size,
                            convert_to_numpy=True,
                            show_progress_bar=False,
                            **prompt,
                        )
                    )
                )
                if len(batch) != len(texts):
                    raise ValueError(
                        "Encoder returned a different number of rows than the input."
                    )
                if normalise:
                    batch /= np.max(np.abs(batch), axis=1, keepdims=True)
                    batch /= np.linalg.norm(batch, axis=1, keepdims=True)
                batch = validate_vectors(batch * scale)
                if vectors is None:
                    manifest["n_components"] = batch.shape[1]
                    vectors = np.lib.format.open_memmap(
                        output / "embeddings.npy",
                        mode="w+",
                        dtype=np.float64,
                        shape=(len(records), batch.shape[1]),
                    )
                if batch.shape[1] != vectors.shape[1]:
                    raise ValueError(
                        "Encoder changed embedding dimensions between batches."
                    )
                vectors[start : start + len(batch)] = batch
            vectors.flush()
        finally:
            # Release the mapping before cleanup, including on platforms that lock open files.
            del vectors
        with (output / "rows.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(
                    json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                )
        manifest["embeddings_sha256"] = file_digest(output / "embeddings.npy")
        write_manifest(output / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="UTF-8 .txt (one record per line) or JSONL"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New output directory"
    )
    parser.add_argument(
        "--model",
        default="minilm",
        help="minilm, granite, snowflake, model ID or local model path",
    )
    parser.add_argument("--role", choices=["document", "query"], default="document")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--revision", help="Hugging Face model revision or commit")
    parser.add_argument(
        "--normalise", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--prompt-name", help="Override the model's named prompt")
    args = vars(parser.parse_args(argv))
    args["source"] = args.pop("input")
    try:
        result = embed_file(**args)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Embedding failed: {exc}\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
