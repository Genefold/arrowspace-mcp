"""Ingest NumPy embeddings locally or register Zarr arrays through arro-server."""

from __future__ import annotations

import argparse
import json
import math
from importlib.metadata import version
from pathlib import Path
from urllib.parse import quote, urlsplit

import httpx
import numpy as np

from arrowspace_mcp.workflow_io import (
    file_digest,
    new_output_directory,
    validate_vectors,
    write_manifest,
)

DEFAULT_GRAPH_PARAMS = {"eps": 1.0, "k": 6, "topk": 3, "p": 2.0, "sigma": None}


def validate_graph_params(params: dict) -> None:
    """Reject malformed graph settings before crossing the native or REST boundary."""
    if not isinstance(params, dict) or params.keys() - DEFAULT_GRAPH_PARAMS.keys():
        raise ValueError(
            "Graph parameters must be an object with eps, k, topk, p or sigma keys."
        )
    for key, value in params.items():
        if key == "sigma" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{key} must be numeric.")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive.")
        if key in {"k", "topk"} and not isinstance(value, int):
            raise ValueError(f"{key} must be an integer.")


def ingest_npy(source: Path, output: Path, *, graph_params: dict | None = None) -> dict:
    """Build a seeded EigenMaps index and persist reloadable Parquet artefacts."""
    from arrowspace import ArrowSpaceBuilder

    if source.suffix.lower() != ".npy":
        raise ValueError(
            "Use .npy for embedded data; use the zarr command for arro-server."
        )
    if graph_params is not None:
        validate_graph_params(graph_params)
    params = DEFAULT_GRAPH_PARAMS | (graph_params or {})
    items = validate_vectors(np.load(source, allow_pickle=False, mmap_mode="r"))
    manifest = {
        "backend": "arrowspace",
        "arrowspace_version": version("arrowspace"),
        "source": str(source.resolve()),
        "input_sha256": file_digest(source),
        "dataset_name": "index",
        "graph_params": params,
        "energy": False,
        "seed": 3407,
        "n_items": int(items.shape[0]),
        "n_components": int(items.shape[1]),
    }
    with new_output_directory(output):
        aspace, _ = (
            ArrowSpaceBuilder()
            .with_seed(3407)
            .build_and_store(
                params,
                items,
                storage_path=str(output.resolve()),
                dataset_name="index",
            )
        )
        lambdas = np.asarray(aspace.lambdas())
        manifest["near_zero_lambdas"] = int(np.count_nonzero(np.abs(lambdas) < 1e-12))
        write_manifest(output / "manifest.json", manifest)
    return manifest


def ingest_zarr(
    server_url: str,
    dataset_id: str,
    server_path: str,
    *,
    graph_params: dict | None = None,
    timeout: float = 300.0,
    client: httpx.Client | None = None,
) -> dict:
    """Register an existing server-visible Zarr array, then request its index."""
    url = urlsplit(server_url)
    if (
        url.scheme not in {"http", "https"}
        or not url.netloc
        or url.query
        or url.fragment
    ):
        raise ValueError(
            "server_url must be an HTTP(S) base URL without query or fragment."
        )
    if not dataset_id or not server_path:
        raise ValueError("dataset_id and server_path are required.")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive.")
    if graph_params is not None:
        validate_graph_params(graph_params)
    if client is None:
        with httpx.Client() as session:
            return ingest_zarr(
                server_url,
                dataset_id,
                server_path,
                graph_params=graph_params,
                timeout=timeout,
                client=session,
            )
    api = server_url.rstrip("/") + "/api"
    registration = client.post(
        f"{api}/upload/commit",
        json={"dataset_id": dataset_id, "fs_path": server_path},
        timeout=timeout,
    )
    registration.raise_for_status()
    # Omitting graph_params preserves the server's tuned/default parameter selection.
    payload = {} if graph_params is None else {"graph_params": graph_params}
    result = client.post(
        f"{api}/datasets/{quote(dataset_id, safe='')}/index",
        json=payload,
        timeout=timeout,
    )
    result.raise_for_status()
    return result.json()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="format", required=True)
    local = commands.add_parser("npy", help="Build and store a local Parquet index")
    local.add_argument("input", type=Path)
    local.add_argument("--output", type=Path, required=True, help="New index directory")
    remote = commands.add_parser(
        "zarr", help="Register/index a Zarr array visible to arro-server"
    )
    remote.add_argument(
        "--server", required=True, help="Server base URL, e.g. http://localhost:8000"
    )
    remote.add_argument(
        "--dataset-id", required=True, help="arro-server ID, e.g. main--vectors.zarr"
    )
    remote.add_argument(
        "--server-path", required=True, help="Zarr array path on the SERVER filesystem"
    )
    remote.add_argument("--timeout", type=float, default=300.0)
    for command in (local, remote):
        command.add_argument(
            "--graph-params",
            type=Path,
            help="JSON object with graph parameter overrides",
        )
    args = parser.parse_args(argv)
    try:
        params = (
            json.loads(args.graph_params.read_text()) if args.graph_params else None
        )
        if args.format == "npy":
            result = ingest_npy(args.input, args.output, graph_params=params)
        else:
            result = ingest_zarr(
                args.server,
                args.dataset_id,
                args.server_path,
                graph_params=params,
                timeout=args.timeout,
            )
    except httpx.HTTPStatusError as exc:
        parser.exit(
            1,
            f"arro-server returned HTTP {exc.response.status_code}: {exc.response.text}\n",
        )
    except (TypeError, ValueError, OSError, httpx.RequestError) as exc:
        parser.exit(1, f"Ingestion failed: {exc}\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
