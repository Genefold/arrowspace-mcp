from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import numpy as np
from mcp.server import FastMCP
from mcp.server.fastmcp import Context

from arrowspace_mcp.file_loader import load_vectors_from_file
from arrowspace_mcp.registry import IndexRegistry


@dataclass
class AppContext:
    registry: IndexRegistry = field(default_factory=IndexRegistry)
    allowed_paths: list[str] = field(default_factory=lambda: ["."])


def build_server(allowed_paths: list[str] | None = None) -> FastMCP:
    if allowed_paths is None:
        allowed_paths = ["."]

    @asynccontextmanager
    async def lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
        ctx = AppContext(allowed_paths=allowed_paths)
        try:
            yield ctx
        finally:
            ctx.registry = None  # type: ignore[assignment]

    server = FastMCP(
        "arrowspace-mcp",
        instructions="MCP server for ArrowSpace spectral vector search. "
        "Build indexes, search with spectral awareness, and inspect per-item λτ scores.",
        lifespan=lifespan,
    )

    @server.tool()
    def build_index(
        ctx: Context,
        vectors: list[list[float]] | None = None,
        file_path: str | None = None,
        file_format: str = "npy",
        eps: float = 1.0,
        k: int = 6,
        topk: int = 3,
        p: float = 2.0,
        sigma: float = 1.0,
    ) -> dict:
        from arrowspace import ArrowSpaceBuilder

        app_ctx: AppContext = ctx.request_context.lifespan_context

        if vectors is not None and file_path is not None:
            return {"error": "Provide either 'vectors' or 'file_path', not both."}
        if vectors is None and file_path is None:
            return {"error": "Provide either 'vectors' or 'file_path'."}

        if file_path is not None:
            items = load_vectors_from_file(
                file_path, file_format, app_ctx.allowed_paths
            )
        else:
            items = np.array(vectors, dtype=np.float64)

        graph_params = {"eps": eps, "k": k, "topk": topk, "p": p, "sigma": sigma}
        aspace, gl = ArrowSpaceBuilder().build(graph_params, items)
        index_id = app_ctx.registry.register(aspace, gl, items)
        return {
            "index_id": index_id,
            "n_items": int(items.shape[0]),
            "n_components": int(items.shape[1]),
        }

    @server.tool()
    def search(
        ctx: Context,
        index_id: str,
        query: list[float],
        tau: float = 1.0,
    ) -> dict:
        from arrowspace import ArrowSpaceBuilder

        app_ctx: AppContext = ctx.request_context.lifespan_context
        entry = app_ctx.registry.get(index_id)
        q = np.array(query, dtype=np.float64)
        hits = entry.aspace.search(q, entry.gl, tau=tau)
        return {
            "hits": [{"index": int(idx), "score": float(sc)} for idx, sc in hits],
        }

    @server.tool()
    def lambdas(
        ctx: Context,
        index_id: str,
        include_stats: bool = False,
    ) -> dict:
        app_ctx: AppContext = ctx.request_context.lifespan_context
        entry = app_ctx.registry.get(index_id)
        scores = entry.aspace.lambdas()
        result: dict = {
            "n_items": int(entry.n_items),
            "lambdas": [float(s) for s in scores],
        }
        if include_stats:
            arr = np.array(scores)
            result["stats"] = {
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
            }
        return result

    @server.tool()
    def lambdas_sorted(
        ctx: Context,
        index_id: str,
        top_n: int | None = None,
    ) -> dict:
        app_ctx: AppContext = ctx.request_context.lifespan_context
        entry = app_ctx.registry.get(index_id)
        ranked = entry.aspace.lambdas_sorted()
        result: dict = {
            "n_items": int(entry.n_items),
            "ranked": [
                {"score": float(sc), "index": int(idx)} for sc, idx in ranked
            ],
        }
        if top_n is not None:
            result["ranked"] = result["ranked"][:top_n]
            result["truncated"] = True
        return result

    return server
