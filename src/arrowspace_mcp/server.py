from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import numpy as np
from mcp.server import FastMCP
from mcp.server.fastmcp import Context

from arrowspace_mcp.config import ServerConfig
from arrowspace_mcp.zzarr_loader import load_vectors_from_zzarr
from arrowspace_mcp.registry import IndexRegistry


@dataclass
class AppContext:
    registry: IndexRegistry
    allowed_paths: list[str]


def build_server(config: ServerConfig | None = None) -> FastMCP:
    if config is None:
        config = ServerConfig()

    @asynccontextmanager
    async def lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
        registry = IndexRegistry(ttl=config.ttl, max_entries=config.max_indexes)
        ctx = AppContext(registry=registry, allowed_paths=config.allowed_paths)
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
        zarr_path: str | None = None,
        eps: float = 1.0,
        k: int = 6,
        topk: int = 3,
        p: float = 2.0,
        sigma: float = 1.0,
    ) -> dict:
        from arrowspace import ArrowSpaceBuilder

        app_ctx: AppContext = ctx.request_context.lifespan_context

        if vectors is not None and zarr_path is not None:
            return {"error": "Provide either 'vectors' or 'zarr_path', not both."}
        if vectors is None and zarr_path is None:
            return {"error": "Provide either 'vectors' or 'zarr_path'."}

        if zarr_path is not None:
            items = load_vectors_from_zzarr(
                zarr_path, app_ctx.allowed_paths
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
        app_ctx: AppContext = ctx.request_context.lifespan_context
        entry = app_ctx.registry.get(index_id)
        q = np.array(query, dtype=np.float64)
        hits = entry.aspace.search(q, entry.gl, tau=tau)
        app_ctx.registry.touch(index_id)
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
        app_ctx.registry.touch(index_id)
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
        app_ctx.registry.touch(index_id)
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

    @server.tool()
    def spectral_analysis(
        ctx: Context,
        index_id: str,
        mode: str = "spectrum",
    ) -> dict:
        try:
            import scipy.sparse as sp
        except ImportError:
            return {
                "error": "scipy is required: pip install arrowspace-mcp[spectral]"
            }

        app_ctx: AppContext = ctx.request_context.lifespan_context
        entry = app_ctx.registry.get(index_id)
        app_ctx.registry.touch(index_id)
        csr = entry.gl.to_csr()
        L = sp.csr_matrix((csr[0], csr[1], csr[2]), shape=csr[3])
        n = L.shape[0]

        if mode == "spectrum":
            eigvals = np.linalg.eigvalsh(L.toarray())
            spectral_gap = 0.0
            sorted_vals = sorted(eigvals)
            for i in range(len(sorted_vals) - 1):
                gap = sorted_vals[i + 1] - sorted_vals[i]
                if gap > 1e-12:
                    spectral_gap = float(gap)
                    break
            estimated_rank = int(np.sum(eigvals > 1e-12))
            return {
                "n_eigenvalues": int(n),
                "eigenvalues": [float(v) for v in eigvals],
                "spectral_gap": spectral_gap,
                "estimated_rank": estimated_rank,
            }

        elif mode == "components":
            eigvals = np.linalg.eigvalsh(L.toarray())
            zero_multiplicity = int(np.sum(np.abs(eigvals) < 1e-10))
            return {
                "n_components": max(zero_multiplicity, 1),
                "zero_eigenvalue_multiplicity": zero_multiplicity,
            }

        elif mode == "clusters":
            k = min(entry.aspace.nclusters, n, 8)
            if k < 1:
                k = 1
            if k >= n:
                eigvecs = np.eye(n)
            else:
                from scipy.sparse.linalg import eigsh

                _, eigvecs = eigsh(L, k=k, which="SM", tol=1e-6)

            try:
                from sklearn.cluster import KMeans
            except ImportError:
                return {"error": "scikit-learn is required: pip install arrowspace-mcp[spectral]"}

            km = KMeans(n_clusters=k, random_state=3407, n_init=10)
            assignments = km.fit_predict(eigvecs)
            return {
                "n_clusters": int(k),
                "assignments": [int(a) for a in assignments],
                "method": "spectral_embedding_kmeans",
            }

        else:
            return {
                "error": f"Unknown mode: {mode}. "
                f"Use 'spectrum', 'components', or 'clusters'."
            }

    @server.tool()
    def delete_index(ctx: Context, index_id: str) -> dict:
        app_ctx: AppContext = ctx.request_context.lifespan_context
        app_ctx.registry.remove(index_id)
        return {"status": "deleted", "index_id": index_id}

    @server.tool()
    def suggest_params(
        n_items: int,
        n_dims: int,
        target_recall: float | None = None,
    ) -> dict:
        try:
            from arrowspace_skills import suggest_params as _sp
            base = _sp(n_items, n_dims)
        except ImportError:
            k = min(max(3, int(n_items / 50)), 25)
            topk = 3 if k <= 5 else 4
            eps = 0.1 if n_dims <= 128 else 0.2 if n_dims <= 768 else 0.5
            base = {"eps": eps, "k": k, "topk": topk, "p": 2.0, "sigma": None}

        notes = []
        if target_recall is not None:
            if target_recall > 0.95:
                base["k"] = min(base["k"] + 5, 50)
                base["topk"] = min(base["topk"] + 2, 10)
                notes.append("Increased k and topk for high recall target.")
            elif target_recall < 0.7:
                base["k"] = max(base["k"] - 2, 3)
                notes.append("Reduced k for lower recall target (faster).")

        if base["sigma"] is None:
            base["sigma"] = base["eps"]
            notes.append("sigma aligned to eps (default).")

        if n_dims > 512:
            notes.append("High-dimensional data; consider PCA or feature selection for better results.")
        if n_items < 100:
            notes.append("Small dataset; results may be unstable. Consider adding more data.")

        base["notes"] = " ".join(notes) if notes else "Defaults suitable for this dataset."
        return base

    import importlib.resources as _res

    _PACKAGE = "arrowspace_mcp"

    @server.resource(
        "arrowspace://skills/core",
        name="core",
        mime_type="text/markdown",
        description="Building an ArrowSpace index: builder, configuration, graph construction",
    )
    def _skill_core() -> str:
        return _res.files(_PACKAGE).joinpath("skills/arrowspace-core.md").read_text(encoding="utf-8")

    @server.resource(
        "arrowspace://skills/search",
        name="search",
        mime_type="text/markdown",
        description="Querying ArrowSpace with λτ scoring and tau tuning",
    )
    def _skill_search() -> str:
        return _res.files(_PACKAGE).joinpath("skills/arrowspace-search.md").read_text(encoding="utf-8")

    @server.resource(
        "arrowspace://skills/spectral",
        name="spectral",
        mime_type="text/markdown",
        description="Spectral analysis, diffusion, and eigenstructure",
    )
    def _skill_spectral() -> str:
        return _res.files(_PACKAGE).joinpath("skills/arrowspace-spectral.md").read_text(encoding="utf-8")

    @server.resource(
        "arrowspace://skills/hyperparameters",
        name="hyperparameters",
        mime_type="text/markdown",
        description="ArrowSpace parameter tuning guide",
    )
    def _skill_hyperparams() -> str:
        return _res.files(_PACKAGE).joinpath("skills/HYPERPARAMETERS.md").read_text(encoding="utf-8")

    @server.resource(
        "arrowspace://paper/abstract",
        name="joss-abstract",
        mime_type="text/plain",
        description="JOSS paper abstract for ArrowSpace",
    )
    def _joss_abstract() -> str:
        return (
            "ArrowSpace is a vector database and search library that augments "
            "nearest-neighbour search with spectral graph features. It computes a "
            "Laplacian over the item graph and uses the Rayleigh quotient to produce "
            "a lambda-tau (λτ) score per item, enabling search that respects both "
            "semantic similarity and structural role."
        )

    return server
