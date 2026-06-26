from __future__ import annotations

from pathlib import Path

import numpy as np


class ZarrLoadError(Exception):
    pass


class PathSecurityError(ZarrLoadError):
    pass


def _resolve_and_check(path_str: str, allowed_paths: list[str]) -> Path:
    path = Path(path_str).resolve()
    allowed = [Path(p).resolve() for p in allowed_paths]
    if not any(path.is_relative_to(base) for base in allowed):
        allowed_str = ", ".join(str(p) for p in allowed)
        raise PathSecurityError(
            f"Access denied: {path} is not in allowed paths ({allowed_str})"
        )
    if not path.exists():
        raise ZarrLoadError(f"Path not found: {path}")
    if not path.is_dir():
        raise ZarrLoadError(f"Expected a Zarr directory, got file: {path}")
    return path


def load_vectors_from_zzarr(
    zarr_path: str,
    allowed_paths: list[str] | None = None,
    row_start: int | None = None,
    row_end: int | None = None,
) -> np.ndarray:
    try:
        import zarr
    except ImportError as exc:
        raise ZarrLoadError(
            "zarr is required: pip install arrowspace-mcp[zarr]"
        ) from exc

    path = _resolve_and_check(zarr_path, allowed_paths or ["."])
    try:
        z = zarr.open(str(path), mode="r")
    except Exception as exc:
        raise ZarrLoadError(f"Failed to open Zarr array at {path}: {exc}") from exc

    if not isinstance(z, zarr.Array):
        raise ZarrLoadError(f"Expected a Zarr Array at {path}, got {type(z).__name__}")

    if len(z.shape) != 2:
        raise ZarrLoadError(
            f"Expected 2D array, got {len(z.shape)}D shape {z.shape}"
        )

    row_start = row_start or 0
    row_end = row_end or z.shape[0]
    arr = z[row_start:row_end]
    return np.ascontiguousarray(arr, dtype=np.float64)
