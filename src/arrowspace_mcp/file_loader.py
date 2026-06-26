from __future__ import annotations

from pathlib import Path

import numpy as np


class FileLoadError(Exception):
    pass


class PathSecurityError(FileLoadError):
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
        raise FileLoadError(f"File not found: {path}")
    return path


def load_vectors_from_file(
    file_path: str,
    file_format: str,
    allowed_paths: list[str] | None = None,
) -> np.ndarray:
    path = _resolve_and_check(file_path, allowed_paths or ["."])

    if file_format == "npy":
        arr = np.load(str(path), mmap_mode=None)
    elif file_format == "csv":
        arr = np.genfromtxt(str(path), delimiter=",", dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
    elif file_format == "parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise FileLoadError(
                "parquet format requires pyarrow: pip install pyarrow"
            ) from None
        table = pq.read_table(str(path))
        arr = table.to_pandas().to_numpy(dtype=np.float64)
    else:
        raise FileLoadError(f"Unsupported file format: {file_format}")

    if arr.ndim != 2:
        raise FileLoadError(
            f"Expected 2D array, got {arr.ndim}D shape {arr.shape}"
        )

    return np.ascontiguousarray(arr, dtype=np.float64)
