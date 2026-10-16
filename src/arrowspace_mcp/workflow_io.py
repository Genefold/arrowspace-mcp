"""Shared validation and output handling for offline workflow scripts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from threading import Lock

import numpy as np
from filelock import FileLock

_QUEUE_GUARD = Lock()
_WRITE_QUEUES: dict[Path, Lock] = {}


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_vectors(items: np.ndarray) -> np.ndarray:
    """Return a finite, non-empty float64 matrix without zero rows."""
    if items.ndim != 2 or 0 in items.shape or items.dtype.kind not in "fiu":
        raise ValueError("Expected a non-empty 2-D real numeric embedding matrix.")
    items = np.ascontiguousarray(items, dtype=np.float64)
    if not np.isfinite(items).all() or np.any(np.all(items == 0, axis=1)):
        raise ValueError("Embeddings must be finite and contain no zero vectors.")
    return items


def write_manifest(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


@contextmanager
def new_output_directory(path: Path):
    """Exclusively claim a new bundle and remove it on a failed write."""
    path = path.resolve()
    with _QUEUE_GUARD:
        queue = _WRITE_QUEUES.setdefault(path, Lock())
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    locks = cache / "arrowspace-mcp" / "workflow-locks"
    key = hashlib.sha256(os.fsencode(path)).hexdigest()
    with queue:
        locks.mkdir(parents=True, exist_ok=True)
        # Lock rendezvous files stay outside user output bundles and are never removed.
        with FileLock(locks / f"{key}.lock", timeout=0):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.mkdir()  # Atomic claim: an existing bundle is never overwritten.
            try:
                yield path
            except BaseException:
                shutil.rmtree(path)
                raise
