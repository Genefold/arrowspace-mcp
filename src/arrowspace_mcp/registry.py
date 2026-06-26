from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from arrowspace import ArrowSpace, GraphLaplacian


@dataclass
class IndexEntry:
    aspace: "ArrowSpace"
    gl: "GraphLaplacian"
    n_items: int
    n_dims: int


class IndexRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, IndexEntry] = {}

    def register(self, aspace: "ArrowSpace", gl: "GraphLaplacian", items: "np.ndarray") -> str:
        index_id = str(uuid.uuid4())
        self._entries[index_id] = IndexEntry(
            aspace=aspace,
            gl=gl,
            n_items=items.shape[0],
            n_dims=items.shape[1],
        )
        return index_id

    def get(self, index_id: str) -> IndexEntry:
        entry = self._entries.get(index_id)
        if entry is None:
            raise ValueError(f"Index not found: {index_id}")
        return entry

    def remove(self, index_id: str) -> None:
        self._entries.pop(index_id, None)

    def __len__(self) -> int:
        return len(self._entries)
