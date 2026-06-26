from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
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
    created_at: float = field(default_factory=time.time)
    last_access_at: float = field(default_factory=time.time)


class IndexRegistry:
    def __init__(
        self,
        ttl: float = 3600.0,
        max_entries: int = 100,
    ) -> None:
        self._entries: dict[str, IndexEntry] = {}
        self._ttl = ttl
        self._max_entries = max_entries

    def register(self, aspace: "ArrowSpace", gl: "GraphLaplacian", items: "np.ndarray") -> str:
        self._evict_expired()
        if len(self._entries) >= self._max_entries:
            self._evict_lru()

        index_id = str(uuid.uuid4())
        now = time.time()
        self._entries[index_id] = IndexEntry(
            aspace=aspace,
            gl=gl,
            n_items=items.shape[0],
            n_dims=items.shape[1],
            created_at=now,
            last_access_at=now,
        )
        return index_id

    def get(self, index_id: str) -> IndexEntry:
        entry = self._entries.get(index_id)
        if entry is None:
            raise ValueError(f"Index not found: {index_id}")
        if time.time() - entry.last_access_at > self._ttl:
            self.remove(index_id)
            raise ValueError(f"Index expired: {index_id}")
        entry.last_access_at = time.time()
        return entry

    def touch(self, index_id: str) -> None:
        entry = self._entries.get(index_id)
        if entry is not None:
            entry.last_access_at = time.time()

    def remove(self, index_id: str) -> bool:
        return self._entries.pop(index_id, None) is not None

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [
            k for k, v in self._entries.items()
            if now - v.last_access_at > self._ttl
        ]
        for k in expired:
            self.remove(k)

    def _evict_lru(self) -> None:
        if not self._entries:
            return
        lru = min(
            self._entries.items(),
            key=lambda kv: kv[1].last_access_at,
        )
        self.remove(lru[0])

    @property
    def entries(self) -> dict[str, IndexEntry]:
        self._evict_expired()
        return dict(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
