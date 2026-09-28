"""Disk-backed KV cache."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path


class KVCacheError(RuntimeError):
    """Raised for invalid KV cache operations."""


class KVCacheStore:
    """Keep hot KV entries in memory and spill older entries to disk."""

    def __init__(self, cache_dir: str | Path, hot_capacity: int = 1) -> None:
        if hot_capacity < 1:
            raise ValueError("hot_capacity must be at least 1")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hot_capacity = hot_capacity
        self._hot: OrderedDict[int, bytes] = OrderedDict()

    def path_for(self, layer_index: int) -> Path:
        if layer_index < 0:
            raise ValueError("layer index must be non-negative")
        return self.cache_dir / f"layer_{layer_index:02d}.cache"

    def put(self, layer_index: int, value: bytes) -> None:
        """Store one layer's KV bytes."""
        if not isinstance(value, bytes):
            raise TypeError("KV value must be bytes")
        self.path_for(layer_index).write_bytes(value)
        self._hot[layer_index] = value
        self._hot.move_to_end(layer_index)
        self._evict_hot()

    def get(self, layer_index: int) -> bytes:
        """Return a KV entry and promote it to hot storage."""
        if layer_index in self._hot:
            value = self._hot.pop(layer_index)
            self._hot[layer_index] = value
            return value
        path = self.path_for(layer_index)
        if not path.is_file():
            raise KVCacheError(f"KV entry does not exist: {path}")
        value = path.read_bytes()
        self._hot[layer_index] = value
        self._evict_hot()
        return value

    def contains(self, layer_index: int) -> bool:
        return layer_index in self._hot or self.path_for(layer_index).is_file()

    def hot_layers(self) -> tuple[int, ...]:
        return tuple(self._hot)

    def clear_hot(self) -> None:
        self._hot.clear()

    def _evict_hot(self) -> None:
        while len(self._hot) > self.hot_capacity:
            self._hot.popitem(last=False)
