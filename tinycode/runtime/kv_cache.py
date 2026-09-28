"""Disk-backed KV cache."""

from __future__ import annotations

from collections import OrderedDict
import io
from pathlib import Path
from typing import Any


class KVCacheError(RuntimeError):
    """Raised for invalid KV cache operations."""


class KVCacheStore:
    """Keep hot KV entries in memory and spill older entries to disk."""

    def __init__(self, cache_dir: str | Path, hot_capacity: int = 1, hot_bytes: int | None = None) -> None:
        if hot_capacity < 1:
            raise ValueError("hot_capacity must be at least 1")
        if hot_bytes is not None and hot_bytes < 1:
            raise ValueError("hot_bytes must be positive")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hot_capacity = hot_capacity
        self.hot_bytes = hot_bytes
        self._hot: OrderedDict[int, bytes] = OrderedDict()
        self._hot_size = 0

    def path_for(self, layer_index: int) -> Path:
        if layer_index < 0:
            raise ValueError("layer index must be non-negative")
        return self.cache_dir / f"layer_{layer_index:02d}.cache"

    def put(self, layer_index: int, value: bytes) -> None:
        """Store one layer's KV bytes."""
        if not isinstance(value, bytes):
            raise TypeError("KV value must be bytes")
        self.path_for(layer_index).write_bytes(value)
        previous = self._hot.pop(layer_index, None)
        if previous is not None:
            self._hot_size -= len(previous)
        self._hot[layer_index] = value
        self._hot_size += len(value)
        self._hot.move_to_end(layer_index)
        self._evict_hot()

    def put_arrays(self, layer_index: int, key: Any, value: Any) -> None:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("NumPy is required for array KV cache") from exc
        buffer = io.BytesIO()
        np.savez(buffer, key=np.asarray(key), value=np.asarray(value))
        self.put(layer_index, buffer.getvalue())

    def get(self, layer_index: int) -> bytes:
        """Return a KV entry and promote it to hot storage."""
        if layer_index in self._hot:
            value = self._hot.pop(layer_index)
            self._hot[layer_index] = value
            self._hot_size += len(value)
            return value
        path = self.path_for(layer_index)
        if not path.is_file():
            raise KVCacheError(f"KV entry does not exist: {path}")
        value = path.read_bytes()
        self._hot[layer_index] = value
        self._evict_hot()
        return value

    def get_arrays(self, layer_index: int) -> tuple[Any, Any]:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("NumPy is required for array KV cache") from exc
        with np.load(io.BytesIO(self.get(layer_index)), allow_pickle=False) as values:
            return values["key"], values["value"]

    def contains(self, layer_index: int) -> bool:
        return layer_index in self._hot or self.path_for(layer_index).is_file()

    def hot_layers(self) -> tuple[int, ...]:
        return tuple(self._hot)

    def clear_hot(self) -> None:
        self._hot.clear()
        self._hot_size = 0

    def clear(self) -> None:
        self._hot.clear()
        self._hot_size = 0
        for path in self.cache_dir.glob("layer_*.cache"):
            path.unlink()

    def _evict_hot(self) -> None:
        while len(self._hot) > self.hot_capacity or (
            self.hot_bytes is not None and self._hot_size > self.hot_bytes
        ):
            _, value = self._hot.popitem(last=False)
            self._hot_size -= len(value)

    @property
    def hot_size(self) -> int:
        return self._hot_size
