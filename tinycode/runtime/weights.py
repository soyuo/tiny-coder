"""Disk-backed layer storage."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import mmap
from pathlib import Path
from typing import Iterator

from .manifest import ModelManifest


class WeightStoreError(RuntimeError):
    """Raised when the on-disk model layout cannot be used."""


@dataclass
class LayerHandle:
    """An open, read-only layer mapping."""

    index: int
    path: Path
    size: int
    mapping: mmap.mmap
    _file: object

    def close(self) -> None:
        self.mapping.close()
        self._file.close()

    def __enter__(self) -> "LayerHandle":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __len__(self) -> int:
        return self.size

    def read(self, start: int = 0, length: int | None = None) -> bytes:
        """Read a byte range from the layer."""
        if start < 0 or start > self.size:
            raise ValueError("start is outside the layer")
        end = self.size if length is None else start + length
        if end < start or end > self.size:
            raise ValueError("requested range is outside the layer")
        return self.mapping[start:end]


class WeightStore:
    """Read layers from disk with a bounded LRU cache."""

    def __init__(
        self,
        model_dir: str | Path,
        cache_size: int = 1,
        manifest: ModelManifest | None = None,
    ) -> None:
        if cache_size < 1:
            raise ValueError("cache_size must be at least 1")
        self.model_dir = Path(model_dir)
        self.cache_size = cache_size
        self.manifest = manifest
        self._cache: OrderedDict[int, LayerHandle] = OrderedDict()

    def layer_path(self, index: int) -> Path:
        if index < 0:
            raise ValueError("layer index must be non-negative")
        digits = self.manifest.layer_digits if self.manifest else 2
        return self.model_dir / f"layer_{index:0{digits}d}.bin"

    def layer_count(self) -> int | None:
        """Return the manifest layer count, if loaded."""
        return self.manifest.num_layers if self.manifest else None

    def load_layer(self, index: int) -> LayerHandle:
        """Map a layer and return its handle."""
        if index in self._cache:
            handle = self._cache.pop(index)
            self._cache[index] = handle
            return handle

        path = self.layer_path(index)
        if not path.is_file():
            raise WeightStoreError(f"layer file does not exist: {path}")

        file_handle = path.open("rb")
        try:
            size = path.stat().st_size
            if size == 0:
                raise WeightStoreError(f"layer file is empty: {path}")
            mapping = mmap.mmap(file_handle.fileno(), length=0, access=mmap.ACCESS_READ)
            handle = LayerHandle(index, path, size, mapping, file_handle)
        except Exception:
            file_handle.close()
            raise

        self._cache[index] = handle
        self._evict_excess()
        return handle

    def unload_layer(self, index: int) -> None:
        """Remove a cached layer mapping."""
        handle = self._cache.pop(index, None)
        if handle is not None:
            handle.close()

    def prefetch_layer(self, index: int) -> None:
        """Load a layer into the cache."""
        self.load_layer(index)

    def close(self) -> None:
        for index in list(self._cache):
            self.unload_layer(index)

    def __enter__(self) -> "WeightStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def cached_layers(self) -> tuple[int, ...]:
        return tuple(self._cache)

    def __iter__(self) -> Iterator[int]:
        return iter(self._cache)

    def _evict_excess(self) -> None:
        while len(self._cache) > self.cache_size:
            oldest_index, oldest = self._cache.popitem(last=False)
            del oldest_index
            oldest.close()
