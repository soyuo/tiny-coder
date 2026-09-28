"""Disk-backed layer storage."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import mmap
from pathlib import Path
from threading import RLock
from typing import Iterator
from contextlib import contextmanager

from .manifest import ModelManifest
from .packed_format import PackedLayer, read_packed_layer
from .tensor import TensorView
from .tensor_format import TensorHeader, TensorFormatError, read_header


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
    leases: int = 0

    def _close_mapping(self) -> None:
        self.mapping.close()
        self._file.close()

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

    def tensor_header(self) -> TensorHeader:
        """Read and validate the layer tensor header."""
        position = self.mapping.tell()
        self.mapping.seek(0)
        try:
            header = read_header(self.mapping)
        finally:
            self.mapping.seek(position)
        if header.data_offset + header.data_size > self.size:
            raise TensorFormatError("tensor payload is truncated")
        return header

    def tensor_view(self) -> TensorView:
        """Return a CPU tensor view over the mapped payload."""
        return TensorView(self.tensor_header(), self.mapping)

    def packed_layer(self) -> PackedLayer:
        """Read the packed tensor index from the mapped layer."""
        position = self.mapping.tell()
        self.mapping.seek(0)
        try:
            return read_packed_layer(self.mapping)
        finally:
            self.mapping.seek(position)


class LayerLease:
    """Owned access to a layer handle."""

    def __init__(self, store: "WeightStore", index: int, handle: LayerHandle) -> None:
        self._store = store
        self._index = index
        self.handle = handle
        self._released = False

    def __enter__(self) -> LayerHandle:
        return self.handle

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._released:
            self._released = True
            self._store.unload_layer(self._index)

    def __getattr__(self, name: str) -> object:
        return getattr(self.handle, name)

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
        self._lock = RLock()
        self._pinned: dict[int, int] = {}
        self._prefetcher = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tinycode-prefetch")

    def layer_path(self, index: int) -> Path:
        if index < 0:
            raise ValueError("layer index must be non-negative")
        digits = self.manifest.layer_digits if self.manifest else 2
        return self.model_dir / f"layer_{index:0{digits}d}.bin"

    def layer_count(self) -> int | None:
        """Return the manifest layer count, if loaded."""
        return self.manifest.num_layers if self.manifest else None

    def load_layer(self, index: int, *, lease: bool = True) -> LayerHandle:
        """Map a layer and return its handle."""
        with self._lock:
            if index in self._cache:
                handle = self._cache.pop(index)
                if lease:
                    handle.leases += 1
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
                handle = LayerHandle(index, path, size, mapping, file_handle, 1 if lease else 0)
            except Exception:
                file_handle.close()
                raise

            self._cache[index] = handle
            self._evict_excess()
            return handle

    def unload_layer(self, index: int) -> None:
        """Remove a cached layer mapping."""
        with self._lock:
            handle = self._cache.get(index)
            if handle is None:
                return
            if handle.leases > 0:
                handle.leases -= 1
                if handle.leases > 0:
                    return
            self._cache.pop(index, None)
            handle._close_mapping()

    @contextmanager
    def hold_layer(self, index: int) -> Iterator[LayerHandle]:
        """Keep a layer out of eviction while it is in use."""
        with self._lock:
            layer = self.load_layer(index)
            self._pinned[index] = self._pinned.get(index, 0) + 1
        try:
            yield layer
        finally:
            with self._lock:
                remaining = self._pinned.get(index, 1) - 1
                if remaining > 0:
                    self._pinned[index] = remaining
                else:
                    self._pinned.pop(index, None)
                    self.unload_layer(index)

    @contextmanager
    def hold_tensor_layer(self, index: int) -> Iterator[LayerHandle]:
        with self.hold_layer(index) as layer:
            layer.tensor_header()
            yield layer

    @contextmanager
    def hold_packed_layer(self, index: int) -> Iterator[LayerHandle]:
        with self.hold_layer(index) as layer:
            layer.packed_layer()
            yield layer

    def load_tensor_layer(self, index: int) -> LayerLease:
        """Map a layer and validate its tensor header."""
        layer = self.load_layer(index)
        try:
            layer.tensor_header()
        except Exception:
            self.unload_layer(index)
            raise
        return LayerLease(self, index, layer)

    def load_packed_layer(self, index: int) -> LayerLease:
        """Map and validate a packed layer."""
        layer = self.load_layer(index)
        try:
            layer.packed_layer()
        except Exception:
            self.unload_layer(index)
            raise
        return LayerLease(self, index, layer)

    def prefetch_layer(self, index: int) -> None:
        """Load a layer into the cache."""
        self.load_layer(index, lease=False)

    def prefetch_layer_async(self, index: int) -> Future[None]:
        """Load a layer on the prefetch worker."""
        return self._prefetcher.submit(self.prefetch_layer, index)

    def close(self) -> None:
        with self._lock:
            if any(handle.leases > 0 for handle in self._cache.values()):
                raise RuntimeError("cannot close weight store while layers are leased")
        self._prefetcher.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            handles = list(self._cache.values())
            self._cache.clear()
            self._pinned.clear()
        for handle in handles:
            handle._close_mapping()

    def __enter__(self) -> "WeightStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def cached_layers(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(self._cache)

    def __iter__(self) -> Iterator[int]:
        return iter(self._cache)

    def _evict_excess(self) -> None:
        while len(self._cache) > self.cache_size:
            candidate = next(((index, handle) for index, handle in self._cache.items() if index not in self._pinned and handle.leases == 0), None)
            if candidate is None:
                return
            oldest_index, oldest = candidate
            del self._cache[oldest_index]
            oldest._close_mapping()
