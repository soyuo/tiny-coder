"""Packed tensor format for one Transformer layer."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import BinaryIO, Iterable

from .tensor import TensorView
from .tensor_format import TensorFormatError, _DTYPE_SIZES, _dtype_code, _dtype_name


MAGIC = b"TCLP"
VERSION = 1
_PREFIX = struct.Struct("<4sBBH")
_ENTRY = struct.Struct("<BBBB")


@dataclass(frozen=True)
class PackedTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_offset: int
    data_size: int


@dataclass(frozen=True)
class PackedLayer:
    tensors: dict[str, PackedTensor]
    buffer: object

    def tensor_view(self, name: str) -> TensorView:
        try:
            tensor = self.tensors[name]
        except KeyError as exc:
            raise TensorFormatError(f"tensor does not exist: {name}") from exc
        from .tensor_format import TensorHeader

        header = TensorHeader(tensor.dtype, tensor.shape, tensor.data_offset, tensor.data_size)
        return TensorView(header, self.buffer)


def pack_tensors(tensors: Iterable[tuple[str, str, tuple[int, ...], bytes]]) -> bytes:
    """Pack named tensor payloads into one layer file."""
    entries = list(tensors)
    metadata = bytearray(_PREFIX.pack(MAGIC, VERSION, 0, len(entries)))
    payload = bytearray()
    for name, dtype, shape, data in entries:
        name_bytes = name.encode("utf-8")
        dimensions = tuple(shape)
        expected = _tensor_size(dtype, dimensions)
        if len(name_bytes) > 255 or len(dimensions) > 255 or len(data) != expected:
            raise TensorFormatError(f"invalid packed tensor: {name}")
        metadata.extend(_ENTRY.pack(len(name_bytes), _dtype_code(dtype), len(dimensions), 0))
        metadata.extend(name_bytes)
        metadata.extend(struct.pack(f"<{len(dimensions)}I", *dimensions))
        payload.extend(data)
    return bytes(metadata + payload)


def read_packed_layer(stream: BinaryIO) -> PackedLayer:
    """Read a packed layer index."""
    prefix = stream.read(_PREFIX.size)
    if len(prefix) != _PREFIX.size:
        raise TensorFormatError("truncated packed layer header")
    magic, version, _, count = _PREFIX.unpack(prefix)
    if magic != MAGIC or version != VERSION:
        raise TensorFormatError("unsupported packed layer format")
    entries: list[tuple[str, str, tuple[int, ...], int]] = []
    for _ in range(count):
        raw_entry = stream.read(_ENTRY.size)
        if len(raw_entry) != _ENTRY.size:
            raise TensorFormatError("truncated packed tensor entry")
        name_size, dtype_code, ndim, _ = _ENTRY.unpack(raw_entry)
        name = stream.read(name_size).decode("utf-8")
        shape_data = stream.read(4 * ndim)
        if len(shape_data) != 4 * ndim or ndim == 0:
            raise TensorFormatError("truncated packed tensor shape")
        shape = struct.unpack(f"<{ndim}I", shape_data)
        dtype = _dtype_name(dtype_code)
        entries.append((name, dtype, tuple(shape), _tensor_size(dtype, shape)))
    data_offset = stream.tell()
    tensors: dict[str, PackedTensor] = {}
    for name, dtype, shape, data_size in entries:
        if name in tensors:
            raise TensorFormatError(f"duplicate packed tensor: {name}")
        tensors[name] = PackedTensor(name, dtype, shape, data_offset, data_size)
        data_offset += data_size
    if hasattr(stream, "seek"):
        stream.seek(0, 2)
        if data_offset > stream.tell():
            raise TensorFormatError("packed layer payload is truncated")
    return PackedLayer(tensors, stream)


def _tensor_size(dtype: str, shape: Iterable[int]) -> int:
    if dtype not in _DTYPE_SIZES:
        raise TensorFormatError(f"unsupported dtype: {dtype}")
    size = _DTYPE_SIZES[dtype]
    dimensions = tuple(shape)
    if not dimensions or any(dim < 1 for dim in dimensions):
        raise TensorFormatError("shape must contain positive dimensions")
    for dimension in dimensions:
        size *= dimension
    return size
