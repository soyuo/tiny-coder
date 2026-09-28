"""Layer tensor header format."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import BinaryIO, Iterable


MAGIC = b"TCLY"
VERSION = 1
_PREFIX = struct.Struct("<4sBBBB")
_DTYPE_SIZES = {"float32": 4, "float16": 2, "int8": 1}


class TensorFormatError(ValueError):
    """Raised for an invalid layer tensor header."""


@dataclass(frozen=True)
class TensorHeader:
    """Metadata for one layer tensor payload."""

    dtype: str
    shape: tuple[int, ...]
    data_offset: int
    data_size: int


def pack_header(dtype: str, shape: Iterable[int]) -> bytes:
    """Build a tensor header."""
    if dtype not in _DTYPE_SIZES:
        raise TensorFormatError(f"unsupported dtype: {dtype}")
    dimensions = tuple(shape)
    if not dimensions or any(not isinstance(dim, int) or dim < 1 for dim in dimensions):
        raise TensorFormatError("shape must contain positive integers")
    if len(dimensions) > 255:
        raise TensorFormatError("shape has too many dimensions")
    prefix = _PREFIX.pack(MAGIC, VERSION, _dtype_code(dtype), len(dimensions), 0)
    return prefix + struct.pack(f"<{len(dimensions)}I", *dimensions)


def read_header(stream: BinaryIO) -> TensorHeader:
    """Read and validate a tensor header."""
    prefix = stream.read(_PREFIX.size)
    if len(prefix) != _PREFIX.size:
        raise TensorFormatError("truncated tensor header")
    magic, version, dtype_code, ndim, _ = _PREFIX.unpack(prefix)
    if magic != MAGIC or version != VERSION:
        raise TensorFormatError("unsupported tensor format")
    if ndim == 0:
        raise TensorFormatError("tensor shape is empty")
    shape_data = stream.read(4 * ndim)
    if len(shape_data) != 4 * ndim:
        raise TensorFormatError("truncated tensor shape")
    shape = struct.unpack(f"<{ndim}I", shape_data)
    dtype = _dtype_name(dtype_code)
    data_offset = _PREFIX.size + 4 * ndim
    data_size = _DTYPE_SIZES[dtype]
    for dimension in shape:
        data_size *= dimension
    return TensorHeader(dtype, tuple(shape), data_offset, data_size)


def _dtype_code(dtype: str) -> int:
    return {"float32": 1, "float16": 2, "int8": 3}[dtype]


def _dtype_name(code: int) -> str:
    try:
        return {1: "float32", 2: "float16", 3: "int8"}[code]
    except KeyError as exc:
        raise TensorFormatError(f"unsupported dtype code: {code}") from exc
