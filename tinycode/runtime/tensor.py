"""CPU tensor views over mapped layer payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .tensor_format import TensorHeader


@dataclass(frozen=True)
class TensorView:
    """A tensor header paired with its mapped byte buffer."""

    header: TensorHeader
    buffer: Any

    def to_numpy(self) -> Any:
        """Return a safe NumPy copy of the mapped payload."""
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("NumPy is required for tensor decoding") from exc
        dtype = np.dtype(self.header.dtype)
        values = np.frombuffer(
            self.buffer,
            dtype=dtype,
            count=self.header.data_size // dtype.itemsize,
            offset=self.header.data_offset,
        ).reshape(self.header.shape)
        return values.copy()
