"""Small CPU tensor operations."""

from __future__ import annotations

from typing import Any


def matmul(left: Any, right: Any) -> Any:
    """Multiply two NumPy matrices on the CPU."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for CPU tensor operations") from exc
    left = np.asarray(left)
    right = np.asarray(right)
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("matmul expects two matrices")
    if left.shape[1] != right.shape[0]:
        raise ValueError("matmul dimensions do not match")
    return left @ right
