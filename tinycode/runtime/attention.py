"""CPU scaled dot-product attention."""

from __future__ import annotations

from typing import Any


def scaled_dot_product_attention(query: Any, key: Any, value: Any) -> Any:
    """Compute attention for 2D sequence matrices on the CPU."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for attention") from exc
    query = np.asarray(query, dtype=np.float32)
    key = np.asarray(key, dtype=np.float32)
    value = np.asarray(value, dtype=np.float32)
    if query.ndim != 2 or key.ndim != 2 or value.ndim != 2:
        raise ValueError("attention expects 2D matrices")
    if query.shape[1] != key.shape[1] or key.shape[0] != value.shape[0]:
        raise ValueError("attention dimensions do not match")
    scores = query @ key.T / np.sqrt(query.shape[1])
    scores -= scores.max(axis=1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities @ value
