"""CPU scaled dot-product attention."""

from __future__ import annotations

from typing import Any


def scaled_dot_product_attention(query: Any, key: Any, value: Any) -> Any:
    """Compute scaled dot-product attention for 2D or batched inputs."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for attention") from exc
    query = np.asarray(query, dtype=np.float32)
    key = np.asarray(key, dtype=np.float32)
    value = np.asarray(value, dtype=np.float32)
    if query.ndim not in (2, 3) or key.ndim != query.ndim or value.ndim != query.ndim:
        raise ValueError("attention expects matching 2D or 3D tensors")
    if query.shape[:-2] != key.shape[:-2] or key.shape[:-2] != value.shape[:-2]:
        raise ValueError("attention batch dimensions do not match")
    if query.shape[-1] != key.shape[-1] or key.shape[-2] != value.shape[-2]:
        raise ValueError("attention dimensions do not match")
    scores = query @ np.swapaxes(key, -1, -2) / np.sqrt(query.shape[-1])
    scores -= scores.max(axis=-1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return probabilities @ value
