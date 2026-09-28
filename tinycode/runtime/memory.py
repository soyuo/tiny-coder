"""Memory limit parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass


class MemoryLimitError(ValueError):
    """Raised for an invalid memory limit."""


@dataclass(frozen=True)
class MemoryPlan:
    total_bytes: int
    weights_bytes: int
    kv_bytes: int
    context_bytes: int
    reserve_bytes: int
    layer_cache: int
    prefetch: bool


@dataclass(frozen=True)
class RuntimeMemoryUsage:
    weights_bytes: int
    kv_bytes: int
    context_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.weights_bytes + self.kv_bytes + self.context_bytes


def collect_memory_usage(
    weight_store: object | None = None,
    kv_store: object | None = None,
    context_bytes: int = 0,
) -> RuntimeMemoryUsage:
    """Collect bytes currently held by runtime caches and context."""
    if context_bytes < 0:
        raise MemoryLimitError("context bytes must be non-negative")
    weights_bytes = 0 if weight_store is None else int(weight_store.cached_bytes)
    kv_bytes = 0 if kv_store is None else int(kv_store.hot_size)
    return RuntimeMemoryUsage(weights_bytes, kv_bytes, context_bytes)


def plan_memory(
    memory_limit: int,
    layer_bytes: int,
    *,
    kv_ratio: float = 0.30,
    context_ratio: float = 0.10,
    reserve_ratio: float = 0.20,
) -> MemoryPlan:
    if memory_limit < 1 or layer_bytes < 1:
        raise MemoryLimitError("memory and layer sizes must be positive")
    ratios = (kv_ratio, context_ratio, reserve_ratio)
    if any(ratio < 0 or ratio >= 1 for ratio in ratios) or sum(ratios) >= 1:
        raise MemoryLimitError("memory ratios must be non-negative and sum below 1")
    kv_bytes = int(memory_limit * kv_ratio)
    context_bytes = int(memory_limit * context_ratio)
    reserve_bytes = int(memory_limit * reserve_ratio)
    weights_bytes = memory_limit - kv_bytes - context_bytes - reserve_bytes
    if weights_bytes < layer_bytes:
        raise MemoryLimitError("memory budget cannot fit one layer")
    layer_cache = weights_bytes // layer_bytes
    prefetch = weights_bytes >= layer_bytes * 2
    return MemoryPlan(
        total_bytes=memory_limit,
        weights_bytes=weights_bytes,
        kv_bytes=kv_bytes,
        context_bytes=context_bytes,
        reserve_bytes=reserve_bytes,
        layer_cache=layer_cache,
        prefetch=prefetch,
    )


def parse_memory_limit(value: str) -> int:
    """Convert a memory value such as ``512M`` to bytes."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]?B?)\s*", value.upper())
    if not match:
        raise MemoryLimitError(f"invalid memory limit: {value}")
    amount = float(match.group(1))
    unit = match.group(2).rstrip("B")
    multiplier = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}[unit]
    result = int(amount * multiplier)
    if result < 1:
        raise MemoryLimitError("memory limit must be positive")
    return result
