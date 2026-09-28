"""Memory limit parsing."""

from __future__ import annotations

import re


class MemoryLimitError(ValueError):
    """Raised for an invalid memory limit."""


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
