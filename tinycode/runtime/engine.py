"""Minimal CPU-only runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

from .weights import WeightStore


LayerExecutor = Callable[[object, object], object]


class InferenceRuntime:
    """Execute layers while keeping weights on disk."""

    def __init__(self, model_dir: str | Path, layer_cache: int = 1) -> None:
        self.weights = WeightStore(model_dir, cache_size=layer_cache)

    def run_layers(
        self,
        hidden_state: object,
        layer_indices: Iterable[int],
        execute_layer: LayerExecutor,
    ) -> object:
        """Apply layers in order."""
        state = hidden_state
        for index in layer_indices:
            layer = self.weights.load_layer(index)
            state = execute_layer(state, layer)
            self.weights.unload_layer(index)
        return state

    def close(self) -> None:
        self.weights.close()

    def __enter__(self) -> "InferenceRuntime":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
