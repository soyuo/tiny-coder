"""Model manifest loading."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


class ManifestError(ValueError):
    """Raised for invalid model metadata."""


@dataclass(frozen=True)
class ModelManifest:
    """Metadata needed to locate model layers."""

    num_layers: int
    layer_digits: int = 2
    vocab_size: int | None = None
    hidden_size: int | None = None
    intermediate_size: int | None = None
    num_heads: int | None = None
    rope_theta: float = 10000.0

    @classmethod
    def load(cls, model_dir: str | Path) -> "ModelManifest":
        path = Path(model_dir) / "model.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ManifestError(f"manifest does not exist: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ManifestError(f"invalid manifest JSON: {path}") from exc

        if not isinstance(data, dict):
            raise ManifestError("manifest must contain a JSON object")
        num_layers = data.get("num_layers")
        layer_digits = data.get("layer_digits", 2)
        if not isinstance(num_layers, int) or num_layers < 1:
            raise ManifestError("num_layers must be a positive integer")
        if not isinstance(layer_digits, int) or layer_digits < 1:
            raise ManifestError("layer_digits must be a positive integer")
        config = {name: data.get(name) for name in ("vocab_size", "hidden_size", "intermediate_size", "num_heads")}
        for name, value in config.items():
            if value is not None and (not isinstance(value, int) or value < 1):
                raise ManifestError(f"{name} must be a positive integer")
        rope_theta = data.get("rope_theta", 10000.0)
        if not isinstance(rope_theta, (int, float)) or rope_theta <= 0:
            raise ManifestError("rope_theta must be positive")
        return cls(num_layers=num_layers, layer_digits=layer_digits, rope_theta=float(rope_theta), **config)
