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
        return cls(num_layers=num_layers, layer_digits=layer_digits)
