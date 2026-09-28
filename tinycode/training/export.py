"""Export PyTorch checkpoints to the TinyCode packed model format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..runtime.packed_format import pack_tensors
from ..runtime.tensor_format import pack_header
from .model import _torch


_LAYER_TENSORS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj", "input_norm", "post_norm",
)


def _tensor_bytes(tensor: Any) -> tuple[str, tuple[int, ...], bytes]:
    torch, _ = _torch()
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype == torch.float32:
        dtype = "float32"
    elif tensor.dtype == torch.float16:
        dtype = "float16"
    elif tensor.dtype == torch.int8:
        dtype = "int8"
    else:
        raise ValueError(f"unsupported checkpoint dtype: {tensor.dtype}")
    return dtype, tuple(tensor.shape), tensor.numpy().tobytes()


def _require(state: dict[str, Any], name: str) -> Any:
    try:
        return state[name]
    except KeyError as exc:
        raise ValueError(f"checkpoint is missing tensor: {name}") from exc


def export_checkpoint(checkpoint: str | Path, output_dir: str | Path) -> Path:
    """Convert a training checkpoint into a runtime model directory."""
    torch, _ = _torch()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model")
    config = payload.get("config")
    if not isinstance(state, dict) or not isinstance(config, dict):
        raise ValueError("checkpoint must contain model and config")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    manifest = {
        "num_layers": config["num_layers"],
        "vocab_size": config["vocab_size"],
        "hidden_size": config["hidden_size"],
        "intermediate_size": config["intermediate_size"],
        "num_heads": config["num_heads"],
        "rope_theta": config["rope_theta"],
    }
    (output / "model.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for name, key in (("embedding.bin", "embedding.weight"), ("norm.bin", "norm.weight"), ("lm_head.bin", "lm_head.weight")):
        dtype, shape, data = _tensor_bytes(_require(state, key))
        (output / name).write_bytes(pack_header(dtype, shape) + data)

    for index in range(config["num_layers"]):
        tensors = []
        for name in _LAYER_TENSORS:
            dtype, shape, data = _tensor_bytes(_require(state, f"layers.{index}.{name}.weight"))
            tensors.append((name, dtype, shape, data))
        (output / f"layer_{index:02d}.bin").write_bytes(pack_tensors(tensors))
    return output
