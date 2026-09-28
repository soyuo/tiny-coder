"""Decoder-only Transformer model."""

from __future__ import annotations

from dataclasses import dataclass, field
import mmap
from pathlib import Path
from typing import Any

from .manifest import ModelManifest
from .kv_cache import KVCacheStore
from .tensor_format import read_header
from .weights import WeightStore

def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for the decoder model") from exc
    return np


@dataclass(frozen=True)
class DecoderConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        if min(self.vocab_size, self.hidden_size, self.intermediate_size, self.num_layers, self.num_heads) < 1:
            raise ValueError("decoder dimensions must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden size must divide evenly across heads")
        if (self.hidden_size // self.num_heads) % 2:
            raise ValueError("head size must be even for rotary embeddings")


@dataclass
class DecoderKVCache:
    keys: list[Any | None] = field(default_factory=list)
    values: list[Any | None] = field(default_factory=list)
    key_buffers: list[Any | None] = field(default_factory=list)
    value_buffers: list[Any | None] = field(default_factory=list)
    sequence_length: int = 0
    disk_store: KVCacheStore | None = None

    def reset(self, num_layers: int) -> None:
        if self.disk_store is not None:
            self.disk_store.clear()
        self.keys = [None] * num_layers
        self.values = [None] * num_layers
        self.key_buffers = [None] * num_layers
        self.value_buffers = [None] * num_layers
        self.sequence_length = 0

    def load_layer(self, index: int) -> tuple[Any | None, Any | None]:
        if self.keys[index] is not None:
            return self.keys[index], self.values[index]
        if self.disk_store is None or not self.disk_store.contains(index):
            return None, None
        return self.disk_store.get_arrays(index)

    def save_layer(
        self,
        index: int,
        key: Any,
        value: Any,
        past_key: Any | None = None,
        past_value: Any | None = None,
    ) -> None:
        if self.disk_store is None:
            length = key.shape[1]
            previous_length = 0 if past_key is None else past_key.shape[1]
            total_length = previous_length + length
            buffer = self.key_buffers[index]
            value_buffer = self.value_buffers[index]
            if buffer is None or buffer.shape[1] < total_length:
                capacity = max(4, total_length, 0 if buffer is None else buffer.shape[1] * 2)
                buffer = _allocate_kv_buffer(key, capacity)
                value_buffer = _allocate_kv_buffer(value, capacity)
                if past_key is not None:
                    buffer[:, :previous_length] = past_key
                    value_buffer[:, :previous_length] = past_value
                self.key_buffers[index] = buffer
                self.value_buffers[index] = value_buffer
            buffer[:, previous_length:total_length] = key
            value_buffer[:, previous_length:total_length] = value
            self.keys[index] = buffer[:, :total_length]
            self.values[index] = value_buffer[:, :total_length]
            return
        np = _numpy()
        if past_key is not None:
            key = np.concatenate((past_key, key), axis=1)
            value = np.concatenate((past_value, value), axis=1)
        self.disk_store.put_arrays(index, key, value)
        self.keys[index] = None
        self.values[index] = None


class RMSNorm:
    def __init__(self, weight: Any, epsilon: float = 1e-6) -> None:
        self.weight = weight
        self.epsilon = epsilon

    def __call__(self, values: Any) -> Any:
        np = _numpy()
        values = np.asarray(values, dtype=np.float32)
        scale = np.sqrt(np.mean(values * values, axis=-1, keepdims=True) + self.epsilon)
        return values / scale * self.weight


class DecoderBlock:
    def __init__(self, config: DecoderConfig, weights: dict[str, Any]) -> None:
        self.config = config
        self.q_proj = weights["q_proj"]
        self.k_proj = weights["k_proj"]
        self.v_proj = weights["v_proj"]
        self.o_proj = weights["o_proj"]
        self.gate_proj = weights["gate_proj"]
        self.up_proj = weights["up_proj"]
        self.down_proj = weights["down_proj"]
        self.input_norm = RMSNorm(weights["input_norm"])
        self.post_norm = RMSNorm(weights["post_norm"])

    def __call__(self, hidden: Any) -> Any:
        np = _numpy()
        normalized = self.input_norm(hidden)
        sequence_length = normalized.shape[0]
        head_size = self.config.hidden_size // self.config.num_heads
        query = (normalized @ self.q_proj.T).reshape(sequence_length, self.config.num_heads, head_size).transpose(1, 0, 2)
        key = (normalized @ self.k_proj.T).reshape(sequence_length, self.config.num_heads, head_size).transpose(1, 0, 2)
        value = (normalized @ self.v_proj.T).reshape(sequence_length, self.config.num_heads, head_size).transpose(1, 0, 2)
        query = _apply_rope(query, self.config.rope_theta)
        key = _apply_rope(key, self.config.rope_theta)
        scores = query @ key.transpose(0, 2, 1) / np.sqrt(head_size)
        scores = np.where(np.triu(np.ones((sequence_length, sequence_length), dtype=bool), 1), -np.inf, scores)
        scores -= np.max(scores, axis=-1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
        attention = probabilities @ value
        attention = attention.transpose(1, 0, 2).reshape(sequence_length, self.config.hidden_size)
        hidden = hidden + attention @ self.o_proj.T

        normalized = self.post_norm(hidden)
        gate = normalized @ self.gate_proj.T
        up = normalized @ self.up_proj.T
        silu = gate / (1.0 + np.exp(-gate))
        hidden = hidden + (silu * up) @ self.down_proj.T
        return hidden

    def forward_cached(self, hidden: Any, past_key: Any | None, past_value: Any | None) -> tuple[Any, Any, Any]:
        np = _numpy()
        normalized = self.input_norm(hidden)
        sequence_length = normalized.shape[0]
        head_size = self.config.hidden_size // self.config.num_heads
        query = (normalized @ self.q_proj.T).reshape(sequence_length, self.config.num_heads, head_size).transpose(1, 0, 2)
        key = (normalized @ self.k_proj.T).reshape(sequence_length, self.config.num_heads, head_size).transpose(1, 0, 2)
        value = (normalized @ self.v_proj.T).reshape(sequence_length, self.config.num_heads, head_size).transpose(1, 0, 2)
        query = _apply_rope(query, self.config.rope_theta, self._position_start(past_key))
        key = _apply_rope(key, self.config.rope_theta, self._position_start(past_key))
        if past_key is None:
            scores = query @ key.transpose(0, 2, 1) / np.sqrt(head_size)
            scores = np.where(np.triu(np.ones((sequence_length, sequence_length), dtype=bool), 1), -np.inf, scores)
            scores -= np.max(scores, axis=-1, keepdims=True)
            probabilities = np.exp(scores)
            probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
            attention = probabilities @ value
        else:
            past_scores = query @ past_key.transpose(0, 2, 1) / np.sqrt(head_size)
            current_scores = query @ key.transpose(0, 2, 1) / np.sqrt(head_size)
            maximum = np.maximum(np.max(past_scores, axis=-1, keepdims=True), current_scores)
            past_weights = np.exp(past_scores - maximum)
            current_weights = np.exp(current_scores - maximum)
            denominator = np.sum(past_weights, axis=-1, keepdims=True) + current_weights
            attention = (past_weights @ past_value + current_weights * value) / denominator
        attention = attention.transpose(1, 0, 2).reshape(sequence_length, self.config.hidden_size)
        hidden = hidden + attention @ self.o_proj.T
        normalized = self.post_norm(hidden)
        gate = normalized @ self.gate_proj.T
        up = normalized @ self.up_proj.T
        silu = gate / (1.0 + np.exp(-gate))
        hidden = hidden + (silu * up) @ self.down_proj.T
        return hidden, key, value

    @staticmethod
    def _position_start(past_key: Any | None) -> int:
        return 0 if past_key is None else int(past_key.shape[1])


class DecoderOnlyTransformer:
    """Run a decoder-only Transformer on a token sequence."""

    def __init__(self, config: DecoderConfig, embedding: Any, blocks: list[DecoderBlock], final_norm: Any, lm_head: Any) -> None:
        if len(blocks) != config.num_layers:
            raise ValueError("block count does not match config")
        self.config = config
        self.embedding = embedding
        self.blocks = blocks
        self.final_norm = RMSNorm(final_norm)
        self.lm_head = lm_head

    def __call__(self, token_ids: Any) -> Any:
        np = _numpy()
        token_ids = np.asarray(token_ids, dtype=np.int64)
        if token_ids.ndim != 1:
            raise ValueError("token IDs must be a 1D sequence")
        if np.any(token_ids < 0) or np.any(token_ids >= self.config.vocab_size):
            raise ValueError("token ID is outside the vocabulary")
        hidden = self.embedding[token_ids]
        for block in self.blocks:
            hidden = block(hidden)
        return self.final_norm(hidden) @ self.lm_head.T

    def new_cache(self, disk_store: KVCacheStore | None = None) -> DecoderKVCache:
        cache = DecoderKVCache()
        cache.disk_store = disk_store
        cache.reset(self.config.num_layers)
        return cache

    def forward_cached(self, token_ids: Any, cache: DecoderKVCache) -> Any:
        np = _numpy()
        token_ids = np.asarray(token_ids, dtype=np.int64)
        if token_ids.ndim != 1 or token_ids.size == 0:
            raise ValueError("token IDs must be a non-empty 1D sequence")
        if np.any(token_ids < 0) or np.any(token_ids >= self.config.vocab_size):
            raise ValueError("token ID is outside the vocabulary")
        if len(cache.keys) != self.config.num_layers or len(cache.values) != self.config.num_layers:
            raise ValueError("cache does not match model layers")
        if cache.sequence_length and token_ids.size != 1:
            raise ValueError("cached decoding accepts one token after the initial sequence")
        hidden = self.embedding[token_ids]
        for index, block in enumerate(self.blocks):
            past_key, past_value = cache.load_layer(index)
            _validate_cached_kv(self.config, cache.sequence_length, past_key, past_value)
            hidden, key, value = block.forward_cached(hidden, past_key, past_value)
            cache.save_layer(index, key, value, past_key, past_value)
        cache.sequence_length += int(token_ids.size)
        return self.final_norm(hidden) @ self.lm_head.T


class DiskDecoderOnlyTransformer:
    """Run decoder layers by loading one packed layer at a time."""

    def __init__(self, model_dir: str | Path, manifest: ModelManifest, layer_cache: int = 1) -> None:
        required = (manifest.vocab_size, manifest.hidden_size, manifest.intermediate_size, manifest.num_heads)
        if any(value is None for value in required):
            raise ValueError("model manifest lacks decoder dimensions")
        self.config = DecoderConfig(
            manifest.vocab_size,
            manifest.hidden_size,
            manifest.intermediate_size,
            manifest.num_layers,
            manifest.num_heads,
            manifest.rope_theta,
        )
        self.model_dir = Path(model_dir)
        self.weights = WeightStore(self.model_dir, cache_size=layer_cache, manifest=manifest)
        self._tensor_resources = []
        self.embedding, resource = _load_tensor_file(self.model_dir / "embedding.bin")
        self._tensor_resources.append(resource)
        self.final_norm, resource = _load_tensor_file(self.model_dir / "norm.bin")
        self._tensor_resources.append(resource)
        self.lm_head, resource = _load_tensor_file(self.model_dir / "lm_head.bin")
        self._tensor_resources.append(resource)

    @classmethod
    def from_model_dir(cls, model_dir: str | Path, layer_cache: int = 1) -> "DiskDecoderOnlyTransformer":
        model_dir = Path(model_dir)
        return cls(model_dir, ModelManifest.load(model_dir), layer_cache=layer_cache)

    def __call__(self, token_ids: Any) -> Any:
        np = _numpy()
        token_ids = np.asarray(token_ids, dtype=np.int64)
        if token_ids.ndim != 1 or token_ids.size == 0 or np.any(token_ids < 0) or np.any(token_ids >= self.config.vocab_size):
            raise ValueError("invalid token sequence")
        hidden = self.embedding[token_ids]
        for index in range(self.config.num_layers):
            with self.weights.hold_packed_layer(index) as layer:
                packed = layer.packed_layer()
                block = DecoderBlock(self.config, _load_block_weights(self.config, packed))
                hidden = block(hidden)
        return RMSNorm(self.final_norm)(hidden) @ self.lm_head.T

    def new_cache(self, disk_store: KVCacheStore | None = None) -> DecoderKVCache:
        cache = DecoderKVCache()
        cache.disk_store = disk_store
        cache.reset(self.config.num_layers)
        return cache

    def forward_cached(self, token_ids: Any, cache: DecoderKVCache) -> Any:
        np = _numpy()
        token_ids = np.asarray(token_ids, dtype=np.int64)
        if token_ids.ndim != 1 or token_ids.size == 0 or np.any(token_ids < 0) or np.any(token_ids >= self.config.vocab_size):
            raise ValueError("invalid token sequence")
        if len(cache.keys) != self.config.num_layers or len(cache.values) != self.config.num_layers:
            raise ValueError("cache does not match model layers")
        if cache.sequence_length and token_ids.size != 1:
            raise ValueError("cached decoding accepts one token after the initial sequence")
        hidden = self.embedding[token_ids]
        for index in range(self.config.num_layers):
            past_key, past_value = cache.load_layer(index)
            _validate_cached_kv(self.config, cache.sequence_length, past_key, past_value)
            with self.weights.hold_packed_layer(index) as layer:
                packed = layer.packed_layer()
                block = DecoderBlock(self.config, _load_block_weights(self.config, packed))
                hidden, key, value = block.forward_cached(hidden, past_key, past_value)
            cache.save_layer(index, key, value, past_key, past_value)
        cache.sequence_length += int(token_ids.size)
        return RMSNorm(self.final_norm)(hidden) @ self.lm_head.T

    def close(self) -> None:
        self.weights.close()
        self.embedding = None
        self.final_norm = None
        self.lm_head = None
        for mapping, file_handle in self._tensor_resources:
            mapping.close()
            file_handle.close()

    def __enter__(self) -> "DiskDecoderOnlyTransformer":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _load_tensor_file(path: Path) -> Any:
    np = _numpy()
    file_handle = path.open("rb")
    try:
        mapping = mmap.mmap(file_handle.fileno(), length=0, access=mmap.ACCESS_READ)
        header = read_header(mapping)
        if header.data_offset + header.data_size > mapping.size():
            raise ValueError(f"tensor payload is truncated: {path}")
        dtype = np.dtype(header.dtype)
        values = np.frombuffer(mapping, dtype=dtype, count=header.data_size // dtype.itemsize, offset=header.data_offset).reshape(header.shape)
        return values, (mapping, file_handle)
    except Exception:
        if "mapping" in locals():
            mapping.close()
        file_handle.close()
        raise


def _load_block_weights(config: DecoderConfig, packed: Any) -> dict[str, Any]:
    expected = {
        "q_proj": (config.hidden_size, config.hidden_size),
        "k_proj": (config.hidden_size, config.hidden_size),
        "v_proj": (config.hidden_size, config.hidden_size),
        "o_proj": (config.hidden_size, config.hidden_size),
        "gate_proj": (config.intermediate_size, config.hidden_size),
        "up_proj": (config.intermediate_size, config.hidden_size),
        "down_proj": (config.hidden_size, config.intermediate_size),
        "input_norm": (config.hidden_size,),
        "post_norm": (config.hidden_size,),
    }
    weights = {}
    for name, shape in expected.items():
        values = packed.tensor_view(name).to_numpy()
        if values.shape != shape:
            raise ValueError(f"{name} has shape {values.shape}, expected {shape}")
        weights[name] = values
    return weights


def _validate_cached_kv(config: DecoderConfig, sequence_length: int, key: Any | None, value: Any | None) -> None:
    if key is None and value is None:
        return
    if key is None or value is None:
        raise ValueError("cached key and value must be provided together")
    expected = (config.num_heads, sequence_length, config.hidden_size // config.num_heads)
    if key.shape != expected or value.shape != expected:
        raise ValueError(f"cached key/value shape must be {expected}")


def _allocate_kv_buffer(values: Any, capacity: int) -> Any:
    np = _numpy()
    return np.empty((values.shape[0], capacity, values.shape[2]), dtype=values.dtype)


def _apply_rope(values: Any, theta: float, position_start: int = 0) -> Any:
    np = _numpy()
    _, sequence_length, head_size = values.shape
    positions = np.arange(position_start, position_start + sequence_length, dtype=np.float32)
    frequencies = theta ** (-np.arange(0, head_size, 2, dtype=np.float32) / head_size)
    angles = positions[:, None] * frequencies[None, :]
    cosines = np.cos(angles)[None, :, :]
    sines = np.sin(angles)[None, :, :]
    even = values[..., 0::2]
    odd = values[..., 1::2]
    rotated = np.empty_like(values)
    rotated[..., 0::2] = even * cosines - odd * sines
    rotated[..., 1::2] = even * sines + odd * cosines
    return rotated
