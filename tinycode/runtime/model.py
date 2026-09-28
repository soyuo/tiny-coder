"""Decoder-only Transformer model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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

    def __post_init__(self) -> None:
        if min(self.vocab_size, self.hidden_size, self.intermediate_size, self.num_layers, self.num_heads) < 1:
            raise ValueError("decoder dimensions must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden size must divide evenly across heads")


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
