"""Small PyTorch decoder-only model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def _torch() -> Any:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for training") from exc
    return torch, nn


@dataclass(frozen=True)
class TorchDecoderConfig:
    vocab_size: int = 256
    hidden_size: int = 256
    intermediate_size: int = 768
    num_layers: int = 4
    num_heads: int = 8
    max_sequence_length: int = 256
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        values = (self.vocab_size, self.hidden_size, self.intermediate_size, self.num_layers, self.num_heads, self.max_sequence_length)
        if min(values) < 1:
            raise ValueError("model dimensions must be positive")
        if self.hidden_size % self.num_heads or (self.hidden_size // self.num_heads) % 2:
            raise ValueError("hidden size must divide into even attention heads")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rotate(values: Any, cosines: Any, sines: Any) -> Any:
    even = values[..., 0::2]
    odd = values[..., 1::2]
    rotated = values.clone()
    rotated[..., 0::2] = even * cosines - odd * sines
    rotated[..., 1::2] = even * sines + odd * cosines
    return rotated


def _build_model(config: TorchDecoderConfig) -> Any:
    torch, nn = _torch()
    head_size = config.hidden_size // config.num_heads

    class RMSNorm(nn.Module):
        def __init__(self, size: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(size))

        def forward(self, values):
            return values * torch.rsqrt(values.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * self.weight

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_norm = RMSNorm(config.hidden_size)
            self.post_norm = RMSNorm(config.hidden_size)
            self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

        def forward(self, hidden):
            batch, sequence, _ = hidden.shape
            normalized = self.input_norm(hidden)
            query = self.q_proj(normalized).view(batch, sequence, config.num_heads, head_size).transpose(1, 2)
            key = self.k_proj(normalized).view(batch, sequence, config.num_heads, head_size).transpose(1, 2)
            value = self.v_proj(normalized).view(batch, sequence, config.num_heads, head_size).transpose(1, 2)
            positions = torch.arange(sequence, device=hidden.device, dtype=hidden.dtype)
            frequencies = config.rope_theta ** (-torch.arange(0, head_size, 2, device=hidden.device, dtype=hidden.dtype) / head_size)
            angles = positions[:, None] * frequencies[None, :]
            cosines = angles.cos()[None, None, :, :]
            sines = angles.sin()[None, None, :, :]
            query = _rotate(query, cosines, sines)
            key = _rotate(key, cosines, sines)
            scores = query @ key.transpose(-2, -1) / head_size**0.5
            mask = torch.triu(torch.ones(sequence, sequence, device=hidden.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
            attention = torch.softmax(scores, dim=-1)
            attended = (attention @ value).transpose(1, 2).reshape(batch, sequence, config.hidden_size)
            hidden = hidden + self.o_proj(attended)
            normalized = self.post_norm(hidden)
            hidden = hidden + self.down_proj(torch.nn.functional.silu(self.gate_proj(normalized)) * self.up_proj(normalized))
            return hidden

    class Decoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
            self.layers = nn.ModuleList(Block() for _ in range(config.num_layers))
            self.norm = RMSNorm(config.hidden_size)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        def forward(self, token_ids):
            if token_ids.ndim != 2 or token_ids.shape[1] > config.max_sequence_length:
                raise ValueError("token sequence exceeds model limits")
            hidden = self.embedding(token_ids)
            for layer in self.layers:
                hidden = layer(hidden)
            return self.lm_head(self.norm(hidden))

    return Decoder()


class TinyCodeDecoder:
    """Factory wrapper that keeps PyTorch optional at import time."""

    def __new__(cls, config: TorchDecoderConfig | None = None):
        return _build_model(config or TorchDecoderConfig())
