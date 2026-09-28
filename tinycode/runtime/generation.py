"""Autoregressive token generation."""

from __future__ import annotations

from typing import Any


def generate_greedy(model: Any, token_ids: list[int], max_new_tokens: int, eos_token_id: int | None = None) -> list[int]:
    """Append the highest-logit token at each step."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for generation") from exc
    generated = list(token_ids)
    for _ in range(max_new_tokens):
        logits = np.asarray(model(np.asarray(generated, dtype=np.int64)))
        if logits.ndim != 2 or logits.shape[0] != len(generated):
            raise ValueError("model must return [sequence, vocabulary] logits")
        next_token = int(np.argmax(logits[-1]))
        generated.append(next_token)
        if eos_token_id is not None and next_token == eos_token_id:
            break
    return generated
