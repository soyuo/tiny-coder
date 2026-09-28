"""Autoregressive token generation."""

from __future__ import annotations

from typing import Any


def generate_greedy(model: Any, token_ids: list[int], max_new_tokens: int, eos_token_id: int | None = None) -> list[int]:
    """Append the highest-logit token at each step."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if not token_ids:
        raise ValueError("token_ids must not be empty")
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


def generate_greedy_cached(model: Any, token_ids: list[int], max_new_tokens: int, eos_token_id: int | None = None) -> list[int]:
    """Generate tokens while reusing decoder key/value states."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if not token_ids:
        raise ValueError("token_ids must not be empty")
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for generation") from exc
    if not hasattr(model, "new_cache") or not hasattr(model, "forward_cached"):
        raise TypeError("model must support cached decoding")
    cache = model.new_cache()
    return _generate_with_cache(model, token_ids, max_new_tokens, eos_token_id, cache)


def generate_greedy_cached_with_cache(
    model: Any,
    token_ids: list[int],
    max_new_tokens: int,
    cache: Any,
    eos_token_id: int | None = None,
) -> list[int]:
    """Generate tokens with a caller-provided KV cache."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if not token_ids:
        raise ValueError("token_ids must not be empty")
    if not hasattr(model, "forward_cached"):
        raise TypeError("model must support cached decoding")
    return _generate_with_cache(model, token_ids, max_new_tokens, eos_token_id, cache)


def _generate_with_cache(
    model: Any,
    token_ids: list[int],
    max_new_tokens: int,
    eos_token_id: int | None,
    cache: Any,
) -> list[int]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for generation") from exc
    generated = list(token_ids)
    logits = np.asarray(model.forward_cached(np.asarray(generated, dtype=np.int64), cache))
    for _ in range(max_new_tokens):
        if logits.ndim != 2 or logits.shape[0] != 1 and logits.shape[0] != len(generated):
            raise ValueError("cached model must return [sequence, vocabulary] logits")
        next_token = int(np.argmax(logits[-1]))
        generated.append(next_token)
        if eos_token_id is not None and next_token == eos_token_id:
            break
        logits = np.asarray(model.forward_cached(np.asarray([next_token], dtype=np.int64), cache))
    return generated


def generate_text(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int, eos_token_id: int | None = None) -> str:
    """Generate text after validating tokenizer vocabulary compatibility."""
    tokenizer.validate_vocab_size(model.config.vocab_size)
    token_ids = tokenizer.encode(prompt)
    return tokenizer.decode(generate_greedy(model, token_ids, max_new_tokens, eos_token_id))
