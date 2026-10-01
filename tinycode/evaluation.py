"""Run generation-based diagnostic evaluations for TinyCode models."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any

from .runtime.generation import generate_greedy_cached_with_cache
from .runtime.kv_cache import KVCacheStore
from .runtime.manifest import ModelManifest
from .runtime.memory import parse_memory_limit, plan_memory
from .runtime.model import DiskDecoderOnlyTransformer
from .runtime.tokenizer import ByteTokenizer, CodeTokenizer
from .training.instruction import komt_conversations


def _normalise(text: str) -> str:
    return " ".join(text.casefold().split())


def _decode_generated(tokenizer: ByteTokenizer | CodeTokenizer, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(token_ids)
    except UnicodeDecodeError:
        byte_ids = [token for token in token_ids if token < 256]
        return bytes(byte_ids).decode("utf-8", errors="replace")


def evaluate_komt(
    model_dir: str | Path,
    dataset: str | Path,
    output: str | Path,
    *,
    memory_limit: str = "1G",
    max_new_tokens: int = 64,
) -> dict[str, Any]:
    model_path = Path(model_dir)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = ModelManifest.load(model_path)
    limit = parse_memory_limit(memory_limit)
    layer_bytes = max((path.stat().st_size for path in model_path.glob("layer_*.bin")), default=limit)
    plan = plan_memory(limit, layer_bytes)
    tokenizer = CodeTokenizer() if manifest.vocab_size == CodeTokenizer.vocab_size else ByteTokenizer()
    conversations = list(komt_conversations(dataset))
    results: list[dict[str, Any]] = []
    reference_matches = 0
    reference_count = 0
    with tempfile.TemporaryDirectory(prefix="tinycode-komt-kv-") as cache_root, DiskDecoderOnlyTransformer.from_model_dir(
        model_path,
        layer_cache=plan.layer_cache,
        weight_budget=plan.weights_bytes,
        prefetch=plan.prefetch,
    ) as model, output_path.open("w", encoding="utf-8", newline="\n") as stream:
        tokenizer.validate_vocab_size(model.config.vocab_size)
        for conversation in conversations:
            history: list[str] = []
            turns: list[dict[str, str]] = []
            for index, turn in enumerate(conversation["turns"]):
                prompt = "\n\n".join(history + [f"사용자: {turn}", "TinyCode:"])
                token_ids = tokenizer.encode_prompt(prompt) if isinstance(tokenizer, CodeTokenizer) else tokenizer.encode(prompt)
                cache_dir = Path(cache_root) / f"{len(results)}-{index}"
                cache = model.new_cache(KVCacheStore(cache_dir, hot_bytes=plan.kv_bytes))
                eos_token_id = tokenizer.EOS if isinstance(tokenizer, CodeTokenizer) else None
                generated = generate_greedy_cached_with_cache(model, token_ids, max_new_tokens, cache, eos_token_id)
                completion = _decode_generated(tokenizer, generated[len(token_ids):])
                reference = conversation["references"][index] if index < len(conversation["references"]) else ""
                exact_match = bool(reference) and _normalise(completion) == _normalise(reference)
                reference_count += bool(reference)
                reference_matches += exact_match
                turns.append({"prompt": prompt, "completion": completion, "reference": reference, "diagnostic_exact_match": str(exact_match).lower()})
                history.extend([f"사용자: {turn}", f"TinyCode: {completion}"])
            result = {"question_id": conversation["question_id"], "category": conversation["category"], "turns": turns}
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            results.append(result)
    return {
        "dataset": str(dataset),
        "model": str(model_path),
        "cases": sum(len(item["turns"]) for item in results),
        "references": reference_count,
        "diagnostic_exact_matches": reference_matches,
        "diagnostic_exact_match_rate": reference_matches / reference_count if reference_count else None,
        "output": str(output_path),
    }
