"""CPU runtime benchmark helpers."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import ctypes
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from .runtime.generation import generate_greedy_cached_with_cache
from .runtime.kv_cache import KVCacheStore
from .runtime.manifest import ModelManifest
from .runtime.memory import parse_memory_limit, plan_memory
from .runtime.model import DiskDecoderOnlyTransformer
from .runtime.tokenizer import ByteTokenizer, CodeTokenizer


@dataclass(frozen=True)
class BenchmarkConfig:
    model_dir: Path
    prompt: str
    language: str | None = None
    memory_limit: str = "1G"
    layer_cache: int | None = None
    kv_cache: str = "memory"
    kv_cache_dir: Path | None = None
    max_new_tokens: int = 32
    iterations: int = 1
    prefetch: bool | None = None


@dataclass(frozen=True)
class BenchmarkResult:
    model: str
    kv_cache: str
    layer_cache: int
    prefetch: bool
    iterations: int
    max_new_tokens: int
    average_seconds: float
    tokens_per_second: float
    rss_before_bytes: int
    rss_after_bytes: int
    rss_delta_bytes: int
    peak_weight_cache_bytes: int
    peak_hot_kv_bytes: int


def current_rss_bytes() -> int:
    if os.name == "nt":
        class MemoryCounters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("page_fault_count", ctypes.c_ulong), ("peak_working_set", ctypes.c_size_t), ("working_set", ctypes.c_size_t)]

        counters = MemoryCounters()
        process = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), ctypes.sizeof(counters))
        return int(counters.working_set)
    try:
        return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return 0


def run_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    if config.kv_cache not in {"memory", "disk"}:
        raise ValueError("kv cache must be memory or disk")
    if config.max_new_tokens < 0 or config.iterations < 1:
        raise ValueError("benchmark parameters must be non-negative and iterations must be positive")
    memory_limit = parse_memory_limit(config.memory_limit)
    manifest = ModelManifest.load(config.model_dir)
    layer_bytes = max((path.stat().st_size for path in config.model_dir.glob("layer_*.bin")), default=memory_limit)
    plan = plan_memory(memory_limit, layer_bytes)
    layer_cache = config.layer_cache or plan.layer_cache
    if layer_cache < 1:
        raise ValueError("layer cache must be at least 1")
    prefetch = plan.prefetch if config.prefetch is None else config.prefetch
    tokenizer = CodeTokenizer() if manifest.vocab_size == CodeTokenizer.vocab_size else ByteTokenizer()
    tokenizer.validate_vocab_size(manifest.vocab_size)
    prompt = config.prompt
    if config.language:
        prompt = f"Language: {config.language}\n\n{prompt}"
    token_ids = tokenizer.encode_prompt(prompt) if isinstance(tokenizer, CodeTokenizer) else tokenizer.encode(prompt)
    rss_before = current_rss_bytes()
    durations: list[float] = []
    peak_weight_bytes = 0
    peak_hot_kv_bytes = 0
    generated_tokens = 0

    with tempfile.TemporaryDirectory(prefix="tinycode-benchmark-") as temporary_dir:
        kv_dir = config.kv_cache_dir or Path(temporary_dir)
        with DiskDecoderOnlyTransformer.from_model_dir(
            config.model_dir,
            layer_cache=layer_cache,
            weight_budget=plan.weights_bytes,
            prefetch=prefetch,
        ) as model:
            for index in range(config.iterations):
                store = KVCacheStore(kv_dir / str(index), hot_bytes=plan.kv_bytes) if config.kv_cache == "disk" else None
                cache = model.new_cache(store)
                started = time.perf_counter()
                eos_token_id = tokenizer.EOS if isinstance(tokenizer, CodeTokenizer) else None
                result = generate_greedy_cached_with_cache(model, token_ids, config.max_new_tokens, cache, eos_token_id)
                durations.append(time.perf_counter() - started)
                generated_tokens += max(0, len(result) - len(token_ids))
                peak_weight_bytes = max(peak_weight_bytes, model.weights.cached_bytes)
                peak_hot_kv_bytes = max(peak_hot_kv_bytes, 0 if store is None else store.hot_size)
                if store is not None:
                    store.clear()
    rss_after = current_rss_bytes()
    total_seconds = sum(durations)
    return asdict(
        BenchmarkResult(
            model=str(config.model_dir),
            kv_cache=config.kv_cache,
            layer_cache=layer_cache,
            prefetch=prefetch,
            iterations=config.iterations,
            max_new_tokens=config.max_new_tokens,
            average_seconds=total_seconds / len(durations),
            tokens_per_second=generated_tokens / total_seconds if total_seconds else 0.0,
            rss_before_bytes=rss_before,
            rss_after_bytes=rss_after,
            rss_delta_bytes=rss_after - rss_before,
            peak_weight_cache_bytes=peak_weight_bytes,
            peak_hot_kv_bytes=peak_hot_kv_bytes,
        )
    )


def benchmark_json(config: BenchmarkConfig) -> str:
    return json.dumps(run_benchmark(config), indent=2)
