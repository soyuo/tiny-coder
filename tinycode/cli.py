"""TinyCode command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from .runtime.generation import generate_greedy_cached_with_cache
from .runtime.kv_cache import KVCacheStore
from .runtime.manifest import ModelManifest, ManifestError
from .runtime.memory import MemoryLimitError, parse_memory_limit, plan_memory
from .runtime.model import DiskDecoderOnlyTransformer
from .runtime.tokenizer import ByteTokenizer, TokenizationError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tinycode")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--memory-limit", default="1G")
    run.add_argument("--layer-cache", type=int)
    run.add_argument("--kv-cache", choices=("disk",), default="disk")
    run.add_argument("--kv-cache-dir", type=Path)
    run.add_argument("--prompt")
    run.add_argument("--max-new-tokens", type=int, default=32)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return run_command(args)
    return 2


def run_command(args: argparse.Namespace) -> int:
    try:
        memory_limit = parse_memory_limit(args.memory_limit)
        manifest = ModelManifest.load(args.model)
    except (MemoryLimitError, ManifestError) as exc:
        print(f"error: {exc}")
        return 2
    if args.layer_cache is not None and args.layer_cache < 1:
        print("error: layer cache must be at least 1")
        return 2
    if args.max_new_tokens < 0:
        print("error: max new tokens must be non-negative")
        return 2
    print("TinyCode CPU runtime ready")
    print(f"model: {args.model}")
    print(f"layers: {manifest.num_layers}")
    print(f"memory limit: {memory_limit} bytes")
    plan = None
    weight_budget = None
    kv_budget = None
    if args.layer_cache is None:
        layer_bytes = max((path.stat().st_size for path in args.model.glob("layer_*.bin")), default=memory_limit)
        try:
            plan = plan_memory(memory_limit, layer_bytes)
        except MemoryLimitError as exc:
            print(f"error: {exc}")
            return 2
        args.layer_cache = plan.layer_cache
        weight_budget = plan.weights_bytes
        kv_budget = plan.kv_bytes
    print(f"layer cache: {args.layer_cache}")
    if plan is not None:
        print(f"weights budget: {plan.weights_bytes} bytes")
        print(f"KV budget: {plan.kv_bytes} bytes")
        print(f"context budget: {plan.context_bytes} bytes")
        print(f"runtime reserve: {plan.reserve_bytes} bytes")
        print(f"prefetch: {'enabled' if plan.prefetch else 'disabled'}")
    print(f"KV cache: {args.kv_cache}")
    if args.prompt is None:
        return 0
    if args.kv_cache_dir is None:
        args.kv_cache_dir = args.model / "kv_cache"
    try:
        tokenizer = ByteTokenizer()
        with DiskDecoderOnlyTransformer.from_model_dir(
            args.model,
            layer_cache=args.layer_cache,
            weight_budget=weight_budget,
        ) as model:
            tokenizer.validate_vocab_size(model.config.vocab_size)
            store = KVCacheStore(args.kv_cache_dir, hot_bytes=kv_budget)
            cache = model.new_cache(store)
            token_ids = tokenizer.encode(args.prompt)
            result = generate_greedy_cached_with_cache(model, token_ids, args.max_new_tokens, cache)
        print(tokenizer.decode(result))
    except (OSError, RuntimeError, TokenizationError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    return 0
