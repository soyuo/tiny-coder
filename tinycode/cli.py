"""TinyCode command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import BenchmarkConfig, benchmark_json
from .runtime.generation import generate_greedy_cached_with_cache
from .runtime.context import RepositoryContext
from .runtime.kv_cache import KVCacheStore
from .runtime.manifest import ModelManifest, ManifestError
from .runtime.memory import MemoryLimitError, parse_memory_limit, plan_memory
from .runtime.model import DiskDecoderOnlyTransformer
from .runtime.tokenizer import ByteTokenizer, TokenizationError
from .training import TorchDecoderConfig, TrainConfig, export_checkpoint, train_jsonl


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
    run.add_argument("--repository", type=Path)
    run.add_argument("--max-new-tokens", type=int, default=32)
    train = commands.add_parser("train")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--block-size", type=int, default=256)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--hidden-size", type=int, default=256)
    train.add_argument("--intermediate-size", type=int, default=768)
    train.add_argument("--layers", type=int, default=4)
    train.add_argument("--heads", type=int, default=8)
    train.add_argument("--device", default="cpu")
    train.add_argument("--validation-data", type=Path)
    train.add_argument("--resume", type=Path)
    export = commands.add_parser("export")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--model", type=Path, required=True)
    benchmark.add_argument("--prompt", required=True)
    benchmark.add_argument("--memory-limit", default="1G")
    benchmark.add_argument("--layer-cache", type=int)
    benchmark.add_argument("--kv-cache", choices=("memory", "disk"), default="memory")
    benchmark.add_argument("--kv-cache-dir", type=Path)
    benchmark.add_argument("--max-new-tokens", type=int, default=32)
    benchmark.add_argument("--iterations", type=int, default=1)
    prefetch = benchmark.add_mutually_exclusive_group()
    prefetch.add_argument("--prefetch", dest="prefetch", action="store_true")
    prefetch.add_argument("--no-prefetch", dest="prefetch", action="store_false")
    benchmark.set_defaults(prefetch=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return run_command(args)
    if args.command == "train":
        return train_command(args)
    if args.command == "export":
        return export_command(args)
    if args.command == "benchmark":
        return benchmark_command(args)
    return 2


def train_command(args: argparse.Namespace) -> int:
    try:
        config = TrainConfig(
            model=TorchDecoderConfig(
                hidden_size=args.hidden_size,
                intermediate_size=args.intermediate_size,
                num_layers=args.layers,
                num_heads=args.heads,
                max_sequence_length=args.block_size,
            ),
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=args.device,
        )
        result = train_jsonl(
            args.data,
            args.output,
            config,
            validation_path=args.validation_data,
            resume_checkpoint=args.resume,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    print(f"checkpoint: {result['checkpoint']}")
    print(f"steps: {result['steps']}")
    print(f"loss: {result['loss']:.6f}")
    if result["validation_loss"] is not None:
        print(f"validation loss: {result['validation_loss']:.6f}")
    return 0


def export_command(args: argparse.Namespace) -> int:
    try:
        output = export_checkpoint(args.checkpoint, args.output)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    print(f"model: {output}")
    return 0


def benchmark_command(args: argparse.Namespace) -> int:
    try:
        print(benchmark_json(BenchmarkConfig(
            model_dir=args.model,
            prompt=args.prompt,
            memory_limit=args.memory_limit,
            layer_cache=args.layer_cache,
            kv_cache=args.kv_cache,
            kv_cache_dir=args.kv_cache_dir,
            max_new_tokens=args.max_new_tokens,
            iterations=args.iterations,
            prefetch=args.prefetch,
        )))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    return 0


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
    context_budget = 64_000
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
        context_budget = plan.context_bytes
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
            prefetch=plan.prefetch if plan is not None else False,
        ) as model:
            tokenizer.validate_vocab_size(model.config.vocab_size)
            store = KVCacheStore(args.kv_cache_dir, hot_bytes=kv_budget)
            prompt = args.prompt
            if args.repository is not None:
                files = RepositoryContext(args.repository, max_bytes=context_budget).search(prompt)
                context = "\n\n".join(f"[{item.path}]\n{item.text}" for item in files)
                if context:
                    prompt = f"{prompt}\n\nRelevant repository context:\n{context}"
                print(f"context files: {len(files)}")
            cache = model.new_cache(store)
            token_ids = tokenizer.encode(prompt)
            result = generate_greedy_cached_with_cache(model, token_ids, args.max_new_tokens, cache)
        print(tokenizer.decode(result))
    except (OSError, RuntimeError, TokenizationError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    return 0
