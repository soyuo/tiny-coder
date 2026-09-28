"""TinyCode command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from .runtime.manifest import ModelManifest, ManifestError
from .runtime.memory import MemoryLimitError, parse_memory_limit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tinycode")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--memory-limit", default="1G")
    run.add_argument("--layer-cache", type=int, default=1)
    run.add_argument("--kv-cache", choices=("disk",), default="disk")
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
    if args.layer_cache < 1:
        print("error: layer cache must be at least 1")
        return 2
    print("TinyCode CPU runtime ready")
    print(f"model: {args.model}")
    print(f"layers: {manifest.num_layers}")
    print(f"memory limit: {memory_limit} bytes")
    print(f"layer cache: {args.layer_cache}")
    print(f"KV cache: {args.kv_cache}")
    return 0
