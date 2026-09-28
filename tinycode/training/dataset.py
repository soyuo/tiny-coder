"""JSONL code dataset for causal language modeling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from ..runtime.tokenizer import ByteTokenizer

try:
    import torch

    _IterableDataset = torch.utils.data.IterableDataset
except ImportError:
    class _IterableDataset:
        pass


def iter_jsonl_blocks(path: str | Path, block_size: int, tokenizer: ByteTokenizer) -> Iterator[tuple[list[int], list[int]]]:
    tokens: list[int] = []
    width = block_size + 1
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}") from exc
            text = record.get("text")
            if not isinstance(text, str):
                raise ValueError(f"line {line_number} must contain a string text field")
            tokens.extend(tokenizer.encode(text))
            while len(tokens) >= width:
                window = tokens[:width]
                del tokens[:block_size]
                yield window[:-1], window[1:]


class JsonlCodeDataset:
    """Turn JSONL text records into fixed-size next-token blocks."""

    def __init__(self, path: str | Path, block_size: int, tokenizer: ByteTokenizer | None = None) -> None:
        if block_size < 2:
            raise ValueError("block_size must be at least 2")
        self.path = Path(path)
        self.block_size = block_size
        self.tokenizer = tokenizer or ByteTokenizer()
        self._blocks = list(self._read_blocks())
        if not self._blocks:
            raise ValueError("dataset contains no complete token blocks")

    def _read_blocks(self) -> Iterator[tuple[list[int], list[int]]]:
        yield from iter_jsonl_blocks(self.path, self.block_size, self.tokenizer)

    def __len__(self) -> int:
        return len(self._blocks)

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        return self._blocks[index]


class JsonlCodeIterableDataset(_IterableDataset):
    """Stream JSONL blocks without retaining the corpus in memory."""

    def __init__(self, path: str | Path, block_size: int, tokenizer: ByteTokenizer | None = None) -> None:
        if block_size < 2:
            raise ValueError("block_size must be at least 2")
        self.path = Path(path)
        self.block_size = block_size
        self.tokenizer = tokenizer or ByteTokenizer()

    def __iter__(self) -> Iterator[tuple[list[int], list[int]]]:
        yield from iter_jsonl_blocks(self.path, self.block_size, self.tokenizer)
