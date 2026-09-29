"""JSONL code dataset for causal language modeling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from ..runtime.tokenizer import ByteTokenizer, CodeTokenizer

try:
    import torch

    _IterableDataset = torch.utils.data.IterableDataset
except ImportError:
    class _IterableDataset:
        pass


def iter_jsonl_blocks(path: str | Path, block_size: int, tokenizer: ByteTokenizer | CodeTokenizer, completion_only: bool = False) -> Iterator[tuple[list[int], list[int]] | tuple[list[int], list[int], list[int]]]:
    tokens: list[int] = []
    active: list[bool] = []
    width = block_size + 1
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}") from exc
            if isinstance(tokenizer, CodeTokenizer):
                try:
                    prompt = record.get("prompt")
                    completion = record.get("completion")
                    if completion_only and isinstance(prompt, str) and isinstance(completion, str):
                        prompt_tokens = tokenizer.encode(prompt)
                        completion_tokens = tokenizer.encode(completion)
                        record_tokens = [tokenizer.BOS, *prompt_tokens, tokenizer.SEP, *completion_tokens, tokenizer.EOS]
                        split_at = 1 + len(prompt_tokens) + 1
                        tokens.extend(record_tokens)
                        active.extend([False] * split_at + [True] * (len(record_tokens) - split_at))
                    else:
                        record_tokens = tokenizer.encode_record(record)
                        tokens.extend(record_tokens)
                        active.extend([True] * len(record_tokens))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"line {line_number} must contain valid code fields") from exc
            else:
                text = record.get("text")
                if not isinstance(text, str):
                    raise ValueError(f"line {line_number} must contain a string text field")
                tokens.extend(tokenizer.encode(text))
                active.extend([True] * len(tokenizer.encode(text)))
            while len(tokens) >= width:
                window = tokens[:width]
                window_active = active[:width]
                del tokens[:block_size]
                del active[:block_size]
                if completion_only:
                    target_active = [1 if value else -100 for value in window_active[1:]]
                    if any(value == 1 for value in target_active):
                        yield window[:-1], window[1:], target_active
                else:
                    yield window[:-1], window[1:]


class JsonlCodeDataset:
    """Turn JSONL text records into fixed-size next-token blocks."""

    def __init__(self, path: str | Path, block_size: int, tokenizer: ByteTokenizer | CodeTokenizer | None = None) -> None:
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

    def __init__(self, path: str | Path, block_size: int, tokenizer: ByteTokenizer | CodeTokenizer | None = None, completion_only: bool = False) -> None:
        if block_size < 2:
            raise ValueError("block_size must be at least 2")
        self.path = Path(path)
        self.block_size = block_size
        self.tokenizer = tokenizer or ByteTokenizer()
        self.completion_only = completion_only

    def __iter__(self) -> Iterator[tuple[list[int], list[int]] | tuple[list[int], list[int], list[int]]]:
        yield from iter_jsonl_blocks(self.path, self.block_size, self.tokenizer, self.completion_only)
