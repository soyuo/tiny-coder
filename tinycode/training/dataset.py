"""JSONL code dataset for causal language modeling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from ..runtime.tokenizer import ByteTokenizer


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
        tokens: list[int] = []
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}") from exc
            text = record.get("text")
            if not isinstance(text, str):
                raise ValueError(f"line {line_number} must contain a string text field")
            tokens.extend(self.tokenizer.encode(text))
        width = self.block_size + 1
        for start in range(0, len(tokens) - width + 1, self.block_size):
            window = tokens[start : start + width]
            yield window[:-1], window[1:]

    def __len__(self) -> int:
        return len(self._blocks)

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        return self._blocks[index]
