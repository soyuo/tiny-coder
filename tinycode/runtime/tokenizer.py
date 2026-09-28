"""Minimal reversible tokenizer."""

from __future__ import annotations


class TokenizationError(ValueError):
    """Raised for invalid token IDs."""


class ByteTokenizer:
    """Encode text as UTF-8 byte token IDs."""

    vocab_size = 256

    def validate_vocab_size(self, vocab_size: int) -> None:
        if vocab_size != self.vocab_size:
            raise TokenizationError(f"byte tokenizer requires vocab size {self.vocab_size}")

    def encode(self, text: str) -> list[int]:
        """Convert text to byte IDs."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        """Convert byte IDs back to text."""
        if any(not isinstance(token, int) or not 0 <= token < self.vocab_size for token in token_ids):
            raise TokenizationError("token ID must be an integer from 0 to 255")
        return bytes(token_ids).decode("utf-8")
