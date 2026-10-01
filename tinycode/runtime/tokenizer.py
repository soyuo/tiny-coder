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


class CodeTokenizer(ByteTokenizer):
    """Byte tokenizer with code-training control tokens."""

    BOS = 256
    EOS = 257
    SEP = 258
    PAD = 259
    vocab_size = 260

    def validate_vocab_size(self, vocab_size: int) -> None:
        if vocab_size != self.vocab_size:
            raise TokenizationError(f"code tokenizer requires vocab size {self.vocab_size}")

    def encode(self, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
        tokens = super().encode(text)
        if bos:
            tokens.insert(0, self.BOS)
        if eos:
            tokens.append(self.EOS)
        return tokens

    def encode_prompt(self, text: str) -> list[int]:
        """Encode a prompt with the control tokens used during training."""
        return [self.BOS, *super().encode(text), self.SEP]

    def encode_record(self, record: dict[str, object]) -> list[int]:
        prompt = record.get("prompt")
        completion = record.get("completion")
        if isinstance(prompt, str) and isinstance(completion, str):
            return [self.BOS, *super().encode(prompt), self.SEP, *super().encode(completion), self.EOS]
        text = record.get("text")
        if not isinstance(text, str):
            raise TokenizationError("record must contain text or prompt/completion")
        return self.encode(text, bos=True, eos=True)

    def decode(self, token_ids: list[int]) -> str:
        if any(not isinstance(token, int) or not 0 <= token < self.vocab_size for token in token_ids):
            raise TokenizationError("token ID must be an integer from 0 to 259")
        return bytes(token for token in token_ids if token < 256).decode("utf-8", errors="replace")
