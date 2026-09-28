"""Repository context search."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class ContextFile:
    """A file selected for model context."""

    path: Path
    text: str
    score: int


class RepositoryContext:
    """Find relevant text files without loading the whole repository."""

    _excluded = {".git", ".venv", "node_modules", "__pycache__", "dist", "build"}
    _extensions = {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".md", ".json"}

    def __init__(self, root: str | Path, max_files: int = 5, max_bytes: int = 64_000) -> None:
        if max_files < 1 or max_bytes < 1:
            raise ValueError("context limits must be positive")
        self.root = Path(root)
        self.max_files = max_files
        self.max_bytes = max_bytes

    def search(self, query: str) -> list[ContextFile]:
        """Return the highest-scoring files for a query."""
        terms = self._terms(query)
        matches: list[ContextFile] = []
        for path in self._files():
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            score = self._score(path, text, terms)
            if score:
                encoded = text.encode("utf-8")[: self.max_bytes]
                limited_text = encoded.decode("utf-8", errors="ignore")
                matches.append(ContextFile(path, limited_text, score))
        matches.sort(key=lambda item: (-item.score, str(item.path)))
        return matches[: self.max_files]

    def _files(self):
        for path in self.root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in self._extensions:
                continue
            if any(part in self._excluded for part in path.relative_to(self.root).parts):
                continue
            yield path

    @staticmethod
    def _terms(query: str) -> tuple[str, ...]:
        return tuple(dict.fromkeys(re.findall(r"\w+", query.lower(), re.UNICODE)))

    @staticmethod
    def _score(path: Path, text: str, terms: tuple[str, ...]) -> int:
        haystack = f"{path.name.lower()}\n{text.lower()}"
        return sum(haystack.count(term) for term in terms)
