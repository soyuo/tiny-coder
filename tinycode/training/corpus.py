"""Build JSONL training corpora from source repositories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path


CODE_EXTENSIONS = {
    ".bash", ".c", ".cc", ".cpp", ".cs", ".css", ".dart", ".go", ".graphql",
    ".h", ".hpp", ".html", ".java", ".js", ".jsx", ".json", ".kt", ".kts",
    ".md", ".php", ".py", ".rb", ".rs", ".s", ".scala", ".scss", ".sh",
    ".sql", ".swift", ".toml", ".ts", ".tsx", ".vue", ".xml", ".yaml", ".yml",
}
SPECIAL_FILES = {"Dockerfile", "Makefile", "CMakeLists.txt", "Gemfile", "Podfile"}
IGNORED_DIRS = {
    ".git", ".gradle", ".next", ".pytest_cache", ".venv", "Pods", "__pycache__",
    "bin", "build", "coverage", "dist", "node_modules", "obj", "out", "target", "vendor", "venv",
}
GENERATED_MARKERS = ("generated", ".generated.", ".min.", ".map")
SECRET_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}


@dataclass
class CorpusStats:
    repositories: int = 0
    files: int = 0
    train_files: int = 0
    validation_files: int = 0
    bytes: int = 0
    skipped_binary: int = 0
    skipped_large: int = 0
    skipped_unsupported: int = 0
    skipped_secret: int = 0
    skipped_duplicate: int = 0
    completion_records: int = 0


def _is_source_file(path: Path) -> bool:
    name = path.name.lower()
    return not any(marker in name for marker in GENERATED_MARKERS) and (
        path.name in SPECIAL_FILES or path.suffix.lower() in CODE_EXTENSIONS
    )


def _is_secret_file(path: Path) -> bool:
    name = path.name.lower()
    return name.startswith(".env") or path.suffix.lower() in SECRET_SUFFIXES or any(
        marker in name for marker in ("secret", "credential", "private-key")
    )


def _repo_bucket(name: str, validation_ratio: float) -> str:
    threshold = int(validation_ratio * 1000)
    value = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big") % 1000
    return "validation" if value < threshold else "train"


def _source_files(repository: Path):
    for root, directories, files in os.walk(repository):
        directories[:] = [directory for directory in directories if directory not in IGNORED_DIRS]
        for filename in files:
            yield Path(root) / filename


def build_corpus(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    validation_ratio: float = 0.1,
    max_file_bytes: int = 262_144,
    completion: bool = False,
    instruction: bool = False,
) -> dict[str, int | str]:
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation ratio must be between 0 and 1")
    if max_file_bytes < 1:
        raise ValueError("max file bytes must be positive")
    source = Path(source_dir)
    output = Path(output_dir)
    if not source.is_dir():
        raise ValueError(f"source directory does not exist: {source}")
    output.mkdir(parents=True, exist_ok=True)
    stats = CorpusStats()
    seen_hashes: set[str] = set()
    train_path = output / "train.jsonl"
    validation_path = output / "validation.jsonl"
    with train_path.open("w", encoding="utf-8", newline="\n") as train_file, validation_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as validation_file:
        for repository in sorted(path for path in source.iterdir() if path.is_dir()):
            stats.repositories += 1
            bucket = _repo_bucket(repository.name, validation_ratio)
            destination = validation_file if bucket == "validation" else train_file
            for path in sorted(_source_files(repository)):
                if any(part in IGNORED_DIRS for part in path.relative_to(repository).parts):
                    continue
                if _is_secret_file(path):
                    stats.skipped_secret += 1
                    continue
                if not _is_source_file(path):
                    stats.skipped_unsupported += 1
                    continue
                if path.stat().st_size > max_file_bytes:
                    stats.skipped_large += 1
                    continue
                raw = path.read_bytes()
                if b"\x00" in raw:
                    stats.skipped_binary += 1
                    continue
                text = raw.decode("utf-8", errors="ignore").strip()
                if not text:
                    continue
                digest = hashlib.sha256(raw).hexdigest()
                if digest in seen_hashes:
                    stats.skipped_duplicate += 1
                    continue
                seen_hashes.add(digest)
                source_name = f"{repository.name}/{path.relative_to(repository).as_posix()}"
                if completion:
                    split = max(1, min(len(text) - 1, int(len(text) * 0.6)))
                    prompt, target = text[:split], text[split:]
                    if instruction:
                        prompt = f"Complete the following code.\n\n{prompt}"
                    record = {"prompt": prompt, "completion": target, "source": source_name}
                    stats.completion_records += 1
                else:
                    record = {"text": text, "source": source_name}
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats.files += 1
                stats.bytes += len(raw)
                if bucket == "validation":
                    stats.validation_files += 1
                else:
                    stats.train_files += 1
    manifest = {**asdict(stats), "source_dir": str(source), "validation_ratio": validation_ratio, "max_file_bytes": max_file_bytes}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def format_corpus(source_dir: str | Path, output_dir: str | Path, *, instruction: bool = True) -> dict[str, int | str]:
    """Convert an existing text corpus into completion records."""
    source = Path(source_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stats = CorpusStats()
    seen: set[str] = set()
    for split in ("train", "validation"):
        with (source / f"{split}.jsonl").open(encoding="utf-8") as input_file, (output / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as output_file:
            for line in input_file:
                record = json.loads(line)
                text = record.get("text")
                if not isinstance(text, str) or len(text) < 2:
                    continue
                digest = hashlib.sha256(text.encode()).hexdigest()
                if digest in seen:
                    stats.skipped_duplicate += 1
                    continue
                seen.add(digest)
                split_at = max(1, min(len(text) - 1, int(len(text) * 0.6)))
                prompt = text[:split_at]
                if instruction:
                    prompt = f"Complete the following code.\n\n{prompt}"
                output_file.write(json.dumps({"prompt": prompt, "completion": text[split_at:], "source": record.get("source", "")}, ensure_ascii=False) + "\n")
                stats.files += 1
                stats.completion_records += 1
                if split == "train":
                    stats.train_files += 1
                else:
                    stats.validation_files += 1
    manifest = {**asdict(stats), "source_dir": str(source), "format": "prompt-completion", "instruction": instruction}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
