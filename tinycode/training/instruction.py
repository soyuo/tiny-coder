"""Prepare instruction datasets from local files or Hugging Face."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def _dataset_id(source: str | Path) -> str:
    value = str(source)
    if value.startswith("https://huggingface.co/datasets/"):
        value = value.removeprefix("https://huggingface.co/datasets/").strip("/")
    return value


def _records(source: str | Path, *, split: str = "train") -> Iterable[dict[str, Any]]:
    path = Path(source)
    if path.is_file():
        if path.suffix.lower() == ".jsonl":
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid JSON on line {line_number}") from exc
                    if not isinstance(record, dict):
                        raise ValueError(f"line {line_number} must contain an object")
                    yield record
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get(split, payload.get("data", []))
        if not isinstance(payload, list):
            raise ValueError("JSON dataset must contain a list of records")
        yield from (record for record in payload if isinstance(record, dict))
        return
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Hugging Face sources require: pip install datasets") from exc
    dataset = load_dataset(_dataset_id(source), split=split)
    yield from dataset


def _prompt(record: dict[str, Any]) -> tuple[str, str] | None:
    instruction = record.get("instruction")
    output = record.get("output", record.get("response"))
    if not isinstance(instruction, str) or not isinstance(output, str):
        return None
    instruction = instruction.strip()
    output = output.strip()
    if not instruction or not output:
        return None
    extra = record.get("input", "")
    if extra is not None and not isinstance(extra, str):
        extra = str(extra)
    prompt = instruction if not str(extra).strip() else f"{instruction}\n\n입력:\n{str(extra).strip()}"
    return prompt, output


@dataclass
class InstructionStats:
    source: str
    records: int = 0
    skipped: int = 0
    train_records: int = 0
    validation_records: int = 0


def prepare_instruction_data(
    source: str | Path,
    output_dir: str | Path,
    *,
    validation_ratio: float = 0.1,
    split: str = "train",
) -> dict[str, Any]:
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation ratio must be between 0 and 1")
    source_path = Path(source)
    output = Path(output_dir)
    if source_path.exists():
        source_resolved = source_path.resolve()
        output_resolved = output.resolve()
        if source_resolved == output_resolved or source_resolved in {
            output_resolved / "train.jsonl",
            output_resolved / "validation.jsonl",
        }:
            raise ValueError("source and output paths must be different")
    output.mkdir(parents=True, exist_ok=True)
    stats = InstructionStats(source=_dataset_id(source))
    seen: set[str] = set()
    train_path = output / "train.jsonl"
    validation_path = output / "validation.jsonl"
    with train_path.open("w", encoding="utf-8", newline="\n") as train_file, validation_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as validation_file:
        for index, record in enumerate(_records(source, split=split)):
            item = _prompt(record)
            if item is None:
                stats.skipped += 1
                continue
            prompt, completion = item
            digest = hashlib.sha256(f"{prompt}\n\0{completion}".encode()).hexdigest()
            if digest in seen:
                stats.skipped += 1
                continue
            seen.add(digest)
            bucket = int(digest[:8], 16) % 1000 < int(validation_ratio * 1000)
            destination = validation_file if bucket else train_file
            destination.write(json.dumps({"prompt": prompt, "completion": completion}, ensure_ascii=False) + "\n")
            stats.records += 1
            if bucket:
                stats.validation_records += 1
            else:
                stats.train_records += 1
    manifest = {**asdict(stats), "format": "prompt-completion", "validation_ratio": validation_ratio}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def komt_conversations(source: str | Path) -> Iterable[dict[str, Any]]:
    for record in _records(source):
        turns = record.get("turns", [])
        references = record.get("reference", [])
        if not isinstance(turns, list):
            continue
        if not isinstance(references, list):
            references = []
        clean_turns = [turn.strip() for turn in turns if isinstance(turn, str) and turn.strip()]
        if clean_turns:
            yield {
                "question_id": str(record.get("question_id", "")),
                "category": str(record.get("category", "")),
                "turns": clean_turns,
                "references": [str(item).strip() for item in references],
            }
