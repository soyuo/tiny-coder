from pathlib import Path
import json

import pytest

from tinycode.training.dataset import JsonlCodeDataset, JsonlCodeIterableDataset
from tinycode.training.corpus import build_corpus
from tinycode.training.model import TorchDecoderConfig
from tinycode.training.train import _collate_blocks


def test_jsonl_code_dataset_builds_shifted_blocks(tmp_path: Path) -> None:
    path = tmp_path / "code.jsonl"
    path.write_text('{"text":"abcde"}\n', encoding="utf-8")

    dataset = JsonlCodeDataset(path, block_size=4)

    assert len(dataset) == 1
    inputs, targets = dataset[0]
    assert inputs == [97, 98, 99, 100]
    assert targets == [98, 99, 100, 101]


def test_jsonl_code_dataset_rejects_invalid_records(tmp_path: Path) -> None:
    path = tmp_path / "code.jsonl"
    path.write_text('{"source":"missing text"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="text field"):
        JsonlCodeDataset(path, block_size=4)


def test_jsonl_code_dataset_keeps_unicode_line_separators(tmp_path: Path) -> None:
    path = tmp_path / "code.jsonl"
    path.write_text('{"text":"first\\u2028second"}\n', encoding="utf-8")

    dataset = JsonlCodeDataset(path, block_size=4)

    assert len(dataset) == 3


def test_jsonl_code_iterable_dataset_streams_blocks(tmp_path: Path) -> None:
    path = tmp_path / "code.jsonl"
    path.write_text('{"text":"abcdefghi"}\n', encoding="utf-8")

    dataset = JsonlCodeIterableDataset(path, block_size=4)

    assert list(dataset) == [([97, 98, 99, 100], [98, 99, 100, 101]), ([101, 102, 103, 104], [102, 103, 104, 105])]


def test_torch_decoder_config_validates_attention_shape() -> None:
    with pytest.raises(ValueError, match="even attention heads"):
        TorchDecoderConfig(hidden_size=6, num_heads=4)


def test_training_collate_builds_batch_tensor() -> None:
    class FakeTorch:
        long = "long"

        @staticmethod
        def tensor(values, dtype=None):
            return values, dtype

    inputs, targets = _collate_blocks([([1, 2], [2, 3]), ([4, 5], [5, 6])], FakeTorch())

    assert inputs == ([[1, 2], [4, 5]], "long")
    assert targets == ([[2, 3], [5, 6]], "long")


def test_build_corpus_filters_files_and_splits_repositories(tmp_path: Path) -> None:
    source = tmp_path / "repos"
    repo = source / "org--repo"
    (repo / "src").mkdir(parents=True)
    (repo / "node_modules").mkdir()
    (repo / "src" / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (repo / "src" / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    (repo / "node_modules" / "ignored.js").write_text("ignored\n", encoding="utf-8")
    (repo / "image.bin").write_bytes(b"\x00\x01")

    result = build_corpus(source, tmp_path / "dataset", validation_ratio=0)

    assert result["files"] == 1
    assert result["train_files"] == 1
    assert result["validation_files"] == 0
    assert "print('ok')" in (tmp_path / "dataset" / "train.jsonl").read_text(encoding="utf-8")


def test_build_corpus_deduplicates_and_builds_completion_records(tmp_path: Path) -> None:
    source = tmp_path / "repos"
    repo = source / "org--repo"
    repo.mkdir(parents=True)
    content = "def add(a, b):\n    return a + b\n"
    (repo / "a.py").write_text(content, encoding="utf-8")
    (repo / "b.py").write_text(content, encoding="utf-8")

    result = build_corpus(source, tmp_path / "dataset", validation_ratio=0, completion=True, instruction=True)
    record = json.loads((tmp_path / "dataset" / "train.jsonl").read_text(encoding="utf-8"))

    assert result["skipped_duplicate"] == 1
    assert record["prompt"].startswith("Complete the following code.")
    assert record["completion"]


def test_train_jsonl_records_validation_loss(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from tinycode.training import TrainConfig, TorchDecoderConfig, train_jsonl

    data = tmp_path / "data.jsonl"
    data.write_text('{"text":"def add(a, b): return a + b;"}\n', encoding="utf-8")
    config = TrainConfig(
        model=TorchDecoderConfig(hidden_size=16, intermediate_size=32, num_layers=1, num_heads=2, max_sequence_length=8),
        batch_size=1,
        epochs=1,
    )

    result = train_jsonl(data, tmp_path / "out", config, validation_path=data)

    assert result["validation_loss"] is not None


def test_train_jsonl_limits_steps(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from tinycode.training import TrainConfig, TorchDecoderConfig, train_jsonl

    data = tmp_path / "data.jsonl"
    data.write_text('{"text":"def add(a, b): return a + b;"}\n', encoding="utf-8")
    config = TrainConfig(
        model=TorchDecoderConfig(hidden_size=16, intermediate_size=32, num_layers=1, num_heads=2, max_sequence_length=8),
        batch_size=1,
        epochs=5,
        max_steps=2,
    )

    result = train_jsonl(data, tmp_path / "out", config)

    assert result["steps"] == 2


def test_train_jsonl_writes_intermediate_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from tinycode.training import TrainConfig, TorchDecoderConfig, train_jsonl

    data = tmp_path / "data.jsonl"
    output = tmp_path / "out"
    data.write_text('{"text":"def add(a, b): return a + b;"}\n', encoding="utf-8")
    config = TrainConfig(
        model=TorchDecoderConfig(hidden_size=16, intermediate_size=32, num_layers=1, num_heads=2, max_sequence_length=8),
        batch_size=1,
        epochs=1,
        checkpoint_interval=1,
    )

    result = train_jsonl(data, output, config)

    assert result["checkpoint"] == str(output / "checkpoint.pt")
    assert (output / "checkpoint.pt").exists()


def test_train_jsonl_resumes_from_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from tinycode.training import TrainConfig, TorchDecoderConfig, train_jsonl

    data = tmp_path / "data.jsonl"
    data.write_text('{"text":"def add(a, b): return a + b;"}\n', encoding="utf-8")
    config = TrainConfig(
        model=TorchDecoderConfig(hidden_size=16, intermediate_size=32, num_layers=1, num_heads=2, max_sequence_length=8),
        batch_size=1,
        epochs=1,
    )

    first = train_jsonl(data, tmp_path / "first", config)
    resumed = train_jsonl(data, tmp_path / "resumed", config, resume_checkpoint=first["checkpoint"])

    assert resumed["resumed"] is True
    assert resumed["steps"] == first["steps"]
