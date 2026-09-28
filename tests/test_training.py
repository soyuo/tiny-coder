from pathlib import Path

import pytest

from tinycode.training.dataset import JsonlCodeDataset
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
