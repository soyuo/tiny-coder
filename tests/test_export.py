from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from tinycode.runtime.model import DiskDecoderOnlyTransformer
from tinycode.training import TinyCodeDecoder, TorchDecoderConfig, export_checkpoint


def test_export_checkpoint_creates_runtime_model(tmp_path: Path) -> None:
    import numpy as np

    config = TorchDecoderConfig(hidden_size=16, intermediate_size=32, num_layers=1, num_heads=2, max_sequence_length=8)
    model = TinyCodeDecoder(config)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "config": config.to_dict()}, checkpoint)

    output = export_checkpoint(checkpoint, tmp_path / "model")

    assert (output / "model.json").is_file()
    assert (output / "layer_00.bin").is_file()
    with DiskDecoderOnlyTransformer.from_model_dir(output) as runtime:
        logits = runtime(np.array([1, 2], dtype=np.int64))
    assert logits.shape == (2, 256)


def test_export_checkpoint_rejects_missing_tensor(tmp_path: Path) -> None:
    config = TorchDecoderConfig(hidden_size=16, intermediate_size=32, num_layers=1, num_heads=2, max_sequence_length=8)
    model = TinyCodeDecoder(config)
    state = model.state_dict()
    del state["layers.0.q_proj.weight"]
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": state, "config": config.to_dict()}, checkpoint)

    with pytest.raises(ValueError, match="missing tensor"):
        export_checkpoint(checkpoint, tmp_path / "model")
