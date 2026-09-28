"""Causal language-model training loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json

from .dataset import JsonlCodeDataset
from .model import TinyCodeDecoder, TorchDecoderConfig, _torch


@dataclass(frozen=True)
class TrainConfig:
    model: TorchDecoderConfig = TorchDecoderConfig()
    batch_size: int = 4
    epochs: int = 1
    learning_rate: float = 3e-4
    device: str = "cpu"


def train_jsonl(path: str | Path, output_dir: str | Path, config: TrainConfig | None = None) -> dict[str, Any]:
    torch, _ = _torch()
    config = config or TrainConfig()
    if config.batch_size < 1 or config.epochs < 1 or config.learning_rate <= 0:
        raise ValueError("training parameters must be positive")
    dataset = JsonlCodeDataset(path, config.model.max_sequence_length)
    model = TinyCodeDecoder(config.model).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model.train()
    losses: list[float] = []
    for _ in range(config.epochs):
        for inputs, targets in loader:
            inputs = inputs.to(config.device)
            targets = targets.to(config.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = loss_fn(logits.reshape(-1, config.model.vocab_size), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "config": config.model.to_dict(), "loss": losses[-1]}, checkpoint)
    (output / "training.json").write_text(
        json.dumps({"config": asdict(config), "steps": len(losses), "loss": losses[-1]}, default=lambda value: asdict(value)),
        encoding="utf-8",
    )
    return {"checkpoint": str(checkpoint), "steps": len(losses), "loss": losses[-1]}


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[Any, TorchDecoderConfig]:
    torch, _ = _torch()
    payload = torch.load(path, map_location=device, weights_only=False)
    config = TorchDecoderConfig(**payload["config"])
    model = TinyCodeDecoder(config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, config
