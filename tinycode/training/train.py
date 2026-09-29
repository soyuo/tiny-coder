"""Causal language-model training loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json

from ..runtime.tokenizer import CodeTokenizer
from .dataset import JsonlCodeIterableDataset
from .model import TinyCodeDecoder, TorchDecoderConfig, _torch


@dataclass(frozen=True)
class TrainConfig:
    model: TorchDecoderConfig = TorchDecoderConfig()
    batch_size: int = 4
    epochs: int = 1
    learning_rate: float = 3e-4
    device: str = "cpu"
    max_steps: int | None = None
    checkpoint_interval: int | None = 500


def _collate_blocks(batch: list[tuple[list[int], ...]], torch: Any) -> tuple[Any, ...]:
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    result = [torch.tensor(list(inputs), dtype=torch.long), torch.tensor(list(targets), dtype=torch.long)]
    if len(batch[0]) == 3:
        result.append(torch.tensor([item[2] for item in batch], dtype=torch.long))
    return tuple(result)


def _evaluate(model: Any, dataset: JsonlCodeIterableDataset, batch_size: int, device: str, torch: Any) -> float:
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: _collate_blocks(batch, torch),
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    total = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch[:2]
            masks = batch[2] if len(batch) == 3 else None
            logits = model(inputs.to(device))
            labels = targets.to(device)
            if masks is not None:
                labels = labels.masked_fill(masks.to(device) == -100, -100)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            total += float(loss.cpu())
            count += 1
    model.train()
    return total / count


def train_jsonl(
    path: str | Path,
    output_dir: str | Path,
    config: TrainConfig | None = None,
    validation_path: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    torch, _ = _torch()
    config = config or TrainConfig()
    if config.batch_size < 1 or config.epochs < 1 or config.learning_rate <= 0 or (config.max_steps is not None and config.max_steps < 1) or (config.checkpoint_interval is not None and config.checkpoint_interval < 1):
        raise ValueError("training parameters must be positive")
    tokenizer = CodeTokenizer() if config.model.vocab_size == CodeTokenizer.vocab_size else None
    completion_only = tokenizer is not None
    dataset = JsonlCodeIterableDataset(path, config.model.max_sequence_length, tokenizer, completion_only)
    model = TinyCodeDecoder(config.model).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    if resume_checkpoint is not None:
        payload = torch.load(resume_checkpoint, map_location=config.device, weights_only=False)
        if payload.get("config") != config.model.to_dict():
            raise ValueError("resume checkpoint config does not match training config")
        model.load_state_dict(payload["model"])
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
    loss_fn = torch.nn.CrossEntropyLoss()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda batch: _collate_blocks(batch, torch),
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(loss_value: float) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config.model.to_dict(),
                "loss": loss_value,
                "validation_loss": None,
            },
            output / "checkpoint.pt",
        )

    model.train()
    losses: list[float] = []
    for _ in range(config.epochs):
        for batch in loader:
            inputs, targets = batch[:2]
            inputs = inputs.to(config.device)
            targets = targets.to(config.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            if len(batch) == 3:
                targets = targets.masked_fill(batch[2].to(config.device) == -100, -100)
            loss = loss_fn(logits.reshape(-1, config.model.vocab_size), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if config.checkpoint_interval and len(losses) % config.checkpoint_interval == 0:
                save_checkpoint(losses[-1])
            if config.max_steps is not None and len(losses) >= config.max_steps:
                break
        if config.max_steps is not None and len(losses) >= config.max_steps:
            break
    validation_loss = None
    if validation_path is not None:
        validation = JsonlCodeIterableDataset(validation_path, config.model.max_sequence_length, tokenizer, completion_only)
        validation_loss = _evaluate(model, validation, config.batch_size, config.device, torch)
    checkpoint = output / "checkpoint.pt"
    save_checkpoint(losses[-1])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["validation_loss"] = validation_loss
    torch.save(payload, checkpoint)
    (output / "training.json").write_text(
        json.dumps({"config": asdict(config), "steps": len(losses), "loss": losses[-1], "validation_loss": validation_loss}, default=lambda value: asdict(value)),
        encoding="utf-8",
    )
    return {
        "checkpoint": str(checkpoint),
        "steps": len(losses),
        "loss": losses[-1],
        "validation_loss": validation_loss,
        "resumed": resume_checkpoint is not None,
    }


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[Any, TorchDecoderConfig]:
    torch, _ = _torch()
    payload = torch.load(path, map_location=device, weights_only=False)
    config = TorchDecoderConfig(**payload["config"])
    model = TinyCodeDecoder(config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, config
