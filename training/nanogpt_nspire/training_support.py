"""Shared, model-agnostic helpers for the learning experiments."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform

import torch
from torch import nn

from nanogpt_nspire.training_dataset import TokenDataset, make_batch


def resolve_device(requested: str) -> torch.device:
    """Resolve `auto`, CPU, or an available CUDA device."""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"invalid device {requested!r}") from error
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("training supports only CPU or CUDA devices")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def bits_per_character(loss: float) -> float:
    """Convert natural-log cross-entropy to bits per character."""

    if not math.isfinite(loss) or loss < 0.0:
        raise ValueError("loss must be finite and non-negative")
    return loss / math.log(2.0)


def evaluate_loss(
    model: nn.Module,
    tokens: torch.Tensor,
    *,
    batch_size: int,
    block_size: int,
    batches: int,
    seed: int,
    device: torch.device,
) -> float:
    """Average loss on deterministic windows selected by a fresh seeded generator."""

    if isinstance(batches, bool) or not isinstance(batches, int) or batches <= 0:
        raise ValueError("batches must be a positive integer")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    was_training = model.training
    model.eval()
    losses: list[float] = []
    with torch.inference_mode():
        for _ in range(batches):
            inputs, targets = make_batch(
                tokens,
                batch_size=batch_size,
                block_size=block_size,
                generator=generator,
                device=device,
            )
            result = model(inputs, targets)
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError("language model forward must return (logits, loss)")
            _, loss = result
            if not isinstance(loss, torch.Tensor):
                raise TypeError("language model did not return a tensor loss")
            losses.append(float(loss.item()))
    model.train(was_training)
    return sum(losses) / len(losses)


def synchronize(device: torch.device) -> None:
    """Wait for queued CUDA work when wall-clock timing requires it."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def sha256_file(path: Path) -> str:
    """Hash a file without loading the whole artifact into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict[str, object]) -> None:
    """Write stable UTF-8 JSON through a replace-on-success temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def dataset_summary(dataset: TokenDataset) -> dict[str, object]:
    """Return bounded hashes and counts from a validated dataset."""

    source = dataset.manifest["source"]
    files = dataset.manifest["files"]
    tokens = dataset.manifest["tokens"]
    assert isinstance(source, dict)
    assert isinstance(files, dict)
    assert isinstance(tokens, dict)
    train_file = files["train.bin"]
    validation_file = files["val.bin"]
    assert isinstance(train_file, dict)
    assert isinstance(validation_file, dict)
    return {
        "schema_version": dataset.manifest["schema_version"],
        "source_sha256": source["sha256"],
        "train_sha256": train_file["sha256"],
        "validation_sha256": validation_file["sha256"],
        "train_tokens": tokens["train"],
        "validation_tokens": tokens["validation"],
        "vocab_size": len(dataset.vocabulary),
    }


def environment_summary(
    device: torch.device,
    *,
    peak_cuda_memory_allocated_bytes: int | None,
) -> dict[str, object]:
    """Describe the exact Python/PyTorch execution environment."""

    gpu_name = None
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
    return {
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu_name": gpu_name,
        "peak_cuda_memory_allocated_bytes": peak_cuda_memory_allocated_bytes,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
    }
