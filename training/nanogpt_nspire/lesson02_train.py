"""Train the Lesson 02 no-attention embedding language model."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
from typing import Sequence

import torch

from nanogpt_nspire.data import DatasetError, decode_tokens
from nanogpt_nspire.models.embedding_lm import EmbeddingLanguageModel
from nanogpt_nspire.training_dataset import (
    TokenDataset,
    load_token_dataset,
    make_batch,
)


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for a reproducible Lesson 02 baseline run."""

    data_dir: Path
    output_dir: Path
    device: str = "auto"
    seed: int = 1337
    steps: int = 1000
    batch_size: int = 64
    block_size: int = 64
    embedding_dim: int = 32
    learning_rate: float = 0.05
    eval_batches: int = 50
    sample_tokens: int = 300
    source_commit: str = "uncommitted"

    def validate(self) -> None:
        for name in (
            "steps",
            "batch_size",
            "block_size",
            "embedding_dim",
            "eval_batches",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.sample_tokens, bool)
            or not isinstance(self.sample_tokens, int)
            or self.sample_tokens < 0
        ):
            raise ValueError("sample_tokens must be a non-negative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not self.source_commit:
            raise ValueError("source_commit must not be empty")


def resolve_device(requested: str) -> torch.device:
    """Resolve `auto`, CPU, or an available CUDA device."""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"invalid device {requested!r}") from error
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Lesson 02 supports only CPU or CUDA devices")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def bits_per_character(loss: float) -> float:
    """Convert natural-log cross-entropy to bits per character."""

    if not math.isfinite(loss) or loss < 0.0:
        raise ValueError("loss must be finite and non-negative")
    return loss / math.log(2.0)


def evaluate_loss(
    model: EmbeddingLanguageModel,
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
            _, loss = model(inputs, targets)
            assert loss is not None
            losses.append(float(loss.item()))
    model.train(was_training)
    return sum(losses) / len(losses)


def sample_token_ids(
    model: EmbeddingLanguageModel,
    prompt_token_id: int,
    *,
    new_tokens: int,
    seed: int,
    temperature: float,
    device: torch.device,
) -> list[int]:
    """Sample a fixed-seed sequence from the current-token-only model."""

    if not 0 <= prompt_token_id < model.vocab_size:
        raise ValueError("prompt_token_id is outside the vocabulary")
    if new_tokens < 0:
        raise ValueError("new_tokens must be non-negative")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    generated = [prompt_token_id]
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for _ in range(new_tokens):
            current = torch.tensor(
                [[generated[-1]]],
                dtype=torch.long,
                device=device,
            )
            logits, _ = model(current)
            probabilities = torch.softmax(
                logits[0, -1].detach().cpu() / temperature,
                dim=-1,
            )
            next_token = int(
                torch.multinomial(
                    probabilities,
                    num_samples=1,
                    generator=generator,
                ).item()
            )
            generated.append(next_token)
    model.train(was_training)
    return generated


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
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


def _dataset_summary(dataset: TokenDataset) -> dict[str, object]:
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


def run_training(config: TrainingConfig) -> dict[str, object]:
    """Train, sample, checkpoint, and return a bounded experiment summary."""

    config.validate()
    device = resolve_device(config.device)
    dataset = load_token_dataset(config.data_dir)
    if dataset.train.numel() < config.block_size + 1:
        raise DatasetError("training split is too short for block_size")
    if dataset.validation.numel() < config.block_size + 1:
        raise DatasetError("validation split is too short for block_size")

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    model = EmbeddingLanguageModel(
        vocab_size=len(dataset.vocabulary),
        embedding_dim=config.embedding_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.0,
    )

    evaluation_seed = config.seed + 1
    initial_validation_loss = evaluate_loss(
        model,
        dataset.validation,
        batch_size=config.batch_size,
        block_size=config.block_size,
        batches=config.eval_batches,
        seed=evaluation_seed,
        device=device,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_generator = torch.Generator(device="cpu").manual_seed(config.seed + 2)
    history: list[dict[str, float | int]] = []
    history_interval = max(1, config.steps // 10)
    model.train()
    _synchronize(device)
    started = time.perf_counter()
    for step in range(1, config.steps + 1):
        inputs, targets = make_batch(
            dataset.train,
            batch_size=config.batch_size,
            block_size=config.block_size,
            generator=training_generator,
            device=device,
        )
        _, loss = model(inputs, targets)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % history_interval == 0 or step == config.steps:
            history.append({"step": step, "training_loss": float(loss.item())})
    _synchronize(device)
    training_seconds = time.perf_counter() - started

    final_validation_loss = evaluate_loss(
        model,
        dataset.validation,
        batch_size=config.batch_size,
        block_size=config.block_size,
        batches=config.eval_batches,
        seed=evaluation_seed,
        device=device,
    )
    generated_tokens = sample_token_ids(
        model,
        prompt_token_id=0,
        new_tokens=config.sample_tokens,
        seed=config.seed + 3,
        temperature=0.8,
        device=device,
    )
    sample_text = decode_tokens(generated_tokens, dataset.vocabulary)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / "embedding_lm.pt"
    checkpoint = {
        "schema_version": 1,
        "model_type": "embedding_lm",
        "model_config": {
            "embedding_dim": config.embedding_dim,
            "vocab_size": len(dataset.vocabulary),
        },
        "model_state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        },
        "source_commit": config.source_commit,
        "vocabulary": list(dataset.vocabulary),
    }
    torch.save(checkpoint, checkpoint_path)
    checkpoint_bytes = checkpoint_path.stat().st_size
    checkpoint_sha256 = _sha256_file(checkpoint_path)

    peak_cuda_bytes = None
    if device.type == "cuda":
        peak_cuda_bytes = int(torch.cuda.max_memory_allocated(device))
    gpu_name = None
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)

    configuration = asdict(config)
    configuration["data_dir"] = str(config.data_dir)
    configuration["output_dir"] = str(config.output_dir)
    summary: dict[str, object] = {
        "artifacts": {
            "checkpoint": {
                "bytes": checkpoint_bytes,
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha256,
            }
        },
        "configuration": configuration,
        "dataset": _dataset_summary(dataset),
        "environment": {
            "cuda_version": torch.version.cuda,
            "device": str(device),
            "gpu_name": gpu_name,
            "peak_cuda_memory_allocated_bytes": peak_cuda_bytes,
            "python": platform.python_version(),
            "torch": str(torch.__version__),
        },
        "history": history,
        "metrics": {
            "final_validation_bpc": bits_per_character(final_validation_loss),
            "final_validation_loss": final_validation_loss,
            "initial_validation_bpc": bits_per_character(initial_validation_loss),
            "initial_validation_loss": initial_validation_loss,
            "training_seconds": training_seconds,
            "uniform_random_loss": math.log(len(dataset.vocabulary)),
        },
        "model": {
            "embedding_dim": config.embedding_dim,
            "parameters": model.parameter_count,
            "type": "embedding_lm_no_attention",
            "vocab_size": len(dataset.vocabulary),
        },
        "sample": {
            "characters": len(sample_text),
            "seed": config.seed + 3,
            "temperature": 0.8,
            "text": sample_text,
        },
        "schema_version": 1,
        "source_commit": config.source_commit,
    }
    _write_json_atomic(config.output_dir / "run.json", summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Lesson 02 embedding-only language model."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--sample-tokens", type=int, default=300)
    parser.add_argument("--source-commit", default="uncommitted")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Lesson 02 training command."""

    parser = _build_parser()
    arguments = parser.parse_args(argv)
    config = TrainingConfig(
        data_dir=arguments.data_dir,
        output_dir=arguments.output_dir,
        device=arguments.device,
        seed=arguments.seed,
        steps=arguments.steps,
        batch_size=arguments.batch_size,
        block_size=arguments.block_size,
        embedding_dim=arguments.embedding_dim,
        learning_rate=arguments.learning_rate,
        eval_batches=arguments.eval_batches,
        sample_tokens=arguments.sample_tokens,
        source_commit=arguments.source_commit,
    )
    try:
        summary = run_training(config)
    except (DatasetError, OSError, RuntimeError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
