"""Train the Lesson 03 single-head causal self-attention model."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Sequence

import torch
from torch import nn

from nanogpt_nspire.data import DatasetError, decode_tokens
from nanogpt_nspire.models.causal_attention_lm import (
    SingleHeadCausalLanguageModel,
)
from nanogpt_nspire.training_dataset import load_token_dataset, make_batch
from nanogpt_nspire.training_support import (
    bits_per_character,
    dataset_summary,
    environment_summary,
    evaluate_loss,
    resolve_device,
    sha256_file,
    synchronize,
    write_json_atomic,
)


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for a reproducible Lesson 03 attention run."""

    data_dir: Path
    output_dir: Path
    device: str = "auto"
    seed: int = 1337
    steps: int = 2000
    batch_size: int = 64
    block_size: int = 64
    embedding_dim: int = 64
    learning_rate: float = 0.003
    eval_batches: int = 50
    sample_tokens: int = 300
    temperature: float = 0.8
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
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        if not self.source_commit:
            raise ValueError("source_commit must not be empty")


def sample_with_context(
    model: nn.Module,
    prompt_token_ids: Sequence[int],
    *,
    new_tokens: int,
    seed: int,
    temperature: float,
    device: torch.device,
) -> list[int]:
    """Sample while cropping model input to its most recent context window."""

    vocab_size = getattr(model, "vocab_size", None)
    block_size = getattr(model, "block_size", None)
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("model must expose a positive vocab_size")
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("model must expose a positive block_size")
    if not prompt_token_ids:
        raise ValueError("prompt_token_ids must not be empty")
    if any(
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or token_id < 0
        or token_id >= vocab_size
        for token_id in prompt_token_ids
    ):
        raise ValueError("prompt contains a token outside the vocabulary")
    if isinstance(new_tokens, bool) or not isinstance(new_tokens, int) or new_tokens < 0:
        raise ValueError("new_tokens must be a non-negative integer")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    generated = list(prompt_token_ids)
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for _ in range(new_tokens):
            context = generated[-block_size:]
            token_ids = torch.tensor(
                [context],
                dtype=torch.long,
                device=device,
            )
            result = model(token_ids)
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError("language model forward must return (logits, loss)")
            logits, _ = result
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


def run_training(config: TrainingConfig) -> dict[str, object]:
    """Train, sample, checkpoint, and summarize the single-head model."""

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
    model = SingleHeadCausalLanguageModel(
        vocab_size=len(dataset.vocabulary),
        embedding_dim=config.embedding_dim,
        block_size=config.block_size,
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
    synchronize(device)
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
    synchronize(device)
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
    generated_tokens = sample_with_context(
        model,
        [0],
        new_tokens=config.sample_tokens,
        seed=config.seed + 3,
        temperature=config.temperature,
        device=device,
    )
    sample_text = decode_tokens(generated_tokens, dataset.vocabulary)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / "single_head_attention_lm.pt"
    checkpoint = {
        "schema_version": 1,
        "model_type": "single_head_causal_attention_lm",
        "model_config": {
            "block_size": config.block_size,
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
    checkpoint_sha256 = sha256_file(checkpoint_path)

    peak_cuda_bytes = None
    if device.type == "cuda":
        peak_cuda_bytes = int(torch.cuda.max_memory_allocated(device))
    configuration = asdict(config)
    configuration["data_dir"] = str(config.data_dir)
    configuration["output_dir"] = str(config.output_dir)
    tokens_processed = config.steps * config.batch_size * config.block_size
    summary: dict[str, object] = {
        "artifacts": {
            "checkpoint": {
                "bytes": checkpoint_bytes,
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha256,
            }
        },
        "configuration": configuration,
        "dataset": dataset_summary(dataset),
        "environment": environment_summary(
            device,
            peak_cuda_memory_allocated_bytes=peak_cuda_bytes,
        ),
        "history": history,
        "metrics": {
            "final_validation_bpc": bits_per_character(final_validation_loss),
            "final_validation_loss": final_validation_loss,
            "initial_validation_bpc": bits_per_character(initial_validation_loss),
            "initial_validation_loss": initial_validation_loss,
            "tokens_per_second": tokens_processed / training_seconds,
            "tokens_processed": tokens_processed,
            "training_seconds": training_seconds,
            "uniform_random_loss": math.log(len(dataset.vocabulary)),
        },
        "model": {
            "block_size": config.block_size,
            "components": [
                "token_embedding",
                "position_embedding",
                "single_head_causal_self_attention",
                "residual_connection",
                "lm_head",
            ],
            "embedding_dim": config.embedding_dim,
            "parameters": model.parameter_count,
            "raw_fp32_parameter_bytes": model.parameter_count * 4,
            "type": "single_head_causal_attention_lm",
            "vocab_size": len(dataset.vocabulary),
        },
        "sample": {
            "characters": len(sample_text),
            "seed": config.seed + 3,
            "temperature": config.temperature,
            "text": sample_text,
        },
        "schema_version": 1,
        "source_commit": config.source_commit,
    }
    write_json_atomic(config.output_dir / "run.json", summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Lesson 03 single-head causal attention model."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--sample-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--source-commit", default="uncommitted")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Lesson 03 training command."""

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
        temperature=arguments.temperature,
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
