"""Deliberately overfit one fixed batch to expose the complete training loop."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Sequence

import torch

from nanogpt_nspire.data import DatasetError
from nanogpt_nspire.models.causal_attention_lm import (
    SingleHeadCausalLanguageModel,
)
from nanogpt_nspire.training_dataset import load_token_dataset, make_batch
from nanogpt_nspire.training_loop import overfit_fixed_batch
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
    """Configuration for a reproducible fixed-batch overfitting run."""

    data_dir: Path
    output_dir: Path
    device: str = "auto"
    seed: int = 1337
    steps: int = 1000
    batch_size: int = 1
    block_size: int = 32
    embedding_dim: int = 64
    learning_rate: float = 0.01
    max_grad_norm: float | None = 1.0
    record_every: int = 100
    eval_batches: int = 50
    target_training_loss: float = 0.05
    source_commit: str = "uncommitted"

    def validate(self) -> None:
        for name in (
            "steps",
            "batch_size",
            "block_size",
            "embedding_dim",
            "record_every",
            "eval_batches",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for name in ("learning_rate", "target_training_loss"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_grad_norm is not None and (
            not math.isfinite(self.max_grad_norm)
            or self.max_grad_norm <= 0.0
        ):
            raise ValueError("max_grad_norm must be finite and positive or None")
        if not self.source_commit:
            raise ValueError("source_commit must not be empty")


def _parameter_vector(model: torch.nn.Module) -> torch.Tensor:
    trainable = [
        parameter.detach().reshape(-1)
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise ValueError("model has no trainable parameters")
    return torch.cat(trainable).clone()


def _parameter_displacement_l2_norm(
    initial_parameters: torch.Tensor,
    model: torch.nn.Module,
) -> float:
    final_parameters = _parameter_vector(model)
    difference = (
        final_parameters.to(torch.float64)
        - initial_parameters.to(torch.float64)
    )
    return float(torch.linalg.vector_norm(difference).item())


def run_overfit_experiment(config: TrainingConfig) -> dict[str, object]:
    """Fit one deterministic training batch and contrast it with validation."""

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

    fixed_batch_seed = config.seed + 1
    fixed_batch_generator = torch.Generator(device="cpu").manual_seed(
        fixed_batch_seed
    )
    fixed_inputs, fixed_targets = make_batch(
        dataset.train,
        batch_size=config.batch_size,
        block_size=config.block_size,
        generator=fixed_batch_generator,
        device=device,
    )
    validation_seed = config.seed + 2
    initial_validation_loss = evaluate_loss(
        model,
        dataset.validation,
        batch_size=config.batch_size,
        block_size=config.block_size,
        batches=config.eval_batches,
        seed=validation_seed,
        device=device,
    )
    initial_parameters = _parameter_vector(model)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    overfit_result = overfit_fixed_batch(
        model,
        optimizer,
        fixed_inputs,
        fixed_targets,
        steps=config.steps,
        record_every=config.record_every,
        max_grad_norm=config.max_grad_norm,
    )
    synchronize(device)
    training_seconds = time.perf_counter() - started

    final_validation_loss = evaluate_loss(
        model,
        dataset.validation,
        batch_size=config.batch_size,
        block_size=config.block_size,
        batches=config.eval_batches,
        seed=validation_seed,
        device=device,
    )
    parameter_displacement = _parameter_displacement_l2_norm(
        initial_parameters,
        model,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / "overfit_attention_lm.pt"
    checkpoint = {
        "experiment_seed": config.seed,
        "fixed_batch": {
            "inputs": fixed_inputs.detach().cpu(),
            "selection_seed": fixed_batch_seed,
            "targets": fixed_targets.detach().cpu(),
        },
        "model_config": {
            "block_size": config.block_size,
            "embedding_dim": config.embedding_dim,
            "vocab_size": len(dataset.vocabulary),
        },
        "model_state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        },
        "model_type": "single_head_causal_attention_lm",
        "schema_version": 1,
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
    initial_fixed_loss = overfit_result.initial.loss
    final_fixed_loss = overfit_result.final.loss
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
        "experiment": {
            "purpose": (
                "Verify optimization wiring by memorizing one deterministic batch; "
                "this is not a generalization benchmark."
            ),
            "type": "fixed_batch_overfit",
        },
        "fixed_batch": {
            "input_token_ids": fixed_inputs.detach().cpu().tolist(),
            "selection_seed": fixed_batch_seed,
            "shape": list(fixed_inputs.shape),
            "target_token_ids": fixed_targets.detach().cpu().tolist(),
            "tokens": fixed_inputs.numel(),
        },
        "history": list(overfit_result.history),
        "metrics": {
            "final_fixed_batch_bpc": bits_per_character(final_fixed_loss),
            "final_fixed_batch_loss": final_fixed_loss,
            "final_fixed_batch_token_accuracy": (
                overfit_result.final.token_accuracy
            ),
            "final_validation_bpc": bits_per_character(final_validation_loss),
            "final_validation_loss": final_validation_loss,
            "fixed_batch_loss_reduction": initial_fixed_loss - final_fixed_loss,
            "fixed_batch_loss_reduction_percent": (
                (initial_fixed_loss - final_fixed_loss)
                / initial_fixed_loss
                * 100.0
            ),
            "generalization_gap": final_validation_loss - final_fixed_loss,
            "initial_fixed_batch_bpc": bits_per_character(initial_fixed_loss),
            "initial_fixed_batch_loss": initial_fixed_loss,
            "initial_fixed_batch_token_accuracy": (
                overfit_result.initial.token_accuracy
            ),
            "initial_validation_bpc": bits_per_character(
                initial_validation_loss
            ),
            "initial_validation_loss": initial_validation_loss,
            "parameter_displacement_l2_norm": parameter_displacement,
            "target_training_loss": config.target_training_loss,
            "target_training_loss_reached": (
                final_fixed_loss <= config.target_training_loss
            ),
            "tokens_per_second": tokens_processed / training_seconds,
            "tokens_processed": tokens_processed,
            "training_seconds": training_seconds,
            "validation_loss_change": (
                final_validation_loss - initial_validation_loss
            ),
        },
        "model": {
            "block_size": config.block_size,
            "embedding_dim": config.embedding_dim,
            "parameters": model.parameter_count,
            "raw_fp32_parameter_bytes": model.parameter_count * 4,
            "type": "single_head_causal_attention_lm",
            "vocab_size": len(dataset.vocabulary),
        },
        "schema_version": 1,
        "source_commit": config.source_commit,
    }
    write_json_atomic(config.output_dir / "run.json", summary)
    return summary


def _optional_float(value: str) -> float | None:
    if value.strip().lower() == "none":
        return None
    try:
        return float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected a number or 'none'"
        ) from error


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Overfit one fixed batch with the Lesson 03 attention model."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=_optional_float, default=1.0)
    parser.add_argument("--record-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--target-training-loss", type=float, default=0.05)
    parser.add_argument("--source-commit", default="uncommitted")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Lesson 04 fixed-batch overfitting command."""

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
        max_grad_norm=arguments.max_grad_norm,
        record_every=arguments.record_every,
        eval_batches=arguments.eval_batches,
        target_training_loss=arguments.target_training_loss,
        source_commit=arguments.source_commit,
    )
    try:
        summary = run_overfit_experiment(config)
    except (DatasetError, FloatingPointError, OSError, RuntimeError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
