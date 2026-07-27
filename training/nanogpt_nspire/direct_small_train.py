"""Train the frozen Direct-Small GPT baseline from random parameters."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Mapping, Protocol, Sequence

import torch
from torch import nn

from nanogpt_nspire.data import DatasetError, decode_tokens
from nanogpt_nspire.lesson03_train import sample_with_context
from nanogpt_nspire.models.direct_small_gpt import (
    DirectSmallConfig,
    DirectSmallGPT,
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


DEPLOYMENT_FILE_LIMIT_BYTES = 6 * 1024 * 1024
DEPLOYMENT_METADATA_RESERVE_BYTES = 64 * 1024


@dataclass(frozen=True)
class TrainingRunIdentity:
    """Name one use of the shared GPT training engine."""

    route: str = "Direct-Small"
    checkpoint_filename: str = "direct_small_gpt.pt"
    deployment_interpretation: str = "fp32_deployment_candidate"
    quality_gate_maximum_selected_validation_loss: float | None = None

    def validate(self) -> None:
        if not self.route.strip():
            raise ValueError("route must not be empty")
        checkpoint_path = Path(self.checkpoint_filename)
        if (
            not self.checkpoint_filename
            or checkpoint_path.name != self.checkpoint_filename
            or checkpoint_path.suffix.lower() != ".pt"
        ):
            raise ValueError(
                "checkpoint_filename must be a basename ending in .pt"
            )
        if not self.deployment_interpretation.strip():
            raise ValueError("deployment_interpretation must not be empty")
        threshold = self.quality_gate_maximum_selected_validation_loss
        if threshold is not None and (
            not math.isfinite(threshold) or threshold <= 0.0
        ):
            raise ValueError(
                "quality_gate maximum loss must be finite and positive or None"
            )


DEFAULT_DIRECT_RUN_IDENTITY = TrainingRunIdentity()


@dataclass(frozen=True)
class TrainingObjectiveStep:
    """One differentiable scalar objective plus finite logging components."""

    loss: torch.Tensor
    metrics: Mapping[str, float]


class TrainingObjective(Protocol):
    """Replace hard-only training while preserving the shared engine."""

    def __call__(
        self,
        model: DirectSmallGPT,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> TrainingObjectiveStep:
        ...

    def summary(self) -> dict[str, object]:
        ...


@dataclass(frozen=True)
class TrainingConfig:
    """Frozen Direct-Small architecture and reproducible training protocol."""

    data_dir: Path
    output_dir: Path
    device: str = "auto"
    seed: int = 1337
    steps: int = 5000
    batch_size: int = 64
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 5
    n_embd: int = 160
    mlp_ratio: int = 4
    dropout: float = 0.1
    bias: bool = False
    tie_embeddings: bool = True
    learning_rate: float = 0.001
    min_learning_rate: float = 0.0001
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.99
    max_grad_norm: float = 1.0
    eval_interval: int = 250
    eval_batches: int = 50
    log_interval: int = 100
    sample_tokens: int = 300
    temperature: float = 0.8
    deployment_file_limit_bytes: int = DEPLOYMENT_FILE_LIMIT_BYTES
    deployment_metadata_reserve_bytes: int = DEPLOYMENT_METADATA_RESERVE_BYTES
    source_commit: str = "uncommitted"

    def model_config(self, *, vocab_size: int) -> DirectSmallConfig:
        return DirectSmallConfig(
            vocab_size=vocab_size,
            block_size=self.block_size,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
            bias=self.bias,
            tie_embeddings=self.tie_embeddings,
        )

    def validate(self) -> None:
        for name in (
            "steps",
            "batch_size",
            "block_size",
            "n_layer",
            "n_head",
            "n_embd",
            "mlp_ratio",
            "warmup_steps",
            "eval_interval",
            "eval_batches",
            "log_interval",
            "deployment_file_limit_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.deployment_metadata_reserve_bytes, bool)
            or not isinstance(self.deployment_metadata_reserve_bytes, int)
            or self.deployment_metadata_reserve_bytes < 0
        ):
            raise ValueError(
                "deployment_metadata_reserve_bytes must be a non-negative integer"
            )
        if (
            isinstance(self.sample_tokens, bool)
            or not isinstance(self.sample_tokens, int)
            or self.sample_tokens < 0
        ):
            raise ValueError("sample_tokens must be a non-negative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.warmup_steps >= self.steps:
            raise ValueError("warmup_steps must be smaller than steps")
        for name in (
            "learning_rate",
            "min_learning_rate",
            "max_grad_norm",
            "temperature",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError(
                "min_learning_rate must not exceed learning_rate"
            )
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        for name in ("beta1", "beta2"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be finite and in (0, 1)")
        self.model_config(vocab_size=1).validate()
        if not self.source_commit:
            raise ValueError("source_commit must not be empty")


def learning_rate_at_step(
    step: int,
    *,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_steps: int,
    max_steps: int,
) -> float:
    """Linear warmup followed by cosine decay to a fixed minimum."""

    for value, name in ((step, "step"), (warmup_steps, "warmup_steps"), (max_steps, "max_steps")):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if warmup_steps >= max_steps:
        raise ValueError("warmup_steps must be smaller than max_steps")
    for value, name in (
        (max_learning_rate, "max_learning_rate"),
        (min_learning_rate, "min_learning_rate"),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if min_learning_rate > max_learning_rate:
        raise ValueError(
            "min_learning_rate must not exceed max_learning_rate"
        )
    if step <= warmup_steps:
        return max_learning_rate * step / warmup_steps
    if step >= max_steps:
        return min_learning_rate
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_learning_rate + cosine * (
        max_learning_rate - min_learning_rate
    )


def configure_adamw(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
) -> torch.optim.AdamW:
    """Apply weight decay to matrices while excluding norm/bias vectors."""

    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError("weight_decay must be finite and non-negative")
    for value, name in ((beta1, "beta1"), (beta2, "beta2")):
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be finite and in (0, 1)")

    unique_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    decay = [parameter for parameter in unique_parameters if parameter.ndim >= 2]
    no_decay = [parameter for parameter in unique_parameters if parameter.ndim < 2]
    if len({id(parameter) for parameter in decay + no_decay}) != len(
        unique_parameters
    ):
        raise RuntimeError("optimizer parameter grouping contains duplicates")
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(beta1, beta2),
    )


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _timed_validation_loss(
    model: nn.Module,
    tokens: torch.Tensor,
    *,
    batch_size: int,
    block_size: int,
    batches: int,
    seed: int,
    device: torch.device,
) -> tuple[float, float]:
    synchronize(device)
    started = time.perf_counter()
    loss = evaluate_loss(
        model,
        tokens,
        batch_size=batch_size,
        block_size=block_size,
        batches=batches,
        seed=seed,
        device=device,
    )
    synchronize(device)
    return loss, time.perf_counter() - started


def run_training(
    config: TrainingConfig,
    *,
    run_identity: TrainingRunIdentity | None = None,
    training_objective: TrainingObjective | None = None,
) -> dict[str, object]:
    """Train, select, sample, checkpoint, and summarize Direct-Small."""

    config.validate()
    identity = run_identity or DEFAULT_DIRECT_RUN_IDENTITY
    identity.validate()
    objective_summary = (
        {"name": "hard_label_cross_entropy"}
        if training_objective is None
        else training_objective.summary()
    )
    if (
        not isinstance(objective_summary, dict)
        or not isinstance(objective_summary.get("name"), str)
        or not objective_summary["name"]
    ):
        raise ValueError(
            "training objective summary must be a dict with a non-empty name"
        )
    device = resolve_device(config.device)
    dataset = load_token_dataset(config.data_dir)
    if dataset.train.numel() < config.block_size + 1:
        raise DatasetError("training split is too short for block_size")
    if dataset.validation.numel() < config.block_size + 1:
        raise DatasetError("validation split is too short for block_size")

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    model_config = config.model_config(vocab_size=len(dataset.vocabulary))
    model = DirectSmallGPT(model_config).to(device)
    optimizer = configure_adamw(
        model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        beta1=config.beta1,
        beta2=config.beta2,
    )

    estimated_deployment_file_bytes = (
        model.raw_fp32_parameter_bytes
        + config.deployment_metadata_reserve_bytes
    )
    estimated_file_eligible = (
        estimated_deployment_file_bytes
        <= config.deployment_file_limit_bytes
    )

    validation_seed = config.seed + 1
    initial_validation_loss, initial_evaluation_seconds = _timed_validation_loss(
        model,
        dataset.validation,
        batch_size=config.batch_size,
        block_size=config.block_size,
        batches=config.eval_batches,
        seed=validation_seed,
        device=device,
    )
    evaluation_seconds = initial_evaluation_seconds
    best_validation_loss = initial_validation_loss
    best_step = 0
    best_state = _cpu_state_dict(model)
    evaluation_history: list[dict[str, float | int | bool]] = [
        {
            "bits_per_character": bits_per_character(initial_validation_loss),
            "is_new_best": True,
            "step": 0,
            "validation_loss": initial_validation_loss,
        }
    ]

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_generator = torch.Generator(device="cpu").manual_seed(
        config.seed + 2
    )
    training_history: list[dict[str, float | int]] = []
    optimizer_update_seconds = 0.0
    segment_started = time.perf_counter()
    final_step_validation_loss = initial_validation_loss
    model.train()
    for step in range(1, config.steps + 1):
        learning_rate = learning_rate_at_step(
            step,
            max_learning_rate=config.learning_rate,
            min_learning_rate=config.min_learning_rate,
            warmup_steps=config.warmup_steps,
            max_steps=config.steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        inputs, targets = make_batch(
            dataset.train,
            batch_size=config.batch_size,
            block_size=config.block_size,
            generator=training_generator,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        objective_metrics: Mapping[str, float] = {}
        if training_objective is None:
            _, loss = model(inputs, targets)
            assert loss is not None
        else:
            objective_step = training_objective(model, inputs, targets)
            if not isinstance(objective_step, TrainingObjectiveStep):
                raise TypeError(
                    "training objective must return TrainingObjectiveStep"
                )
            loss = objective_step.loss
            objective_metrics = objective_step.metrics
        if (
            not isinstance(loss, torch.Tensor)
            or loss.ndim != 0
            or not loss.requires_grad
        ):
            raise TypeError(
                "training objective loss must be a differentiable scalar tensor"
            )
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(
                f"training loss became non-finite at step {step}"
            )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=config.max_grad_norm,
        )
        if not bool(torch.isfinite(gradient_norm).item()):
            raise FloatingPointError(
                f"gradient norm became non-finite at step {step}"
            )
        optimizer.step()

        if step == 1 or step % config.log_interval == 0 or step == config.steps:
            reserved_metrics = {
                "gradient_l2_norm_before_clip",
                "learning_rate",
                "step",
                "training_loss",
            }
            logged_objective_metrics: dict[str, float] = {}
            for name, value in objective_metrics.items():
                if not isinstance(name, str) or not name:
                    raise ValueError(
                        "training objective metric names must be non-empty strings"
                    )
                if name in reserved_metrics:
                    raise ValueError(
                        f"training objective metric {name!r} is reserved"
                    )
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError(
                        f"training objective metric {name!r} must be finite"
                    )
                logged_objective_metrics[name] = float(value)
            training_history.append(
                {
                    "gradient_l2_norm_before_clip": float(
                        gradient_norm.item()
                    ),
                    "learning_rate": learning_rate,
                    "step": step,
                    "training_loss": float(loss.detach().item()),
                    **logged_objective_metrics,
                }
            )

        if step % config.eval_interval == 0 or step == config.steps:
            synchronize(device)
            optimizer_update_seconds += time.perf_counter() - segment_started
            validation_loss, elapsed = _timed_validation_loss(
                model,
                dataset.validation,
                batch_size=config.batch_size,
                block_size=config.block_size,
                batches=config.eval_batches,
                seed=validation_seed,
                device=device,
            )
            evaluation_seconds += elapsed
            final_step_validation_loss = validation_loss
            is_new_best = validation_loss < best_validation_loss
            if is_new_best:
                best_validation_loss = validation_loss
                best_step = step
                best_state = _cpu_state_dict(model)
            evaluation_history.append(
                {
                    "bits_per_character": bits_per_character(validation_loss),
                    "is_new_best": is_new_best,
                    "step": step,
                    "validation_loss": validation_loss,
                }
            )
            model.train()
            segment_started = time.perf_counter()

    model.load_state_dict(best_state, strict=True)
    selected_validation_loss, selected_evaluation_seconds = _timed_validation_loss(
        model,
        dataset.validation,
        batch_size=config.batch_size,
        block_size=config.block_size,
        batches=config.eval_batches,
        seed=validation_seed,
        device=device,
    )
    evaluation_seconds += selected_evaluation_seconds
    generated_tokens = sample_with_context(
        model,
        [0],
        new_tokens=config.sample_tokens,
        seed=config.seed + 3,
        temperature=config.temperature,
        device=device,
    )
    sample_text = decode_tokens(generated_tokens, dataset.vocabulary)
    quality_threshold = (
        identity.quality_gate_maximum_selected_validation_loss
    )
    quality_gate_passed = (
        None
        if quality_threshold is None
        else selected_validation_loss <= quality_threshold
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / identity.checkpoint_filename
    checkpoint = {
        "best_step": best_step,
        "deployment_interpretation": (
            identity.deployment_interpretation
        ),
        "model_config": asdict(model_config),
        "model_state_dict": _cpu_state_dict(model),
        "model_type": "direct_small_gpt",
        "quality_gate_maximum_selected_validation_loss": (
            quality_threshold
        ),
        "quality_gate_passed": quality_gate_passed,
        "route": identity.route,
        "schema_version": 1,
        "selected_validation_loss": selected_validation_loss,
        "source_commit": config.source_commit,
        "training_objective": objective_summary,
        "training_seed": config.seed,
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
    training_tokens = (
        config.steps * config.batch_size * config.block_size
    )
    kv_cache_fp32_bytes = (
        2
        * config.n_layer
        * config.block_size
        * config.n_embd
        * 4
    )
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
        "deployment": {
            "actual_deployment_file": {
                "reason": "unified model exporter is implemented in Lesson 08",
                "status": "pending",
            },
            "estimated_deployment_file_bytes": (
                estimated_deployment_file_bytes
            ),
            "estimated_file_eligible": estimated_file_eligible,
            "file_limit_bytes": config.deployment_file_limit_bytes,
            "host_c_alignment": {
                "reason": "Host C runtime is implemented in Lesson 08",
                "status": "pending",
            },
            "host_peak_ram": {
                "reason": "Host C runtime is implemented in Lesson 08",
                "status": "pending",
            },
            "kv_cache_fp32_bytes": kv_cache_fp32_bytes,
            "metadata_reserve_bytes": (
                config.deployment_metadata_reserve_bytes
            ),
            "nspire_peak_ram": {
                "reason": "CX II measurement is performed in Lesson 09",
                "status": "pending",
            },
            "raw_fp32_parameter_bytes": (
                model.raw_fp32_parameter_bytes
            ),
            "route_interpretation": (
                identity.deployment_interpretation
            ),
        },
        "environment": environment_summary(
            device,
            peak_cuda_memory_allocated_bytes=peak_cuda_bytes,
        ),
        "evaluation_history": evaluation_history,
        "metrics": {
            "best_step": best_step,
            "evaluation_seconds": evaluation_seconds,
            "final_step_validation_bpc": bits_per_character(
                final_step_validation_loss
            ),
            "final_step_validation_loss": final_step_validation_loss,
            "initial_validation_bpc": bits_per_character(
                initial_validation_loss
            ),
            "initial_validation_loss": initial_validation_loss,
            "optimizer_update_seconds": optimizer_update_seconds,
            "quality_gate_maximum_selected_validation_loss": (
                quality_threshold
            ),
            "quality_gate_passed": quality_gate_passed,
            "selected_validation_bpc": bits_per_character(
                selected_validation_loss
            ),
            "selected_validation_loss": selected_validation_loss,
            "training_tokens": training_tokens,
            "update_tokens_per_second": (
                training_tokens / optimizer_update_seconds
            ),
        },
        "model": {
            **asdict(model_config),
            "head_dim": config.n_embd // config.n_head,
            "parameters": model.parameter_count,
            "raw_fp32_parameter_bytes": model.raw_fp32_parameter_bytes,
            "type": "direct_small_gpt",
        },
        "optimizer": {
            "beta1": config.beta1,
            "beta2": config.beta2,
            "gradient_norm_cap": config.max_grad_norm,
            "maximum_learning_rate": config.learning_rate,
            "minimum_learning_rate": config.min_learning_rate,
            "name": "AdamW",
            "schedule": "linear_warmup_then_cosine",
            "warmup_steps": config.warmup_steps,
            "weight_decay": config.weight_decay,
            "weight_decay_rule": "parameter.ndim >= 2",
        },
        "route": identity.route,
        "run_identity": asdict(identity),
        "sample": {
            "characters": len(sample_text),
            "seed": config.seed + 3,
            "temperature": config.temperature,
            "text": sample_text,
        },
        "schema_version": 1,
        "source_commit": config.source_commit,
        "training_objective": objective_summary,
        "training_history": training_history,
    }
    write_json_atomic(config.output_dir / "run.json", summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the frozen deployable Direct-Small GPT baseline."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=5)
    parser.add_argument("--n-embd", type=int, default=160)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--min-learning-rate", type=float, default=0.0001)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--sample-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument(
        "--deployment-file-limit-bytes",
        type=int,
        default=DEPLOYMENT_FILE_LIMIT_BYTES,
    )
    parser.add_argument(
        "--deployment-metadata-reserve-bytes",
        type=int,
        default=DEPLOYMENT_METADATA_RESERVE_BYTES,
    )
    parser.add_argument("--source-commit", default="uncommitted")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Direct-Small baseline command."""

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
        n_layer=arguments.n_layer,
        n_head=arguments.n_head,
        n_embd=arguments.n_embd,
        mlp_ratio=arguments.mlp_ratio,
        dropout=arguments.dropout,
        learning_rate=arguments.learning_rate,
        min_learning_rate=arguments.min_learning_rate,
        warmup_steps=arguments.warmup_steps,
        weight_decay=arguments.weight_decay,
        beta1=arguments.beta1,
        beta2=arguments.beta2,
        max_grad_norm=arguments.max_grad_norm,
        eval_interval=arguments.eval_interval,
        eval_batches=arguments.eval_batches,
        log_interval=arguments.log_interval,
        sample_tokens=arguments.sample_tokens,
        temperature=arguments.temperature,
        deployment_file_limit_bytes=(
            arguments.deployment_file_limit_bytes
        ),
        deployment_metadata_reserve_bytes=(
            arguments.deployment_metadata_reserve_bytes
        ),
        source_commit=arguments.source_commit,
    )
    try:
        summary = run_training(config)
    except (
        DatasetError,
        FloatingPointError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
