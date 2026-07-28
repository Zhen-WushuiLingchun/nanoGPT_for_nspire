"""Train Lesson 12 CPT/SFT stages from a strictly declared parent checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import platform
import re
import time
from typing import Mapping

import numpy as np
import torch

from nanogpt_nspire.base_train import (
    _atomic_torch_save,
    _autocast_context,
    _cpu_state_dict,
    _dataset_summary,
    _evaluate_model,
    _loss_metrics,
    _run_overfit_gate,
    evaluate_sequential_split,
    load_packed_dataset,
    make_packed_batch,
    masked_cross_entropy,
)
from nanogpt_nspire.byte_tokenizer import VOCAB_SIZE
from nanogpt_nspire.direct_small_train import (
    configure_adamw,
    learning_rate_at_step,
)
from nanogpt_nspire.models.direct_small_gpt import (
    DirectSmallConfig,
    DirectSmallGPT,
)
from nanogpt_nspire.training_support import (
    resolve_device,
    sha256_file,
    synchronize,
    write_json_atomic,
)


STAGE_ROUTES = {
    "cpt": "Math-Physics-CPT",
    "sft": "Role-Aware-SFT",
}
EXPECTED_PARENT_ROUTES = {
    "cpt": "English-Base-Pilot",
    "sft": "Math-Physics-CPT",
}
CHECKPOINT_FILENAMES = {
    "cpt": "cpt_best.pt",
    "sft": "sft_best.pt",
}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class StageTrainingConfig:
    """One reproducible CPT or role-aware SFT optimization stage."""

    stage: str
    data_dir: Path
    output_dir: Path
    parent_checkpoint: Path
    parent_checkpoint_sha256: str
    expected_parent_route: str
    source_commit: str
    device: str = "auto"
    seed: int = 20260728
    steps: int = 1000
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    vocab_size: int = VOCAB_SIZE
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    mlp_ratio: int = 4
    dropout: float = 0.1
    bias: bool = False
    tie_embeddings: bool = True
    learning_rate: float = 0.0003
    min_learning_rate: float = 0.00003
    warmup_steps: int = 50
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 20
    log_interval: int = 25
    overfit_gate_steps: int = 20
    use_bfloat16: bool = True
    route_override: str | None = None
    checkpoint_filename_override: str | None = None
    required_parent_route_override: str | None = None

    @property
    def effective_batch_tokens(self) -> int:
        return (
            self.micro_batch_size
            * self.gradient_accumulation_steps
            * self.block_size
        )

    @property
    def route(self) -> str:
        return self.route_override or STAGE_ROUTES[self.stage]

    @property
    def checkpoint_filename(self) -> str:
        return (
            self.checkpoint_filename_override
            or CHECKPOINT_FILENAMES[self.stage]
        )

    def model_config(self) -> DirectSmallConfig:
        return DirectSmallConfig(
            vocab_size=self.vocab_size,
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
        if self.stage not in STAGE_ROUTES:
            raise ValueError("stage must be 'cpt' or 'sft'")
        required_parent = (
            self.required_parent_route_override
            or EXPECTED_PARENT_ROUTES[self.stage]
        )
        if self.expected_parent_route != required_parent:
            label = self.stage.upper()
            raise ValueError(
                f"{label} must start from {required_parent}"
            )
        if (
            not isinstance(self.parent_checkpoint_sha256, str)
            or _SHA256_PATTERN.fullmatch(
                self.parent_checkpoint_sha256
            )
            is None
        ):
            raise ValueError(
                "parent_checkpoint_sha256 must be 64 lowercase hex characters"
            )
        if not isinstance(self.source_commit, str) or not self.source_commit:
            raise ValueError("source_commit must be non-empty")
        for name in (
            "steps",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "vocab_size",
            "block_size",
            "n_layer",
            "n_head",
            "n_embd",
            "mlp_ratio",
            "warmup_steps",
            "eval_interval",
            "eval_batches",
            "log_interval",
            "overfit_gate_steps",
        ):
            _positive_integer(getattr(self, name), name)
        _non_negative_integer(self.seed, "seed")
        if self.vocab_size != VOCAB_SIZE:
            raise ValueError(f"vocab_size must equal {VOCAB_SIZE}")
        if self.warmup_steps >= self.steps:
            raise ValueError("warmup_steps must be smaller than steps")
        for name in (
            "learning_rate",
            "min_learning_rate",
            "max_grad_norm",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError(
                "min_learning_rate must not exceed learning_rate"
            )
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        for name in ("beta1", "beta2"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must be finite and in (0, 1)")
        if (
            not math.isfinite(self.dropout)
            or not 0 <= self.dropout < 1
        ):
            raise ValueError("dropout must be finite and in [0, 1)")
        if not isinstance(self.use_bfloat16, bool):
            raise ValueError("use_bfloat16 must be boolean")
        for name in ("route_override", "required_parent_route_override"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"{name} must be null or non-empty")
        if self.checkpoint_filename_override is not None and (
            not isinstance(self.checkpoint_filename_override, str)
            or not self.checkpoint_filename_override.endswith(".pt")
            or Path(self.checkpoint_filename_override).name
            != self.checkpoint_filename_override
        ):
            raise ValueError(
                "checkpoint_filename_override must be one local .pt filename"
            )
        self.model_config().validate()


def frozen_student_stage_config(
    *,
    stage: str,
    data_dir: str | Path,
    output_dir: str | Path,
    parent_checkpoint: str | Path,
    parent_checkpoint_sha256: str,
    source_commit: str,
    **overrides: object,
) -> StageTrainingConfig:
    """Create a frozen 6x384 Lesson 12 stage with stage-specific LR."""

    if stage not in STAGE_ROUTES:
        raise ValueError("stage must be 'cpt' or 'sft'")
    defaults: dict[str, object] = {}
    if stage == "sft":
        defaults.update(
            {
                "learning_rate": 0.0001,
                "min_learning_rate": 0.00001,
            }
        )
    defaults.update(overrides)
    return StageTrainingConfig(
        stage=stage,
        data_dir=Path(data_dir),
        output_dir=Path(output_dir),
        parent_checkpoint=Path(parent_checkpoint),
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        expected_parent_route=EXPECTED_PARENT_ROUTES[stage],
        source_commit=source_commit,
        **defaults,
    )


def load_parent_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_route: str,
    expected_model_config: DirectSmallConfig,
) -> dict[str, object]:
    """Verify lineage, tokenizer, architecture, tensor keys, shapes, and values."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if (
        not isinstance(expected_sha256, str)
        or _SHA256_PATTERN.fullmatch(expected_sha256) is None
    ):
        raise ValueError("expected SHA-256 must be 64 lowercase hex characters")
    actual_sha256 = sha256_file(checkpoint_path)
    if actual_sha256 != expected_sha256:
        raise ValueError("parent checkpoint SHA-256 mismatch")
    try:
        raw = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise ValueError("parent checkpoint could not be loaded safely") from error
    if not isinstance(raw, Mapping):
        raise ValueError("parent checkpoint must contain a mapping")
    if raw.get("schema_version") != 1:
        raise ValueError("parent checkpoint schema_version must be 1")
    if raw.get("route") != expected_route:
        raise ValueError(
            f"parent checkpoint route must be {expected_route}"
        )
    tokenizer = raw.get("tokenizer")
    if (
        not isinstance(tokenizer, Mapping)
        or tokenizer.get("kind")
        != "byte_plus_fixed_special_tokens"
        or tokenizer.get("vocab_size") != VOCAB_SIZE
    ):
        raise ValueError("parent checkpoint tokenizer contract is invalid")
    expected_configuration = asdict(expected_model_config)
    if raw.get("model_config") != expected_configuration:
        raise ValueError("parent checkpoint model configuration mismatch")
    state = raw.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("parent checkpoint model_state_dict is missing")
    reference_state = DirectSmallGPT(expected_model_config).state_dict()
    if set(state) != set(reference_state):
        raise ValueError("parent checkpoint tensor keys mismatch")
    checked_state: dict[str, torch.Tensor] = {}
    for name, reference in reference_state.items():
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"parent checkpoint tensor {name} is invalid")
        if tensor.shape != reference.shape or tensor.dtype != reference.dtype:
            raise ValueError(
                f"parent checkpoint tensor {name} shape or dtype mismatch"
            )
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(
                f"parent checkpoint tensor {name} contains non-finite values"
            )
        checked_state[name] = tensor.detach().cpu()
    return {
        "model_state_dict": checked_state,
        "route": expected_route,
        "sha256": actual_sha256,
        "source_commit": raw.get("source_commit"),
    }


def run_stage_training(
    config: StageTrainingConfig,
) -> dict[str, object]:
    """Optimize one declared CPT/SFT stage and select by validation loss."""

    if not isinstance(config, StageTrainingConfig):
        raise ValueError("config must be a StageTrainingConfig")
    config.validate()
    if config.output_dir.exists():
        raise ValueError(
            f"output directory already exists: {config.output_dir}"
        )
    dataset = load_packed_dataset(config.data_dir)
    for name, split in (
        ("train", dataset.train),
        ("validation", dataset.validation),
        ("test", dataset.test),
    ):
        if split.token_count < config.block_size + 1:
            raise ValueError(f"{name} split is too short for block_size")
        if int(np.asarray(split.loss_mask).sum()) == 0:
            raise ValueError(f"{name} split has no eligible targets")
    device = resolve_device(config.device)
    parent = load_parent_checkpoint(
        config.parent_checkpoint,
        expected_sha256=config.parent_checkpoint_sha256,
        expected_route=config.expected_parent_route,
        expected_model_config=config.model_config(),
    )
    overfit_gate = _run_overfit_gate(dataset, config, device=device)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.reset_peak_memory_stats(device)
    model = DirectSmallGPT(config.model_config()).to(device)
    model.load_state_dict(parent["model_state_dict"], strict=True)
    optimizer = configure_adamw(
        model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        beta1=config.beta1,
        beta2=config.beta2,
    )
    validation_seed = config.seed + 1
    synchronize(device)
    evaluation_started = time.perf_counter()
    initial_validation_loss = _evaluate_model(
        model,
        dataset.validation,
        config=config,
        device=device,
        seed=validation_seed,
    )
    initial_full_validation = evaluate_sequential_split(
        model,
        dataset.validation,
        block_size=config.block_size,
        batch_size=config.micro_batch_size,
        device=device,
        use_bfloat16=config.use_bfloat16,
    )
    synchronize(device)
    evaluation_seconds = time.perf_counter() - evaluation_started
    best_validation_loss = initial_validation_loss
    best_step = 0
    best_state = _cpu_state_dict(model)
    evaluation_history: list[dict[str, object]] = [
        {
            **_loss_metrics(initial_validation_loss),
            "is_new_best": True,
            "step": 0,
        }
    ]
    training_history: list[dict[str, object]] = []
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 2)
    optimizer_update_seconds = 0.0
    sampled_eligible_targets = 0
    synchronize(device)
    segment_started = time.perf_counter()
    wall_started = segment_started
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
        optimizer.zero_grad(set_to_none=True)
        micro_losses: list[float] = []
        for _ in range(config.gradient_accumulation_steps):
            batch = make_packed_batch(
                dataset.train,
                batch_size=config.micro_batch_size,
                block_size=config.block_size,
                generator=generator,
                device=device,
            )
            sampled_eligible_targets += int(batch.target_mask.sum().item())
            with _autocast_context(
                device,
                enabled=config.use_bfloat16,
            ):
                logits, _ = model(batch.inputs)
                loss = masked_cross_entropy(
                    logits,
                    batch.targets,
                    batch.target_mask,
                )
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(
                    f"training loss became non-finite at step {step}"
                )
            (loss / config.gradient_accumulation_steps).backward()
            micro_losses.append(float(loss.detach().item()))
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.max_grad_norm,
        )
        if not bool(torch.isfinite(gradient_norm).item()):
            raise FloatingPointError(
                f"gradient norm became non-finite at step {step}"
            )
        optimizer.step()
        if (
            step == 1
            or step % config.log_interval == 0
            or step == config.steps
        ):
            training_history.append(
                {
                    "gradient_l2_norm_before_clip": float(
                        gradient_norm.item()
                    ),
                    "learning_rate": learning_rate,
                    "step": step,
                    "training_loss": sum(micro_losses) / len(micro_losses),
                }
            )
        if step % config.eval_interval == 0 or step == config.steps:
            synchronize(device)
            optimizer_update_seconds += (
                time.perf_counter() - segment_started
            )
            evaluation_started = time.perf_counter()
            validation_loss = _evaluate_model(
                model,
                dataset.validation,
                config=config,
                device=device,
                seed=validation_seed,
            )
            synchronize(device)
            evaluation_seconds += (
                time.perf_counter() - evaluation_started
            )
            final_step_validation_loss = validation_loss
            is_new_best = validation_loss < best_validation_loss
            if is_new_best:
                best_validation_loss = validation_loss
                best_step = step
                best_state = _cpu_state_dict(model)
            evaluation_history.append(
                {
                    **_loss_metrics(validation_loss),
                    "is_new_best": is_new_best,
                    "step": step,
                }
            )
            model.train()
            synchronize(device)
            segment_started = time.perf_counter()
    synchronize(device)
    wall_seconds = time.perf_counter() - wall_started

    model.load_state_dict(best_state, strict=True)
    selected_validation_loss = _evaluate_model(
        model,
        dataset.validation,
        config=config,
        device=device,
        seed=validation_seed,
    )
    selected_full_validation = evaluate_sequential_split(
        model,
        dataset.validation,
        block_size=config.block_size,
        batch_size=config.micro_batch_size,
        device=device,
        use_bfloat16=config.use_bfloat16,
    )
    selected_full_test = evaluate_sequential_split(
        model,
        dataset.test,
        block_size=config.block_size,
        batch_size=config.micro_batch_size,
        device=device,
        use_bfloat16=config.use_bfloat16,
    )
    peak_cuda_bytes = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )
    checkpoint = {
        "best_step": best_step,
        "dataset_manifest_sha256": sha256_file(dataset.manifest_path),
        "model_config": asdict(config.model_config()),
        "model_state_dict": best_state,
        "parent_checkpoint_sha256": parent["sha256"],
        "parent_route": parent["route"],
        "route": config.route,
        "schema_version": 1,
        "selected_validation_loss": selected_validation_loss,
        "source_commit": config.source_commit,
        "tokenizer": {
            "kind": "byte_plus_fixed_special_tokens",
            "vocab_size": VOCAB_SIZE,
        },
        "training_seed": config.seed,
    }
    checkpoint_path = config.output_dir / config.checkpoint_filename
    _atomic_torch_save(checkpoint, checkpoint_path)
    configuration = asdict(config)
    for name in (
        "data_dir",
        "output_dir",
        "parent_checkpoint",
    ):
        configuration[name] = str(configuration[name])
    eligible_training_targets = int(
        np.asarray(dataset.train.loss_mask).sum()
    )
    training_tokens = config.steps * config.effective_batch_tokens
    summary: dict[str, object] = {
        "artifacts": {
            "checkpoint": {
                "bytes": checkpoint_path.stat().st_size,
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
            },
        },
        "configuration": configuration,
        "dataset": _dataset_summary(dataset),
        "environment": {
            "bfloat16_autocast": (
                config.use_bfloat16 and device.type == "cuda"
            ),
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "gpu_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "peak_cuda_memory_allocated_bytes": peak_cuda_bytes,
            "python": platform.python_version(),
            "torch": str(torch.__version__),
        },
        "evaluation_history": evaluation_history,
        "lineage": {
            "parent_route": parent["route"],
            "parent_sha256": parent["sha256"],
            "parent_source_commit": parent["source_commit"],
        },
        "metrics": {
            "approximate_eligible_target_epochs": (
                sampled_eligible_targets / eligible_training_targets
            ),
            "best_step": best_step,
            "eligible_sampled_training_targets": sampled_eligible_targets,
            "eligible_training_targets": eligible_training_targets,
            "evaluation_seconds": evaluation_seconds,
            "final_step_validation": _loss_metrics(
                final_step_validation_loss
            ),
            "initial_full_validation": initial_full_validation,
            "initial_validation": _loss_metrics(
                initial_validation_loss
            ),
            "optimizer_update_seconds": optimizer_update_seconds,
            "selected_full_test": selected_full_test,
            "selected_full_validation": selected_full_validation,
            "selected_validation": _loss_metrics(
                selected_validation_loss
            ),
            "training_tokens": training_tokens,
            "update_tokens_per_second": (
                training_tokens / optimizer_update_seconds
            ),
            "wall_seconds": wall_seconds,
        },
        "model": {
            **asdict(config.model_config()),
            "parameters": model.parameter_count,
            "raw_fp32_parameter_bytes": model.raw_fp32_parameter_bytes,
        },
        "overfit_gate": overfit_gate,
        "route": config.route,
        "schema_version": 1,
        "source_commit": config.source_commit,
        "training_history": training_history,
    }
    write_json_atomic(config.output_dir / "run.json", summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGE_ROUTES), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-checkpoint-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
    )
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--min-learning-rate", type=float)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--overfit-gate-steps", type=int, default=20)
    parser.add_argument("--no-bfloat16", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    overrides: dict[str, object] = {
        "device": arguments.device,
        "eval_batches": arguments.eval_batches,
        "eval_interval": arguments.eval_interval,
        "gradient_accumulation_steps": (
            arguments.gradient_accumulation_steps
        ),
        "log_interval": arguments.log_interval,
        "micro_batch_size": arguments.micro_batch_size,
        "overfit_gate_steps": arguments.overfit_gate_steps,
        "seed": arguments.seed,
        "steps": arguments.steps,
        "use_bfloat16": not arguments.no_bfloat16,
        "warmup_steps": arguments.warmup_steps,
    }
    if arguments.learning_rate is not None:
        overrides["learning_rate"] = arguments.learning_rate
    if arguments.min_learning_rate is not None:
        overrides["min_learning_rate"] = arguments.min_learning_rate
    config = frozen_student_stage_config(
        stage=arguments.stage,
        data_dir=arguments.data_dir,
        output_dir=arguments.output_dir,
        parent_checkpoint=arguments.parent_checkpoint,
        parent_checkpoint_sha256=arguments.parent_checkpoint_sha256,
        source_commit=arguments.source_commit,
        **overrides,
    )
    result = run_stage_training(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

