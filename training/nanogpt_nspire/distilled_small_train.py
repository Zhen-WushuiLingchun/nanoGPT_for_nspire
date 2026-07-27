"""Train Direct-Small with hard labels and a passing Teacher-v2."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from nanogpt_nspire.data import DatasetError
from nanogpt_nspire.direct_small_train import (
    TrainingConfig,
    TrainingRunIdentity,
    run_training,
)
from nanogpt_nspire.distillation import DistillationObjective
from nanogpt_nspire.models.direct_small_gpt import DirectSmallGPT
from nanogpt_nspire.quantize_teacher import (
    validate_teacher_source_metadata,
)
from nanogpt_nspire.training_dataset import (
    load_token_dataset,
    make_batch,
)
from nanogpt_nspire.training_support import (
    dataset_summary,
    resolve_device,
    sha256_file,
    synchronize,
    write_json_atomic,
)


DIRECT_SMALL_SELECTED_VALIDATION_LOSS = 1.4997899746894836
DIRECT_SMALL_TRAINING_TOKENS = 40_960_000
DISTILLATION_TEMPERATURE = 2.0
DISTILLATION_ALPHA = 0.5
DISTILLED_SMALL_RUN_IDENTITY = TrainingRunIdentity(
    route="Distilled-Small",
    checkpoint_filename="distilled_small_gpt.pt",
    deployment_interpretation=(
        "fp32_student_trained_with_teacher_v2_soft_targets"
    ),
    quality_gate_maximum_selected_validation_loss=(
        DIRECT_SMALL_SELECTED_VALIDATION_LOSS
    ),
)
DISTILLED_SMALL_EXTENDED_RUN_IDENTITY = TrainingRunIdentity(
    route="Distilled-Small-Extended",
    checkpoint_filename="distilled_small_extended_gpt.pt",
    deployment_interpretation=(
        "fp32_student_extended_training_ablation_not_same_token_budget"
    ),
    quality_gate_maximum_selected_validation_loss=(
        DIRECT_SMALL_SELECTED_VALIDATION_LOSS
    ),
)


def frozen_distilled_student_config(
    *,
    data_dir: Path,
    output_dir: Path,
    source_commit: str,
    device: str = "auto",
) -> TrainingConfig:
    """Return the exact Direct-Small v1 architecture and training protocol."""

    return TrainingConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        device=device,
        source_commit=source_commit,
    )


def frozen_extended_distilled_student_config(
    *,
    data_dir: Path,
    output_dir: Path,
    source_commit: str,
    device: str = "auto",
) -> TrainingConfig:
    """Reproduce the base trajectory, then continue 5,000 minimum-LR steps."""

    return TrainingConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        device=device,
        source_commit=source_commit,
        steps=10_000,
        learning_rate_decay_steps=5_000,
    )


def run_distilled_training(
    config: TrainingConfig,
    *,
    teacher: DirectSmallGPT,
    teacher_provenance: Mapping[str, object],
    temperature: float = DISTILLATION_TEMPERATURE,
    alpha: float = DISTILLATION_ALPHA,
    run_identity: TrainingRunIdentity = DISTILLED_SMALL_RUN_IDENTITY,
) -> dict[str, object]:
    """Run the shared student engine with a frozen distillation objective."""

    if not isinstance(teacher, DirectSmallGPT):
        raise ValueError("teacher must be a DirectSmallGPT")
    device = resolve_device(config.device)
    teacher = teacher.to(device)
    objective = DistillationObjective(
        teacher,
        temperature=temperature,
        alpha=alpha,
        teacher_provenance=teacher_provenance,
    )
    summary = run_training(
        config,
        run_identity=run_identity,
        training_objective=objective,
    )
    selected_loss = float(summary["metrics"]["selected_validation_loss"])
    student_training_tokens = (
        config.steps * config.batch_size * config.block_size
    )
    same_training_tokens = (
        student_training_tokens == DIRECT_SMALL_TRAINING_TOKENS
    )
    summary["comparison_to_direct_small"] = {
        "comparison_scope": (
            "base_same_student_token_budget"
            if same_training_tokens
            else "extended_training_ablation_not_same_token_budget"
        ),
        "direct_small_selected_validation_loss": (
            DIRECT_SMALL_SELECTED_VALIDATION_LOSS
        ),
        "loss_absolute_improvement": (
            DIRECT_SMALL_SELECTED_VALIDATION_LOSS - selected_loss
        ),
        "loss_relative_improvement_percent": (
            100.0
            * (DIRECT_SMALL_SELECTED_VALIDATION_LOSS - selected_loss)
            / DIRECT_SMALL_SELECTED_VALIDATION_LOSS
        ),
        "same_student_architecture": True,
        "same_student_training_tokens": same_training_tokens,
        "status": (
            "distillation_improved_validation_loss"
            if selected_loss < DIRECT_SMALL_SELECTED_VALIDATION_LOSS
            else "distillation_did_not_improve_validation_loss"
        ),
        "student_training_token_ratio": (
            student_training_tokens / DIRECT_SMALL_TRAINING_TOKENS
        ),
    }
    write_json_atomic(config.output_dir / "run.json", summary)
    return summary


@dataclass(frozen=True)
class DistilledExperimentConfig:
    """Paths and fixed settings for the official Distilled-Small run."""

    data_dir: Path
    teacher_checkpoint: Path
    output_dir: Path
    device: str = "auto"
    source_commit: str = "uncommitted"
    temperature: float = DISTILLATION_TEMPERATURE
    alpha: float = DISTILLATION_ALPHA
    teacher_benchmark_batches: int = 50
    student_profile: str = "base"

    def validate(self) -> None:
        if not self.source_commit:
            raise ValueError("source_commit must not be empty")
        if (
            not math.isfinite(self.temperature)
            or self.temperature <= 0
        ):
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(self.alpha) or not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be finite and in [0, 1]")
        if (
            isinstance(self.teacher_benchmark_batches, bool)
            or not isinstance(self.teacher_benchmark_batches, int)
            or self.teacher_benchmark_batches <= 0
        ):
            raise ValueError(
                "teacher_benchmark_batches must be a positive integer"
            )
        if self.student_profile not in {"base", "extended"}:
            raise ValueError("student_profile must be 'base' or 'extended'")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _benchmark_teacher(
    teacher: DirectSmallGPT,
    validation_tokens: torch.Tensor,
    *,
    batch_size: int,
    block_size: int,
    batches: int,
    device: torch.device,
) -> dict[str, float | int]:
    generator = torch.Generator(device="cpu").manual_seed(1341)
    inputs = [
        make_batch(
            validation_tokens,
            batch_size=batch_size,
            block_size=block_size,
            generator=generator,
            device=device,
        )[0]
        for _ in range(batches)
    ]
    teacher.eval()
    with torch.inference_mode():
        teacher(inputs[0])
        synchronize(device)
        started = time.perf_counter()
        for batch in inputs:
            teacher(batch)
        synchronize(device)
    elapsed = time.perf_counter() - started
    tokens = batches * batch_size * block_size
    return {
        "batch_size": batch_size,
        "batches": batches,
        "block_size": block_size,
        "seconds": elapsed,
        "tokens": tokens,
        "tokens_per_second": tokens / elapsed,
    }


def run_distilled_experiment(
    experiment: DistilledExperimentConfig,
) -> dict[str, object]:
    """Validate Teacher-v2 provenance, train, benchmark and summarize."""

    experiment.validate()
    teacher_run_path = experiment.teacher_checkpoint.with_name("run.json")
    if not experiment.teacher_checkpoint.is_file():
        raise FileNotFoundError(experiment.teacher_checkpoint)
    if not teacher_run_path.is_file():
        raise FileNotFoundError(teacher_run_path)
    with teacher_run_path.open("r", encoding="utf-8") as stream:
        teacher_run = json.load(stream)
    teacher_sha256 = sha256_file(experiment.teacher_checkpoint)
    checkpoint = torch.load(
        experiment.teacher_checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    teacher_config = validate_teacher_source_metadata(
        checkpoint,
        teacher_run,
        checkpoint_sha256=teacher_sha256,
    )
    if checkpoint.get("route") != "Teacher-v2":
        raise ValueError("Distilled-Small requires a passing Teacher-v2")
    if teacher_config.dropout != 0.3:
        raise ValueError("Teacher-v2 dropout must be 0.3")

    dataset = load_token_dataset(experiment.data_dir)
    current_dataset = dataset_summary(dataset)
    if current_dataset != dict(
        _mapping(teacher_run.get("dataset"), "teacher dataset")
    ):
        raise DatasetError("Teacher-v2 dataset hashes or counts do not match")
    if list(dataset.vocabulary) != checkpoint.get("vocabulary"):
        raise DatasetError("Teacher-v2 vocabulary does not match dataset")

    teacher = DirectSmallGPT(teacher_config)
    teacher.load_state_dict(
        _mapping(
            checkpoint.get("model_state_dict"),
            "Teacher-v2 model_state_dict",
        ),
        strict=True,
    )
    if teacher.token_embedding.weight is not teacher.lm_head.weight:
        raise RuntimeError("Teacher-v2 tied embedding identity was not preserved")
    teacher_provenance = {
        "checkpoint_bytes": experiment.teacher_checkpoint.stat().st_size,
        "checkpoint_sha256": teacher_sha256,
        "route": checkpoint["route"],
        "selected_validation_loss": checkpoint[
            "selected_validation_loss"
        ],
        "source_commit": checkpoint["source_commit"],
    }
    if experiment.student_profile == "base":
        student_config = frozen_distilled_student_config(
            data_dir=experiment.data_dir,
            output_dir=experiment.output_dir,
            device=experiment.device,
            source_commit=experiment.source_commit,
        )
        run_identity = DISTILLED_SMALL_RUN_IDENTITY
    else:
        student_config = frozen_extended_distilled_student_config(
            data_dir=experiment.data_dir,
            output_dir=experiment.output_dir,
            device=experiment.device,
            source_commit=experiment.source_commit,
        )
        run_identity = DISTILLED_SMALL_EXTENDED_RUN_IDENTITY
    summary = run_distilled_training(
        student_config,
        teacher=teacher,
        teacher_provenance=teacher_provenance,
        temperature=experiment.temperature,
        alpha=experiment.alpha,
        run_identity=run_identity,
    )
    device = resolve_device(experiment.device)
    teacher_benchmark = _benchmark_teacher(
        teacher.to(device),
        dataset.validation,
        batch_size=student_config.batch_size,
        block_size=student_config.block_size,
        batches=experiment.teacher_benchmark_batches,
        device=device,
    )
    summary["distillation"] = {
        "alpha": experiment.alpha,
        "student_training_tokens": (
            student_config.steps
            * student_config.batch_size
            * student_config.block_size
        ),
        "student_profile": experiment.student_profile,
        "teacher_forward_benchmark": teacher_benchmark,
        "teacher_provenance": teacher_provenance,
        "temperature": experiment.temperature,
    }
    summary["experiment_configuration"] = {
        **asdict(experiment),
        "data_dir": str(experiment.data_dir),
        "teacher_checkpoint": str(experiment.teacher_checkpoint),
        "output_dir": str(experiment.output_dir),
    }
    write_json_atomic(experiment.output_dir / "run.json", summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the exact Direct-Small student with hard labels and "
            "temperature-scaled soft targets from a passing Teacher-v2."
        )
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--extended-training",
        action="store_true",
        help=(
            "run the separately labelled 10,000-step ablation: reproduce "
            "the base 5,000-step schedule, then continue at minimum LR"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    experiment = DistilledExperimentConfig(
        data_dir=arguments.data_dir,
        teacher_checkpoint=arguments.teacher_checkpoint,
        output_dir=arguments.output_dir,
        device=arguments.device,
        source_commit=arguments.source_commit,
        student_profile=(
            "extended" if arguments.extended_training else "base"
        ),
    )
    try:
        summary = run_distilled_experiment(experiment)
    except (
        DatasetError,
        FileNotFoundError,
        FloatingPointError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        raise SystemExit(f"error: {error}") from error
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
