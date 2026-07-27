"""Convert a quality-gated Teacher checkpoint to packed groupwise INT4."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from nanogpt_nspire.data import DatasetError, decode_tokens
from nanogpt_nspire.lesson03_train import sample_with_context
from nanogpt_nspire.models.direct_small_gpt import (
    DirectSmallConfig,
    DirectSmallGPT,
)
from nanogpt_nspire.quantization.model_state import (
    dequantize_model_state,
    quantize_model_state,
    reconstruct_dequantized_reference,
)
from nanogpt_nspire.teacher_train import (
    TEACHER_QUALITY_GATE_MAXIMUM_LOSS,
)
from nanogpt_nspire.training_dataset import load_token_dataset
from nanogpt_nspire.training_support import (
    bits_per_character,
    dataset_summary,
    environment_summary,
    evaluate_loss,
    resolve_device,
    sha256_file,
    write_json_atomic,
)


INT4_MAXIMUM_LOSS_DEGRADATION = 0.05
DEPLOYMENT_FILE_LIMIT_BYTES = 6 * 1024 * 1024
DEPLOYMENT_METADATA_RESERVE_BYTES = 64 * 1024
TEACHER_TRAINING_TOKENS = 81_920_000
FROZEN_TEACHER_ARCHITECTURE = {
    "block_size": 128,
    "n_layer": 6,
    "n_head": 6,
    "n_embd": 384,
    "mlp_ratio": 4,
    "bias": False,
    "tie_embeddings": True,
}
FROZEN_TEACHER_ROUTE_DROPOUT = {
    "Teacher": 0.2,
    "Teacher-v2": 0.3,
}


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class QuantizeTeacherConfig:
    """Frozen evaluation and deployment gates for the INT4 experiment."""

    data_dir: Path
    teacher_checkpoint: Path
    output_dir: Path
    device: str = "auto"
    group_size: int = 64
    batch_size: int = 64
    eval_batches: int = 50
    validation_seed: int = 1338
    sample_seed: int = 1340
    sample_tokens: int = 300
    temperature: float = 0.8
    maximum_loss_degradation: float = INT4_MAXIMUM_LOSS_DEGRADATION
    file_limit_bytes: int = DEPLOYMENT_FILE_LIMIT_BYTES
    metadata_reserve_bytes: int = DEPLOYMENT_METADATA_RESERVE_BYTES
    source_commit: str = "uncommitted"
    diagnostic_allow_failed_teacher: bool = False

    def validate(self) -> None:
        for name in (
            "group_size",
            "batch_size",
            "eval_batches",
            "file_limit_bytes",
        ):
            _positive_integer(getattr(self, name), name)
        for name in ("validation_seed", "sample_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            isinstance(self.sample_tokens, bool)
            or not isinstance(self.sample_tokens, int)
            or self.sample_tokens < 0
        ):
            raise ValueError("sample_tokens must be a non-negative integer")
        if (
            isinstance(self.metadata_reserve_bytes, bool)
            or not isinstance(self.metadata_reserve_bytes, int)
            or self.metadata_reserve_bytes < 0
        ):
            raise ValueError(
                "metadata_reserve_bytes must be a non-negative integer"
            )
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if (
            not math.isfinite(self.maximum_loss_degradation)
            or self.maximum_loss_degradation < 0
        ):
            raise ValueError(
                "maximum_loss_degradation must be finite and non-negative"
            )
        if not self.source_commit:
            raise ValueError("source_commit must not be empty")
        if not isinstance(self.diagnostic_allow_failed_teacher, bool):
            raise ValueError(
                "diagnostic_allow_failed_teacher must be boolean"
            )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def experiment_id_for_teacher_route(teacher_route: str) -> str:
    """Keep the reusable quantizer's evidence tied to its source lesson."""

    if teacher_route == "Teacher":
        return "lesson06-int4-diagnostic"
    if teacher_route == "Teacher-v2":
        return "lesson07-int4-teacher-v2"
    raise ValueError(f"unsupported teacher route {teacher_route!r}")


def _finite_loss(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


def validate_teacher_source_metadata(
    checkpoint: Mapping[str, Any],
    teacher_run: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    require_quality_gate_passed: bool = True,
) -> DirectSmallConfig:
    """Reject any source that is not the preregistered passing Teacher."""

    checkpoint = _mapping(checkpoint, "teacher checkpoint")
    teacher_run = _mapping(teacher_run, "teacher run")
    if checkpoint.get("schema_version") != 1:
        raise ValueError("teacher checkpoint schema_version must be 1")
    if checkpoint.get("model_type") != "direct_small_gpt":
        raise ValueError("teacher checkpoint model type is not direct_small_gpt")
    source_route = checkpoint.get("route")
    if (
        not isinstance(source_route, str)
        or source_route not in FROZEN_TEACHER_ROUTE_DROPOUT
        or teacher_run.get("route") != source_route
    ):
        raise ValueError(
            "teacher source route must be a matching Teacher or Teacher-v2"
        )
    if not isinstance(require_quality_gate_passed, bool):
        raise ValueError("require_quality_gate_passed must be boolean")
    checkpoint_gate = checkpoint.get("quality_gate_passed")
    run_gate = _mapping(
        teacher_run.get("metrics"),
        "teacher metrics",
    ).get("quality_gate_passed")
    if (
        not isinstance(checkpoint_gate, bool)
        or not isinstance(run_gate, bool)
        or checkpoint_gate != run_gate
    ):
        raise ValueError("teacher quality gate metadata disagrees")
    if require_quality_gate_passed and not checkpoint_gate:
        raise ValueError("teacher quality gate must have passed")

    checkpoint_threshold = _finite_loss(
        checkpoint.get("quality_gate_maximum_selected_validation_loss"),
        "teacher quality threshold",
    )
    run_metrics = _mapping(teacher_run.get("metrics"), "teacher metrics")
    run_threshold = _finite_loss(
        run_metrics.get("quality_gate_maximum_selected_validation_loss"),
        "teacher run quality threshold",
    )
    if (
        checkpoint_threshold != TEACHER_QUALITY_GATE_MAXIMUM_LOSS
        or run_threshold != TEACHER_QUALITY_GATE_MAXIMUM_LOSS
    ):
        raise ValueError("teacher quality gate threshold was not preregistered")

    selected_loss = _finite_loss(
        checkpoint.get("selected_validation_loss"),
        "teacher selected validation loss",
    )
    run_selected_loss = _finite_loss(
        run_metrics.get("selected_validation_loss"),
        "teacher run selected validation loss",
    )
    if (
        require_quality_gate_passed
        and selected_loss > TEACHER_QUALITY_GATE_MAXIMUM_LOSS
    ):
        raise ValueError("teacher selected validation loss exceeds its gate")
    if selected_loss != run_selected_loss:
        raise ValueError(
            "teacher checkpoint and run selected validation loss disagree"
        )
    if run_metrics.get("training_tokens") != TEACHER_TRAINING_TOKENS:
        raise ValueError("teacher training token budget is not the frozen value")

    raw_model_config = _mapping(
        checkpoint.get("model_config"),
        "teacher model_config",
    )
    try:
        model_config = DirectSmallConfig(**dict(raw_model_config))
    except (TypeError, ValueError) as error:
        raise ValueError("teacher model_config is invalid") from error
    model_config.validate()
    for field, expected in FROZEN_TEACHER_ARCHITECTURE.items():
        if getattr(model_config, field) != expected:
            raise ValueError(
                f"teacher {field} does not match the frozen teacher architecture"
            )
    expected_dropout = FROZEN_TEACHER_ROUTE_DROPOUT[source_route]
    if model_config.dropout != expected_dropout:
        raise ValueError(
            "teacher dropout does not match its frozen source route"
        )

    vocabulary = checkpoint.get("vocabulary")
    if (
        not isinstance(vocabulary, list)
        or len(vocabulary) != model_config.vocab_size
        or any(not isinstance(token, str) or len(token) != 1 for token in vocabulary)
        or len(set(vocabulary)) != len(vocabulary)
    ):
        raise ValueError("teacher vocabulary is invalid")

    source_commit = checkpoint.get("source_commit")
    if (
        not isinstance(source_commit, str)
        or not source_commit
        or teacher_run.get("source_commit") != source_commit
    ):
        raise ValueError("teacher source commit provenance disagrees")
    if teacher_run.get("schema_version") != 1:
        raise ValueError("teacher run schema_version must be 1")

    run_model = _mapping(teacher_run.get("model"), "teacher run model")
    for field, value in asdict(model_config).items():
        if run_model.get(field) != value:
            raise ValueError(
                f"teacher run model field {field} disagrees with checkpoint"
            )
    if (
        run_model.get("type") != "direct_small_gpt"
        or run_model.get("parameters") != 10_695_936
        or run_model.get("raw_fp32_parameter_bytes") != 42_783_744
    ):
        raise ValueError("teacher run model size is not the frozen value")

    teacher_artifacts = _mapping(
        teacher_run.get("artifacts"),
        "teacher artifacts",
    )
    checkpoint_artifact = _mapping(
        teacher_artifacts.get("checkpoint"),
        "teacher checkpoint artifact",
    )
    if checkpoint_artifact.get("sha256") != checkpoint_sha256:
        raise ValueError("teacher checkpoint SHA-256 disagrees with run.json")
    return model_config


def _tensor_error_summary(
    original_state: Mapping[str, torch.Tensor],
    dequantized_state: Mapping[str, torch.Tensor],
    canonical_names: Sequence[str],
) -> tuple[dict[str, dict[str, float]], float, float]:
    per_tensor: dict[str, dict[str, float]] = {}
    squared_error_sum = 0.0
    value_count = 0
    maximum_error = 0.0
    for name in canonical_names:
        original = original_state[name].detach().cpu().to(torch.float32)
        reconstructed = dequantized_state[name].detach().cpu()
        difference = (original - reconstructed).to(torch.float64)
        tensor_maximum = float(difference.abs().max().item())
        tensor_rmse = float(torch.sqrt(torch.mean(difference.square())).item())
        per_tensor[name] = {
            "max_absolute_error": tensor_maximum,
            "rmse": tensor_rmse,
        }
        maximum_error = max(maximum_error, tensor_maximum)
        squared_error_sum += float(difference.square().sum().item())
        value_count += difference.numel()
    overall_rmse = math.sqrt(squared_error_sum / value_count)
    return per_tensor, maximum_error, overall_rmse


def run_teacher_quantization(
    config: QuantizeTeacherConfig,
) -> dict[str, object]:
    """Validate, quantize, evaluate, save and summarize the Teacher."""

    config.validate()
    device = resolve_device(config.device)
    teacher_run_path = config.teacher_checkpoint.with_name("run.json")
    if not config.teacher_checkpoint.is_file():
        raise FileNotFoundError(config.teacher_checkpoint)
    if not teacher_run_path.is_file():
        raise FileNotFoundError(teacher_run_path)
    with teacher_run_path.open("r", encoding="utf-8") as stream:
        teacher_run = json.load(stream)
    checkpoint_sha256 = sha256_file(config.teacher_checkpoint)
    checkpoint = torch.load(
        config.teacher_checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    model_config = validate_teacher_source_metadata(
        checkpoint,
        teacher_run,
        checkpoint_sha256=checkpoint_sha256,
        require_quality_gate_passed=(
            not config.diagnostic_allow_failed_teacher
        ),
    )
    teacher_quality_gate_passed = bool(checkpoint["quality_gate_passed"])
    original_state = _mapping(
        checkpoint.get("model_state_dict"),
        "teacher model_state_dict",
    )
    teacher = DirectSmallGPT(model_config)
    teacher.load_state_dict(original_state, strict=True)
    if teacher.token_embedding.weight is not teacher.lm_head.weight:
        raise RuntimeError("teacher tied embedding identity was not preserved")

    dataset = load_token_dataset(config.data_dir)
    current_dataset = dataset_summary(dataset)
    recorded_dataset = _mapping(
        teacher_run.get("dataset"),
        "teacher dataset",
    )
    if current_dataset != dict(recorded_dataset):
        raise DatasetError("teacher dataset hashes or counts do not match")
    if list(dataset.vocabulary) != checkpoint["vocabulary"]:
        raise DatasetError("teacher checkpoint vocabulary does not match dataset")
    if dataset.validation.numel() < model_config.block_size + 1:
        raise DatasetError("validation split is too short for teacher block_size")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    teacher = teacher.to(device).eval()
    fp32_loss = evaluate_loss(
        teacher,
        dataset.validation,
        batch_size=config.batch_size,
        block_size=model_config.block_size,
        batches=config.eval_batches,
        seed=config.validation_seed,
        device=device,
    )
    recorded_loss = float(checkpoint["selected_validation_loss"])
    if not math.isclose(fp32_loss, recorded_loss, rel_tol=0.0, abs_tol=1e-7):
        raise RuntimeError(
            "fixed-window FP32 teacher loss does not reproduce checkpoint: "
            f"{fp32_loss} != {recorded_loss}"
        )
    teacher = teacher.cpu()

    quantized_state = quantize_model_state(
        teacher,
        group_size=config.group_size,
    )
    dequantized_state = dequantize_model_state(quantized_state)
    canonical_names = list(quantized_state["tensors"])
    tensor_errors, weight_max_error, weight_rmse = _tensor_error_summary(
        original_state,
        dequantized_state,
        canonical_names,
    )
    quantized_reference = reconstruct_dequantized_reference(
        model_config,
        quantized_state,
    ).to(device).eval()
    int4_loss = evaluate_loss(
        quantized_reference,
        dataset.validation,
        batch_size=config.batch_size,
        block_size=model_config.block_size,
        batches=config.eval_batches,
        seed=config.validation_seed,
        device=device,
    )
    loss_degradation = int4_loss - fp32_loss

    probe = (
        dataset.validation[: model_config.block_size]
        .to(device)
        .unsqueeze(0)
    )
    teacher = teacher.to(device).eval()
    with torch.inference_mode():
        teacher_logits, _ = teacher(probe)
        int4_logits, _ = quantized_reference(probe)
    logits_difference = (
        teacher_logits.to(torch.float64) - int4_logits.to(torch.float64)
    )
    logits_max_error = float(logits_difference.abs().max().item())
    logits_rmse = float(
        torch.sqrt(torch.mean(logits_difference.square())).item()
    )
    generated_tokens = sample_with_context(
        quantized_reference,
        [0],
        new_tokens=config.sample_tokens,
        seed=config.sample_seed,
        temperature=config.temperature,
        device=device,
    )
    sample_text = decode_tokens(generated_tokens, dataset.vocabulary)

    storage = dict(quantized_state["storage"])
    logical_with_reserve = (
        storage["logical_payload_bytes"] + config.metadata_reserve_bytes
    )
    size_gate_passed = logical_with_reserve <= config.file_limit_bytes
    quality_gate_passed = (
        loss_degradation <= config.maximum_loss_degradation
    )
    candidate_gate_passed = (
        teacher_quality_gate_passed
        and size_gate_passed
        and quality_gate_passed
    )
    route = (
        "Quantized-Small"
        if teacher_quality_gate_passed
        else "Quantized-Small-Diagnostic"
    )

    artifact = {
        "model_config": asdict(model_config),
        "model_type": "direct_small_gpt_int4",
        "provenance": {
            "dataset": current_dataset,
            "teacher_route": checkpoint["route"],
            "teacher_checkpoint_bytes": (
                config.teacher_checkpoint.stat().st_size
            ),
            "teacher_checkpoint_sha256": checkpoint_sha256,
            "teacher_selected_validation_loss": recorded_loss,
            "teacher_source_commit": checkpoint["source_commit"],
        },
        "quantized_model_state": quantized_state,
        "route": route,
        "runtime_status": {
            "integer_C_runtime": "pending Lesson 08",
            "nspire_measurement": "pending Lesson 09",
            "reference": "dequantized PyTorch reference",
            "state": "packed_weight_and_dequantized_reference_complete",
        },
        "schema_version": 1,
        "source_commit": config.source_commit,
        "vocabulary": list(dataset.vocabulary),
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = config.output_dir / "teacher_int4.pt"
    torch.save(artifact, artifact_path)
    artifact_bytes = artifact_path.stat().st_size
    artifact_sha256 = sha256_file(artifact_path)
    peak_cuda_bytes = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )

    summary: dict[str, object] = {
        "artifacts": {
            "int4_checkpoint": {
                "bytes": artifact_bytes,
                "path": str(artifact_path),
                "sha256": artifact_sha256,
            },
            "teacher_checkpoint": {
                "bytes": config.teacher_checkpoint.stat().st_size,
                "path": str(config.teacher_checkpoint),
                "sha256": checkpoint_sha256,
            },
        },
        "configuration": {
            **asdict(config),
            "data_dir": str(config.data_dir),
            "teacher_checkpoint": str(config.teacher_checkpoint),
            "output_dir": str(config.output_dir),
        },
        "dataset": current_dataset,
        "deployment": {
            "candidate_gate_passed": candidate_gate_passed,
            "file_limit_bytes": config.file_limit_bytes,
            "integer_C_runtime": {
                "reason": "W4A8/int32 kernel is implemented in Lesson 08",
                "status": "pending",
            },
            "logical_payload_plus_metadata_reserve_bytes": (
                logical_with_reserve
            ),
            "metadata_reserve_bytes": config.metadata_reserve_bytes,
            "nspire_measurement": {
                "reason": "CX II measurement is performed in Lesson 09",
                "status": "pending",
            },
            "packed_checkpoint_interpretation": (
                "weights are packed, but this lesson evaluates a "
                "dequantized PyTorch reference"
            ),
            "size_gate_passed": size_gate_passed,
            "storage": storage,
            "teacher_quality_gate_passed": teacher_quality_gate_passed,
        },
        "environment": environment_summary(
            device,
            peak_cuda_memory_allocated_bytes=peak_cuda_bytes,
        ),
        "experiment_id": experiment_id_for_teacher_route(
            checkpoint["route"]
        ),
        "metrics": {
            "fp32_teacher_validation_bpc": bits_per_character(fp32_loss),
            "fp32_teacher_validation_loss": fp32_loss,
            "int4_validation_bpc": bits_per_character(int4_loss),
            "int4_validation_loss": int4_loss,
            "logits_max_absolute_error": logits_max_error,
            "logits_rmse": logits_rmse,
            "loss_absolute_degradation": loss_degradation,
            "loss_relative_degradation_percent": (
                100.0 * loss_degradation / fp32_loss
            ),
            "maximum_allowed_loss_degradation": (
                config.maximum_loss_degradation
            ),
            "quality_gate_passed": quality_gate_passed,
            "weight_max_absolute_error": weight_max_error,
            "weight_rmse": weight_rmse,
        },
        "model": {
            **asdict(model_config),
            "parameters": teacher.parameter_count,
            "raw_fp32_parameter_bytes": teacher.raw_fp32_parameter_bytes,
            "type": "direct_small_gpt",
        },
        "provenance": {
            "teacher_route": checkpoint["route"],
            "teacher_run_json": str(teacher_run_path),
            "teacher_source_commit": checkpoint["source_commit"],
        },
        "route": route,
        "runtime_status": artifact["runtime_status"],
        "sample": {
            "characters": len(sample_text),
            "seed": config.sample_seed,
            "temperature": config.temperature,
            "text": sample_text,
        },
        "schema_version": 1,
        "source_commit": config.source_commit,
        "tensor_errors": tensor_errors,
    }
    write_json_atomic(config.output_dir / "run.json", summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pack the preregistered Teacher as groupwise signed INT4 and "
            "evaluate a dequantized PyTorch reference; failed teachers "
            "require explicit diagnostic mode."
        )
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--diagnostic-allow-failed-teacher",
        action="store_true",
        help=(
            "measure a failed teacher without promoting it to the "
            "Quantized-Small candidate route"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    config = QuantizeTeacherConfig(
        data_dir=arguments.data_dir,
        teacher_checkpoint=arguments.teacher_checkpoint,
        output_dir=arguments.output_dir,
        device=arguments.device,
        source_commit=arguments.source_commit,
        diagnostic_allow_failed_teacher=(
            arguments.diagnostic_allow_failed_teacher
        ),
    )
    try:
        summary = run_teacher_quantization(config)
    except (
        DatasetError,
        FileNotFoundError,
        RuntimeError,
        ValueError,
    ) as error:
        raise SystemExit(f"error: {error}") from error
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
