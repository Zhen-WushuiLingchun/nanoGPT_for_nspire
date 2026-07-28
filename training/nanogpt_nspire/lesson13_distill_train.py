"""Fair Lesson 13 sequence, local-logit, and combined student routes."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict
import json
from pathlib import Path
import re
import time

import torch

from nanogpt_nspire.distillation import DistillationObjective
from nanogpt_nspire.local_teacher_train import (
    LOCAL_TEACHER_ARCHITECTURE,
    LOCAL_TEACHER_SFT_ROUTE,
)
from nanogpt_nspire.models.direct_small_gpt import (
    DirectSmallConfig,
    DirectSmallGPT,
)
from nanogpt_nspire.stage_train import (
    StageTrainingConfig,
    run_stage_training,
)
from nanogpt_nspire.base_train import (
    load_packed_dataset,
    make_packed_batch,
)
from nanogpt_nspire.byte_tokenizer import VOCAB_SIZE
from nanogpt_nspire.training_support import (
    resolve_device,
    sha256_file,
    synchronize,
    write_json_atomic,
)


SEQUENCE_ROUTE = "Verified-Sequence-SFT"
LOCAL_LOGIT_ROUTE = "Local-Logit-Distilled-SFT"
COMBINED_ROUTE = "Combined-Sequence-Logit-SFT"
ROUTES = {
    "sequence": SEQUENCE_ROUTE,
    "logit": LOCAL_LOGIT_ROUTE,
    "combined": COMBINED_ROUTE,
}
CHECKPOINT_FILENAMES = {
    "sequence": "verified_sequence_sft.pt",
    "logit": "local_logit_distilled_sft.pt",
    "combined": "combined_sequence_logit_sft.pt",
}
STUDENT_ARCHITECTURE = {
    "block_size": 256,
    "n_layer": 6,
    "n_head": 6,
    "n_embd": 384,
    "mlp_ratio": 4,
    "dropout": 0.1,
    "bias": False,
    "tie_embeddings": True,
}
_ARCHITECTURE_FIELDS = frozenset(STUDENT_ARCHITECTURE)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def local_teacher_model_config() -> DirectSmallConfig:
    return DirectSmallConfig(
        vocab_size=VOCAB_SIZE,
        **LOCAL_TEACHER_ARCHITECTURE,
    )


def frozen_lesson13_student_config(
    *,
    kind: str,
    data_dir: str | Path,
    output_dir: str | Path,
    parent_checkpoint: str | Path,
    parent_checkpoint_sha256: str,
    source_commit: str,
    **overrides: object,
) -> StageTrainingConfig:
    """Freeze every student field except data and supervision route."""

    if kind not in ROUTES:
        raise ValueError("kind must be 'sequence', 'logit', or 'combined'")
    changed = sorted(_ARCHITECTURE_FIELDS & frozenset(overrides))
    if changed:
        raise ValueError(
            "Lesson 13 student architecture is frozen; remove overrides: "
            + ", ".join(changed)
        )
    defaults: dict[str, object] = {
        **STUDENT_ARCHITECTURE,
        "device": "auto",
        "steps": 1000,
        "micro_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 0.0001,
        "min_learning_rate": 0.00001,
        "warmup_steps": 50,
        "eval_interval": 100,
        "eval_batches": 20,
        "log_interval": 25,
        "overfit_gate_steps": 20,
        "route_override": ROUTES[kind],
        "checkpoint_filename_override": CHECKPOINT_FILENAMES[kind],
    }
    defaults.update(overrides)
    config = StageTrainingConfig(
        stage="sft",
        data_dir=Path(data_dir),
        output_dir=Path(output_dir),
        parent_checkpoint=Path(parent_checkpoint),
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        expected_parent_route="Math-Physics-CPT",
        source_commit=source_commit,
        **defaults,
    )
    config.validate()
    return config


def load_local_teacher_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_model_config: DirectSmallConfig | None = None,
) -> tuple[DirectSmallGPT, dict[str, object]]:
    """Strictly reload a shared-tokenizer Local-Teacher-SFT checkpoint."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if (
        not isinstance(expected_sha256, str)
        or _SHA256_PATTERN.fullmatch(expected_sha256) is None
    ):
        raise ValueError("expected SHA-256 must be lowercase hexadecimal")
    actual_sha256 = sha256_file(checkpoint_path)
    if actual_sha256 != expected_sha256:
        raise ValueError("local teacher checkpoint SHA-256 mismatch")
    try:
        raw = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise ValueError(
            "local teacher checkpoint could not be loaded safely"
        ) from error
    if not isinstance(raw, Mapping):
        raise ValueError("local teacher checkpoint must be a mapping")
    if raw.get("schema_version") != 1:
        raise ValueError("local teacher checkpoint schema version mismatch")
    if raw.get("route") != LOCAL_TEACHER_SFT_ROUTE:
        raise ValueError(
            f"local teacher route must be {LOCAL_TEACHER_SFT_ROUTE}"
        )
    tokenizer = raw.get("tokenizer")
    if (
        not isinstance(tokenizer, Mapping)
        or tokenizer.get("kind") != "byte_plus_fixed_special_tokens"
        or tokenizer.get("vocab_size") != VOCAB_SIZE
    ):
        raise ValueError("local teacher tokenizer contract mismatch")
    model_config = expected_model_config or local_teacher_model_config()
    if raw.get("model_config") != asdict(model_config):
        raise ValueError("local teacher model configuration mismatch")
    state = raw.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("local teacher model state is missing")
    model = DirectSmallGPT(model_config)
    reference = model.state_dict()
    if set(state) != set(reference):
        raise ValueError("local teacher tensor keys mismatch")
    checked: dict[str, torch.Tensor] = {}
    for name, expected in reference.items():
        value = state[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != expected.shape
            or value.dtype != expected.dtype
        ):
            raise ValueError(
                f"local teacher tensor {name} shape or dtype mismatch"
            )
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(
                f"local teacher tensor {name} contains non-finite values"
            )
        checked[name] = value.detach().cpu()
    model.load_state_dict(checked, strict=True)
    if model.token_embedding.weight is not model.lm_head.weight:
        raise ValueError("local teacher tied embedding identity was lost")
    provenance = {
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": actual_sha256,
        "route": LOCAL_TEACHER_SFT_ROUTE,
        "selected_validation_loss": raw.get("selected_validation_loss"),
        "source_commit": raw.get("source_commit"),
    }
    return model, provenance


def _benchmark_teacher(
    teacher: DirectSmallGPT,
    config: StageTrainingConfig,
    *,
    device: torch.device,
    batches: int = 20,
) -> dict[str, object]:
    dataset = load_packed_dataset(config.data_dir)
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 91)
    inputs = [
        make_packed_batch(
            dataset.validation,
            batch_size=config.micro_batch_size,
            block_size=config.block_size,
            generator=generator,
            device=device,
        ).inputs
        for _ in range(batches)
    ]
    teacher.eval()
    with torch.inference_mode():
        teacher(inputs[0])
        synchronize(device)
        started = time.perf_counter()
        for tensor in inputs:
            teacher(tensor)
        synchronize(device)
    elapsed = time.perf_counter() - started
    tokens = batches * config.micro_batch_size * config.block_size
    return {
        "batch_size": config.micro_batch_size,
        "batches": batches,
        "seconds": elapsed,
        "tokens": tokens,
        "tokens_per_second": tokens / elapsed,
    }


def run_lesson13_student(
    *,
    kind: str,
    config: StageTrainingConfig,
    teacher_checkpoint: str | Path | None = None,
    teacher_checkpoint_sha256: str | None = None,
    temperature: float = 2.0,
    alpha: float = 0.5,
) -> dict[str, object]:
    """Run one frozen student route and attach teacher cost when relevant."""

    if kind not in ROUTES or config.route != ROUTES[kind]:
        raise ValueError("kind and student route disagree")
    if kind == "sequence":
        if teacher_checkpoint is not None or teacher_checkpoint_sha256 is not None:
            raise ValueError(
                "sequence route must not declare a local teacher"
            )
        return run_stage_training(config)
    if teacher_checkpoint is None or teacher_checkpoint_sha256 is None:
        raise ValueError("logit routes require a local teacher checkpoint")
    teacher, provenance = load_local_teacher_checkpoint(
        teacher_checkpoint,
        expected_sha256=teacher_checkpoint_sha256,
    )
    device = resolve_device(config.device)
    teacher = teacher.to(device)
    objective = DistillationObjective(
        teacher,
        temperature=temperature,
        alpha=alpha,
        teacher_provenance=provenance,
    )
    benchmark = _benchmark_teacher(teacher, config, device=device)
    result = run_stage_training(
        config,
        training_objective=objective,
    )
    result["teacher_forward_benchmark"] = benchmark
    write_json_atomic(config.output_dir / "run.json", result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=tuple(ROUTES), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-checkpoint-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--teacher-checkpoint-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    config = frozen_lesson13_student_config(
        kind=arguments.kind,
        data_dir=arguments.data_dir,
        output_dir=arguments.output_dir,
        parent_checkpoint=arguments.parent_checkpoint,
        parent_checkpoint_sha256=arguments.parent_checkpoint_sha256,
        source_commit=arguments.source_commit,
        device=arguments.device,
    )
    result = run_lesson13_student(
        kind=arguments.kind,
        config=config,
        teacher_checkpoint=arguments.teacher_checkpoint,
        teacher_checkpoint_sha256=(
            arguments.teacher_checkpoint_sha256
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
