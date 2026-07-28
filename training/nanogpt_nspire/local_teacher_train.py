"""Train a larger local teacher with the student's exact tokenizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nanogpt_nspire.base_train import (
    BaseTrainingConfig,
    run_base_training,
)
from nanogpt_nspire.stage_train import (
    StageTrainingConfig,
    run_stage_training,
)


LOCAL_TEACHER_CPT_ROUTE = "Local-Teacher-CPT"
LOCAL_TEACHER_SFT_ROUTE = "Local-Teacher-SFT"
LOCAL_TEACHER_CPT_CHECKPOINT = "local_teacher_cpt.pt"
LOCAL_TEACHER_SFT_CHECKPOINT = "local_teacher_sft.pt"
LOCAL_TEACHER_ARCHITECTURE = {
    "block_size": 256,
    "n_layer": 12,
    "n_head": 10,
    "n_embd": 640,
    "mlp_ratio": 4,
    "dropout": 0.1,
    "bias": False,
    "tie_embeddings": True,
}
_ARCHITECTURE_FIELDS = frozenset(LOCAL_TEACHER_ARCHITECTURE)


def _reject_architecture_overrides(overrides: dict[str, object]) -> None:
    changed = sorted(_ARCHITECTURE_FIELDS & frozenset(overrides))
    if changed:
        raise ValueError(
            "local teacher architecture is frozen; remove overrides: "
            + ", ".join(changed)
        )


def frozen_local_teacher_pretrain_config(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    source_commit: str,
    **overrides: object,
) -> BaseTrainingConfig:
    """Pretrain 12x640 on the exact replay-aware Lesson 12 CPT mixture."""

    _reject_architecture_overrides(overrides)
    defaults: dict[str, object] = {
        **LOCAL_TEACHER_ARCHITECTURE,
        "route": LOCAL_TEACHER_CPT_ROUTE,
        "checkpoint_filename": LOCAL_TEACHER_CPT_CHECKPOINT,
        "steps": 2000,
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate": 0.0004,
        "min_learning_rate": 0.00004,
        "warmup_steps": 100,
        "eval_interval": 200,
        "eval_batches": 20,
        "log_interval": 25,
        "overfit_gate_steps": 20,
        "sample_prompts": (
            "Force is",
            "Calculate 12 * 7",
            "The kinetic energy",
        ),
    }
    defaults.update(overrides)
    config = BaseTrainingConfig(
        data_dir=Path(data_dir),
        output_dir=Path(output_dir),
        source_commit=source_commit,
        **defaults,
    )
    config.validate()
    return config


def frozen_local_teacher_sft_config(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    parent_checkpoint: str | Path,
    parent_checkpoint_sha256: str,
    source_commit: str,
    **overrides: object,
) -> StageTrainingConfig:
    """Continue the local teacher on role-aware assistant-only SFT."""

    _reject_architecture_overrides(overrides)
    defaults: dict[str, object] = {
        **LOCAL_TEACHER_ARCHITECTURE,
        "device": "auto",
        "steps": 1000,
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate": 0.0001,
        "min_learning_rate": 0.00001,
        "warmup_steps": 50,
        "eval_interval": 100,
        "eval_batches": 20,
        "log_interval": 25,
        "overfit_gate_steps": 20,
        "route_override": LOCAL_TEACHER_SFT_ROUTE,
        "checkpoint_filename_override": LOCAL_TEACHER_SFT_CHECKPOINT,
        "required_parent_route_override": LOCAL_TEACHER_CPT_ROUTE,
    }
    defaults.update(overrides)
    config = StageTrainingConfig(
        stage="sft",
        data_dir=Path(data_dir),
        output_dir=Path(output_dir),
        parent_checkpoint=Path(parent_checkpoint),
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        expected_parent_route=LOCAL_TEACHER_CPT_ROUTE,
        source_commit=source_commit,
        **defaults,
    )
    config.validate()
    return config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("pretrain", "sft"),
        required=True,
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--parent-checkpoint", type=Path)
    parser.add_argument("--parent-checkpoint-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    overrides: dict[str, object] = {"device": arguments.device}
    if arguments.steps is not None:
        overrides["steps"] = arguments.steps
    if arguments.stage == "pretrain":
        config = frozen_local_teacher_pretrain_config(
            data_dir=arguments.data_dir,
            output_dir=arguments.output_dir,
            source_commit=arguments.source_commit,
            **overrides,
        )
        result = run_base_training(config)
    else:
        if (
            arguments.parent_checkpoint is None
            or arguments.parent_checkpoint_sha256 is None
        ):
            raise SystemExit(
                "error: SFT requires --parent-checkpoint and "
                "--parent-checkpoint-sha256"
            )
        config = frozen_local_teacher_sft_config(
            data_dir=arguments.data_dir,
            output_dir=arguments.output_dir,
            parent_checkpoint=arguments.parent_checkpoint,
            parent_checkpoint_sha256=(
                arguments.parent_checkpoint_sha256
            ),
            source_commit=arguments.source_commit,
            **overrides,
        )
        result = run_stage_training(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
