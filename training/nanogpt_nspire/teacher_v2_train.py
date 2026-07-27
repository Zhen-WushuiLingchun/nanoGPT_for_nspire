"""Train the preregistered dropout-only Teacher v2 candidate."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

from nanogpt_nspire.data import DatasetError
from nanogpt_nspire.direct_small_train import (
    TrainingConfig,
    TrainingRunIdentity,
    run_training,
)
from nanogpt_nspire.teacher_train import (
    TEACHER_QUALITY_GATE_MAXIMUM_LOSS,
    frozen_teacher_config,
)


TEACHER_V2_RUN_IDENTITY = TrainingRunIdentity(
    route="Teacher-v2",
    checkpoint_filename="teacher_v2_gpt.pt",
    deployment_interpretation=(
        "dropout_only_teacher_candidate_for_int4_and_distillation"
    ),
    quality_gate_maximum_selected_validation_loss=(
        TEACHER_QUALITY_GATE_MAXIMUM_LOSS
    ),
)


def frozen_teacher_v2_config(
    *,
    data_dir: Path,
    output_dir: Path,
    source_commit: str,
    device: str = "auto",
) -> TrainingConfig:
    """Return Teacher v1's protocol with dropout as the sole model change."""

    teacher_v1 = frozen_teacher_config(
        data_dir=data_dir,
        output_dir=output_dir,
        source_commit=source_commit,
        device=device,
    )
    return replace(teacher_v1, dropout=0.3)


def run_teacher_v2_training(
    config: TrainingConfig,
) -> dict[str, object]:
    """Run the shared trainer with the separate Teacher-v2 identity."""

    return run_training(
        config,
        run_identity=TEACHER_V2_RUN_IDENTITY,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the preregistered 6-layer Teacher v2 whose only "
            "experimental change is dropout 0.2 -> 0.3."
        )
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--source-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the frozen Teacher-v2 command."""

    parser = _build_parser()
    arguments = parser.parse_args(argv)
    config = frozen_teacher_v2_config(
        data_dir=arguments.data_dir,
        output_dir=arguments.output_dir,
        source_commit=arguments.source_commit,
        device=arguments.device,
    )
    try:
        summary = run_teacher_v2_training(config)
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
