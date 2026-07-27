"""Train the frozen provisional teacher used by quantization and distillation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from nanogpt_nspire.data import DatasetError
from nanogpt_nspire.direct_small_train import (
    TrainingConfig,
    TrainingRunIdentity,
    run_training,
)


TEACHER_QUALITY_GATE_MAXIMUM_LOSS = 1.4797899746894836
TEACHER_RUN_IDENTITY = TrainingRunIdentity(
    route="Teacher",
    checkpoint_filename="teacher_gpt.pt",
    deployment_interpretation="fp32_source_for_int4_and_distillation",
    quality_gate_maximum_selected_validation_loss=(
        TEACHER_QUALITY_GATE_MAXIMUM_LOSS
    ),
)


def frozen_teacher_config(
    *,
    data_dir: Path,
    output_dir: Path,
    source_commit: str,
    device: str = "auto",
) -> TrainingConfig:
    """Return the preregistered teacher architecture and training protocol."""

    return TrainingConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        device=device,
        seed=1337,
        steps=10_000,
        batch_size=64,
        block_size=128,
        n_layer=6,
        n_head=6,
        n_embd=384,
        mlp_ratio=4,
        dropout=0.2,
        bias=False,
        tie_embeddings=True,
        learning_rate=0.001,
        min_learning_rate=0.0001,
        warmup_steps=100,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.99,
        max_grad_norm=1.0,
        eval_interval=250,
        eval_batches=50,
        log_interval=100,
        sample_tokens=300,
        temperature=0.8,
        source_commit=source_commit,
    )


def run_teacher_training(config: TrainingConfig) -> dict[str, object]:
    """Run the shared trainer with the frozen Teacher identity."""

    return run_training(
        config,
        run_identity=TEACHER_RUN_IDENTITY,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the frozen 6-layer provisional teacher and evaluate its "
            "pre-registered quality gate."
        )
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--source-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the frozen teacher command."""

    parser = _build_parser()
    arguments = parser.parse_args(argv)
    config = frozen_teacher_config(
        data_dir=arguments.data_dir,
        output_dir=arguments.output_dir,
        source_commit=arguments.source_commit,
        device=arguments.device,
    )
    try:
        summary = run_teacher_training(config)
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
