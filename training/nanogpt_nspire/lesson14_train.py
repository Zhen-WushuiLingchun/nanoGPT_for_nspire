"""Frozen Direct, short-CoT, and hybrid SFT routes for Lesson 14."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nanogpt_nspire.stage_train import (
    StageTrainingConfig,
    run_stage_training,
)


DIRECT_CONTROL_ROUTE = "Direct-Control-SFT"
SHORT_COT_ROUTE = "Short-CoT-SFT"
HYBRID_CONTROL_ROUTE = "Hybrid-Control-SFT"
ROUTES = {
    "direct": DIRECT_CONTROL_ROUTE,
    "cot": SHORT_COT_ROUTE,
    "hybrid": HYBRID_CONTROL_ROUTE,
}
CHECKPOINT_FILENAMES = {
    "direct": "direct_control_sft.pt",
    "cot": "short_cot_sft.pt",
    "hybrid": "hybrid_control_sft.pt",
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


def frozen_lesson14_student_config(
    *,
    kind: str,
    data_dir: str | Path,
    output_dir: str | Path,
    parent_checkpoint: str | Path,
    parent_checkpoint_sha256: str,
    source_commit: str,
    **overrides: object,
) -> StageTrainingConfig:
    """Keep architecture and optimization fixed across all three routes."""

    if kind not in ROUTES:
        raise ValueError("kind must be 'direct', 'cot', or 'hybrid'")
    changed = sorted(_ARCHITECTURE_FIELDS & frozenset(overrides))
    if changed:
        raise ValueError(
            "Lesson 14 student architecture is frozen; remove overrides: "
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=tuple(ROUTES), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-checkpoint-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    config = frozen_lesson14_student_config(
        kind=arguments.kind,
        data_dir=arguments.data_dir,
        output_dir=arguments.output_dir,
        parent_checkpoint=arguments.parent_checkpoint,
        parent_checkpoint_sha256=arguments.parent_checkpoint_sha256,
        source_commit=arguments.source_commit,
        device=arguments.device,
    )
    result = run_stage_training(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
