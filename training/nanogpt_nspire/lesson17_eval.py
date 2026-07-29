"""Evaluate one Lesson 17 policy on both frozen sets and both modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from nanogpt_nspire.efficient_context import (
    GQA_ALIBI_SFT_ROUTE,
    GQA_ALIBI_SFT_V2_ROUTE,
)
from nanogpt_nspire.lesson16_eval import (
    run_lesson16_challenge_evaluation,
)
from nanogpt_nspire.lesson17_routes import TRAINABLE_ROUTES
from nanogpt_nspire.reasoning_eval import run_reasoning_evaluation
from nanogpt_nspire.reasoning_format import DIRECT_MODE, THINK_MODE
from nanogpt_nspire.training_support import write_json_atomic


EVALUATION_ROUTES = frozenset(
    {
        GQA_ALIBI_SFT_ROUTE,
        GQA_ALIBI_SFT_V2_ROUTE,
        *TRAINABLE_ROUTES,
    }
)


def _mode_metrics(
    evaluation: Mapping[str, object],
) -> dict[str, object]:
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("evaluation metrics are missing")
    examples = metrics.get("examples")
    accuracy = metrics.get("task_accuracy")
    format_rate = metrics.get("format_valid_rate")
    mode_rate = metrics.get("mode_compliance_rate")
    if (
        isinstance(examples, bool)
        or not isinstance(examples, int)
        or examples <= 0
        or isinstance(accuracy, bool)
        or not isinstance(accuracy, (int, float))
        or isinstance(format_rate, bool)
        or not isinstance(format_rate, (int, float))
        or isinstance(mode_rate, bool)
        or not isinstance(mode_rate, (int, float))
    ):
        raise ValueError("evaluation metrics are invalid")
    correct_float = float(accuracy) * examples
    correct = round(correct_float)
    if abs(correct_float - correct) > 1e-8:
        raise ValueError("evaluation accuracy does not encode an exact count")
    return {
        "correct": correct,
        "examples": examples,
        "format_valid_rate": float(format_rate),
        "mode_compliance_rate": float(mode_rate),
        "task_accuracy": float(accuracy),
    }


def _set_summary(
    by_mode: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    if set(by_mode) != {DIRECT_MODE, THINK_MODE}:
        raise ValueError("both direct and think evaluations are required")
    direct = _mode_metrics(by_mode[DIRECT_MODE])
    think = _mode_metrics(by_mode[THINK_MODE])
    examples = int(direct["examples"]) + int(think["examples"])
    correct = int(direct["correct"]) + int(think["correct"])
    return {
        DIRECT_MODE: direct,
        THINK_MODE: think,
        "combined": {
            "correct": correct,
            "examples": examples,
            "format_valid_rate": (
                float(direct["format_valid_rate"])
                * int(direct["examples"])
                + float(think["format_valid_rate"])
                * int(think["examples"])
            )
            / examples,
            "mode_compliance_rate": (
                float(direct["mode_compliance_rate"])
                * int(direct["examples"])
                + float(think["mode_compliance_rate"])
                * int(think["examples"])
            )
            / examples,
            "task_accuracy": correct / examples,
        },
    }


def summarize_policy_evaluations(
    *,
    primary_by_mode: Mapping[str, Mapping[str, object]],
    challenge_by_mode: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    return {
        "challenge": _set_summary(challenge_by_mode),
        "primary": _set_summary(primary_by_mode),
    }


def run_lesson17_policy_evaluation(
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    route: str,
    primary_evaluation: Path,
    challenge_evaluation: Path,
    output_dir: Path,
    device: str = "cuda",
    use_bfloat16: bool = True,
) -> dict[str, object]:
    if route not in EVALUATION_ROUTES:
        raise ValueError("evaluation route is unsupported")
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    primary: dict[str, dict[str, object]] = {}
    challenge: dict[str, dict[str, object]] = {}
    for mode in (DIRECT_MODE, THINK_MODE):
        primary[mode] = run_reasoning_evaluation(
            checkpoint_path=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            expected_route=route,
            evaluation_path=primary_evaluation,
            output_path=output_dir / f"primary-{mode}.json",
            mode=mode,
            device_name=device,
            max_per_task=32,
            max_new_tokens=256,
            use_bfloat16=use_bfloat16,
        )
        challenge[mode] = run_lesson16_challenge_evaluation(
            checkpoint_path=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            expected_route=route,
            evaluation_path=challenge_evaluation,
            output_path=output_dir / f"challenge-{mode}.json",
            mode=mode,
            device_name=device,
            max_new_tokens=256,
            use_bfloat16=use_bfloat16,
        )
    summary = {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "contract": {
            "challenge_examples_per_mode": 256,
            "decoding": "greedy",
            "max_new_tokens": 256,
            "primary_examples_per_mode": 128,
        },
        "route": route,
        "schema_version": 1,
        **summarize_policy_evaluations(
            primary_by_mode=primary,
            challenge_by_mode=challenge,
        ),
    }
    write_json_atomic(output_dir / "summary.json", summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument(
        "--route",
        choices=tuple(sorted(EVALUATION_ROUTES)),
        required=True,
    )
    parser.add_argument("--primary-evaluation", type=Path, required=True)
    parser.add_argument("--challenge-evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-bfloat16", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    summary = run_lesson17_policy_evaluation(
        checkpoint=arguments.checkpoint,
        checkpoint_sha256=arguments.checkpoint_sha256,
        route=arguments.route,
        primary_evaluation=arguments.primary_evaluation,
        challenge_evaluation=arguments.challenge_evaluation,
        output_dir=arguments.output_dir,
        device=arguments.device,
        use_bfloat16=not arguments.no_bfloat16,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
