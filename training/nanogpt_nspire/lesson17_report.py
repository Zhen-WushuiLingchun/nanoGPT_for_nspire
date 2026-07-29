"""Aggregate Lesson 17 training, evaluations, and pre-registered gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Mapping, Sequence

from nanogpt_nspire.assistant_eval import load_evaluation_records
from nanogpt_nspire.lesson17_data import FORMAL_POLICY_SEEDS
from nanogpt_nspire.lesson17_routes import (
    COMBINED_ROUTE,
    DIRECT_RLAIF_ROUTE,
    RLVR_ROUTE,
)
from nanogpt_nspire.secret_safety import (
    assert_secret_free,
    assert_secret_free_tree,
)
from nanogpt_nspire.training_support import (
    sha256_file,
    write_json_atomic,
)


ROUTE_DIRECTORIES = {
    "rlvr": (RLVR_ROUTE, "rlvr-seed"),
    "rlaif": (DIRECT_RLAIF_ROUTE, "rlaif-seed"),
    "combined": (COMBINED_ROUTE, "combined-seed"),
}


def public_evaluation_summary(
    summary: Mapping[str, object],
) -> dict[str, object]:
    """Drop machine-local paths while preserving the audited evaluation."""

    required = {
        "challenge",
        "checkpoint_sha256",
        "contract",
        "primary",
        "route",
        "schema_version",
    }
    missing = sorted(required - summary.keys())
    if missing:
        raise ValueError(
            "evaluation summary is missing fields: "
            + ", ".join(missing)
        )
    return {
        key: summary[key]
        for key in sorted(required)
    }


def _public_checkpoint(
    checkpoint: object,
) -> dict[str, object]:
    if not isinstance(checkpoint, Mapping):
        raise ValueError("training checkpoint record is missing")
    required = {
        "bytes",
        "model_state_sha256",
        "path",
        "sha256",
    }
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError(
            "training checkpoint is missing fields: "
            + ", ".join(missing)
        )
    return {
        "bytes": checkpoint["bytes"],
        "filename": Path(str(checkpoint["path"])).name,
        "model_state_sha256": checkpoint["model_state_sha256"],
        "sha256": checkpoint["sha256"],
    }


def _combined(
    summary: Mapping[str, object],
    set_name: str,
) -> Mapping[str, object]:
    value = summary.get(set_name)
    if not isinstance(value, Mapping):
        raise ValueError(f"{set_name} summary is missing")
    combined = value.get("combined")
    if not isinstance(combined, Mapping):
        raise ValueError(f"{set_name} combined summary is missing")
    return combined


def _correct(
    summary: Mapping[str, object],
    set_name: str,
) -> int:
    value = _combined(summary, set_name).get("correct")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{set_name} correct count is invalid")
    return value


def _rate(
    summary: Mapping[str, object],
    set_name: str,
    field: str,
) -> float:
    value = _combined(summary, set_name).get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{set_name} {field} is invalid")
    return float(value)


def aggregate_route_evaluations(
    *,
    baseline: Mapping[str, object],
    seeds: Sequence[Mapping[str, object]],
    no_holdout_overlap: bool,
) -> dict[str, object]:
    """Apply the frozen both-set, three-seed, format, and leakage gates."""

    if len(seeds) != 3:
        raise ValueError("exactly three seed summaries are required")
    if not isinstance(no_holdout_overlap, bool):
        raise ValueError("no_holdout_overlap must be boolean")
    baseline_primary = _correct(baseline, "primary")
    baseline_challenge = _correct(baseline, "challenge")
    primary = [_correct(seed, "primary") for seed in seeds]
    challenge = [_correct(seed, "challenge") for seed in seeds]
    mean_primary = statistics.fmean(primary)
    mean_challenge = statistics.fmean(challenge)
    improving_both = sum(
        primary_value > baseline_primary
        and challenge_value > baseline_challenge
        for primary_value, challenge_value in zip(
            primary,
            challenge,
            strict=True,
        )
    )
    minimum_format = min(
        _rate(seed, set_name, "format_valid_rate")
        for seed in seeds
        for set_name in ("primary", "challenge")
    )
    minimum_mode = min(
        _rate(seed, set_name, "mode_compliance_rate")
        for seed in seeds
        for set_name in ("primary", "challenge")
    )
    gates = {
        "both_set_means_improve": (
            mean_primary > baseline_primary
            and mean_challenge > baseline_challenge
        ),
        "format_at_least_95_percent": (
            minimum_format >= 0.95
            and minimum_mode >= 0.95
        ),
        "no_holdout_overlap": no_holdout_overlap,
        "two_of_three_seeds_improve_both_sets": improving_both >= 2,
    }
    gates["ability_improvement"] = all(gates.values())
    return {
        "baseline": {
            "challenge_correct": baseline_challenge,
            "primary_correct": baseline_primary,
        },
        "challenge": {
            "mean_correct": mean_challenge,
            "per_seed_correct": challenge,
            "population_std_correct": statistics.pstdev(challenge),
        },
        "claim_gate": gates,
        "minimum_format_valid_rate": minimum_format,
        "minimum_mode_compliance_rate": minimum_mode,
        "primary": {
            "mean_correct": mean_primary,
            "per_seed_correct": primary,
            "population_std_correct": statistics.pstdev(primary),
        },
        "seeds_improving_both_sets": improving_both,
    }


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8", errors="strict")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    try:
        lines = path.read_text(
            encoding="ascii",
            errors="strict",
        ).splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"could not load {path}") from error
    for line in lines:
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path} contains a non-object row")
        rows.append(value)
    return tuple(rows)


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    centered_left = [value - mean_left for value in left]
    centered_right = [value - mean_right for value in right]
    denominator = math.sqrt(
        sum(value * value for value in centered_left)
        * sum(value * value for value in centered_right)
    )
    if denominator == 0.0:
        return None
    return sum(
        left_value * right_value
        for left_value, right_value in zip(
            centered_left,
            centered_right,
            strict=True,
        )
    ) / denominator


def _training_audit(
    run: Mapping[str, object],
    trajectories: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    ai_values: list[float] = []
    numeric_values: list[float] = []
    reward_values: list[float] = []
    exact = 0
    format_valid = 0
    high_ai_wrong = 0
    low_ai_correct = 0
    for row in trajectories:
        score = row.get("local_score")
        reward = row.get("reward")
        if not isinstance(score, Mapping) or not isinstance(reward, Mapping):
            raise ValueError("training audit row is malformed")
        numeric = bool(score.get("numeric_correct"))
        valid = bool(score.get("format_valid"))
        task_correct = bool(score.get("task_correct"))
        total = reward.get("total")
        if not isinstance(total, (int, float)):
            raise ValueError("training reward is invalid")
        reward_values.append(float(total))
        exact += int(task_correct)
        format_valid += int(valid)
        ai = row.get("ai_reward")
        if ai is not None:
            if not isinstance(ai, (int, float)):
                raise ValueError("AI reward is invalid")
            ai_value = float(ai)
            ai_values.append(ai_value)
            numeric_values.append(float(numeric))
            high_ai_wrong += int(ai_value >= 0.75 and not numeric)
            low_ai_correct += int(ai_value <= 0.25 and numeric)
    updates = run.get("updates")
    if not isinstance(updates, list):
        raise ValueError("training run updates are missing")
    cache_hits = 0
    judgments = 0
    retry_attempts = 0
    zero_variance = 0
    for update in updates:
        if not isinstance(update, Mapping):
            raise ValueError("training update is malformed")
        zero_variance += int(update.get("zero_variance_groups", 0))
        ai_judgments = update.get("ai_judgments")
        if isinstance(ai_judgments, Mapping):
            for judgment in ai_judgments.values():
                if not isinstance(judgment, Mapping):
                    raise ValueError("AI judgment is malformed")
                judgments += 1
                cache_hits += int(bool(judgment.get("cache_hit")))
                attempts = judgment.get("transport_attempts")
                if isinstance(attempts, int):
                    retry_attempts += attempts
    return {
        "ai_numeric_pearson": _pearson(ai_values, numeric_values),
        "ai_reward_mean": (
            statistics.fmean(ai_values) if ai_values else None
        ),
        "cache_hits": cache_hits,
        "exact_completions": exact,
        "format_valid_rate": format_valid / len(trajectories),
        "high_ai_wrong_numeric": high_ai_wrong,
        "judge_groups": judgments,
        "low_ai_correct_numeric": low_ai_correct,
        "reward_mean": statistics.fmean(reward_values),
        "sampled_completions": len(trajectories),
        "transport_attempts_for_uncached_groups": retry_attempts,
        "zero_variance_groups": zero_variance,
    }


def build_lesson17_experiment(
    *,
    start_screen: Path,
    formal_root: Path,
    evaluation_root: Path,
    primary_evaluation: Path,
    challenge_evaluation: Path,
) -> dict[str, object]:
    baseline_raw = _load_json(
        evaluation_root / "sft-only" / "summary.json"
    )
    baseline = public_evaluation_summary(baseline_raw)
    holdout_families = {
        str(row["family_id"])
        for path in (primary_evaluation, challenge_evaluation)
        for row in load_evaluation_records(path)
    }
    routes: dict[str, object] = {}
    for label, (expected_route, prefix) in ROUTE_DIRECTORIES.items():
        seed_records: list[dict[str, object]] = []
        no_overlap = True
        for seed in FORMAL_POLICY_SEEDS:
            training_dir = formal_root / f"{prefix}-{seed}"
            evaluation_dir = evaluation_root / f"{prefix}-{seed}"
            run = _load_json(training_dir / "run.json")
            evaluation = _load_json(evaluation_dir / "summary.json")
            if (
                run.get("route") != expected_route
                or evaluation.get("route") != expected_route
            ):
                raise ValueError(f"{label}/{seed} route mismatch")
            pool = _load_jsonl(training_dir / "problem_pool.jsonl")
            pool_families = {
                str(row["family_id"]) for row in pool
            }
            overlap = sorted(pool_families & holdout_families)
            no_overlap = no_overlap and not overlap
            trajectories = _load_jsonl(
                training_dir / "trajectories.jsonl"
            )
            seed_records.append(
                {
                    "checkpoint": _public_checkpoint(run["checkpoint"]),
                    "evaluation": public_evaluation_summary(evaluation),
                    "holdout_overlap_count": len(overlap),
                    "policy_seed": seed,
                    "training_audit": _training_audit(
                        run,
                        trajectories,
                    ),
                }
            )
        routes[label] = {
            "aggregate": aggregate_route_evaluations(
                baseline=baseline,
                seeds=[
                    record["evaluation"]  # type: ignore[list-item]
                    for record in seed_records
                ],
                no_holdout_overlap=no_overlap,
            ),
            "route": expected_route,
            "seeds": seed_records,
        }
    result = {
        "baseline": baseline,
        "claim_policy": {
            "ability_improvement_requires": [
                "mean exact count above SFT-only on primary and challenge",
                "at least two of three seeds improve both sets",
                "format and mode compliance at least 95 percent",
                "zero training-family overlap with either holdout",
            ]
        },
        "evaluation_contract": {
            "challenge_filename": challenge_evaluation.name,
            "challenge_sha256": sha256_file(challenge_evaluation),
            "max_new_tokens": 256,
            "primary_filename": primary_evaluation.name,
            "primary_sha256": sha256_file(primary_evaluation),
        },
        "routes": routes,
        "schema_version": 1,
        "start_screen": _load_json(start_screen),
    }
    assert_secret_free(result, context="Lesson 17 experiment")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-screen", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--primary-evaluation", type=Path, required=True)
    parser.add_argument("--challenge-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    if arguments.output.exists():
        raise SystemExit(f"output already exists: {arguments.output}")
    result = build_lesson17_experiment(
        start_screen=arguments.start_screen,
        formal_root=arguments.formal_root,
        evaluation_root=arguments.evaluation_root,
        primary_evaluation=arguments.primary_evaluation,
        challenge_evaluation=arguments.challenge_evaluation,
    )
    write_json_atomic(arguments.output, result)
    assert_secret_free_tree(arguments.output.parent)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
