"""No-update dual-start exploitability screen for Lesson 17."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch

from nanogpt_nspire.assistant_eval import load_evaluation_records
from nanogpt_nspire.efficient_context import (
    GQA_ALIBI_SFT_ROUTE,
    GQA_ALIBI_SFT_V2_ROUTE,
    lesson15_efficient_config,
    load_efficient_checkpoint,
)
from nanogpt_nspire.lesson17_data import (
    LESSON17_DATA_SEED,
    ScheduledPrompt,
    build_lesson17_problem_pool,
    build_prompt_schedule,
    canonical_problem_pool_bytes,
)
from nanogpt_nspire.models.efficient_long_context_gpt import ALIBI_POSITIONS
from nanogpt_nspire.reasoning_eval import score_mode_completion
from nanogpt_nspire.rl_rollout import sample_mode_group
from nanogpt_nspire.secret_safety import assert_secret_free
from nanogpt_nspire.training_support import (
    resolve_device,
    sha256_file,
    write_json_atomic,
)


SCREEN_POLICY_SEED = 20260731
SCREEN_UPDATES = 8
SCREEN_PROMPTS = 32
SCREEN_GROUP_SIZE = 8
SCREEN_TEMPERATURE = 0.8
SCREEN_MAX_NEW_TOKENS = 256
START_ROUTES = frozenset(
    {GQA_ALIBI_SFT_ROUTE, GQA_ALIBI_SFT_V2_ROUTE}
)


@dataclass(frozen=True)
class StartCandidate:
    name: str
    checkpoint: Path
    checkpoint_sha256: str
    route: str

    def validate(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("start candidate name must be non-empty")
        if self.route not in START_ROUTES:
            raise ValueError("start candidate route is unsupported")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        if sha256_file(self.checkpoint) != self.checkpoint_sha256:
            raise ValueError("start candidate checkpoint hash mismatch")
        assert_secret_free(asdict(self), context="start candidate")


def summarize_candidate_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not rows:
        raise ValueError("candidate rows must be non-empty")
    grouped: dict[str, list[bool]] = defaultdict(list)
    exact = 0
    invalid = 0
    for row in rows:
        schedule_id = row.get("schedule_id")
        score = row.get("score")
        if not isinstance(schedule_id, str) or not isinstance(score, Mapping):
            raise ValueError("candidate screen row is invalid")
        task_correct = score.get("task_correct")
        format_valid = score.get("format_valid")
        mode_compliant = score.get("mode_compliant")
        if any(
            not isinstance(value, bool)
            for value in (task_correct, format_valid, mode_compliant)
        ):
            raise ValueError("candidate screen score is invalid")
        correct = bool(task_correct)
        grouped[schedule_id].append(correct)
        exact += int(correct)
        invalid += int(
            not (bool(format_valid) and bool(mode_compliant))
        )
    mixed = sum(
        any(values) and not all(values)
        for values in grouped.values()
    )
    return {
        "completions": len(rows),
        "exact_completion_rate": exact / len(rows),
        "exact_completions": exact,
        "groups": len(grouped),
        "invalid_format_rate": invalid / len(rows),
        "invalid_formats": invalid,
        "mixed_exact_group_fraction": mixed / len(grouped),
        "mixed_exact_groups": mixed,
    }


def select_start_candidate(
    candidates: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Apply the pre-registered exploitability order with the v2 tie break."""

    if len(candidates) != 2:
        raise ValueError("exactly two start candidates are required")

    def rank(candidate: Mapping[str, object]) -> tuple[float, ...]:
        metrics = candidate.get("metrics")
        route = candidate.get("route")
        if not isinstance(metrics, Mapping) or route not in START_ROUTES:
            raise ValueError("candidate selection record is invalid")
        values = (
            metrics.get("mixed_exact_group_fraction"),
            metrics.get("exact_completion_rate"),
            metrics.get("invalid_format_rate"),
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            for value in values
        ):
            raise ValueError("candidate selection metrics are invalid")
        return (
            float(values[0]),
            float(values[1]),
            -float(values[2]),
            float(route == GQA_ALIBI_SFT_V2_ROUTE),
        )

    return dict(max(candidates, key=rank))


def _holdout_families(paths: Sequence[Path]) -> frozenset[str]:
    families: set[str] = set()
    for path in paths:
        families.update(
            str(row["family_id"])
            for row in load_evaluation_records(path)
        )
    return frozenset(families)


def run_start_screen(
    *,
    candidates: Sequence[StartCandidate],
    primary_evaluation: Path,
    challenge_evaluation: Path,
    output_dir: Path,
    device_name: str = "cuda",
    use_bfloat16: bool = True,
) -> dict[str, object]:
    """Sample both starts under identical prompts and RNG without updates."""

    if len(candidates) != 2:
        raise ValueError("exactly two candidates are required")
    for candidate in candidates:
        candidate.validate()
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")
    excluded = _holdout_families(
        (primary_evaluation, challenge_evaluation)
    )
    pool = build_lesson17_problem_pool(
        count_per_task=32,
        seed=LESSON17_DATA_SEED,
        excluded_families=excluded,
    )
    schedule = build_prompt_schedule(
        pool,
        seed=SCREEN_POLICY_SEED,
        updates=SCREEN_UPDATES,
        prompts_per_update=4,
    )
    if len(schedule) != SCREEN_PROMPTS:
        raise AssertionError("screen schedule size drifted")
    device = resolve_device(device_name)
    summaries: list[dict[str, object]] = []
    raw_by_name: dict[str, list[dict[str, object]]] = {}
    for candidate in candidates:
        model, parent = load_efficient_checkpoint(
            candidate.checkpoint,
            expected_sha256=candidate.checkpoint_sha256,
            expected_route=candidate.route,
            expected_model_config=lesson15_efficient_config(
                ALIBI_POSITIONS
            ),
        )
        model.to(device)
        model.eval()
        generator = torch.Generator(device="cpu").manual_seed(
            SCREEN_POLICY_SEED
        )
        rows: list[dict[str, object]] = []
        for scheduled in schedule:
            trajectories = sample_mode_group(
                model,
                scheduled.problem.prompt,
                mode=scheduled.mode,
                schedule_id=scheduled.schedule_id,
                family_id=scheduled.problem.family_id,
                group_size=SCREEN_GROUP_SIZE,
                max_new_tokens=SCREEN_MAX_NEW_TOKENS,
                temperature=SCREEN_TEMPERATURE,
                device=device,
                generator=generator,
                use_bfloat16=use_bfloat16,
            )
            for trajectory in trajectories:
                rows.append(
                    {
                        "candidate_id": trajectory.candidate_id,
                        "completion": dict(trajectory.completion),
                        "family_id": scheduled.problem.family_id,
                        "mode": scheduled.mode,
                        "schedule_id": scheduled.schedule_id,
                        "score": score_mode_completion(
                            scheduled.problem.evaluation_record(),
                            trajectory.completion,
                        ),
                        "task": scheduled.problem.task,
                    }
                )
        metrics = summarize_candidate_rows(rows)
        summaries.append(
            {
                "checkpoint_sha256": candidate.checkpoint_sha256,
                "metrics": metrics,
                "name": candidate.name,
                "parent": parent,
                "route": candidate.route,
            }
        )
        raw_by_name[candidate.name] = rows
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    selected = select_start_candidate(summaries)
    result: dict[str, object] = {
        "candidates": summaries,
        "configuration": {
            "api_calls": 0,
            "group_size": SCREEN_GROUP_SIZE,
            "max_new_tokens": SCREEN_MAX_NEW_TOKENS,
            "policy_seed": SCREEN_POLICY_SEED,
            "prompt_count": SCREEN_PROMPTS,
            "selection_order": [
                "mixed_exact_group_fraction",
                "exact_completion_rate",
                "negative_invalid_format_rate",
                "sft_v2_exact_tie_break",
            ],
            "temperature": SCREEN_TEMPERATURE,
            "updates": 0,
            "use_bfloat16": use_bfloat16 and device.type == "cuda",
        },
        "holdout_family_count": len(excluded),
        "problem_pool_family_count": len(pool),
        "selected": selected,
        "schema_version": 1,
    }
    assert_secret_free(result, context="start screen result")
    output_dir.mkdir(parents=True)
    (output_dir / "problem_pool.jsonl").write_bytes(
        canonical_problem_pool_bytes(pool)
    )
    for name, rows in raw_by_name.items():
        write_json_atomic(output_dir / f"{name}-raw.json", rows)
    write_json_atomic(output_dir / "run.json", result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--v1-sha256", required=True)
    parser.add_argument("--v2-checkpoint", type=Path, required=True)
    parser.add_argument("--v2-sha256", required=True)
    parser.add_argument("--primary-evaluation", type=Path, required=True)
    parser.add_argument("--challenge-evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-bfloat16", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    result = run_start_screen(
        candidates=(
            StartCandidate(
                name="lesson15-sft",
                checkpoint=arguments.v1_checkpoint,
                checkpoint_sha256=arguments.v1_sha256,
                route=GQA_ALIBI_SFT_ROUTE,
            ),
            StartCandidate(
                name="lesson16-sft-v2",
                checkpoint=arguments.v2_checkpoint,
                checkpoint_sha256=arguments.v2_sha256,
                route=GQA_ALIBI_SFT_V2_ROUTE,
            ),
        ),
        primary_evaluation=arguments.primary_evaluation,
        challenge_evaluation=arguments.challenge_evaluation,
        output_dir=arguments.output_dir,
        device_name=arguments.device,
        use_bfloat16=not arguments.no_bfloat16,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
