"""Slice-aware exact-answer evaluation for Lesson 16."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Mapping, Sequence

from nanogpt_nspire.assistant_eval import (
    EvaluationError,
    load_evaluation_model,
    load_evaluation_records,
)
from nanogpt_nspire.efficient_context import (
    GQA_ALIBI_SFT_ROUTE,
    GQA_ALIBI_SFT_V2_ROUTE,
)
from nanogpt_nspire.reasoning_eval import (
    generate_mode_completion,
    score_mode_completion,
)
from nanogpt_nspire.reasoning_format import SUPPORTED_MODES
from nanogpt_nspire.training_support import (
    resolve_device,
    sha256_file,
    write_json_atomic,
)


CHALLENGE_SLICES = frozenset(
    {
        "in_range",
        "range_shifted",
        "sign_shifted",
        "substitution_adversarial",
    }
)
CHALLENGE_ROUTES = frozenset(
    {GQA_ALIBI_SFT_ROUTE, GQA_ALIBI_SFT_V2_ROUTE}
)


def _rate(rows: Sequence[Mapping[str, object]], field: str) -> float:
    return sum(
        bool(dict(row["score"])[field])  # type: ignore[arg-type]
        for row in rows
    ) / len(rows)


def _group_metrics(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    generated = sum(
        int(dict(row["completion"])["generated_tokens"])  # type: ignore[arg-type]
        for row in rows
    )
    correct = sum(
        bool(dict(row["score"])["task_correct"])  # type: ignore[arg-type]
        for row in rows
    )
    return {
        "budget_truncation_rate": sum(
            bool(dict(row["completion"])["budget_exhausted"])  # type: ignore[arg-type]
            for row in rows
        )
        / len(rows),
        "context_truncation_rate": sum(
            bool(dict(row["completion"])["context_exhausted"])  # type: ignore[arg-type]
            for row in rows
        )
        / len(rows),
        "correct": correct,
        "correct_per_1000_generated_tokens": (
            correct * 1000 / generated if generated else None
        ),
        "count": len(rows),
        "final_transition_rate": sum(
            bool(dict(row["completion"])["final_transition"])  # type: ignore[arg-type]
            for row in rows
        )
        / len(rows),
        "format_valid_rate": _rate(rows, "format_valid"),
        "generated_tokens": generated,
        "mean_final_tokens": sum(
            int(dict(row["completion"])["final_tokens"])  # type: ignore[arg-type]
            for row in rows
        )
        / len(rows),
        "mean_reasoning_tokens": sum(
            int(dict(row["completion"])["reasoning_tokens"])  # type: ignore[arg-type]
            for row in rows
        )
        / len(rows),
        "mode_compliance_rate": _rate(rows, "mode_compliant"),
        "role_leak_rate": _rate(rows, "special_token_leak"),
        "task_accuracy": correct / len(rows),
    }


def summarize_challenge_results(
    results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Aggregate one run without hiding range or substitution failures."""

    if not results:
        raise EvaluationError("challenge results are empty")
    slices = {str(row.get("slice")) for row in results}
    if not slices <= CHALLENGE_SLICES:
        raise EvaluationError("challenge result slice is unsupported")
    tasks = {str(row.get("task")) for row in results}
    if not tasks <= {"arithmetic", "physics_numeric"}:
        raise EvaluationError("challenge result task is unsupported")
    return {
        "metrics": _group_metrics(results),
        "per_slice": {
            slice_name: _group_metrics(
                [
                    row
                    for row in results
                    if row.get("slice") == slice_name
                ]
            )
            for slice_name in sorted(slices)
        },
        "per_task": {
            task: _group_metrics(
                [row for row in results if row.get("task") == task]
            )
            for task in sorted(tasks)
        },
    }


def run_lesson16_challenge_evaluation(
    *,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    expected_route: str,
    evaluation_path: str | Path,
    output_path: str | Path,
    mode: str,
    device_name: str = "auto",
    max_new_tokens: int = 256,
    use_bfloat16: bool = True,
) -> dict[str, object]:
    """Evaluate all 256 held-out challenge families for one mode."""

    if expected_route not in CHALLENGE_ROUTES:
        raise EvaluationError("expected route is not a Lesson 16 route")
    if mode not in SUPPORTED_MODES:
        raise EvaluationError("mode is unsupported")
    output = Path(output_path)
    if output.exists():
        raise EvaluationError(f"output already exists: {output}")
    records = load_evaluation_records(evaluation_path)
    slice_counts: Counter[str] = Counter()
    selected: list[dict[str, object]] = []
    for raw in records:
        slice_name = raw.get("slice")
        if slice_name not in CHALLENGE_SLICES:
            raise EvaluationError("challenge row slice is invalid")
        slice_counts[str(slice_name)] += 1
        selected.append(raw)
    if set(slice_counts) != CHALLENGE_SLICES or any(
        count != 64 for count in slice_counts.values()
    ):
        raise EvaluationError(
            "challenge evaluation must contain 64 rows per slice"
        )
    selected.sort(
        key=lambda row: (str(row["slice"]), str(row["family_id"]))
    )
    device = resolve_device(device_name)
    model, checkpoint = load_evaluation_model(
        checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        expected_route=expected_route,
        device=device,
    )
    results: list[dict[str, object]] = []
    for record in selected:
        completion = generate_mode_completion(
            model,
            str(record["prompt"]),
            mode=mode,
            max_new_tokens=max_new_tokens,
            device=device,
            use_bfloat16=use_bfloat16,
        )
        results.append(
            {
                "completion": completion,
                "expected_answer": record["expected_answer"],
                "expected_unit": record["expected_unit"],
                "family_id": record["family_id"],
                "prompt": record["prompt"],
                "score": score_mode_completion(record, completion),
                "slice": record["slice"],
                "source_id": record["source_id"],
                "task": record["task"],
            }
        )
    aggregated = summarize_challenge_results(results)
    total_seconds = sum(
        float(dict(row["completion"])["elapsed_seconds"])
        for row in results
    )
    total_generated = int(aggregated["metrics"]["generated_tokens"])  # type: ignore[index]
    summary: dict[str, object] = {
        "checkpoint": checkpoint,
        "configuration": {
            "device": str(device),
            "evaluation_path": str(evaluation_path),
            "max_new_tokens": max_new_tokens,
            "mode": mode,
            "use_bfloat16": use_bfloat16 and device.type == "cuda",
        },
        "evaluation_file_sha256": sha256_file(Path(evaluation_path)),
        **aggregated,
        "results": results,
        "schema_version": 1,
    }
    metrics = summary["metrics"]
    assert isinstance(metrics, dict)
    metrics["tokens_per_second"] = (
        total_generated / total_seconds if total_seconds else None
    )
    write_json_atomic(output, summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument(
        "--expected-route",
        choices=tuple(sorted(CHALLENGE_ROUTES)),
        required=True,
    )
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=tuple(sorted(SUPPORTED_MODES)),
        required=True,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--no-bfloat16", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    summary = run_lesson16_challenge_evaluation(
        checkpoint_path=arguments.checkpoint,
        checkpoint_sha256=arguments.checkpoint_sha256,
        expected_route=arguments.expected_route,
        evaluation_path=arguments.evaluation,
        output_path=arguments.output,
        mode=arguments.mode,
        device_name=arguments.device,
        max_new_tokens=arguments.max_new_tokens,
        use_bfloat16=not arguments.no_bfloat16,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
