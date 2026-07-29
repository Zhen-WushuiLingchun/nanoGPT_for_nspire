from __future__ import annotations

import pytest

from nanogpt_nspire.assistant_eval import EvaluationError
from nanogpt_nspire.lesson16_eval import summarize_challenge_results


def _row(
    *,
    slice_name: str,
    task: str,
    correct: bool,
    generated_tokens: int,
) -> dict[str, object]:
    return {
        "completion": {
            "budget_exhausted": not correct,
            "context_exhausted": False,
            "elapsed_seconds": 1.0,
            "final_tokens": 2,
            "final_transition": True,
            "generated_tokens": generated_tokens,
            "reasoning_tokens": 3,
        },
        "score": {
            "format_valid": correct,
            "mode_compliant": correct,
            "special_token_leak": False,
            "task_correct": correct,
        },
        "slice": slice_name,
        "task": task,
    }


def test_challenge_summary_preserves_slice_and_token_efficiency() -> None:
    rows = [
        _row(
            slice_name="in_range",
            task="arithmetic",
            correct=True,
            generated_tokens=10,
        ),
        _row(
            slice_name="range_shifted",
            task="physics_numeric",
            correct=False,
            generated_tokens=30,
        ),
    ]

    summary = summarize_challenge_results(rows)

    assert summary["metrics"]["task_accuracy"] == 0.5
    assert summary["metrics"]["correct_per_1000_generated_tokens"] == 25.0
    assert summary["per_slice"]["in_range"]["task_accuracy"] == 1.0
    assert summary["per_slice"]["range_shifted"]["task_accuracy"] == 0.0


def test_challenge_summary_rejects_unknown_slice() -> None:
    rows = [
        _row(
            slice_name="training",
            task="arithmetic",
            correct=True,
            generated_tokens=10,
        )
    ]

    with pytest.raises(EvaluationError, match="slice"):
        summarize_challenge_results(rows)
