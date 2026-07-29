from __future__ import annotations

from nanogpt_nspire.lesson17_eval import summarize_policy_evaluations


def _evaluation(*, examples: int, correct: int) -> dict[str, object]:
    return {
        "metrics": {
            "examples": examples,
            "format_valid_rate": 1.0,
            "mode_compliance_rate": 1.0,
            "task_accuracy": correct / examples,
        }
    }


def _challenge(*, examples: int, correct: int) -> dict[str, object]:
    return {
        "metrics": {
            "examples": examples,
            "format_valid_rate": 0.99,
            "mode_compliance_rate": 0.98,
            "task_accuracy": correct / examples,
        }
    }


def test_policy_summary_keeps_modes_and_combines_raw_counts() -> None:
    summary = summarize_policy_evaluations(
        primary_by_mode={
            "direct": _evaluation(examples=128, correct=3),
            "think": _evaluation(examples=128, correct=5),
        },
        challenge_by_mode={
            "direct": _challenge(examples=256, correct=2),
            "think": _challenge(examples=256, correct=4),
        },
    )

    assert summary["primary"]["direct"]["correct"] == 3
    assert summary["primary"]["think"]["correct"] == 5
    assert summary["primary"]["combined"]["examples"] == 256
    assert summary["primary"]["combined"]["correct"] == 8
    assert summary["challenge"]["combined"]["examples"] == 512
    assert summary["challenge"]["combined"]["correct"] == 6
    assert summary["challenge"]["combined"]["format_valid_rate"] == 0.99
    assert (
        summary["challenge"]["combined"]["mode_compliance_rate"]
        == 0.98
    )
