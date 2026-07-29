from __future__ import annotations

from nanogpt_nspire.efficient_context import (
    GQA_ALIBI_SFT_ROUTE,
    GQA_ALIBI_SFT_V2_ROUTE,
)
from nanogpt_nspire.lesson17_start_screen import (
    select_start_candidate,
    summarize_candidate_rows,
)


def _row(*, correct: bool, valid: bool) -> dict[str, object]:
    return {
        "schedule_id": "group",
        "score": {
            "format_valid": valid,
            "mode_compliant": valid,
            "task_correct": correct,
        },
    }


def test_candidate_summary_reports_mixed_groups_and_rates() -> None:
    rows = [
        {**_row(correct=True, valid=True), "schedule_id": "a"},
        {**_row(correct=False, valid=True), "schedule_id": "a"},
        {**_row(correct=True, valid=True), "schedule_id": "b"},
        {**_row(correct=True, valid=True), "schedule_id": "b"},
    ]

    summary = summarize_candidate_rows(rows)

    assert summary["groups"] == 2
    assert summary["mixed_exact_group_fraction"] == 0.5
    assert summary["exact_completion_rate"] == 0.75
    assert summary["invalid_format_rate"] == 0.0


def test_selection_uses_frozen_lexicographic_order() -> None:
    candidates = [
        {
            "name": "v1",
            "route": GQA_ALIBI_SFT_ROUTE,
            "metrics": {
                "mixed_exact_group_fraction": 0.5,
                "exact_completion_rate": 0.2,
                "invalid_format_rate": 0.0,
            },
        },
        {
            "name": "v2",
            "route": GQA_ALIBI_SFT_V2_ROUTE,
            "metrics": {
                "mixed_exact_group_fraction": 0.4,
                "exact_completion_rate": 0.9,
                "invalid_format_rate": 0.0,
            },
        },
    ]

    assert select_start_candidate(candidates)["name"] == "v1"


def test_exact_tie_falls_to_sft_v2() -> None:
    metrics = {
        "mixed_exact_group_fraction": 0.25,
        "exact_completion_rate": 0.1,
        "invalid_format_rate": 0.2,
    }
    candidates = [
        {
            "name": "v1",
            "route": GQA_ALIBI_SFT_ROUTE,
            "metrics": metrics,
        },
        {
            "name": "v2",
            "route": GQA_ALIBI_SFT_V2_ROUTE,
            "metrics": metrics,
        },
    ]

    assert select_start_candidate(candidates)["name"] == "v2"
