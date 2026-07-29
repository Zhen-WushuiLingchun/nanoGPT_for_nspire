from __future__ import annotations

import pytest

from nanogpt_nspire.rl_rewards import (
    combined_reward,
    direct_rlaif_reward,
    verifier_reward,
)


def _score(
    *,
    numeric: bool,
    unit: bool,
    format_valid: bool,
) -> dict[str, object]:
    return {
        "format_valid": format_valid,
        "numeric_correct": numeric,
        "task_correct": numeric and unit and format_valid,
        "unit_correct": unit,
    }


def test_verifier_reward_is_exact_dominated() -> None:
    correct = verifier_reward(
        _score(numeric=True, unit=True, format_valid=True)
    )
    wrong = verifier_reward(
        _score(numeric=False, unit=True, format_valid=True)
    )

    assert correct.total == 1.0
    assert correct.numeric == 0.8
    assert correct.unit == 0.15
    assert correct.format == 0.05
    assert wrong.total == 0.05


def test_combined_ai_reward_cannot_override_wrong_numeric_result() -> None:
    correct = verifier_reward(
        _score(numeric=True, unit=True, format_valid=False)
    )
    wrong = verifier_reward(
        _score(numeric=False, unit=True, format_valid=True)
    )

    assert combined_reward(correct, ai_reward=0.0) == pytest.approx(0.95)
    assert combined_reward(wrong, ai_reward=1.0) == pytest.approx(0.25)
    assert combined_reward(correct, ai_reward=0.0) > combined_reward(
        wrong,
        ai_reward=1.0,
    )


def test_direct_rlaif_uses_format_as_guard_not_numeric_reward() -> None:
    valid_wrong = _score(
        numeric=False,
        unit=False,
        format_valid=True,
    )
    invalid_correct = _score(
        numeric=True,
        unit=True,
        format_valid=False,
    )

    assert direct_rlaif_reward(valid_wrong, ai_reward=0.75) == 0.75
    assert direct_rlaif_reward(invalid_correct, ai_reward=1.0) == 0.0
