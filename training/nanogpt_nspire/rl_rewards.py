"""Reward composition for Lesson 17 RLVR and direct-RLAIF."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


class RewardError(ValueError):
    """Raised when verifier or AI reward fields violate the contract."""


def _required_bool(score: Mapping[str, object], field: str) -> bool:
    value = score.get(field)
    if not isinstance(value, bool):
        raise RewardError(f"{field} must be boolean")
    return value


def _ai_value(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 <= float(value) <= 1
    ):
        raise RewardError("ai_reward must be in [0, 1]")
    return float(value)


@dataclass(frozen=True)
class VerifierReward:
    numeric: float
    unit: float
    format: float
    total: float

    def __post_init__(self) -> None:
        for name in ("numeric", "unit", "format", "total"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise RewardError(f"{name} reward is invalid")
        if not math.isclose(
            self.total,
            self.numeric + self.unit + self.format,
            abs_tol=1e-12,
        ):
            raise RewardError("reward components do not sum to total")


def verifier_reward(score: Mapping[str, object]) -> VerifierReward:
    """Return an exact-dominated reward with at most 0.05 for wrong values."""

    if not isinstance(score, Mapping):
        raise RewardError("score must be a mapping")
    numeric_correct = _required_bool(score, "numeric_correct")
    unit_correct = _required_bool(score, "unit_correct")
    format_valid = _required_bool(score, "format_valid")
    numeric = 0.8 if numeric_correct else 0.0
    unit = 0.15 if numeric_correct and unit_correct else 0.0
    format_component = 0.05 if format_valid else 0.0
    return VerifierReward(
        numeric=numeric,
        unit=unit,
        format=format_component,
        total=numeric + unit + format_component,
    )


def direct_rlaif_reward(
    score: Mapping[str, object],
    *,
    ai_reward: float,
) -> float:
    """Use local format only as a guard around direct AI feedback."""

    if not isinstance(score, Mapping):
        raise RewardError("score must be a mapping")
    value = _ai_value(ai_reward)
    return value if _required_bool(score, "format_valid") else 0.0


def combined_reward(
    verifier: VerifierReward,
    *,
    ai_reward: float,
) -> float:
    """Bound AI feedback so it cannot override exact numeric correctness."""

    if not isinstance(verifier, VerifierReward):
        raise RewardError("verifier must be VerifierReward")
    return verifier.total + 0.2 * _ai_value(ai_reward)
