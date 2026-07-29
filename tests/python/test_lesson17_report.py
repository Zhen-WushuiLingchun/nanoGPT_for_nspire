from __future__ import annotations

from nanogpt_nspire.lesson17_report import (
    aggregate_route_evaluations,
    public_evaluation_summary,
)


def _summary(
    *,
    primary: int,
    challenge: int,
    format_rate: float = 1.0,
) -> dict[str, object]:
    return {
        "checkpoint_sha256": "a" * 64,
        "primary": {
            "combined": {
                "correct": primary,
                "examples": 256,
                "format_valid_rate": format_rate,
                "mode_compliance_rate": format_rate,
            }
        },
        "challenge": {
            "combined": {
                "correct": challenge,
                "examples": 512,
                "format_valid_rate": format_rate,
                "mode_compliance_rate": format_rate,
            }
        },
    }


def test_claim_gate_requires_both_set_means_and_two_of_three_support() -> None:
    baseline = _summary(primary=4, challenge=6)
    result = aggregate_route_evaluations(
        baseline=baseline,
        seeds=(
            _summary(primary=5, challenge=7),
            _summary(primary=6, challenge=8),
            _summary(primary=3, challenge=5),
        ),
        no_holdout_overlap=True,
    )

    assert result["primary"]["mean_correct"] > 4
    assert result["challenge"]["mean_correct"] > 6
    assert result["seeds_improving_both_sets"] == 2
    assert result["claim_gate"]["ability_improvement"] is True


def test_one_lucky_seed_and_format_regression_fail_gate() -> None:
    baseline = _summary(primary=4, challenge=6)
    result = aggregate_route_evaluations(
        baseline=baseline,
        seeds=(
            _summary(primary=20, challenge=20, format_rate=0.9),
            _summary(primary=3, challenge=5),
            _summary(primary=3, challenge=5),
        ),
        no_holdout_overlap=True,
    )

    assert result["seeds_improving_both_sets"] == 1
    assert result["claim_gate"]["format_at_least_95_percent"] is False
    assert result["claim_gate"]["ability_improvement"] is False


def test_overlap_alone_blocks_claim() -> None:
    baseline = _summary(primary=4, challenge=6)
    result = aggregate_route_evaluations(
        baseline=baseline,
        seeds=(
            _summary(primary=5, challenge=7),
            _summary(primary=5, challenge=7),
            _summary(primary=5, challenge=7),
        ),
        no_holdout_overlap=False,
    )

    assert result["claim_gate"]["no_holdout_overlap"] is False
    assert result["claim_gate"]["ability_improvement"] is False


def test_public_evaluation_summary_drops_machine_local_path() -> None:
    summary = _summary(primary=4, challenge=6)
    summary.update(
        {
            "checkpoint_path": r"F:\private\checkpoint.pt",
            "contract": {"decoding": "greedy"},
            "route": "test-route",
            "schema_version": 1,
        }
    )

    public = public_evaluation_summary(summary)

    assert "checkpoint_path" not in public
    assert public["checkpoint_sha256"] == "a" * 64
