from __future__ import annotations

import json

import pytest

from nanogpt_nspire.efficient_context import GQA_ALIBI_SFT_V2_ROUTE
from nanogpt_nspire.group_policy_train import (
    build_judge_problem,
    compose_route_rewards,
    frozen_policy_training_config,
    smoke_policy_training_config,
)
from nanogpt_nspire.lesson17_data import RLProblem, ScheduledPrompt
from nanogpt_nspire.lesson17_routes import (
    COMBINED_ROUTE,
    DIRECT_RLAIF_ROUTE,
    RLVR_ROUTE,
)
from nanogpt_nspire.rl_rollout import RolloutTrajectory


def _score(
    *,
    numeric: bool,
    unit: bool = True,
    valid: bool = True,
) -> dict[str, object]:
    return {
        "format_valid": valid,
        "numeric_correct": numeric,
        "unit_correct": unit,
    }


def test_route_reward_sources_remain_separate_and_bounded() -> None:
    scores = (
        _score(numeric=True),
        _score(numeric=False),
        _score(numeric=False, valid=False),
    )
    ai = {"a": 0.0, "b": 1.0, "c": 1.0}
    identifiers = ("a", "b", "c")

    rlvr = compose_route_rewards(
        route=RLVR_ROUTE,
        candidate_ids=identifiers,
        local_scores=scores,
        ai_rewards=None,
    )
    rlaif = compose_route_rewards(
        route=DIRECT_RLAIF_ROUTE,
        candidate_ids=identifiers,
        local_scores=scores,
        ai_rewards=ai,
    )
    combined = compose_route_rewards(
        route=COMBINED_ROUTE,
        candidate_ids=identifiers,
        local_scores=scores,
        ai_rewards=ai,
    )

    assert [item.total for item in rlvr] == [1.0, 0.05, 0.0]
    assert [item.total for item in rlaif] == [0.0, 1.0, 0.0]
    assert [item.total for item in combined] == [1.0, 0.25, 0.2]
    assert combined[0].verifier_total == 1.0
    assert combined[1].ai_reward == 1.0


def test_rlaif_route_requires_one_ai_score_per_candidate() -> None:
    with pytest.raises(ValueError, match="AI reward"):
        compose_route_rewards(
            route=DIRECT_RLAIF_ROUTE,
            candidate_ids=("a", "b"),
            local_scores=(
                _score(numeric=True),
                _score(numeric=False),
            ),
            ai_rewards={"a": 1.0},
        )


def test_judge_problem_contains_no_locally_verified_answer() -> None:
    problem = RLProblem(
        record_id="record",
        family_id="family",
        task="arithmetic",
        prompt="Calculate 12 * 7.",
        expected_answer="84",
        expected_unit=None,
        formula=None,
        difficulty="integer",
        source_id="local",
    )
    scheduled = ScheduledPrompt(
        schedule_id="schedule",
        update=1,
        slot=0,
        mode="direct",
        problem=problem,
    )
    trajectories = tuple(
        RolloutTrajectory(
            candidate_id=f"candidate-{index}",
            schedule_id="schedule",
            family_id="family",
            mode="direct",
            prompt_tokens=(256, 258, 65, 259, 262),
            full_tokens=(256, 258, 65, 259, 262, 56, 52, 257),
            generated_tokens=(56, 52, 257),
            old_log_probs=(-1.0, -1.0, -1.0),
            completion={
                "budget_exhausted": False,
                "context_exhausted": False,
                "final_text": "84",
                "final_transition": False,
                "leaked_token": None,
                "reasoning_text": "",
                "special_token_leak": False,
                "terminated": True,
            },
        )
        for index in range(2)
    )

    judge = build_judge_problem(scheduled, trajectories)
    serialized = json.dumps(
        {
            "mode": judge.mode,
            "prompt": judge.prompt,
            "responses": [item.response for item in judge.candidates],
            "task": judge.task,
        }
    )

    assert "expected_answer" not in serialized
    assert "ground_truth" not in serialized
    assert judge.prompt == problem.prompt


def test_frozen_policy_config_has_16_rollouts_and_32_steps(tmp_path) -> None:
    config = frozen_policy_training_config(
        route=RLVR_ROUTE,
        output_dir=tmp_path / "output",
        start_checkpoint=tmp_path / "start.pt",
        start_checkpoint_sha256="a" * 64,
        start_route=GQA_ALIBI_SFT_V2_ROUTE,
        source_commit="source",
        seed=20260731,
    )

    assert config.rollout_updates == 16
    assert config.policy_epochs == 2
    assert config.optimizer_steps == 32
    assert config.prompt_groups_per_update == 4
    assert config.group_size == 8
    assert config.max_new_tokens == 256
    assert config.learning_rate == 5e-6
    assert config.policy_micro_batch_size == 4

    with pytest.raises(ValueError, match="frozen"):
        frozen_policy_training_config(
            route=RLVR_ROUTE,
            output_dir=tmp_path / "other",
            start_checkpoint=tmp_path / "start.pt",
            start_checkpoint_sha256="a" * 64,
            start_route=GQA_ALIBI_SFT_V2_ROUTE,
            source_commit="source",
            seed=20260731,
            group_size=4,
        )


def test_smoke_profile_is_small_and_cannot_be_mistaken_for_formal(
    tmp_path,
) -> None:
    config = smoke_policy_training_config(
        route=RLVR_ROUTE,
        output_dir=tmp_path / "smoke",
        start_checkpoint=tmp_path / "start.pt",
        start_checkpoint_sha256="a" * 64,
        start_route=GQA_ALIBI_SFT_V2_ROUTE,
        source_commit="source",
        seed=20260731,
    )

    assert config.rollout_updates == 1
    assert config.group_size == 2
    assert config.max_new_tokens == 32
    assert config.policy_epochs == 1
    assert config.optimizer_steps == 1
    assert config.public_record()["profile"] == "smoke"
