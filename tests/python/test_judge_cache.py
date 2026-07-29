from __future__ import annotations

import json

import pytest

from nanogpt_nspire.judge_cache import (
    JudgeCache,
    JudgeCacheError,
    judge_with_cache,
    render_judge_response,
)
from nanogpt_nspire.preference_judge import (
    JudgeCandidate,
    JudgeProblem,
    PreferenceJudgeClient,
    PreferenceJudgeConfig,
)
from nanogpt_nspire.rl_rollout import RolloutTrajectory
from nanogpt_nspire.secret_safety import assert_secret_free_tree

from test_preference_judge import FAKE_SECRET, provider_response


def problem() -> JudgeProblem:
    return JudgeProblem(
        schedule_id="schedule-cache",
        task="arithmetic",
        mode="direct",
        prompt="Calculate 12 * 7.",
        candidates=(
            JudgeCandidate(candidate_id="candidate-a", response="84"),
            JudgeCandidate(candidate_id="candidate-b", response="74"),
        ),
    )


def test_cache_prevents_a_second_paid_request(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_SECRET)
    calls = 0

    def transport(*_: object) -> bytes:
        nonlocal calls
        calls += 1
        return provider_response()

    client = PreferenceJudgeClient(
        PreferenceJudgeConfig(
            max_requests=1,
            maximum_attempts=1,
        ),
        transport=transport,
    )
    cache_path = tmp_path / "judge-cache.jsonl"
    cache = JudgeCache(cache_path)

    first, first_hit = judge_with_cache(
        client=client,
        problem=problem(),
        cache=cache,
    )
    second, second_hit = judge_with_cache(
        client=client,
        problem=problem(),
        cache=JudgeCache(cache_path),
    )

    assert first_hit is False
    assert second_hit is True
    assert first.public_record() == second.public_record()
    assert calls == 1
    assert client.logical_requests == 1
    assert_secret_free_tree(tmp_path)


def test_conflicting_duplicate_cache_record_is_rejected(tmp_path) -> None:
    path = tmp_path / "judge-cache.jsonl"
    base = {
        "answer": {
            "preferred_candidate_id": "candidate-a",
            "provider_model": "deepseek-v4-pro",
            "provider_request_id": "request-one",
            "rationale": "A is correct.",
            "request_sha256": "a" * 64,
            "scores": [
                {"candidate_id": "candidate-a", "score": 4},
                {"candidate_id": "candidate-b", "score": 0},
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        },
        "request_sha256": "a" * 64,
        "schema_version": 1,
    }
    conflict = json.loads(json.dumps(base))
    conflict["answer"]["provider_request_id"] = "request-two"
    path.write_text(
        json.dumps(base) + "\n" + json.dumps(conflict) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(JudgeCacheError, match="conflicting duplicate"):
        JudgeCache(path)


def test_trajectory_render_exposes_mode_segments_and_format_status() -> None:
    trajectory = RolloutTrajectory(
        candidate_id="candidate-a",
        schedule_id="schedule",
        family_id="family",
        mode="think",
        prompt_tokens=(256, 258, 65, 259, 261),
        full_tokens=(256, 258, 65, 259, 261, 49, 262, 50, 257),
        generated_tokens=(49, 262, 50, 257),
        old_log_probs=(-1.0, -1.0, -1.0, -1.0),
        completion={
            "budget_exhausted": False,
            "context_exhausted": False,
            "final_text": "2",
            "final_transition": True,
            "leaked_token": None,
            "reasoning_text": "1",
            "special_token_leak": False,
            "terminated": True,
        },
    )

    rendered = render_judge_response(trajectory)

    assert rendered.startswith("<THINK>1<FINAL>2")
    assert "terminated=true" in rendered
    assert "special_token_leak=false" in rendered
