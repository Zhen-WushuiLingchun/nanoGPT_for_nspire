from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json

import pytest

from nanogpt_nspire.preference_judge import (
    JudgeCandidate,
    JudgeProblem,
    PreferenceJudgeClient,
    PreferenceJudgeConfig,
    PreferenceJudgeError,
)
from nanogpt_nspire.secret_safety import assert_secret_free


FAKE_SECRET = "sk-" + "j" * 32


def judge_problem() -> JudgeProblem:
    return JudgeProblem(
        schedule_id="policy-1:update-01:slot-0",
        task="arithmetic",
        mode="direct",
        prompt="Calculate 12 * 7.",
        candidates=(
            JudgeCandidate(candidate_id="candidate-a", response="84"),
            JudgeCandidate(candidate_id="candidate-b", response="74"),
        ),
    )


def provider_response(
    *,
    content: object | None = None,
    reasoning_content: str = "private judge scratch work",
) -> bytes:
    if content is None:
        content = {
            "scores": [
                {"candidate_id": "candidate-a", "score": 4},
                {"candidate_id": "candidate-b", "score": 0},
            ],
            "preferred_candidate_id": "candidate-a",
            "rationale": "Candidate A is correct and concise.",
        }
    return json.dumps(
        {
            "id": "judge-request-123",
            "model": "deepseek-v4-pro",
            "choices": [
                {
                    "message": {
                        "content": json.dumps(content),
                        "reasoning_content": reasoning_content,
                        "role": "assistant",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 180,
                "completion_tokens": 50,
                "total_tokens": 230,
            },
        }
    ).encode("utf-8")


def test_plan_contains_candidates_but_never_local_ground_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    client = PreferenceJudgeClient(
        PreferenceJudgeConfig(max_requests=1),
    )

    plan = client.plan(judge_problem())
    serialized = json.dumps(plan, sort_keys=True)

    assert plan["endpoint"] == "https://api.deepseek.com/chat/completions"
    assert plan["body"]["model"] == "deepseek-v4-pro"
    assert plan["body"]["max_tokens"] == 4096
    assert "expected_answer" not in serialized
    assert "ground_truth" not in serialized
    assert "84" in serialized
    assert "74" in serialized
    assert "Authorization" not in serialized
    assert len(plan["request_sha256"]) == 64
    assert_secret_free(plan, context="preference judge plan")


def test_candidate_order_is_deterministic_and_not_input_order() -> None:
    first = PreferenceJudgeClient().plan(judge_problem())
    reversed_problem = JudgeProblem(
        schedule_id=judge_problem().schedule_id,
        task="arithmetic",
        mode="direct",
        prompt=judge_problem().prompt,
        candidates=tuple(reversed(judge_problem().candidates)),
    )
    second = PreferenceJudgeClient().plan(reversed_problem)

    assert first["body"] == second["body"]
    assert first["request_sha256"] == second["request_sha256"]


def test_live_judgment_returns_normalized_public_scores_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_SECRET)
    observed: dict[str, object] = {}

    def transport(
        url: str,
        headers: dict[str, str],
        body: bytes,
        timeout_seconds: float,
    ) -> bytes:
        observed.update(
            {
                "url": url,
                "headers": headers,
                "body": json.loads(body),
                "timeout": timeout_seconds,
            }
        )
        return provider_response()

    client = PreferenceJudgeClient(
        PreferenceJudgeConfig(
            max_requests=1,
            maximum_attempts=1,
        ),
        transport=transport,
    )
    answer = client.judge(judge_problem())

    assert observed["headers"] == {
        "Authorization": f"Bearer {FAKE_SECRET}",
        "Content-Type": "application/json",
    }
    assert answer.reward_by_candidate() == {
        "candidate-a": 1.0,
        "candidate-b": 0.0,
    }
    assert answer.preferred_candidate_id == "candidate-a"
    assert answer.transport_attempts == 1
    public = answer.public_record()
    serialized = json.dumps(public)
    assert "reasoning_content" not in serialized
    assert "private judge scratch work" not in serialized
    assert FAKE_SECRET not in serialized


@pytest.mark.parametrize(
    "content, message",
    [
        (
            {
                "scores": [
                    {"candidate_id": "candidate-a", "score": 4},
                    {"candidate_id": "candidate-b", "score": 0},
                ],
                "preferred_candidate_id": "candidate-b",
                "rationale": "Bad preference.",
            },
            "preferred",
        ),
        (
            {
                "scores": [
                    {"candidate_id": "candidate-a", "score": 5},
                    {"candidate_id": "candidate-b", "score": 0},
                ],
                "preferred_candidate_id": "candidate-a",
                "rationale": "Out of range.",
            },
            "score",
        ),
        (
            {
                "scores": [
                    {"candidate_id": "candidate-a", "score": 4},
                ],
                "preferred_candidate_id": "candidate-a",
                "rationale": "Missing candidate.",
            },
            "candidate",
        ),
    ],
)
def test_response_schema_and_preference_are_strict(
    monkeypatch: pytest.MonkeyPatch,
    content: object,
    message: str,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_SECRET)
    client = PreferenceJudgeClient(
        PreferenceJudgeConfig(
            max_requests=1,
            maximum_attempts=1,
        ),
        transport=lambda *_: provider_response(content=content),
    )
    with pytest.raises(PreferenceJudgeError, match=message):
        client.judge(judge_problem())


def test_score_object_is_strictly_normalized_to_public_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_SECRET)
    content = {
        "scores": {
            "candidate-a": 4,
            "candidate-b": 0,
        },
        "preferred_candidate_id": "candidate-a",
        "rationale": "A is correct.",
    }
    client = PreferenceJudgeClient(
        PreferenceJudgeConfig(
            max_requests=1,
            maximum_attempts=1,
        ),
        transport=lambda *_: provider_response(content=content),
    )

    answer = client.judge(judge_problem())

    assert answer.reward_by_candidate() == {
        "candidate-a": 1.0,
        "candidate-b": 0.0,
    }
    assert isinstance(answer.public_record()["scores"], list)


def test_empty_score_container_reports_only_safe_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_SECRET)
    content = {
        "scores": {},
        "preferred_candidate_id": "candidate-a",
        "rationale": "No scores.",
    }
    client = PreferenceJudgeClient(
        PreferenceJudgeConfig(
            max_requests=1,
            maximum_attempts=1,
        ),
        transport=lambda *_: provider_response(content=content),
    )

    with pytest.raises(
        PreferenceJudgeError,
        match="dict with 0 entries",
    ) as captured:
        client.judge(judge_problem())

    assert "candidate-a" not in str(captured.value)
    assert FAKE_SECRET not in str(captured.value)


def test_request_budget_is_exact_under_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_SECRET)
    client = PreferenceJudgeClient(
        PreferenceJudgeConfig(max_requests=3),
        transport=lambda *_: provider_response(),
    )

    def attempt(_: int) -> bool:
        try:
            client.judge(judge_problem())
        except PreferenceJudgeError as error:
            assert "request budget" in str(error)
            return False
        return True

    with ThreadPoolExecutor(max_workers=6) as executor:
        completed = list(executor.map(attempt, range(6)))

    assert sum(completed) == 3
    assert client.logical_requests == 3
