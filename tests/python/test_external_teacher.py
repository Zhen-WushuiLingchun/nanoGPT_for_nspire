from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json

import pytest

from nanogpt_nspire.external_teacher import (
    ExternalTeacherClient,
    ExternalTeacherConfig,
    ExternalTeacherError,
    ProviderTransportError,
    TeacherProblem,
)
from nanogpt_nspire.secret_safety import assert_secret_free


FAKE_SECRET = "sk-" + "y" * 32


def arithmetic_problem() -> TeacherProblem:
    return TeacherProblem(
        record_id="arith-demo",
        family_id="arith-family",
        task="arithmetic",
        prompt="Calculate 12 * 7.",
        expected_answer="84",
        expected_unit=None,
        formula=None,
    )


def provider_response(
    *,
    content: object | None = None,
    reasoning_content: str = "private scratch work",
) -> bytes:
    if content is None:
        content = {
            "answer_text": "Multiply 12 by 7. The answer is 84.",
            "final_answer": "84",
            "unit": None,
        }
    return json.dumps(
        {
            "id": "request-123",
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
                "prompt_tokens": 120,
                "completion_tokens": 40,
                "total_tokens": 160,
            },
        }
    ).encode("utf-8")


def test_config_freezes_current_provider_contract_without_a_key() -> None:
    config = ExternalTeacherConfig(max_requests=3)

    public = config.public_metadata()

    assert public["base_url"] == "https://api.deepseek.com"
    assert public["model"] == "deepseek-v4-pro"
    assert public["thinking"] == {"type": "enabled"}
    assert public["reasoning_effort"] == "high"
    assert public["max_requests"] == 3
    assert_secret_free(public, context="public provider config")


def test_plan_builds_json_request_without_reading_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    client = ExternalTeacherClient(ExternalTeacherConfig(max_requests=1))

    plan = client.plan(arithmetic_problem())

    assert plan["endpoint"] == "https://api.deepseek.com/chat/completions"
    assert plan["body"]["model"] == "deepseek-v4-pro"
    assert plan["body"]["response_format"] == {"type": "json_object"}
    assert plan["body"]["thinking"] == {"type": "enabled"}
    assert plan["body"]["reasoning_effort"] == "high"
    assert "Authorization" not in json.dumps(plan)
    assert_secret_free(plan, context="request plan")


def test_live_call_uses_runtime_key_but_returns_only_public_provenance(
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

    client = ExternalTeacherClient(
        ExternalTeacherConfig(
            max_requests=1,
            maximum_attempts=1,
        ),
        transport=transport,
    )
    answer = client.generate(arithmetic_problem())

    assert observed["headers"] == {
        "Authorization": f"Bearer {FAKE_SECRET}",
        "Content-Type": "application/json",
    }
    assert answer.answer_text.endswith("84.")
    assert answer.final_answer == "84"
    assert answer.unit is None
    assert answer.provider_request_id == "request-123"
    assert answer.usage.total_tokens == 160
    serialized = answer.public_record()
    assert "reasoning_content" not in json.dumps(serialized)
    assert "private scratch work" not in json.dumps(serialized)
    assert FAKE_SECRET not in json.dumps(serialized)


def test_response_schema_is_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_SECRET)

    def transport(*_: object) -> bytes:
        return provider_response(
            content={
                "answer_text": "The answer is 84.",
                "final_answer": "84",
                "unit": None,
                "unexpected": True,
            }
        )

    client = ExternalTeacherClient(
        ExternalTeacherConfig(max_requests=1),
        transport=transport,
    )
    with pytest.raises(ExternalTeacherError, match="response content schema"):
        client.generate(arithmetic_problem())


def test_empty_or_invalid_final_content_is_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_SECRET)
    attempts = 0

    def transport(*_: object) -> bytes:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return json.dumps(
                {
                    "id": "request-empty",
                    "model": "deepseek-v4-pro",
                    "choices": [{"message": {"content": ""}}],
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 1024,
                        "total_tokens": 1044,
                    },
                }
            ).encode("utf-8")
        return provider_response()

    client = ExternalTeacherClient(
        ExternalTeacherConfig(
            max_requests=1,
            maximum_attempts=2,
            retry_delay_seconds=0.0,
        ),
        transport=transport,
    )

    assert client.generate(arithmetic_problem()).final_answer == "84"
    assert attempts == 2
    assert client.logical_requests == 1
    assert client.transport_attempts == 2


def test_retryable_transport_error_retries_without_exposing_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_SECRET)
    attempts = 0

    def transport(*_: object) -> bytes:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ProviderTransportError(
                status_code=503,
                detail=f"temporary failure {FAKE_SECRET}",
            )
        return provider_response()

    client = ExternalTeacherClient(
        ExternalTeacherConfig(
            max_requests=1,
            maximum_attempts=2,
            retry_delay_seconds=0.0,
        ),
        transport=transport,
    )
    assert client.generate(arithmetic_problem()).final_answer == "84"
    assert attempts == 2
    assert client.transport_attempts == 2


def test_non_retryable_provider_error_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_SECRET)

    def transport(*_: object) -> bytes:
        raise ProviderTransportError(
            status_code=401,
            detail=f"Authorization: Bearer {FAKE_SECRET}",
        )

    client = ExternalTeacherClient(
        ExternalTeacherConfig(max_requests=1),
        transport=transport,
    )
    with pytest.raises(ExternalTeacherError) as captured:
        client.generate(arithmetic_problem())

    assert FAKE_SECRET not in str(captured.value)
    assert "401" in str(captured.value)


def test_request_budget_is_enforced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_SECRET)
    client = ExternalTeacherClient(
        ExternalTeacherConfig(max_requests=1),
        transport=lambda *_: provider_response(),
    )

    client.generate(arithmetic_problem())
    with pytest.raises(ExternalTeacherError, match="request budget"):
        client.generate(arithmetic_problem())


def test_request_budget_remains_exact_under_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_SECRET)
    client = ExternalTeacherClient(
        ExternalTeacherConfig(max_requests=4),
        transport=lambda *_: provider_response(),
    )

    def attempt(_: int) -> bool:
        try:
            client.generate(arithmetic_problem())
        except ExternalTeacherError as error:
            assert "request budget" in str(error)
            return False
        return True

    with ThreadPoolExecutor(max_workers=8) as executor:
        completed = list(executor.map(attempt, range(8)))

    assert sum(completed) == 4
    assert client.logical_requests == 4
