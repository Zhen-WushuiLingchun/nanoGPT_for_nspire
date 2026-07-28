"""Credential-isolated DeepSeek sequence-teacher client."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
import json
import math
import threading
import time
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from nanogpt_nspire.secret_safety import (
    CredentialSafetyError,
    assert_secret_free,
    get_deepseek_api_key,
    redact_text,
)


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-pro"
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_CONTENT_FIELDS = frozenset({"answer_text", "final_answer", "unit"})


class ExternalTeacherError(RuntimeError):
    """Raised for bounded, sanitized external-teacher failures."""


class ProviderTransportError(RuntimeError):
    """One HTTP-like failure whose detail must never be reflected verbatim."""

    def __init__(self, *, status_code: int | None, detail: str) -> None:
        super().__init__("provider transport failure")
        self.status_code = status_code
        self.detail = detail

    @property
    def retryable(self) -> bool:
        return self.status_code in RETRYABLE_STATUS_CODES


@dataclass(frozen=True)
class TeacherProblem:
    """One independently solved problem offered to the sequence teacher."""

    record_id: str
    family_id: str
    task: str
    prompt: str
    expected_answer: str
    expected_unit: str | None
    formula: str | None

    def validate(self) -> None:
        for field in (
            "record_id",
            "family_id",
            "prompt",
            "expected_answer",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ExternalTeacherError(f"{field} must be non-empty")
        if self.task not in {"arithmetic", "physics_numeric"}:
            raise ExternalTeacherError(
                "task must be 'arithmetic' or 'physics_numeric'"
            )
        if self.task == "arithmetic":
            if self.expected_unit is not None or self.formula is not None:
                raise ExternalTeacherError(
                    "arithmetic problem must not declare unit or formula"
                )
        else:
            if (
                not isinstance(self.expected_unit, str)
                or not self.expected_unit
                or not isinstance(self.formula, str)
                or not self.formula
            ):
                raise ExternalTeacherError(
                    "physics problem requires unit and formula"
                )
        assert_secret_free(asdict(self), context="teacher problem")


@dataclass(frozen=True)
class ExternalTeacherConfig:
    """Public provider settings; credentials are deliberately absent."""

    base_url: str = DEEPSEEK_BASE_URL
    model: str = DEEPSEEK_MODEL
    max_requests: int = 512
    max_tokens: int = 1024
    maximum_attempts: int = 3
    retry_delay_seconds: float = 1.0
    timeout_seconds: float = 90.0
    maximum_request_bytes: int = 16_384
    maximum_response_bytes: int = 256_000
    reasoning_effort: str = "high"

    def validate(self) -> None:
        if self.base_url != DEEPSEEK_BASE_URL:
            raise ExternalTeacherError(
                f"base_url must be {DEEPSEEK_BASE_URL}"
            )
        if self.model not in {"deepseek-v4-pro", "deepseek-v4-flash"}:
            raise ExternalTeacherError("unsupported current DeepSeek model")
        for field in (
            "max_requests",
            "max_tokens",
            "maximum_attempts",
            "maximum_request_bytes",
            "maximum_response_bytes",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ExternalTeacherError(f"{field} must be positive")
        for field in ("retry_delay_seconds", "timeout_seconds"):
            value = getattr(self, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ExternalTeacherError(
                    f"{field} must be finite and non-negative"
                )
        if self.timeout_seconds == 0:
            raise ExternalTeacherError("timeout_seconds must be positive")
        if self.reasoning_effort not in {"low", "medium", "high"}:
            raise ExternalTeacherError("reasoning_effort is invalid")
        assert_secret_free(
            asdict(self),
            context="external teacher configuration",
        )

    def public_metadata(self) -> dict[str, object]:
        self.validate()
        return {
            **asdict(self),
            "credential_source": "DEEPSEEK_API_KEY",
            "response_format": {"type": "json_object"},
            "stream": False,
            "thinking": {"type": "enabled"},
        }


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class TeacherAnswer:
    """Only the final public sequence plus provider provenance."""

    answer_text: str
    final_answer: str
    unit: str | None
    provider_request_id: str
    provider_model: str
    usage: TokenUsage

    def public_record(self) -> dict[str, object]:
        record = {
            "answer_text": self.answer_text,
            "final_answer": self.final_answer,
            "provider_model": self.provider_model,
            "provider_request_id": self.provider_request_id,
            "unit": self.unit,
            "usage": asdict(self.usage),
        }
        assert_secret_free(record, context="teacher answer record")
        return record


Transport = Callable[[str, dict[str, str], bytes, float], bytes]


def _default_transport(
    url: str,
    headers: dict[str, str],
    body: bytes,
    timeout_seconds: float,
) -> bytes:
    request = urllib_request.Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib_request.urlopen(
            request,
            timeout=timeout_seconds,
        ) as response:
            return response.read()
    except urllib_error.HTTPError as error:
        raise ProviderTransportError(
            status_code=error.code,
            detail="provider returned an HTTP error",
        ) from None
    except urllib_error.URLError as error:
        raise ProviderTransportError(
            status_code=None,
            detail=redact_text(str(error.reason)),
        ) from None
    except OSError as error:
        raise ProviderTransportError(
            status_code=None,
            detail=redact_text(str(error)),
        ) from None


def _required_string(mapping: Mapping[str, object], field: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ExternalTeacherError(
            f"provider response {field} must be non-empty"
        )
    return value.strip()


def _usage_from_response(value: object) -> TokenUsage:
    if not isinstance(value, Mapping):
        raise ExternalTeacherError("provider response usage must be an object")
    fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    parsed: dict[str, int] = {}
    for field in fields:
        item = value.get(field)
        if (
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < 0
        ):
            raise ExternalTeacherError(
                f"provider response usage {field} is invalid"
            )
        parsed[field] = item
    if parsed["total_tokens"] < (
        parsed["prompt_tokens"] + parsed["completion_tokens"]
    ):
        raise ExternalTeacherError(
            "provider response total token count is inconsistent"
        )
    return TokenUsage(**parsed)


def _parse_provider_response(
    payload: bytes,
    *,
    expected_model: str,
) -> TeacherAnswer:
    try:
        assert_secret_free(payload, context="provider response")
        raw = json.loads(payload.decode("utf-8", errors="strict"))
    except CredentialSafetyError as error:
        raise ExternalTeacherError(str(error)) from None
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ExternalTeacherError(
            "provider response is not valid UTF-8 JSON"
        ) from None
    if not isinstance(raw, Mapping):
        raise ExternalTeacherError("provider response must be an object")
    provider_request_id = _required_string(raw, "id")
    provider_model = _required_string(raw, "model")
    if provider_model != expected_model:
        raise ExternalTeacherError("provider response model mismatch")
    choices = raw.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ExternalTeacherError(
            "provider response must contain exactly one choice"
        )
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise ExternalTeacherError("provider response choice is invalid")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise ExternalTeacherError("provider response message is invalid")
    content_text = _required_string(message, "content")
    try:
        content = json.loads(content_text)
    except json.JSONDecodeError:
        raise ExternalTeacherError(
            "provider response content is not valid JSON"
        ) from None
    if not isinstance(content, Mapping) or frozenset(content) != _CONTENT_FIELDS:
        raise ExternalTeacherError("provider response content schema is invalid")
    answer_text = _required_string(content, "answer_text")
    final_answer = _required_string(content, "final_answer")
    unit = content.get("unit")
    if unit is not None and (
        not isinstance(unit, str) or not unit.strip()
    ):
        raise ExternalTeacherError("provider response unit is invalid")
    answer = TeacherAnswer(
        answer_text=answer_text,
        final_answer=final_answer,
        unit=None if unit is None else unit.strip(),
        provider_request_id=provider_request_id,
        provider_model=provider_model,
        usage=_usage_from_response(raw.get("usage")),
    )
    answer.public_record()
    return answer


def _request_body(
    config: ExternalTeacherConfig,
    problem: TeacherProblem,
) -> dict[str, object]:
    problem.validate()
    problem_payload = {
        "expected_final_answer": problem.expected_answer,
        "expected_unit": problem.expected_unit,
        "formula": problem.formula,
        "question": problem.prompt,
        "task": problem.task,
    }
    system = (
        "You create concise English supervision for a tiny student model. "
        "Use the supplied independently verified ground truth. Explain the "
        "essential calculation in at most three short sentences. Return one "
        "JSON object with exactly answer_text, final_answer, and unit. "
        "Do not include role labels, markdown, hidden reasoning, or extra keys."
    )
    user = (
        "Produce the verified teaching answer for this JSON problem:\n"
        + json.dumps(
            problem_payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    body: dict[str, object] = {
        "max_tokens": config.max_tokens,
        "messages": [
            {"content": system, "role": "system"},
            {"content": user, "role": "user"},
        ],
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "response_format": {"type": "json_object"},
        "stream": False,
        "thinking": {"type": "enabled"},
    }
    assert_secret_free(body, context="provider request body")
    return body


class ExternalTeacherClient:
    """Bounded request planner and live client with runtime-only credentials."""

    def __init__(
        self,
        config: ExternalTeacherConfig | None = None,
        *,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config or ExternalTeacherConfig()
        self.config.validate()
        self._transport = transport or _default_transport
        self._sleep = sleep
        self._logical_requests = 0
        self._request_lock = threading.Lock()

    @property
    def logical_requests(self) -> int:
        with self._request_lock:
            return self._logical_requests

    def plan(self, problem: TeacherProblem) -> dict[str, object]:
        body = _request_body(self.config, problem)
        encoded = json.dumps(
            body,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if len(encoded) > self.config.maximum_request_bytes:
            raise ExternalTeacherError("provider request exceeds byte budget")
        plan = {
            "body": body,
            "endpoint": f"{self.config.base_url}/chat/completions",
            "family_id": problem.family_id,
            "record_id": problem.record_id,
            "request_bytes": len(encoded),
        }
        assert_secret_free(plan, context="provider request plan")
        return plan

    def generate(self, problem: TeacherProblem) -> TeacherAnswer:
        plan = self.plan(problem)
        body = json.dumps(
            plan["body"],
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        with self._request_lock:
            if self._logical_requests >= self.config.max_requests:
                raise ExternalTeacherError(
                    "external teacher request budget exhausted"
                )
            self._logical_requests += 1
        api_key = get_deepseek_api_key()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        for attempt in range(1, self.config.maximum_attempts + 1):
            try:
                response = self._transport(
                    str(plan["endpoint"]),
                    headers,
                    body,
                    self.config.timeout_seconds,
                )
                if not isinstance(response, bytes):
                    raise ExternalTeacherError(
                        "provider transport must return bytes"
                    )
                if len(response) > self.config.maximum_response_bytes:
                    raise ExternalTeacherError(
                        "provider response exceeds byte budget"
                    )
                return _parse_provider_response(
                    response,
                    expected_model=self.config.model,
                )
            except ProviderTransportError as error:
                if (
                    error.retryable
                    and attempt < self.config.maximum_attempts
                ):
                    self._sleep(self.config.retry_delay_seconds)
                    continue
                status = (
                    "unknown"
                    if error.status_code is None
                    else str(error.status_code)
                )
                raise ExternalTeacherError(
                    f"provider transport failed with status {status}"
                ) from None
            except ExternalTeacherError:
                raise
            except BaseException as error:
                safe = redact_text(str(error))
                raise ExternalTeacherError(
                    f"provider transport failed: {safe}"
                ) from None
        raise AssertionError("unreachable external teacher retry state")
