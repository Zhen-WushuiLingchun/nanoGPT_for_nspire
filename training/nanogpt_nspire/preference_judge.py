"""Credential-isolated direct-RLAIF judge for Lesson 17."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import threading
import time
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from nanogpt_nspire.external_teacher import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    ProviderTransportError,
    RETRYABLE_STATUS_CODES,
    TokenUsage,
)
from nanogpt_nspire.reasoning_format import SUPPORTED_MODES
from nanogpt_nspire.secret_safety import (
    CredentialSafetyError,
    assert_secret_free,
    get_deepseek_api_key,
    redact_text,
)


JUDGE_PROMPT_SCHEMA_VERSION = 1
_CONTENT_FIELDS = frozenset(
    {"scores", "preferred_candidate_id", "rationale"}
)
_SCORE_FIELDS = frozenset({"candidate_id", "score"})


class PreferenceJudgeError(RuntimeError):
    """Raised for bounded and sanitized direct-RLAIF failures."""


@dataclass(frozen=True)
class JudgeCandidate:
    candidate_id: str
    response: str

    def validate(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise PreferenceJudgeError("candidate_id must be non-empty")
        if not isinstance(self.response, str) or not self.response:
            raise PreferenceJudgeError("candidate response must be non-empty")
        assert_secret_free(asdict(self), context="judge candidate")


@dataclass(frozen=True)
class JudgeProblem:
    schedule_id: str
    task: str
    mode: str
    prompt: str
    candidates: tuple[JudgeCandidate, ...]

    def validate(self) -> None:
        for name in ("schedule_id", "prompt"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise PreferenceJudgeError(f"{name} must be non-empty")
        if self.task not in {"arithmetic", "physics_numeric"}:
            raise PreferenceJudgeError("judge task is unsupported")
        if self.mode not in SUPPORTED_MODES:
            raise PreferenceJudgeError("judge mode is unsupported")
        if len(self.candidates) < 2:
            raise PreferenceJudgeError(
                "judge problem requires at least two candidates"
            )
        for candidate in self.candidates:
            if not isinstance(candidate, JudgeCandidate):
                raise PreferenceJudgeError(
                    "all candidates must be JudgeCandidate"
                )
            candidate.validate()
        identifiers = [
            candidate.candidate_id for candidate in self.candidates
        ]
        if len(set(identifiers)) != len(identifiers):
            raise PreferenceJudgeError("candidate IDs must be unique")
        assert_secret_free(asdict(self), context="judge problem")


@dataclass(frozen=True)
class PreferenceJudgeConfig:
    """Public provider settings; no credential can be serialized here."""

    base_url: str = DEEPSEEK_BASE_URL
    model: str = DEEPSEEK_MODEL
    max_requests: int = 512
    max_tokens: int = 1024
    maximum_attempts: int = 3
    retry_delay_seconds: float = 1.0
    timeout_seconds: float = 90.0
    maximum_request_bytes: int = 65_536
    maximum_response_bytes: int = 256_000
    reasoning_effort: str = "high"

    def validate(self) -> None:
        if self.base_url != DEEPSEEK_BASE_URL:
            raise PreferenceJudgeError(
                f"base_url must be {DEEPSEEK_BASE_URL}"
            )
        if self.model not in {"deepseek-v4-pro", "deepseek-v4-flash"}:
            raise PreferenceJudgeError("unsupported current DeepSeek model")
        for name in (
            "max_requests",
            "max_tokens",
            "maximum_attempts",
            "maximum_request_bytes",
            "maximum_response_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise PreferenceJudgeError(f"{name} must be positive")
        for name in ("retry_delay_seconds", "timeout_seconds"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise PreferenceJudgeError(
                    f"{name} must be finite and non-negative"
                )
        if self.timeout_seconds == 0:
            raise PreferenceJudgeError("timeout_seconds must be positive")
        if self.reasoning_effort not in {"low", "medium", "high"}:
            raise PreferenceJudgeError("reasoning_effort is invalid")
        assert_secret_free(
            asdict(self),
            context="preference judge configuration",
        )

    def public_metadata(self) -> dict[str, object]:
        self.validate()
        return {
            **asdict(self),
            "credential_source": "DEEPSEEK_API_KEY",
            "prompt_schema_version": JUDGE_PROMPT_SCHEMA_VERSION,
            "response_format": {"type": "json_object"},
            "stream": False,
            "thinking": {"type": "enabled"},
        }


@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    score: int

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise PreferenceJudgeError("score candidate_id is invalid")
        if (
            isinstance(self.score, bool)
            or not isinstance(self.score, int)
            or not 0 <= self.score <= 4
        ):
            raise PreferenceJudgeError(
                "candidate score must be an integer in [0, 4]"
            )


@dataclass(frozen=True)
class JudgeAnswer:
    scores: tuple[CandidateScore, ...]
    preferred_candidate_id: str
    rationale: str
    provider_request_id: str
    provider_model: str
    usage: TokenUsage
    request_sha256: str

    def __post_init__(self) -> None:
        if len(self.scores) < 2:
            raise PreferenceJudgeError(
                "judge answer requires at least two scores"
            )
        if any(
            not isinstance(item, CandidateScore) for item in self.scores
        ):
            raise PreferenceJudgeError(
                "judge answer scores must be CandidateScore"
            )
        identifiers = [item.candidate_id for item in self.scores]
        if len(set(identifiers)) != len(identifiers):
            raise PreferenceJudgeError(
                "judge answer candidate IDs must be unique"
            )
        if self.preferred_candidate_id not in identifiers:
            raise PreferenceJudgeError(
                "judge answer preferred candidate is unknown"
            )
        maximum = max(item.score for item in self.scores)
        preferred_score = next(
            item.score
            for item in self.scores
            if item.candidate_id == self.preferred_candidate_id
        )
        if preferred_score != maximum:
            raise PreferenceJudgeError(
                "judge answer preferred candidate is not maximal"
            )
        for name in (
            "rationale",
            "provider_request_id",
            "provider_model",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise PreferenceJudgeError(
                    f"judge answer {name} must be non-empty"
                )
        if (
            not isinstance(self.request_sha256, str)
            or len(self.request_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.request_sha256
            )
        ):
            raise PreferenceJudgeError(
                "judge answer request_sha256 is invalid"
            )
        if not isinstance(self.usage, TokenUsage):
            raise PreferenceJudgeError(
                "judge answer usage must be TokenUsage"
            )

    def reward_by_candidate(self) -> dict[str, float]:
        return {
            item.candidate_id: item.score / 4.0
            for item in self.scores
        }

    def public_record(self) -> dict[str, object]:
        record = {
            "preferred_candidate_id": self.preferred_candidate_id,
            "provider_model": self.provider_model,
            "provider_request_id": self.provider_request_id,
            "rationale": self.rationale,
            "request_sha256": self.request_sha256,
            "scores": [asdict(item) for item in self.scores],
            "usage": asdict(self.usage),
        }
        assert_secret_free(record, context="judge answer record")
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


def _required_string(mapping: Mapping[str, object], name: str) -> str:
    value = mapping.get(name)
    if not isinstance(value, str) or not value.strip():
        raise PreferenceJudgeError(
            f"provider response {name} must be non-empty"
        )
    return value.strip()


def _parse_usage(value: object) -> TokenUsage:
    if not isinstance(value, Mapping):
        raise PreferenceJudgeError(
            "provider response usage must be an object"
        )
    parsed: dict[str, int] = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        item = value.get(name)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise PreferenceJudgeError(
                f"provider response usage {name} is invalid"
            )
        parsed[name] = item
    if parsed["total_tokens"] < (
        parsed["prompt_tokens"] + parsed["completion_tokens"]
    ):
        raise PreferenceJudgeError(
            "provider response token usage is inconsistent"
        )
    return TokenUsage(**parsed)


def ordered_candidates(
    problem: JudgeProblem,
) -> tuple[JudgeCandidate, ...]:
    """Return the stable request permutation used for position-bias audits."""

    problem.validate()
    return tuple(
        sorted(
            problem.candidates,
            key=lambda candidate: hashlib.sha256(
                (
                    f"{problem.schedule_id}:"
                    f"{candidate.candidate_id}"
                ).encode("utf-8")
            ).digest(),
        )
    )


def _request_body(
    config: PreferenceJudgeConfig,
    problem: JudgeProblem,
) -> dict[str, object]:
    problem.validate()
    payload = {
        "candidates": [
            {
                "candidate_id": item.candidate_id,
                "response": item.response,
            }
            for item in ordered_candidates(problem)
        ],
        "mode": problem.mode,
        "question": problem.prompt,
        "task": problem.task,
    }
    system = (
        "You are an independent English math and physics response judge. "
        "Solve the question yourself; no reference answer is supplied. Score "
        "every candidate from 0 to 4: 0 is invalid or entirely wrong, 1 has "
        "little correct content, 2 is partially correct, 3 is essentially "
        "correct with a minor issue, and 4 is correct, coherent, concise, and "
        "consistent between reasoning and final answer. Check arithmetic, "
        "units, unsupported claims, requested direct/think mode, and whether "
        "the final answer follows the reasoning. Do not reward verbosity. "
        "Return exactly one JSON object with scores, preferred_candidate_id, "
        "and rationale. scores must contain exactly one object with "
        "candidate_id and integer score for every supplied candidate. The "
        "preferred candidate must have a maximum score. Keep rationale under "
        "three short sentences. Do not return markdown or extra keys."
    )
    user = (
        "Judge this candidate group:\n"
        + json.dumps(
            payload,
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
    assert_secret_free(body, context="preference judge request")
    return body


def _parse_response(
    payload: bytes,
    *,
    expected_model: str,
    expected_candidate_ids: frozenset[str],
    request_sha256: str,
) -> JudgeAnswer:
    try:
        assert_secret_free(payload, context="preference judge response")
        raw = json.loads(payload.decode("utf-8", errors="strict"))
    except CredentialSafetyError as error:
        raise PreferenceJudgeError(str(error)) from None
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise PreferenceJudgeError(
            "provider response is not valid UTF-8 JSON"
        ) from None
    if not isinstance(raw, Mapping):
        raise PreferenceJudgeError("provider response must be an object")
    provider_request_id = _required_string(raw, "id")
    provider_model = _required_string(raw, "model")
    if provider_model != expected_model:
        raise PreferenceJudgeError("provider response model mismatch")
    choices = raw.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise PreferenceJudgeError(
            "provider response must contain exactly one choice"
        )
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise PreferenceJudgeError("provider response choice is invalid")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise PreferenceJudgeError("provider response message is invalid")
    try:
        content = json.loads(_required_string(message, "content"))
    except json.JSONDecodeError:
        raise PreferenceJudgeError(
            "provider response content is not valid JSON"
        ) from None
    if not isinstance(content, Mapping) or frozenset(content) != _CONTENT_FIELDS:
        raise PreferenceJudgeError(
            "provider response content schema is invalid"
        )
    raw_scores = content.get("scores")
    if not isinstance(raw_scores, list) or not raw_scores:
        raise PreferenceJudgeError("provider scores must be a non-empty list")
    scores: list[CandidateScore] = []
    for raw_score in raw_scores:
        if (
            not isinstance(raw_score, Mapping)
            or frozenset(raw_score) != _SCORE_FIELDS
        ):
            raise PreferenceJudgeError(
                "provider score schema is invalid"
            )
        candidate_id = _required_string(raw_score, "candidate_id")
        score = raw_score.get("score")
        scores.append(
            CandidateScore(candidate_id=candidate_id, score=score)  # type: ignore[arg-type]
        )
    returned_ids = [item.candidate_id for item in scores]
    if (
        len(set(returned_ids)) != len(returned_ids)
        or frozenset(returned_ids) != expected_candidate_ids
    ):
        raise PreferenceJudgeError(
            "provider candidate score set is invalid"
        )
    preferred = _required_string(content, "preferred_candidate_id")
    if preferred not in expected_candidate_ids:
        raise PreferenceJudgeError(
            "provider preferred candidate is unknown"
        )
    maximum = max(item.score for item in scores)
    if next(
        item.score for item in scores if item.candidate_id == preferred
    ) != maximum:
        raise PreferenceJudgeError(
            "provider preferred candidate does not have maximum score"
        )
    answer = JudgeAnswer(
        scores=tuple(scores),
        preferred_candidate_id=preferred,
        rationale=_required_string(content, "rationale"),
        provider_request_id=provider_request_id,
        provider_model=provider_model,
        usage=_parse_usage(raw.get("usage")),
        request_sha256=request_sha256,
    )
    answer.public_record()
    return answer


class PreferenceJudgeClient:
    """Bounded d-RLAIF planner and client with runtime-only credentials."""

    def __init__(
        self,
        config: PreferenceJudgeConfig | None = None,
        *,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config or PreferenceJudgeConfig()
        self.config.validate()
        self._transport = transport or _default_transport
        self._sleep = sleep
        self._logical_requests = 0
        self._transport_attempts = 0
        self._request_lock = threading.Lock()

    @property
    def logical_requests(self) -> int:
        with self._request_lock:
            return self._logical_requests

    @property
    def transport_attempts(self) -> int:
        with self._request_lock:
            return self._transport_attempts

    def plan(self, problem: JudgeProblem) -> dict[str, Any]:
        body = _request_body(self.config, problem)
        encoded = json.dumps(
            body,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if len(encoded) > self.config.maximum_request_bytes:
            raise PreferenceJudgeError(
                "preference judge request exceeds byte budget"
            )
        plan: dict[str, Any] = {
            "body": body,
            "endpoint": f"{self.config.base_url}/chat/completions",
            "request_bytes": len(encoded),
            "request_sha256": hashlib.sha256(encoded).hexdigest(),
            "schedule_id": problem.schedule_id,
        }
        assert_secret_free(plan, context="preference judge plan")
        return plan

    def judge(self, problem: JudgeProblem) -> JudgeAnswer:
        plan = self.plan(problem)
        body = json.dumps(
            plan["body"],
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        with self._request_lock:
            if self._logical_requests >= self.config.max_requests:
                raise PreferenceJudgeError(
                    "preference judge request budget exhausted"
                )
            self._logical_requests += 1
        api_key = get_deepseek_api_key()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        expected_ids = frozenset(
            item.candidate_id for item in problem.candidates
        )
        for attempt in range(1, self.config.maximum_attempts + 1):
            try:
                with self._request_lock:
                    self._transport_attempts += 1
                response = self._transport(
                    str(plan["endpoint"]),
                    headers,
                    body,
                    self.config.timeout_seconds,
                )
                if not isinstance(response, bytes):
                    raise PreferenceJudgeError(
                        "preference judge transport must return bytes"
                    )
                if len(response) > self.config.maximum_response_bytes:
                    raise PreferenceJudgeError(
                        "preference judge response exceeds byte budget"
                    )
                return _parse_response(
                    response,
                    expected_model=self.config.model,
                    expected_candidate_ids=expected_ids,
                    request_sha256=str(plan["request_sha256"]),
                )
            except ProviderTransportError as error:
                if (
                    error.status_code in RETRYABLE_STATUS_CODES
                    and attempt < self.config.maximum_attempts
                ):
                    self._sleep(self.config.retry_delay_seconds)
                    continue
                status = (
                    "unknown"
                    if error.status_code is None
                    else str(error.status_code)
                )
                raise PreferenceJudgeError(
                    f"preference judge transport failed with status {status}"
                ) from None
            except PreferenceJudgeError:
                if attempt < self.config.maximum_attempts:
                    self._sleep(self.config.retry_delay_seconds)
                    continue
                raise
            except BaseException as error:
                safe = redact_text(str(error))
                raise PreferenceJudgeError(
                    f"preference judge transport failed: {safe}"
                ) from None
        raise AssertionError("unreachable preference judge retry state")
