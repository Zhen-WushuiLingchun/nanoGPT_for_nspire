"""Crash-safe public cache and candidate rendering for direct RLAIF."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import threading
from typing import Mapping

from nanogpt_nspire.external_teacher import TokenUsage
from nanogpt_nspire.preference_judge import (
    CandidateScore,
    JudgeAnswer,
    JudgeProblem,
    PreferenceJudgeClient,
    PreferenceJudgeError,
)
from nanogpt_nspire.reasoning_format import THINK_MODE
from nanogpt_nspire.rl_rollout import RolloutTrajectory
from nanogpt_nspire.secret_safety import assert_secret_free


JUDGE_CACHE_SCHEMA_VERSION = 1
_CACHE_FIELDS = frozenset(
    {"answer", "request_sha256", "schema_version"}
)
_ANSWER_FIELDS = frozenset(
    {
        "preferred_candidate_id",
        "provider_model",
        "provider_request_id",
        "rationale",
        "request_sha256",
        "scores",
        "usage",
    }
)


class JudgeCacheError(RuntimeError):
    """Raised when a public judge cache is malformed or conflicting."""


def render_judge_response(trajectory: RolloutTrajectory) -> str:
    """Expose visible response segments and local format status to the judge."""

    if not isinstance(trajectory, RolloutTrajectory):
        raise JudgeCacheError("trajectory must be RolloutTrajectory")
    completion = trajectory.completion
    reasoning = completion.get("reasoning_text")
    final = completion.get("final_text")
    if not isinstance(reasoning, str) or not isinstance(final, str):
        raise JudgeCacheError("trajectory completion text is invalid")
    if trajectory.mode == THINK_MODE:
        response = f"<THINK>{reasoning}<FINAL>{final}"
    else:
        response = final
    if not response:
        response = "<EMPTY>"
    status = (
        "STATUS "
        f"terminated={str(bool(completion.get('terminated'))).lower()} "
        "special_token_leak="
        f"{str(bool(completion.get('special_token_leak'))).lower()} "
        "budget_exhausted="
        f"{str(bool(completion.get('budget_exhausted'))).lower()} "
        "context_exhausted="
        f"{str(bool(completion.get('context_exhausted'))).lower()}"
    )
    leaked = completion.get("leaked_token")
    if leaked is not None:
        if not isinstance(leaked, str):
            raise JudgeCacheError("leaked token diagnostic is invalid")
        status += f" leaked_token={leaked}"
    rendered = response + "\n" + status
    assert_secret_free(rendered, context="rendered judge response")
    return rendered


def _required_string(mapping: Mapping[str, object], name: str) -> str:
    value = mapping.get(name)
    if not isinstance(value, str) or not value:
        raise JudgeCacheError(f"cached {name} must be non-empty")
    return value


def _answer_from_public(value: object) -> JudgeAnswer:
    if not isinstance(value, Mapping) or frozenset(value) != _ANSWER_FIELDS:
        raise JudgeCacheError("cached judge answer schema is invalid")
    raw_scores = value.get("scores")
    if not isinstance(raw_scores, list):
        raise JudgeCacheError("cached judge scores must be a list")
    scores: list[CandidateScore] = []
    for item in raw_scores:
        if (
            not isinstance(item, Mapping)
            or frozenset(item) != frozenset({"candidate_id", "score"})
        ):
            raise JudgeCacheError("cached judge score schema is invalid")
        candidate_id = _required_string(item, "candidate_id")
        try:
            scores.append(
                CandidateScore(
                    candidate_id=candidate_id,
                    score=item.get("score"),  # type: ignore[arg-type]
                )
            )
        except PreferenceJudgeError as error:
            raise JudgeCacheError(str(error)) from None
    raw_usage = value.get("usage")
    if not isinstance(raw_usage, Mapping):
        raise JudgeCacheError("cached judge usage is invalid")
    usage_fields: dict[str, int] = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        item = raw_usage.get(name)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise JudgeCacheError(f"cached usage {name} is invalid")
        usage_fields[name] = item
    if usage_fields["total_tokens"] < (
        usage_fields["prompt_tokens"]
        + usage_fields["completion_tokens"]
    ):
        raise JudgeCacheError("cached usage total is inconsistent")
    try:
        return JudgeAnswer(
            scores=tuple(scores),
            preferred_candidate_id=_required_string(
                value,
                "preferred_candidate_id",
            ),
            rationale=_required_string(value, "rationale"),
            provider_request_id=_required_string(
                value,
                "provider_request_id",
            ),
            provider_model=_required_string(value, "provider_model"),
            usage=TokenUsage(**usage_fields),
            request_sha256=_required_string(
                value,
                "request_sha256",
            ),
        )
    except PreferenceJudgeError as error:
        raise JudgeCacheError(str(error)) from None


def _canonical_record(answer: JudgeAnswer) -> dict[str, object]:
    public = answer.public_record()
    record = {
        "answer": public,
        "request_sha256": answer.request_sha256,
        "schema_version": JUDGE_CACHE_SCHEMA_VERSION,
    }
    assert_secret_free(record, context="judge cache record")
    return record


def _canonical_bytes(record: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            record,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


class JudgeCache:
    """Append-only JSONL cache keyed by canonical provider request hash."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._answers: dict[str, JudgeAnswer] = {}
        self._records: dict[str, bytes] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        try:
            lines = self.path.read_bytes().splitlines()
        except OSError as error:
            raise JudgeCacheError("could not read judge cache") from error
        for line_number, line in enumerate(lines, start=1):
            if not line:
                continue
            try:
                assert_secret_free(line, context="judge cache line")
                raw = json.loads(line.decode("ascii", errors="strict"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise JudgeCacheError(
                    f"judge cache line {line_number} is invalid JSON"
                ) from error
            if not isinstance(raw, Mapping) or frozenset(raw) != _CACHE_FIELDS:
                raise JudgeCacheError(
                    f"judge cache line {line_number} schema is invalid"
                )
            if raw.get("schema_version") != JUDGE_CACHE_SCHEMA_VERSION:
                raise JudgeCacheError(
                    f"judge cache line {line_number} version is invalid"
                )
            request_sha256 = _required_string(raw, "request_sha256")
            answer = _answer_from_public(raw.get("answer"))
            if answer.request_sha256 != request_sha256:
                raise JudgeCacheError(
                    f"judge cache line {line_number} hash mismatch"
                )
            canonical = _canonical_bytes(_canonical_record(answer))
            prior = self._records.get(request_sha256)
            if prior is not None and prior != canonical:
                raise JudgeCacheError(
                    "judge cache has conflicting duplicate request"
                )
            self._records[request_sha256] = canonical
            self._answers[request_sha256] = answer

    def get(self, request_sha256: str) -> JudgeAnswer | None:
        with self._lock:
            return self._answers.get(request_sha256)

    def put(self, answer: JudgeAnswer) -> None:
        if not isinstance(answer, JudgeAnswer):
            raise JudgeCacheError("answer must be JudgeAnswer")
        record = _canonical_record(answer)
        payload = _canonical_bytes(record)
        request_sha256 = answer.request_sha256
        with self._lock:
            prior = self._records.get(request_sha256)
            if prior is not None:
                if prior != payload:
                    raise JudgeCacheError(
                        "judge cache has conflicting duplicate request"
                    )
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with self.path.open("ab") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            except OSError as error:
                raise JudgeCacheError(
                    "could not append judge cache"
                ) from error
            self._records[request_sha256] = payload
            self._answers[request_sha256] = answer


def judge_with_cache(
    *,
    client: PreferenceJudgeClient,
    problem: JudgeProblem,
    cache: JudgeCache,
) -> tuple[JudgeAnswer, bool]:
    """Return a cached answer or execute and durably append one live call."""

    if not isinstance(client, PreferenceJudgeClient):
        raise JudgeCacheError("client must be PreferenceJudgeClient")
    if not isinstance(cache, JudgeCache):
        raise JudgeCacheError("cache must be JudgeCache")
    plan = client.plan(problem)
    request_sha256 = str(plan["request_sha256"])
    cached = cache.get(request_sha256)
    if cached is not None:
        return cached, True
    answer = client.judge(problem)
    if answer.request_sha256 != request_sha256:
        raise JudgeCacheError("live judge request hash mismatch")
    cache.put(answer)
    return answer, False
