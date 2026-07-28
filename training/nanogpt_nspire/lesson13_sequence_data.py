"""Verified external-teacher problems, answers, plans, and packed shards."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from nanogpt_nspire.base_corpus import (
    CorpusRecord,
    build_corpus,
    stable_family_split,
)
from nanogpt_nspire.base_train import load_packed_dataset
from nanogpt_nspire.byte_tokenizer import (
    SPECIAL_TOKEN_NAMES,
    TOKENIZER_SCHEMA_VERSION,
    VOCAB_SIZE,
    ByteTokenizerError,
    ConversationTurn,
    format_conversation,
)
from nanogpt_nspire.external_teacher import (
    ExternalTeacherClient,
    ExternalTeacherConfig,
    ExternalTeacherError,
    TeacherAnswer,
    TeacherProblem,
)
from nanogpt_nspire.lesson12_curriculum import (
    PhysicsExample,
    canonical_jsonl_bytes,
    generate_physics_examples,
)
from nanogpt_nspire.math_curriculum import (
    ArithmeticExample,
    generate_arithmetic_examples,
)
from nanogpt_nspire.secret_safety import assert_secret_free


LESSON13_SEQUENCE_SPLIT_SEED = "lesson12-domain-v1"
SEQUENCE_SOURCE_ID = "deepseek-v4-pro-generated"
SEQUENCE_LICENSE_ID = "DeepSeek-Output-Terms-2026-03-27"
MAX_CONVERSATION_TOKENS = 256


class SequenceTeacherDataError(ValueError):
    """Raised when teacher sequences are unsafe, unverifiable, or unstable."""


@dataclass(frozen=True)
class TeacherVerification:
    problem: TeacherProblem
    answer: TeacherAnswer
    accepted: bool
    reason: str

    def public_record(self) -> dict[str, object]:
        record = {
            "accepted": self.accepted,
            "answer": self.answer.public_record(),
            "problem": asdict(self.problem),
            "reason": self.reason,
        }
        assert_secret_free(record, context="teacher verification record")
        return record


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_metadata(payload: bytes) -> dict[str, object]:
    return {"bytes": len(payload), "sha256": _sha256_bytes(payload)}


def _stable_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _publish_artifact(
    output_dir: Path,
    files: Mapping[str, bytes],
) -> None:
    if output_dir.exists():
        raise SequenceTeacherDataError(
            f"output destination already exists: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=output_dir.parent,
        )
    )
    try:
        for name, payload in files.items():
            path = temporary / name
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_bytes(path, payload)
        if output_dir.exists():
            raise SequenceTeacherDataError(
                f"output destination appeared during build: {output_dir}"
            )
        os.replace(temporary, output_dir)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _evaluation_families(path: str | Path | None) -> frozenset[str]:
    if path is None:
        return frozenset()
    source = Path(path)
    families: set[str] = set()
    try:
        lines = source.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as error:
        raise SequenceTeacherDataError(
            f"could not read evaluation file: {source}"
        ) from error
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise SequenceTeacherDataError(
                f"evaluation line {line_number} is empty"
            )
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise SequenceTeacherDataError(
                f"evaluation line {line_number} is invalid JSON"
            ) from error
        if not isinstance(row, Mapping):
            raise SequenceTeacherDataError(
                f"evaluation line {line_number} must be an object"
            )
        family = row.get("family_id")
        if not isinstance(family, str) or not family:
            raise SequenceTeacherDataError(
                f"evaluation line {line_number} has no family_id"
            )
        if family in families:
            raise SequenceTeacherDataError(
                f"duplicate evaluation family {family}"
            )
        families.add(family)
    if not families:
        raise SequenceTeacherDataError("evaluation file contains no families")
    return frozenset(families)


def _rank_problem(problem: TeacherProblem) -> tuple[bytes, str]:
    digest = hashlib.sha256(
        f"lesson13-sequence:{problem.task}:{problem.family_id}".encode("utf-8")
    ).digest()
    return digest, problem.family_id


def make_teacher_problems(
    *,
    arithmetic: Iterable[ArithmeticExample],
    physics: Iterable[PhysicsExample],
    evaluation_path: str | Path | None,
    max_per_task: int,
) -> tuple[TeacherProblem, ...]:
    """Select deterministic training-only families absent from frozen eval."""

    if (
        isinstance(max_per_task, bool)
        or not isinstance(max_per_task, int)
        or max_per_task <= 0
    ):
        raise SequenceTeacherDataError("max_per_task must be positive")
    arithmetic_examples = tuple(arithmetic)
    physics_examples = tuple(physics)
    if not arithmetic_examples or any(
        not isinstance(item, ArithmeticExample)
        for item in arithmetic_examples
    ):
        raise SequenceTeacherDataError(
            "arithmetic must contain ArithmeticExample values"
        )
    if not physics_examples or any(
        not isinstance(item, PhysicsExample) for item in physics_examples
    ):
        raise SequenceTeacherDataError(
            "physics must contain PhysicsExample values"
        )
    frozen_evaluation = _evaluation_families(evaluation_path)

    arithmetic_problems = [
        TeacherProblem(
            record_id=f"{item.example_id}-deepseek-v4-pro",
            family_id=item.family_id,
            task="arithmetic",
            prompt=item.question,
            expected_answer=item.exact_answer,
            expected_unit=None,
            formula=None,
        )
        for item in arithmetic_examples
        if stable_family_split(
            item.family_id,
            split_seed=LESSON13_SEQUENCE_SPLIT_SEED,
        )
        == "train"
        and item.family_id not in frozen_evaluation
    ]
    physics_problems = [
        TeacherProblem(
            record_id=f"{item.record_id}-deepseek-v4-pro",
            family_id=item.family_id,
            task="physics_numeric",
            prompt=item.question,
            expected_answer=item.exact_answer,
            expected_unit=item.unit,
            formula=item.formula,
        )
        for item in physics_examples
        if stable_family_split(
            item.family_id,
            split_seed=LESSON13_SEQUENCE_SPLIT_SEED,
        )
        == "train"
        and item.family_id not in frozen_evaluation
    ]
    selected_arithmetic = sorted(
        arithmetic_problems,
        key=_rank_problem,
    )[:max_per_task]
    selected_physics = sorted(
        physics_problems,
        key=_rank_problem,
    )[:max_per_task]
    if (
        len(selected_arithmetic) != max_per_task
        or len(selected_physics) != max_per_task
    ):
        raise SequenceTeacherDataError(
            "not enough eligible training families for teacher plan"
        )
    selected = tuple(selected_arithmetic + selected_physics)
    families = [item.family_id for item in selected]
    if len(set(families)) != len(families):
        raise SequenceTeacherDataError(
            "teacher problem families must be unique"
        )
    for problem in selected:
        problem.validate()
    return selected


def _parse_decimal(text: str) -> Decimal:
    try:
        value = Decimal(text.strip())
    except (InvalidOperation, ValueError):
        raise SequenceTeacherDataError("answer is not a finite decimal") from None
    if not value.is_finite():
        raise SequenceTeacherDataError("answer is not a finite decimal")
    return value


def verify_teacher_answer(
    problem: TeacherProblem,
    answer: TeacherAnswer,
) -> TeacherVerification:
    """Apply deterministic numeric, unit, role, language, and length gates."""

    problem.validate()
    if not isinstance(answer, TeacherAnswer):
        raise SequenceTeacherDataError("answer must be TeacherAnswer")
    try:
        expected = _parse_decimal(problem.expected_answer)
        actual = _parse_decimal(answer.final_answer)
    except SequenceTeacherDataError:
        return TeacherVerification(
            problem=problem,
            answer=answer,
            accepted=False,
            reason="invalid_final_answer",
        )
    if actual != expected:
        return TeacherVerification(
            problem=problem,
            answer=answer,
            accepted=False,
            reason="final_answer_mismatch",
        )
    if problem.task == "arithmetic":
        if answer.unit is not None:
            return TeacherVerification(
                problem=problem,
                answer=answer,
                accepted=False,
                reason="unexpected_unit",
            )
    elif answer.unit != problem.expected_unit:
        return TeacherVerification(
            problem=problem,
            answer=answer,
            accepted=False,
            reason="unit_mismatch",
        )

    text = " ".join(answer.answer_text.split())
    if not text or not text.isascii():
        return TeacherVerification(
            problem=problem,
            answer=answer,
            accepted=False,
            reason="non_ascii_or_empty_answer",
        )
    upper = text.upper()
    if any(name.upper() in upper for name in SPECIAL_TOKEN_NAMES.values()):
        return TeacherVerification(
            problem=problem,
            answer=answer,
            accepted=False,
            reason="role_token_leak",
        )
    if answer.final_answer not in text:
        return TeacherVerification(
            problem=problem,
            answer=answer,
            accepted=False,
            reason="final_answer_missing_from_text",
        )
    if problem.expected_unit is not None and problem.expected_unit not in text:
        return TeacherVerification(
            problem=problem,
            answer=answer,
            accepted=False,
            reason="unit_missing_from_text",
        )
    try:
        format_conversation(
            (
                ConversationTurn("user", problem.prompt),
                ConversationTurn("assistant", text),
            ),
            context_limit=MAX_CONVERSATION_TOKENS,
        )
    except ByteTokenizerError:
        return TeacherVerification(
            problem=problem,
            answer=answer,
            accepted=False,
            reason="conversation_exceeds_context_or_is_invalid",
        )
    normalized_answer = TeacherAnswer(
        answer_text=text,
        final_answer=answer.final_answer.strip(),
        unit=None if answer.unit is None else answer.unit.strip(),
        provider_request_id=answer.provider_request_id,
        provider_model=answer.provider_model,
        usage=answer.usage,
    )
    return TeacherVerification(
        problem=problem,
        answer=normalized_answer,
        accepted=True,
        reason="verified",
    )


def sequence_record(verification: TeacherVerification) -> CorpusRecord:
    """Convert one accepted final sequence to role-aware training data."""

    if not isinstance(verification, TeacherVerification):
        raise SequenceTeacherDataError(
            "verification must be TeacherVerification"
        )
    if not verification.accepted:
        raise SequenceTeacherDataError(
            "only verified teacher answers can become corpus records"
        )
    return CorpusRecord.conversation(
        record_id=verification.problem.record_id,
        family_id=verification.problem.family_id,
        turns=(
            ConversationTurn("user", verification.problem.prompt),
            ConversationTurn(
                "assistant",
                verification.answer.answer_text,
            ),
        ),
        source_id=SEQUENCE_SOURCE_ID,
        license_id=SEQUENCE_LICENSE_ID,
    )


def build_request_plan_artifact(
    problems: Sequence[TeacherProblem],
    output_dir: str | Path,
    *,
    client: ExternalTeacherClient,
) -> dict[str, object]:
    """Write a deterministic no-key dry-run plan for later paid calls."""

    materialized = tuple(problems)
    if not materialized:
        raise SequenceTeacherDataError("request plan requires problems")
    if len(materialized) > client.config.max_requests:
        raise SequenceTeacherDataError(
            "problem count exceeds configured request budget"
        )
    plans = tuple(client.plan(problem) for problem in materialized)
    plan_payload = canonical_jsonl_bytes(plans)
    task_counts = Counter(problem.task for problem in materialized)
    manifest: dict[str, object] = {
        "files": {
            "request-plan.jsonl": _file_metadata(plan_payload),
        },
        "problems": {
            "families": len({problem.family_id for problem in materialized}),
            "tasks": dict(sorted(task_counts.items())),
            "total": len(materialized),
        },
        "provider": client.config.public_metadata(),
        "schema_version": 1,
        "status": "dry_run_no_network",
    }
    assert_secret_free(manifest, context="request plan manifest")
    manifest_payload = _stable_json_bytes(manifest)
    _publish_artifact(
        Path(output_dir),
        {
            "manifest.json": manifest_payload,
            "request-plan.jsonl": plan_payload,
        },
    )
    return manifest


def _validated_train_payloads(
    sequence_corpus_dir: Path,
) -> tuple[bytes, bytes, dict[str, object]]:
    manifest_path = sequence_corpus_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SequenceTeacherDataError(
            "sequence corpus manifest is invalid"
        ) from error
    if not isinstance(manifest, dict):
        raise SequenceTeacherDataError(
            "sequence corpus manifest must be an object"
        )
    files = manifest.get("files")
    tokens = manifest.get("tokens")
    tokenizer = manifest.get("tokenizer")
    if (
        not isinstance(files, dict)
        or not isinstance(tokens, dict)
        or not isinstance(tokenizer, dict)
        or tokenizer.get("vocab_size") != VOCAB_SIZE
    ):
        raise SequenceTeacherDataError(
            "sequence corpus manifest contract is invalid"
        )
    payloads: list[bytes] = []
    for filename in ("train.tokens.bin", "train.loss.bin"):
        metadata = files.get(filename)
        path = sequence_corpus_dir / filename
        if not isinstance(metadata, dict) or not path.is_file():
            raise SequenceTeacherDataError(
                f"sequence corpus is missing {filename}"
            )
        payload = path.read_bytes()
        if (
            len(payload) != metadata.get("bytes")
            or _sha256_bytes(payload) != metadata.get("sha256")
        ):
            raise SequenceTeacherDataError(
                f"sequence corpus {filename} hash mismatch"
            )
        payloads.append(payload)
    if not payloads[0] or len(payloads[0]) != 2 * len(payloads[1]):
        raise SequenceTeacherDataError(
            "sequence training token/mask bytes are invalid"
        )
    if tokens.get("train") != len(payloads[1]):
        raise SequenceTeacherDataError(
            "sequence training token count mismatch"
        )
    return payloads[0], payloads[1], manifest


def assemble_fixed_evaluation_sft(
    *,
    sequence_corpus_dir: str | Path,
    reference_sft_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """Replace only SFT train data while preserving reference val/test bytes."""

    sequence_root = Path(sequence_corpus_dir)
    reference = load_packed_dataset(reference_sft_dir)
    train_tokens, train_mask, sequence_manifest = _validated_train_payloads(
        sequence_root
    )
    payloads: dict[str, bytes] = {
        "train.tokens.bin": train_tokens,
        "train.loss.bin": train_mask,
        "validation.tokens.bin": (
            reference.validation.token_path.read_bytes()
        ),
        "validation.loss.bin": reference.validation.mask_path.read_bytes(),
        "test.tokens.bin": reference.test.token_path.read_bytes(),
        "test.loss.bin": reference.test.mask_path.read_bytes(),
    }
    token_counts = {
        split: len(payloads[f"{split}.loss.bin"])
        for split in ("train", "validation", "test")
    }
    eligible_targets = {
        split: sum(payloads[f"{split}.loss.bin"])
        for split in ("train", "validation", "test")
    }
    manifest: dict[str, object] = {
        "comparison_contract": {
            "test_split": "byte_identical_to_reference_sft",
            "training_split": "verified_external_sequences",
            "validation_split": "byte_identical_to_reference_sft",
        },
        "eligible_targets": {
            **eligible_targets,
            "total": sum(eligible_targets.values()),
        },
        "files": {
            filename: _file_metadata(payload)
            for filename, payload in sorted(payloads.items())
        },
        "schema_version": 1,
        "sources": {
            "reference_sft_manifest_sha256": _sha256_file(
                reference.manifest_path
            ),
            "sequence_manifest_sha256": _sha256_file(
                sequence_root / "manifest.json"
            ),
            "sequence_records": sequence_manifest.get("records"),
        },
        "split": {
            "kind": "external-train-reference-evaluation-v1",
            "seed": LESSON13_SEQUENCE_SPLIT_SEED,
        },
        "tokenizer": dict(reference.manifest["tokenizer"]),
        "tokens": {
            **token_counts,
            "total": sum(token_counts.values()),
        },
    }
    assert_secret_free(manifest, context="assembled sequence SFT manifest")
    files = {
        **payloads,
        "manifest.json": _stable_json_bytes(manifest),
    }
    _publish_artifact(Path(output_dir), files)
    load_packed_dataset(output_dir)
    return manifest


def build_sequence_teacher_artifact(
    problems: Sequence[TeacherProblem],
    output_dir: str | Path,
    *,
    client: ExternalTeacherClient,
    registry_path: str | Path,
    reference_sft_dir: str | Path,
) -> dict[str, object]:
    """Call the provider, verify outputs, and publish train-only sequences."""

    materialized = tuple(problems)
    if not materialized:
        raise SequenceTeacherDataError(
            "live sequence build requires problems"
        )
    if len(materialized) > client.config.max_requests:
        raise SequenceTeacherDataError(
            "problem count exceeds configured request budget"
        )
    destination = Path(output_dir)
    if destination.exists():
        raise SequenceTeacherDataError(
            f"output destination already exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )
    try:
        verifications: list[TeacherVerification] = []
        for problem in materialized:
            verification = verify_teacher_answer(
                problem,
                client.generate(problem),
            )
            verifications.append(verification)
        accepted = tuple(
            item for item in verifications if item.accepted
        )
        rejected = tuple(
            item for item in verifications if not item.accepted
        )
        if not accepted:
            raise SequenceTeacherDataError(
                "external teacher produced no verified sequences"
            )
        records = tuple(sequence_record(item) for item in accepted)
        accepted_payload = canonical_jsonl_bytes(
            item.public_record() for item in accepted
        )
        rejected_payload = canonical_jsonl_bytes(
            item.public_record() for item in rejected
        )
        _write_bytes(temporary / "accepted.jsonl", accepted_payload)
        _write_bytes(temporary / "rejected.jsonl", rejected_payload)
        sequence_manifest = build_corpus(
            records,
            temporary / "sequence_corpus",
            registry_path=registry_path,
            split_seed=LESSON13_SEQUENCE_SPLIT_SEED,
            require_all_splits=False,
        )
        assembled_manifest = assemble_fixed_evaluation_sft(
            sequence_corpus_dir=temporary / "sequence_corpus",
            reference_sft_dir=reference_sft_dir,
            output_dir=temporary / "sft",
        )
        rejection_reasons = Counter(item.reason for item in rejected)
        usage = {
            "completion_tokens": sum(
                item.answer.usage.completion_tokens
                for item in verifications
            ),
            "prompt_tokens": sum(
                item.answer.usage.prompt_tokens
                for item in verifications
            ),
            "total_tokens": sum(
                item.answer.usage.total_tokens
                for item in verifications
            ),
        }
        manifest: dict[str, object] = {
            "files": {
                "accepted.jsonl": _file_metadata(accepted_payload),
                "rejected.jsonl": _file_metadata(rejected_payload),
                "sequence_corpus/manifest.json": {
                    "bytes": (
                        temporary
                        / "sequence_corpus"
                        / "manifest.json"
                    ).stat().st_size,
                    "sha256": _sha256_file(
                        temporary
                        / "sequence_corpus"
                        / "manifest.json"
                    ),
                },
                "sft/manifest.json": {
                    "bytes": (
                        temporary / "sft" / "manifest.json"
                    ).stat().st_size,
                    "sha256": _sha256_file(
                        temporary / "sft" / "manifest.json"
                    ),
                },
            },
            "provider": client.config.public_metadata(),
            "provider_usage": usage,
            "requests": client.logical_requests,
            "schema_version": 1,
            "sequence_corpus": {
                "records": sequence_manifest.get("records"),
                "tokens": sequence_manifest.get("tokens"),
            },
            "sft": {
                "comparison_contract": assembled_manifest[
                    "comparison_contract"
                ],
                "tokens": assembled_manifest["tokens"],
            },
            "status": "verified_live_build",
            "verification": {
                "accepted": len(accepted),
                "rejected": len(rejected),
                "rejection_reasons": dict(
                    sorted(rejection_reasons.items())
                ),
                "total": len(verifications),
            },
        }
        assert_secret_free(
            manifest,
            context="sequence teacher artifact manifest",
        )
        _write_bytes(
            temporary / "manifest.json",
            _stable_json_bytes(manifest),
        )
        if destination.exists():
            raise SequenceTeacherDataError(
                f"output destination appeared during build: {destination}"
            )
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-per-task", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--registry-path", type=Path)
    parser.add_argument("--reference-sft-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    problems = make_teacher_problems(
        arithmetic=generate_arithmetic_examples(
            count=12_000,
            seed=arguments.seed,
        ),
        physics=generate_physics_examples(
            count=4_000,
            seed=arguments.seed,
        ),
        evaluation_path=arguments.evaluation,
        max_per_task=arguments.max_per_task,
    )
    client = ExternalTeacherClient(
        ExternalTeacherConfig(max_requests=len(problems))
    )
    try:
        if arguments.dry_run:
            summary = build_request_plan_artifact(
                problems,
                arguments.output_dir,
                client=client,
            )
        else:
            if (
                arguments.registry_path is None
                or arguments.reference_sft_dir is None
            ):
                raise SequenceTeacherDataError(
                    "live build requires --registry-path and "
                    "--reference-sft-dir"
                )
            summary = build_sequence_teacher_artifact(
                problems,
                arguments.output_dir,
                client=client,
                registry_path=arguments.registry_path,
                reference_sft_dir=arguments.reference_sft_dir,
            )
    except (
        ExternalTeacherError,
        OSError,
        SequenceTeacherDataError,
        ValueError,
    ) as error:
        raise SystemExit(f"error: {error}") from None
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
