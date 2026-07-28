from __future__ import annotations

import json
from pathlib import Path

from nanogpt_nspire.base_corpus import (
    CorpusRecord,
    build_corpus,
    stable_family_split,
)
from nanogpt_nspire.base_train import load_packed_dataset
from nanogpt_nspire.byte_tokenizer import ConversationTurn
from nanogpt_nspire.external_teacher import (
    ExternalTeacherClient,
    ExternalTeacherConfig,
    TeacherAnswer,
    TokenUsage,
)
from nanogpt_nspire.lesson12_curriculum import generate_physics_examples
from nanogpt_nspire.lesson13_sequence_data import (
    LESSON13_SEQUENCE_SPLIT_SEED,
    assemble_fixed_evaluation_sft,
    build_request_plan_artifact,
    build_sequence_teacher_artifact,
    make_teacher_problems,
    sequence_record,
    verify_teacher_answer,
)
from nanogpt_nspire.math_curriculum import generate_arithmetic_examples


def answer(
    *,
    text: str,
    final: str,
    unit: str | None,
) -> TeacherAnswer:
    return TeacherAnswer(
        answer_text=text,
        final_answer=final,
        unit=unit,
        provider_request_id="request-test",
        provider_model="deepseek-v4-pro",
        usage=TokenUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
    )


def _family_for_split(split: str) -> str:
    for index in range(100_000):
        family = f"reference-{split}-{index}"
        if (
            stable_family_split(
                family,
                split_seed=LESSON13_SEQUENCE_SPLIT_SEED,
            )
            == split
        ):
            return family
    raise AssertionError("could not find split family")


def test_problem_selection_uses_training_families_and_excludes_evaluation(
    tmp_path: Path,
) -> None:
    arithmetic = generate_arithmetic_examples(count=200, seed=20260728)
    physics = generate_physics_examples(count=100, seed=20260728)
    evaluation_family = next(
        item.family_id
        for item in arithmetic
        if stable_family_split(
            item.family_id,
            split_seed=LESSON13_SEQUENCE_SPLIT_SEED,
        )
        == "train"
    )
    evaluation = tmp_path / "evaluation.jsonl"
    evaluation.write_text(
        json.dumps(
            {
                "expected_answer": "0",
                "expected_unit": None,
                "family_id": evaluation_family,
                "prompt": "held out",
                "source_id": "project",
                "task": "arithmetic",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    problems = make_teacher_problems(
        arithmetic=arithmetic,
        physics=physics,
        evaluation_path=evaluation,
        max_per_task=8,
    )

    assert len(problems) == 16
    assert evaluation_family not in {item.family_id for item in problems}
    assert {item.task for item in problems} == {
        "arithmetic",
        "physics_numeric",
    }
    assert all(
        stable_family_split(
            item.family_id,
            split_seed=LESSON13_SEQUENCE_SPLIT_SEED,
        )
        == "train"
        for item in problems
    )


def test_exact_arithmetic_answer_is_accepted() -> None:
    problem = make_teacher_problems(
        arithmetic=generate_arithmetic_examples(count=100, seed=10),
        physics=generate_physics_examples(count=100, seed=10),
        evaluation_path=None,
        max_per_task=1,
    )[0]

    result = verify_teacher_answer(
        problem,
        answer(
            text=f"Compute carefully. The answer is {problem.expected_answer}.",
            final=problem.expected_answer,
            unit=None,
        ),
    )

    assert result.accepted is True
    assert result.reason == "verified"
    record = sequence_record(result)
    assert record.source_id == "deepseek-v4-pro-generated"
    assert record.turns[0].content == problem.prompt


def test_wrong_answer_role_leak_and_wrong_unit_are_rejected() -> None:
    problems = make_teacher_problems(
        arithmetic=generate_arithmetic_examples(count=100, seed=11),
        physics=generate_physics_examples(count=100, seed=11),
        evaluation_path=None,
        max_per_task=1,
    )
    arithmetic = next(item for item in problems if item.task == "arithmetic")
    physics = next(item for item in problems if item.task == "physics_numeric")

    wrong = verify_teacher_answer(
        arithmetic,
        answer(text="The answer is 999.", final="999", unit=None),
    )
    leaked = verify_teacher_answer(
        arithmetic,
        answer(
            text=f"<USER> The answer is {arithmetic.expected_answer}.",
            final=arithmetic.expected_answer,
            unit=None,
        ),
    )
    wrong_unit = verify_teacher_answer(
        physics,
        answer(
            text=f"The answer is {physics.expected_answer} kg.",
            final=physics.expected_answer,
            unit="kg",
        ),
    )

    assert wrong.reason == "final_answer_mismatch"
    assert leaked.reason == "role_token_leak"
    assert wrong_unit.reason == "unit_mismatch"


def test_dry_run_writes_deterministic_secret_free_plan(
    tmp_path: Path,
) -> None:
    problems = make_teacher_problems(
        arithmetic=generate_arithmetic_examples(count=100, seed=12),
        physics=generate_physics_examples(count=100, seed=12),
        evaluation_path=None,
        max_per_task=2,
    )
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_manifest = build_request_plan_artifact(
        problems,
        first,
        client=ExternalTeacherClient(
            ExternalTeacherConfig(max_requests=4)
        ),
    )
    second_manifest = build_request_plan_artifact(
        problems,
        second,
        client=ExternalTeacherClient(
            ExternalTeacherConfig(max_requests=4)
        ),
    )

    assert first_manifest == second_manifest
    assert (first / "request-plan.jsonl").read_bytes() == (
        second / "request-plan.jsonl"
    ).read_bytes()
    payload = (first / "request-plan.jsonl").read_text(encoding="utf-8")
    assert "Authorization" not in payload
    assert "sk-" not in payload


def test_assembled_corpus_replaces_only_training_split(
    tmp_path: Path,
) -> None:
    registry = Path(__file__).resolve().parents[2] / "experiments" / (
        "lesson10-public-sources.json"
    )
    external = tmp_path / "external"
    reference = tmp_path / "reference"
    destination = tmp_path / "assembled"
    external_family = _family_for_split("train")
    external_record = CorpusRecord.conversation(
        record_id="external-1",
        family_id=external_family,
        turns=(
            ConversationTurn("user", "Calculate 1 + 1."),
            ConversationTurn("assistant", "The answer is 2."),
        ),
        source_id="deepseek-v4-pro-generated",
        license_id="DeepSeek-Output-Terms-2026-03-27",
    )
    build_corpus(
        (external_record,),
        external,
        registry_path=registry,
        split_seed=LESSON13_SEQUENCE_SPLIT_SEED,
        require_all_splits=False,
    )
    reference_records = tuple(
        CorpusRecord.conversation(
            record_id=f"reference-{split}",
            family_id=_family_for_split(split),
            turns=(
                ConversationTurn("user", f"{split} question"),
                ConversationTurn("assistant", f"{split} answer"),
            ),
            source_id="project-arithmetic-v1",
            license_id="MIT",
        )
        for split in ("train", "validation", "test")
    )
    build_corpus(
        reference_records,
        reference,
        registry_path=registry,
        split_seed=LESSON13_SEQUENCE_SPLIT_SEED,
    )

    manifest = assemble_fixed_evaluation_sft(
        sequence_corpus_dir=external,
        reference_sft_dir=reference,
        output_dir=destination,
    )
    assembled = load_packed_dataset(destination)
    external_loaded = load_packed_dataset(
        destination
    )
    reference_loaded = load_packed_dataset(reference)

    assert manifest["comparison_contract"]["training_split"] == (
        "verified_external_sequences"
    )
    assert assembled.train.token_path.read_bytes() == (
        external / "train.tokens.bin"
    ).read_bytes()
    assert assembled.validation.token_path.read_bytes() == (
        reference_loaded.validation.token_path.read_bytes()
    )
    assert assembled.test.mask_path.read_bytes() == (
        reference_loaded.test.mask_path.read_bytes()
    )
    assert external_loaded.train.token_count == assembled.train.token_count


def test_live_artifact_keeps_only_verified_public_sequences(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-" + "z" * 32)
    registry = Path(__file__).resolve().parents[2] / "experiments" / (
        "lesson10-public-sources.json"
    )
    reference = tmp_path / "reference"
    reference_records = tuple(
        CorpusRecord.conversation(
            record_id=f"reference-{split}",
            family_id=_family_for_split(split),
            turns=(
                ConversationTurn("user", f"{split} question"),
                ConversationTurn("assistant", f"{split} answer"),
            ),
            source_id="project-arithmetic-v1",
            license_id="MIT",
        )
        for split in ("train", "validation", "test")
    )
    build_corpus(
        reference_records,
        reference,
        registry_path=registry,
        split_seed=LESSON13_SEQUENCE_SPLIT_SEED,
    )
    problems = make_teacher_problems(
        arithmetic=generate_arithmetic_examples(count=100, seed=13),
        physics=generate_physics_examples(count=100, seed=13),
        evaluation_path=None,
        max_per_task=1,
    )
    calls = 0

    def transport(
        _url: str,
        _headers: dict[str, str],
        body: bytes,
        _timeout: float,
    ) -> bytes:
        nonlocal calls
        calls += 1
        request = json.loads(body)
        user = request["messages"][1]["content"]
        problem_payload = json.loads(user.split("\n", 1)[1])
        final = problem_payload["expected_final_answer"]
        unit = problem_payload["expected_unit"]
        suffix = "" if unit is None else f" {unit}"
        content = {
            "answer_text": f"Apply the given rule. The answer is {final}{suffix}.",
            "final_answer": final,
            "unit": unit,
        }
        return json.dumps(
            {
                "id": f"request-{calls}",
                "model": "deepseek-v4-pro",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(content),
                            "reasoning_content": "discarded",
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                    "total_tokens": 30,
                },
            }
        ).encode("utf-8")

    output = tmp_path / "live"
    manifest = build_sequence_teacher_artifact(
        problems,
        output,
        client=ExternalTeacherClient(
            ExternalTeacherConfig(max_requests=2),
            transport=transport,
        ),
        registry_path=registry,
        reference_sft_dir=reference,
    )

    assert calls == 2
    assert manifest["verification"]["accepted"] == 2
    assert manifest["verification"]["rejected"] == 0
    accepted = (output / "accepted.jsonl").read_text(encoding="utf-8")
    assert "reasoning_content" not in accepted
    assert "discarded" not in accepted
    assert "sk-" not in accepted
    load_packed_dataset(output / "sft")
