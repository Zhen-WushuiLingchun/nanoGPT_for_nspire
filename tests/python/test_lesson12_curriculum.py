from dataclasses import replace
from decimal import Decimal

import pytest

from nanogpt_nspire.lesson12_curriculum import (
    CurriculumError,
    GSM8KExample,
    OASSTPair,
    PhysicsExample,
    canonical_jsonl_bytes,
    generate_physics_examples,
    parse_gsm8k_final_answer,
    verify_physics_example,
)


def test_gsm8k_final_answer_parser_is_strict_and_canonical() -> None:
    answer = (
        "She buys 3 bags at $12 each, so the total is "
        "<<3*12=36>>$36.\n#### 1,200.50"
    )

    assert parse_gsm8k_final_answer(answer) == "1200.5"


@pytest.mark.parametrize(
    "answer",
    (
        "The result is 42.",
        "#### 42 extra",
        "#### NaN",
        "#### 1/2",
        "#### 1,20",
    ),
)
def test_gsm8k_final_answer_parser_rejects_ambiguous_values(answer: str) -> None:
    with pytest.raises(CurriculumError, match="final answer"):
        parse_gsm8k_final_answer(answer)


def test_gsm8k_example_preserves_original_split_and_short_target() -> None:
    example = GSM8KExample.from_record(
        {
            "question": "A box has 12 rows of 7 bolts. How many bolts?",
            "answer": "There are <<12*7=84>>84 bolts.\n#### 84",
        },
        source_split="train",
        row_index=9,
    )

    assert example.exact_answer == "84"
    assert example.direct_answer == "The answer is 84."
    assert example.source_split == "train"
    assert example.record_id.endswith(":train:9")
    assert example.family_id.startswith("gsm8k-")


def test_gsm8k_example_rejects_bad_schema_and_split() -> None:
    with pytest.raises(CurriculumError, match="question"):
        GSM8KExample.from_record(
            {"question": "", "answer": "#### 1"},
            source_split="train",
            row_index=0,
        )
    with pytest.raises(CurriculumError, match="source_split"):
        GSM8KExample.from_record(
            {"question": "Q?", "answer": "#### 1"},
            source_split="validation",
            row_index=0,
        )


def _oasst_prompt() -> dict[str, object]:
    return {
        "message_id": "prompt-1",
        "parent_id": None,
        "message_tree_id": "tree-1",
        "role": "prompter",
        "lang": "en",
        "text": "What is inertia?",
        "deleted": False,
        "rank": None,
    }


def _oasst_answer() -> dict[str, object]:
    return {
        "message_id": "answer-1",
        "parent_id": "prompt-1",
        "message_tree_id": "tree-1",
        "role": "assistant",
        "lang": "en",
        "text": (
            "Inertia is an object's tendency to resist a change in its "
            "motion."
        ),
        "deleted": False,
        "rank": 0,
    }


def test_oasst_pair_requires_english_top_rank_direct_reply() -> None:
    pair = OASSTPair.from_messages(_oasst_prompt(), _oasst_answer())

    assert pair.question == "What is inertia?"
    assert pair.answer.startswith("Inertia is")
    assert pair.family_id == "oasst1-tree-1"
    assert pair.record_id == "oasst1-answer-1"


@pytest.mark.parametrize(
    "field,value,message",
    (
        ("rank", 1, "rank"),
        ("lang", "de", "English"),
        ("deleted", True, "deleted"),
        ("parent_id", "other", "direct reply"),
    ),
)
def test_oasst_pair_rejects_ineligible_assistant_rows(
    field: str,
    value: object,
    message: str,
) -> None:
    answer = _oasst_answer()
    answer[field] = value
    with pytest.raises(CurriculumError, match=message):
        OASSTPair.from_messages(_oasst_prompt(), answer)


def test_physics_examples_are_exact_deterministic_and_balanced() -> None:
    first = generate_physics_examples(count=120, seed=20260728)
    second = generate_physics_examples(count=120, seed=20260728)

    assert first == second
    assert len({example.family_id for example in first}) == 120
    assert all(verify_physics_example(example) for example in first)
    assert {example.formula_id for example in first} == {
        "density",
        "force",
        "kinetic-energy",
        "momentum",
        "ohms-law",
        "power",
        "pressure",
        "speed",
        "wave-speed",
        "weight",
    }
    assert all(example.unit in example.direct_answer for example in first)


def test_physics_curriculum_supports_full_lesson12_unique_count() -> None:
    examples = generate_physics_examples(count=4_000, seed=20260728)

    assert len(examples) == 4_000
    assert len({example.family_id for example in examples}) == 4_000


def test_physics_verifier_rejects_tampering() -> None:
    example = PhysicsExample.create(
        formula_id="force",
        left=Decimal("12"),
        right=Decimal("7"),
    )

    assert example.exact_answer == "84"
    assert verify_physics_example(example)
    assert not verify_physics_example(
        replace(example, exact_answer="85")
    )


def test_physics_examples_reject_unknown_formula_and_nonpositive_inputs() -> None:
    with pytest.raises(CurriculumError, match="formula"):
        PhysicsExample.create(
            formula_id="teleportation",
            left=Decimal("1"),
            right=Decimal("2"),
        )
    with pytest.raises(CurriculumError, match="positive"):
        PhysicsExample.create(
            formula_id="speed",
            left=Decimal("10"),
            right=Decimal("0"),
        )


def test_canonical_jsonl_is_order_stable_and_rejects_nonfinite_values() -> None:
    rows = ({"b": 2, "a": 1}, {"name": "Δ"})

    assert canonical_jsonl_bytes(rows) == (
        b'{"a":1,"b":2}\n'
        + '{"name":"Δ"}\n'.encode("utf-8")
    )
    with pytest.raises(CurriculumError, match="JSON"):
        canonical_jsonl_bytes(({"bad": float("nan")},))
