from __future__ import annotations

from decimal import Decimal

import pytest

from nanogpt_nspire.lesson12_curriculum import PhysicsExample
from nanogpt_nspire.lesson16_data import (
    Lesson16DataError,
    build_compact_examples,
    compact_gsm8k_example,
    parse_verified_calculation,
)
from nanogpt_nspire.math_curriculum import ArithmeticExample


def _gsm_record(
    answer: str = (
        "There are <<12*7=84>>84 bolts in the box.\n"
        "#### 84"
    ),
) -> dict[str, str]:
    return {
        "question": "A box has 12 rows of 7 bolts. How many bolts are there?",
        "answer": answer,
    }


def test_verified_calculation_parser_is_exact_and_bounded() -> None:
    calculation = parse_verified_calculation(
        "12 * (7 + 1) = 96"
    )

    assert calculation.expression == "12 * (7 + 1)"
    assert calculation.result == "96"
    assert calculation.value == Decimal("96")


@pytest.mark.parametrize(
    "text",
    (
        "12 * 7 = 83",
        "1 / 0 = 0",
        "__import__('os') = 1",
        "2 ** 100 = 1267650600228229401496703205376",
        "1 + (2 * (3 + (4 * (5 + 6)))) = 51",
    ),
)
def test_verified_calculation_parser_rejects_wrong_or_unsafe_text(
    text: str,
) -> None:
    with pytest.raises(Lesson16DataError):
        parse_verified_calculation(text)


def test_gsm_annotations_become_compact_verified_steps() -> None:
    example = compact_gsm8k_example(_gsm_record(), row_index=9)

    assert example.task == "gsm8k"
    assert example.exact_answer == "84"
    assert example.reasoning == "Compute 12 * 7 = 84."
    assert example.reasoning_steps == 1
    assert len(example.reasoning.encode("utf-8")) <= 160


@pytest.mark.parametrize(
    "answer",
    (
        "There are 84 bolts.\n#### 84",
        "There are <<12*7=83>>83 bolts.\n#### 84",
        (
            "<<1+1=2>><<2+1=3>><<3+1=4>><<4+1=5>>"
            "<<5+1=6>>\n#### 6"
        ),
    ),
)
def test_gsm_compaction_requires_one_to_four_verified_final_steps(
    answer: str,
) -> None:
    with pytest.raises(Lesson16DataError):
        compact_gsm8k_example(_gsm_record(answer), row_index=0)


def test_project_examples_use_short_verified_reasoning() -> None:
    arithmetic = ArithmeticExample.create(
        left=12,
        operator="*",
        right=7,
    )
    physics = PhysicsExample.create(
        formula_id="force",
        left=Decimal("12"),
        right=Decimal("7"),
    )

    examples, report = build_compact_examples(
        arithmetic=(arithmetic,),
        physics=(physics,),
        gsm8k_rows=(_gsm_record(),),
    )

    by_task = {item.task: item for item in examples}
    assert by_task["arithmetic"].reasoning == "Compute 12 * 7 = 84."
    assert by_task["physics_numeric"].reasoning == (
        "Formula F = m a. Substitute F = 12 * 7 = 84 N."
    )
    assert all(1 <= item.reasoning_steps <= 4 for item in examples)
    assert all(
        len(item.reasoning.encode("utf-8")) <= 160
        for item in examples
    )
    assert report["accepted_by_task"] == {
        "arithmetic": 1,
        "gsm8k": 1,
        "physics_numeric": 1,
    }
