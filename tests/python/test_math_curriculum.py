from dataclasses import replace
from decimal import Decimal

import pytest

from nanogpt_nspire.math_curriculum import (
    ArithmeticError,
    ArithmeticExample,
    generate_arithmetic_examples,
    verify_arithmetic_example,
)


@pytest.mark.parametrize(
    "left, operator, right, expected",
    [
        (12, "+", 7, "19"),
        (12, "-", 7, "5"),
        (12, "*", 7, "84"),
        (12, "/", 7, "12/7"),
        (Decimal("1.20"), "+", Decimal("2.3"), "3.5"),
        (Decimal("1.20"), "-", Decimal("2.3"), "-1.1"),
        (Decimal("1.20"), "*", Decimal("2.3"), "2.76"),
    ],
)
def test_binary_examples_have_exact_canonical_answers(
    left,
    operator,
    right,
    expected,
):
    example = ArithmeticExample.create(
        left=left,
        operator=operator,
        right=right,
    )

    assert example.exact_answer == expected
    assert verify_arithmetic_example(example)
    assert example.question == f"Calculate {example.expression}."
    assert expected in example.direct_answer
    assert expected in example.worked_answer


def test_parenthesized_expression_is_verified_without_eval():
    example = ArithmeticExample.create(
        left=7,
        operator="+",
        right=5,
        outer_operator="*",
        outer_right=3,
    )

    assert example.expression == "(7 + 5) * 3"
    assert example.exact_answer == "36"
    assert example.worked_answer == (
        "First, 7 + 5 = 12. Then, 12 * 3 = 36."
    )
    assert verify_arithmetic_example(example)


def test_family_id_is_independent_of_response_style_and_paraphrase():
    example = ArithmeticExample.create(left=12, operator="*", right=7)

    direct_id = example.variant_id(
        style="direct",
        question="Calculate 12 * 7.",
    )
    worked_id = example.variant_id(
        style="worked",
        question="What is twelve times seven?",
    )

    assert direct_id != worked_id
    assert direct_id.startswith(example.family_id + "-")
    assert worked_id.startswith(example.family_id + "-")


def test_seeded_generation_is_deterministic_balanced_and_unique():
    first = generate_arithmetic_examples(count=64, seed=20260728)
    second = generate_arithmetic_examples(count=64, seed=20260728)
    different = generate_arithmetic_examples(count=64, seed=20260729)

    assert first == second
    assert first != different
    assert len({example.example_id for example in first}) == 64
    assert len({example.family_id for example in first}) == 64
    assert all(verify_arithmetic_example(example) for example in first)
    assert {example.category for example in first} == {
        "integer",
        "decimal",
        "parenthesized",
    }
    assert {example.operator for example in first} >= {"+", "-", "*", "/"}


def test_verifier_rejects_tampered_answer():
    example = ArithmeticExample.create(left=12, operator="*", right=7)
    tampered = replace(example, exact_answer="85")

    assert not verify_arithmetic_example(tampered)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"left": 1, "operator": "^", "right": 2}, "operator"),
        ({"left": 1, "operator": "/", "right": 0}, "zero"),
        (
            {
                "left": Decimal("1"),
                "operator": "/",
                "right": Decimal("2"),
            },
            "decimal division",
        ),
        ({"left": Decimal("NaN"), "operator": "+", "right": 2}, "finite"),
        ({"left": True, "operator": "+", "right": 2}, "operand"),
        (
            {"left": 10**25, "operator": "+", "right": 2},
            "digit limit",
        ),
    ],
)
def test_invalid_expressions_fail_closed(kwargs, message):
    with pytest.raises(ArithmeticError, match=message):
        ArithmeticExample.create(**kwargs)


def test_generator_rejects_invalid_count_and_seed_types():
    with pytest.raises(ArithmeticError, match="count"):
        generate_arithmetic_examples(count=0, seed=1)

    with pytest.raises(ArithmeticError, match="seed"):
        generate_arithmetic_examples(count=1, seed=True)
