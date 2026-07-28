"""Deterministic, exactly verifiable arithmetic curriculum generation."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import random


SUPPORTED_OPERATORS = frozenset({"+", "-", "*", "/"})
MAX_OPERAND_DIGITS = 12


class ArithmeticError(ValueError):
    """Raised when a curriculum expression cannot be exactly verified."""


def _digit_count(text: str) -> int:
    return sum(character.isdigit() for character in text)


def _normalize_operand(value: object) -> tuple[str, str]:
    if isinstance(value, bool):
        raise ArithmeticError("operand must be an integer or finite Decimal")
    if isinstance(value, int):
        text = str(value)
        if _digit_count(text) > MAX_OPERAND_DIGITS:
            raise ArithmeticError("operand exceeds digit limit")
        return text, "integer"
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ArithmeticError("decimal operand must be finite")
        normalized = value.normalize()
        text = format(normalized, "f")
        if text in {"-0", ""}:
            text = "0"
        if _digit_count(text) > MAX_OPERAND_DIGITS:
            raise ArithmeticError("operand exceeds digit limit")
        return text, "decimal"
    raise ArithmeticError("operand must be an integer or finite Decimal")


def _parse_operand(text: str, numeric_kind: str) -> Fraction | Decimal:
    if numeric_kind == "decimal":
        return Decimal(text)
    return Fraction(int(text), 1)


def _apply_operator(
    left: Fraction | Decimal,
    operator: str,
    right: Fraction | Decimal,
) -> Fraction | Decimal:
    if operator not in SUPPORTED_OPERATORS:
        raise ArithmeticError(f"unsupported operator {operator!r}")
    if operator == "/" and right == 0:
        raise ArithmeticError("division by zero is not allowed")
    if isinstance(left, Decimal) or isinstance(right, Decimal):
        if not isinstance(left, Decimal) or not isinstance(right, Decimal):
            raise ArithmeticError("mixed numeric kinds are not canonical")
        if operator == "/":
            raise ArithmeticError("decimal division is not supported")
        if operator == "+":
            return left + right
        if operator == "-":
            return left - right
        return left * right
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    return left / right


def _format_result(value: Fraction | Decimal) -> str:
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ArithmeticError("result must be finite")
        text = format(value.normalize(), "f")
        return "0" if text == "-0" else text
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _stable_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ArithmeticExample:
    """One canonical arithmetic family with direct and worked responses."""

    category: str
    numeric_kind: str
    left: str
    operator: str
    right: str
    outer_operator: str | None
    outer_right: str | None
    expression: str
    exact_answer: str
    question: str
    direct_answer: str
    worked_answer: str
    family_id: str
    example_id: str
    difficulty: str

    @classmethod
    def create(
        cls,
        *,
        left: int | Decimal,
        operator: str,
        right: int | Decimal,
        outer_operator: str | None = None,
        outer_right: int | Decimal | None = None,
        difficulty: str = "easy",
    ) -> ArithmeticExample:
        """Build and exactly evaluate a binary or one-parenthesis expression."""

        if operator not in SUPPORTED_OPERATORS:
            raise ArithmeticError(f"unsupported operator {operator!r}")
        if not isinstance(difficulty, str) or not difficulty:
            raise ArithmeticError("difficulty must be a non-empty string")
        left_text, left_kind = _normalize_operand(left)
        right_text, right_kind = _normalize_operand(right)
        numeric_kind = (
            "decimal"
            if "decimal" in {left_kind, right_kind}
            else "integer"
        )
        if numeric_kind == "decimal":
            if left_kind == "integer":
                left_text, left_kind = _normalize_operand(Decimal(left_text))
            if right_kind == "integer":
                right_text, right_kind = _normalize_operand(Decimal(right_text))

        if outer_operator is None and outer_right is not None:
            raise ArithmeticError("outer_right requires an outer operator")
        if outer_operator is not None and outer_right is None:
            raise ArithmeticError("outer operator requires outer_right")
        if outer_operator is not None and outer_operator not in SUPPORTED_OPERATORS:
            raise ArithmeticError(f"unsupported operator {outer_operator!r}")

        left_value = _parse_operand(left_text, numeric_kind)
        right_value = _parse_operand(right_text, numeric_kind)
        inner_result = _apply_operator(left_value, operator, right_value)
        inner_answer = _format_result(inner_result)

        if outer_operator is None:
            category = "decimal" if numeric_kind == "decimal" else "integer"
            outer_text = None
            result = inner_result
            expression = f"{left_text} {operator} {right_text}"
            worked_answer = f"{expression} = {_format_result(result)}."
        else:
            assert outer_right is not None
            outer_text, outer_kind = _normalize_operand(outer_right)
            if numeric_kind == "decimal" and outer_kind == "integer":
                outer_text, outer_kind = _normalize_operand(Decimal(outer_text))
            if numeric_kind == "integer" and outer_kind == "decimal":
                numeric_kind = "decimal"
                left_text, _ = _normalize_operand(Decimal(left_text))
                right_text, _ = _normalize_operand(Decimal(right_text))
                outer_text, _ = _normalize_operand(Decimal(outer_text))
                left_value = _parse_operand(left_text, numeric_kind)
                right_value = _parse_operand(right_text, numeric_kind)
                inner_result = _apply_operator(left_value, operator, right_value)
                inner_answer = _format_result(inner_result)
            outer_value = _parse_operand(outer_text, numeric_kind)
            result = _apply_operator(inner_result, outer_operator, outer_value)
            category = "parenthesized"
            expression = (
                f"({left_text} {operator} {right_text}) "
                f"{outer_operator} {outer_text}"
            )
            worked_answer = (
                f"First, {left_text} {operator} {right_text} = {inner_answer}. "
                f"Then, {inner_answer} {outer_operator} {outer_text} = "
                f"{_format_result(result)}."
            )

        exact_answer = _format_result(result)
        if _digit_count(exact_answer) > 2 * MAX_OPERAND_DIGITS:
            raise ArithmeticError("result exceeds digit limit")
        family_payload = {
            "left": left_text,
            "numeric_kind": numeric_kind,
            "operator": operator,
            "outer_operator": outer_operator,
            "outer_right": outer_text,
            "right": right_text,
        }
        family_id = f"arith-{_stable_digest(family_payload)[:20]}"
        return cls(
            category=category,
            numeric_kind=numeric_kind,
            left=left_text,
            operator=operator,
            right=right_text,
            outer_operator=outer_operator,
            outer_right=outer_text,
            expression=expression,
            exact_answer=exact_answer,
            question=f"Calculate {expression}.",
            direct_answer=f"The answer is {exact_answer}.",
            worked_answer=worked_answer,
            family_id=family_id,
            example_id=f"{family_id}-canonical",
            difficulty=difficulty,
        )

    def variant_id(self, *, style: str, question: str) -> str:
        """Return a stable variant ID while retaining the family prefix."""

        if not isinstance(style, str) or not style:
            raise ArithmeticError("style must be a non-empty string")
        if not isinstance(question, str) or not question:
            raise ArithmeticError("question must be a non-empty string")
        suffix = _stable_digest({"question": question, "style": style})[:16]
        return f"{self.family_id}-{suffix}"


def verify_arithmetic_example(example: object) -> bool:
    """Rebuild an example from its operation fields and compare invariants."""

    if not isinstance(example, ArithmeticExample):
        return False
    try:
        def parse_original(text: str) -> int | Decimal:
            return Decimal(text) if example.numeric_kind == "decimal" else int(text)

        rebuilt = ArithmeticExample.create(
            left=parse_original(example.left),
            operator=example.operator,
            right=parse_original(example.right),
            outer_operator=example.outer_operator,
            outer_right=(
                None
                if example.outer_right is None
                else parse_original(example.outer_right)
            ),
            difficulty=example.difficulty,
        )
    except (ArithmeticError, ValueError):
        return False
    return (
        example.category == rebuilt.category
        and example.expression == rebuilt.expression
        and example.exact_answer == rebuilt.exact_answer
        and example.family_id == rebuilt.family_id
        and example.example_id == rebuilt.example_id
    )


def generate_arithmetic_examples(
    *,
    count: int,
    seed: int,
) -> tuple[ArithmeticExample, ...]:
    """Generate a balanced, deterministic set of unique arithmetic families."""

    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ArithmeticError("count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ArithmeticError("seed must be an integer")

    generator = random.Random(seed)
    examples: list[ArithmeticExample] = []
    seen: set[str] = set()
    attempts = 0
    while len(examples) < count:
        attempts += 1
        if attempts > count * 100:
            raise ArithmeticError("could not generate enough unique examples")
        route = len(examples) % 8
        if route == 0:
            example = ArithmeticExample.create(
                left=generator.randint(-99, 999),
                operator="+",
                right=generator.randint(-99, 999),
            )
        elif route == 1:
            example = ArithmeticExample.create(
                left=generator.randint(-99, 999),
                operator="-",
                right=generator.randint(-99, 999),
            )
        elif route == 2:
            example = ArithmeticExample.create(
                left=generator.randint(-30, 99),
                operator="*",
                right=generator.randint(-30, 99),
            )
        elif route == 3:
            divisor = generator.randint(1, 20)
            quotient = generator.randint(-50, 50)
            example = ArithmeticExample.create(
                left=divisor * quotient,
                operator="/",
                right=divisor,
            )
        elif route == 4:
            example = ArithmeticExample.create(
                left=Decimal(generator.randint(-999, 999)) / Decimal(10),
                operator="+",
                right=Decimal(generator.randint(-999, 999)) / Decimal(100),
            )
        elif route == 5:
            example = ArithmeticExample.create(
                left=Decimal(generator.randint(-99, 99)) / Decimal(10),
                operator="*",
                right=Decimal(generator.randint(-99, 99)) / Decimal(10),
            )
        elif route == 6:
            example = ArithmeticExample.create(
                left=generator.randint(-20, 50),
                operator="+",
                right=generator.randint(-20, 50),
                outer_operator="*",
                outer_right=generator.randint(-10, 10),
            )
        else:
            example = ArithmeticExample.create(
                left=Decimal(generator.randint(-999, 999)) / Decimal(10),
                operator="-",
                right=Decimal(generator.randint(-999, 999)) / Decimal(100),
            )
        if example.family_id in seen:
            continue
        seen.add(example.family_id)
        examples.append(example)
    return tuple(examples)
