"""Compact, exactly verified SFT-v2 data for Lesson 16."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Iterable, Mapping, Sequence

from nanogpt_nspire.assistant_eval import load_evaluation_records
from nanogpt_nspire.lesson12_curriculum import (
    GSM8KExample,
    PhysicsExample,
    canonical_jsonl_bytes,
    generate_physics_examples,
    verify_physics_example,
)
from nanogpt_nspire.lesson12_data import PINNED_INPUTS
from nanogpt_nspire.lesson14_data import (
    Lesson14Example,
    build_mode_corpus,
    select_paired_examples,
)
from nanogpt_nspire.math_curriculum import (
    ArithmeticExample,
    generate_arithmetic_examples,
    verify_arithmetic_example,
)
from nanogpt_nspire.reasoning_format import DIRECT_MODE, THINK_MODE


LESSON16_SCHEMA_VERSION = 1
LESSON16_SPLIT_SEED = "lesson16-sft-v2-split-v1"
LESSON16_DATA_SEED = 20260729
LESSON16_CHALLENGE_SEED = 20260730
MAX_REASONING_BYTES = 160
MAX_CALCULATION_STEPS = 4
MAX_EXPRESSION_TOKENS = 63
MAX_EXPRESSION_DEPTH = 3
GSM_ANNOTATION = re.compile(r"<<([^<>]+)>>")
NUMBER_TOKEN = re.compile(
    r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
)
SIGNED_NUMBER = re.compile(
    r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
)
PHYSICS_FORMULA_IDS = (
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
)


class Lesson16DataError(ValueError):
    """Raised when an SFT-v2 record is not compact and exactly verified."""


def _format_decimal(value: Decimal) -> str:
    if not value.is_finite():
        raise Lesson16DataError("numeric value must be finite")
    text = format(value.normalize(), "f")
    return "0" if text == "-0" else text


def _stable_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _load_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    try:
        lines = path.read_text(
            encoding="utf-8",
            errors="strict",
        ).splitlines()
    except (OSError, UnicodeError) as error:
        raise Lesson16DataError(f"invalid JSONL file: {path}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise Lesson16DataError(
                f"invalid JSON at line {line_number}"
            ) from error
        if not isinstance(row, dict):
            raise Lesson16DataError("JSONL rows must be objects")
        rows.append(row)
    if not rows:
        raise Lesson16DataError("JSONL file is empty")
    return tuple(rows)


def _tokenize_expression(expression: str) -> tuple[str, ...]:
    if not isinstance(expression, str) or not expression.strip():
        raise Lesson16DataError("calculation expression is empty")
    tokens: list[str] = []
    cursor = 0
    while cursor < len(expression):
        if expression[cursor].isspace():
            cursor += 1
            continue
        number = NUMBER_TOKEN.match(expression, cursor)
        if number is not None:
            tokens.append(number.group(0))
            cursor = number.end()
            continue
        character = expression[cursor]
        if character in "+-*/()":
            tokens.append(character)
            cursor += 1
            continue
        raise Lesson16DataError(
            "calculation contains an unsupported token"
        )
    if not tokens or len(tokens) > MAX_EXPRESSION_TOKENS:
        raise Lesson16DataError("calculation token count is invalid")
    return tuple(tokens)


def _fraction_from_number(token: str) -> Fraction:
    normalized = token.replace(",", "")
    try:
        decimal = Decimal(normalized)
    except InvalidOperation as error:
        raise Lesson16DataError("calculation number is invalid") from error
    if not decimal.is_finite() or len(decimal.as_tuple().digits) > 18:
        raise Lesson16DataError("calculation number exceeds limits")
    return Fraction(decimal)


class _ExpressionParser:
    def __init__(self, tokens: Sequence[str]) -> None:
        self.tokens = tokens
        self.index = 0

    def parse(self) -> Fraction:
        value = self._expression(depth=0)
        if self.index != len(self.tokens):
            raise Lesson16DataError("calculation has trailing tokens")
        return value

    def _peek(self) -> str | None:
        return (
            self.tokens[self.index]
            if self.index < len(self.tokens)
            else None
        )

    def _take(self) -> str:
        token = self._peek()
        if token is None:
            raise Lesson16DataError("calculation ended unexpectedly")
        self.index += 1
        return token

    def _expression(self, *, depth: int) -> Fraction:
        value = self._term(depth=depth)
        while self._peek() in {"+", "-"}:
            operator = self._take()
            right = self._term(depth=depth)
            value = value + right if operator == "+" else value - right
        return value

    def _term(self, *, depth: int) -> Fraction:
        value = self._factor(depth=depth)
        while self._peek() in {"*", "/"}:
            operator = self._take()
            right = self._factor(depth=depth)
            if operator == "/" and right == 0:
                raise Lesson16DataError("division by zero is not allowed")
            value = value * right if operator == "*" else value / right
        return value

    def _factor(self, *, depth: int) -> Fraction:
        token = self._peek()
        if token in {"+", "-"}:
            operator = self._take()
            value = self._factor(depth=depth)
            return value if operator == "+" else -value
        if token == "(":
            if depth >= MAX_EXPRESSION_DEPTH:
                raise Lesson16DataError(
                    "calculation nesting exceeds the limit"
                )
            self._take()
            value = self._expression(depth=depth + 1)
            if self._take() != ")":
                raise Lesson16DataError("calculation parentheses mismatch")
            return value
        if token is None or NUMBER_TOKEN.fullmatch(token) is None:
            raise Lesson16DataError("calculation expected a number")
        self._take()
        return _fraction_from_number(token)


def _finite_decimal(value: Fraction) -> Decimal:
    denominator = value.denominator
    for factor in (2, 5):
        while denominator % factor == 0:
            denominator //= factor
    if denominator != 1:
        raise Lesson16DataError(
            "calculation result is not a finite decimal"
        )
    result = Decimal(value.numerator) / Decimal(value.denominator)
    if not result.is_finite() or len(result.as_tuple().digits) > 24:
        raise Lesson16DataError("calculation result exceeds limits")
    return result


def _canonical_expression(tokens: Sequence[str]) -> str:
    text = " ".join(tokens)
    text = text.replace("( ", "(").replace(" )", ")")
    return text


@dataclass(frozen=True)
class VerifiedCalculation:
    expression: str
    result: str
    value: Decimal


def parse_verified_calculation(text: str) -> VerifiedCalculation:
    """Safely parse and exactly verify one GSM8K calculation annotation."""

    if not isinstance(text, str) or text.count("=") != 1:
        raise Lesson16DataError(
            "calculation must contain exactly one equals sign"
        )
    expression_text, result_text = (
        part.strip() for part in text.split("=", maxsplit=1)
    )
    if SIGNED_NUMBER.fullmatch(result_text) is None:
        raise Lesson16DataError(
            "calculation result must be a decimal literal"
        )
    tokens = _tokenize_expression(expression_text)
    computed = _finite_decimal(_ExpressionParser(tokens).parse())
    try:
        declared = Decimal(result_text.replace(",", ""))
    except InvalidOperation as error:
        raise Lesson16DataError(
            "calculation result is invalid"
        ) from error
    if not declared.is_finite() or computed != declared:
        raise Lesson16DataError(
            "calculation result does not match expression"
        )
    return VerifiedCalculation(
        expression=_canonical_expression(tokens),
        result=_format_decimal(declared),
        value=computed,
    )


@dataclass(frozen=True)
class Lesson16Example(Lesson14Example):
    reasoning_steps: int

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            isinstance(self.reasoning_steps, bool)
            or not isinstance(self.reasoning_steps, int)
            or not 1 <= self.reasoning_steps <= MAX_CALCULATION_STEPS
        ):
            raise Lesson16DataError(
                "reasoning_steps must be between one and four"
            )
        if len(self.reasoning.encode("utf-8")) > MAX_REASONING_BYTES:
            raise Lesson16DataError("reasoning exceeds the byte budget")


def compact_gsm8k_example(
    record: Mapping[str, object],
    *,
    row_index: int,
) -> Lesson16Example:
    """Convert one GSM8K row to one-to-four exactly verified steps."""

    if not isinstance(record, Mapping):
        raise Lesson16DataError("GSM8K record must be a mapping")
    try:
        base = GSM8KExample.from_record(
            record,
            source_split="train",
            row_index=row_index,
        )
    except ValueError as error:
        raise Lesson16DataError(str(error)) from error
    raw_answer = record.get("answer")
    if not isinstance(raw_answer, str):
        raise Lesson16DataError("GSM8K answer must be text")
    annotations = GSM_ANNOTATION.findall(raw_answer)
    if not 1 <= len(annotations) <= MAX_CALCULATION_STEPS:
        raise Lesson16DataError(
            "GSM8K requires one to four calculation annotations"
        )
    calculations = tuple(
        parse_verified_calculation(value) for value in annotations
    )
    try:
        expected = Decimal(base.exact_answer)
    except InvalidOperation as error:
        raise Lesson16DataError(
            "GSM8K final answer must be decimal"
        ) from error
    if calculations[-1].value != expected:
        raise Lesson16DataError(
            "last GSM8K calculation does not equal final answer"
        )
    reasoning = " ".join(
        f"Compute {item.expression} = {item.result}."
        for item in calculations
    )
    return Lesson16Example(
        record_id=f"{base.record_id}-compact",
        family_id=base.family_id,
        task="gsm8k",
        prompt=base.question,
        reasoning=reasoning,
        final_answer=base.direct_answer,
        exact_answer=base.exact_answer,
        expected_unit=None,
        source_id="gsm8k",
        license_id="MIT",
        reasoning_steps=len(calculations),
    )


def _compact_arithmetic(item: ArithmeticExample) -> Lesson16Example:
    if not verify_arithmetic_example(item):
        raise Lesson16DataError("arithmetic example failed recomputation")
    if item.outer_operator is None:
        reasoning = f"Compute {item.expression} = {item.exact_answer}."
        steps = 1
    else:
        reasoning = item.worked_answer
        steps = 2
    return Lesson16Example(
        record_id=f"{item.example_id}-sft-v2",
        family_id=item.family_id,
        task="arithmetic",
        prompt=item.question,
        reasoning=reasoning,
        final_answer=item.direct_answer,
        exact_answer=item.exact_answer,
        expected_unit=None,
        source_id="project-arithmetic-v1",
        license_id="MIT",
        reasoning_steps=steps,
    )


def _physics_substitution(item: PhysicsExample) -> str:
    if item.formula_id == "kinetic-energy":
        return f"0.5 * {item.left} * {item.right}^2"
    if item.formula_id in {"density", "power", "pressure", "speed"}:
        return f"{item.left} / {item.right}"
    return f"{item.left} * {item.right}"


def _compact_physics(item: PhysicsExample) -> Lesson16Example:
    if not verify_physics_example(item):
        raise Lesson16DataError("physics example failed recomputation")
    left_symbol = item.formula.split("=", maxsplit=1)[0].strip()
    reasoning = (
        f"Formula {item.formula}. Substitute {left_symbol} = "
        f"{_physics_substitution(item)} = {item.exact_answer} {item.unit}."
    )
    return Lesson16Example(
        record_id=f"{item.record_id}-sft-v2",
        family_id=item.family_id,
        task="physics_numeric",
        prompt=item.question,
        reasoning=reasoning,
        final_answer=item.direct_answer,
        exact_answer=item.exact_answer,
        expected_unit=item.unit,
        source_id="project-arithmetic-v1",
        license_id="MIT",
        reasoning_steps=2,
    )


def build_compact_examples(
    *,
    arithmetic: Iterable[ArithmeticExample],
    physics: Iterable[PhysicsExample],
    gsm8k_rows: Iterable[Mapping[str, object]],
) -> tuple[tuple[Lesson16Example, ...], dict[str, object]]:
    """Build exact, compact positive SFT examples and rejection evidence."""

    examples: list[Lesson16Example] = []
    for item in arithmetic:
        if not isinstance(item, ArithmeticExample):
            raise Lesson16DataError(
                "arithmetic must contain ArithmeticExample values"
            )
        examples.append(_compact_arithmetic(item))
    for item in physics:
        if not isinstance(item, PhysicsExample):
            raise Lesson16DataError(
                "physics must contain PhysicsExample values"
            )
        examples.append(_compact_physics(item))
    rejections: Counter[str] = Counter()
    gsm_rows = tuple(gsm8k_rows)
    for row_index, row in enumerate(gsm_rows):
        try:
            examples.append(
                compact_gsm8k_example(row, row_index=row_index)
            )
        except Lesson16DataError as error:
            rejections[str(error)] += 1
    examples.sort(key=lambda item: item.record_id)
    if not examples:
        raise Lesson16DataError("compact example set is empty")
    if len({item.record_id for item in examples}) != len(examples):
        raise Lesson16DataError("compact record IDs must be unique")
    return tuple(examples), {
        "accepted": len(examples),
        "accepted_by_task": dict(
            sorted(Counter(item.task for item in examples).items())
        ),
        "gsm8k_rows": len(gsm_rows),
        "reasoning_steps": dict(
            sorted(
                Counter(
                    str(item.reasoning_steps) for item in examples
                ).items()
            )
        ),
        "rejected": sum(rejections.values()),
        "rejection_reasons": dict(sorted(rejections.items())),
    }


def _evaluation_row(
    example: ArithmeticExample | PhysicsExample,
    *,
    slice_name: str,
) -> dict[str, object]:
    if isinstance(example, ArithmeticExample):
        return {
            "expected_answer": example.exact_answer,
            "expected_unit": None,
            "family_id": example.family_id,
            "prompt": example.question,
            "slice": slice_name,
            "source_id": "project-arithmetic-v1",
            "task": "arithmetic",
        }
    return {
        "expected_answer": example.exact_answer,
        "expected_unit": example.unit,
        "family_id": example.family_id,
        "prompt": example.question,
        "slice": slice_name,
        "source_id": "project-arithmetic-v1",
        "task": "physics_numeric",
    }


def _wrong_number(expected: str) -> str:
    try:
        return _format_decimal(Decimal(expected) + Decimal("1"))
    except InvalidOperation as error:
        raise Lesson16DataError("challenge answer is not decimal") from error


def _hard_negative(row: Mapping[str, object]) -> dict[str, object]:
    wrong = _wrong_number(str(row["expected_answer"]))
    unit = row["expected_unit"]
    suffix = f" {unit}" if isinstance(unit, str) else ""
    return {
        "family_id": row["family_id"],
        "negative_kind": "wrong_substitution_or_final",
        "prompt": row["prompt"],
        "slice": row["slice"],
        "wrong_response": f"The answer is {wrong}{suffix}.",
    }


def build_challenge_evaluation(
    *,
    excluded_families: Iterable[str],
    seed: int = LESSON16_CHALLENGE_SEED,
) -> tuple[
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
    dict[str, object],
]:
    """Build deterministic range/sign/substitution challenge slices."""

    excluded = set(excluded_families)
    rows: list[dict[str, object]] = []
    families = set(excluded)

    def add(
        example: ArithmeticExample | PhysicsExample,
        slice_name: str,
    ) -> bool:
        if example.family_id in families:
            return False
        row = _evaluation_row(example, slice_name=slice_name)
        rows.append(row)
        families.add(example.family_id)
        return True

    for item in generate_arithmetic_examples(count=256, seed=seed):
        if (
            sum(row["slice"] == "in_range" for row in rows) >= 32
            or add(item, "in_range")
        ):
            if sum(row["slice"] == "in_range" for row in rows) >= 32:
                break
    in_range_physics = 0
    for item in generate_physics_examples(count=256, seed=seed):
        if add(item, "in_range"):
            in_range_physics += 1
        if in_range_physics == 32:
            break

    range_arithmetic = 0
    candidate = 0
    while range_arithmetic < 32:
        candidate += 1
        left = 1_000 + candidate * 37
        right = 900 + candidate * 19
        item = ArithmeticExample.create(
            left=left,
            operator="+" if candidate % 2 else "*",
            right=right,
        )
        if add(item, "range_shifted"):
            range_arithmetic += 1
    range_physics = 0
    candidate = 0
    while range_physics < 32:
        candidate += 1
        item = PhysicsExample.create(
            formula_id=PHYSICS_FORMULA_IDS[
                candidate % len(PHYSICS_FORMULA_IDS)
            ],
            left=Decimal(500 + candidate * 11) / Decimal("2"),
            right=(
                Decimal("9.8")
                if PHYSICS_FORMULA_IDS[
                    candidate % len(PHYSICS_FORMULA_IDS)
                ]
                == "weight"
                else Decimal(40 + candidate) / Decimal("2")
            ),
        )
        if add(item, "range_shifted"):
            range_physics += 1

    sign_count = 0
    candidate = 0
    while sign_count < 64:
        candidate += 1
        route = candidate % 4
        if route == 0:
            item = ArithmeticExample.create(
                left=-(candidate + 10),
                operator="+",
                right=candidate - 30,
            )
        elif route == 1:
            item = ArithmeticExample.create(
                left=candidate - 70,
                operator="-",
                right=-(candidate + 3),
            )
        elif route == 2:
            item = ArithmeticExample.create(
                left=-(candidate % 31 + 2),
                operator="*",
                right=candidate % 17 + 2,
            )
        else:
            divisor = -(candidate % 19 + 1)
            quotient = candidate % 41 - 20
            item = ArithmeticExample.create(
                left=divisor * quotient,
                operator="/",
                right=divisor,
            )
        if add(item, "sign_shifted"):
            sign_count += 1

    substitution_count = 0
    candidate = 0
    while substitution_count < 64:
        candidate += 1
        formula_id = PHYSICS_FORMULA_IDS[
            candidate % len(PHYSICS_FORMULA_IDS)
        ]
        item = PhysicsExample.create(
            formula_id=formula_id,
            left=Decimal(candidate * 7 + 3) / Decimal("2"),
            right=(
                Decimal("9.8")
                if formula_id == "weight"
                else Decimal(candidate * 5 + 1) / Decimal("2")
            ),
        )
        if add(item, "substitution_adversarial"):
            substitution_count += 1

    rows.sort(
        key=lambda row: (str(row["slice"]), str(row["family_id"]))
    )
    if len(rows) != 256:
        raise Lesson16DataError("challenge set must contain 256 rows")
    negatives = tuple(_hard_negative(row) for row in rows)
    report = {
        "families": len(rows),
        "hard_negatives_training_eligible": False,
        "slices": dict(
            sorted(Counter(str(row["slice"]) for row in rows).items())
        ),
        "tasks": dict(
            sorted(Counter(str(row["task"]) for row in rows).items())
        ),
    }
    return tuple(rows), negatives, report


def _magnitude_bucket(values: Sequence[str]) -> str:
    largest = max(abs(Decimal(value)) for value in values)
    if largest < 10:
        return "lt10"
    if largest < 100:
        return "lt100"
    if largest < 1_000:
        return "lt1000"
    return "gte1000"


def _training_audit(
    selected: Sequence[Lesson16Example],
    arithmetic: Sequence[ArithmeticExample],
    physics: Sequence[PhysicsExample],
) -> dict[str, object]:
    families = {item.family_id for item in selected}
    selected_arithmetic = [
        item for item in arithmetic if item.family_id in families
    ]
    selected_physics = [
        item for item in physics if item.family_id in families
    ]
    return {
        "arithmetic": {
            "magnitude_buckets": dict(
                sorted(
                    Counter(
                        _magnitude_bucket(
                            tuple(
                                value
                                for value in (
                                    item.left,
                                    item.right,
                                    item.outer_right,
                                )
                                if value is not None
                            )
                        )
                        for item in selected_arithmetic
                    ).items()
                )
            ),
            "negative_operand": dict(
                sorted(
                    Counter(
                        str(
                            any(
                                value is not None
                                and str(value).startswith("-")
                                for value in (
                                    item.left,
                                    item.right,
                                    item.outer_right,
                                )
                            )
                        ).lower()
                        for item in selected_arithmetic
                    ).items()
                )
            ),
            "operators": dict(
                sorted(
                    Counter(
                        item.operator for item in selected_arithmetic
                    ).items()
                )
            ),
        },
        "physics": {
            "formulas": dict(
                sorted(
                    Counter(
                        item.formula_id for item in selected_physics
                    ).items()
                )
            ),
            "units": dict(
                sorted(
                    Counter(item.unit for item in selected_physics).items()
                )
            ),
        },
        "reasoning_bytes": {
            "maximum": max(
                len(item.reasoning.encode("utf-8")) for item in selected
            ),
            "mean": sum(
                len(item.reasoning.encode("utf-8")) for item in selected
            )
            / len(selected),
        },
        "reasoning_steps": dict(
            sorted(
                Counter(str(item.reasoning_steps) for item in selected).items()
            )
        ),
        "tasks": dict(
            sorted(Counter(item.task for item in selected).items())
        ),
    }


def build_lesson16_data(
    *,
    gsm8k_train_path: str | Path,
    evaluation_path: str | Path,
    output_dir: str | Path,
    registry_path: str | Path,
    arithmetic: Iterable[ArithmeticExample],
    physics: Iterable[PhysicsExample],
) -> dict[str, object]:
    """Build the SFT-v2 corpus, challenge set, and hard negatives atomically."""

    gsm_path = Path(gsm8k_train_path)
    expected_gsm_hash = PINNED_INPUTS["gsm8k-train.jsonl"]
    if _sha256_file(gsm_path) != expected_gsm_hash:
        raise Lesson16DataError("pinned GSM8K train hash mismatch")
    primary_evaluation_path = Path(evaluation_path)
    primary_evaluation = load_evaluation_records(
        primary_evaluation_path
    )
    primary_families = {
        str(row["family_id"]) for row in primary_evaluation
    }
    challenge, negatives, challenge_report = build_challenge_evaluation(
        excluded_families=primary_families,
    )
    challenge_families = {
        str(row["family_id"]) for row in challenge
    }
    arithmetic_rows = tuple(arithmetic)
    physics_rows = tuple(physics)
    examples, example_report = build_compact_examples(
        arithmetic=arithmetic_rows,
        physics=physics_rows,
        gsm8k_rows=_load_jsonl(gsm_path),
    )
    selected, selection_report = select_paired_examples(
        examples,
        context_limit=512,
        excluded_families=primary_families | challenge_families,
    )
    destination = Path(output_dir)
    if destination.exists():
        raise Lesson16DataError(
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
        corpus = build_mode_corpus(
            selected,
            temporary / "hybrid_512",
            registry_path=registry_path,
            split_seed=LESSON16_SPLIT_SEED,
            modes=(DIRECT_MODE, THINK_MODE),
            context_limit=512,
        )
        challenge_payload = canonical_jsonl_bytes(challenge)
        negative_payload = canonical_jsonl_bytes(negatives)
        _write_bytes(
            temporary / "challenge_evaluation.jsonl",
            challenge_payload,
        )
        _write_bytes(
            temporary / "hard_negatives.jsonl",
            negative_payload,
        )
        manifest: dict[str, object] = {
            "audit": _training_audit(
                selected,
                arithmetic_rows,
                physics_rows,
            ),
            "challenge": {
                **challenge_report,
                "evaluation_sha256": _sha256_bytes(challenge_payload),
                "hard_negatives_sha256": _sha256_bytes(negative_payload),
            },
            "corpus": {
                "families": corpus["families"],
                "manifest_sha256": _sha256_file(
                    temporary / "hybrid_512" / "manifest.json"
                ),
                "modes": corpus["modes"],
                "records": corpus["records"],
                "tokens": corpus["tokens"],
            },
            "examples": example_report,
            "frozen_evaluation": {
                "challenge_families": len(challenge_families),
                "primary_families": len(primary_families),
                "primary_sha256": _sha256_file(
                    primary_evaluation_path
                ),
                "training_eligible": False,
            },
            "gsm8k_train_sha256": expected_gsm_hash,
            "schema_version": LESSON16_SCHEMA_VERSION,
            "selection": selection_report,
            "split_seed": LESSON16_SPLIT_SEED,
        }
        _write_bytes(
            temporary / "manifest.json",
            _stable_json_bytes(manifest),
        )
        if destination.exists():
            raise Lesson16DataError(
                "output destination appeared during build"
            )
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsm8k-train", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--registry-path", type=Path, required=True)
    parser.add_argument("--arithmetic-count", type=int, default=24_000)
    parser.add_argument("--physics-count", type=int, default=12_000)
    parser.add_argument("--seed", type=int, default=LESSON16_DATA_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    manifest = build_lesson16_data(
        gsm8k_train_path=arguments.gsm8k_train,
        evaluation_path=arguments.evaluation,
        output_dir=arguments.output_dir,
        registry_path=arguments.registry_path,
        arithmetic=generate_arithmetic_examples(
            count=arguments.arithmetic_count,
            seed=arguments.seed,
        ),
        physics=generate_physics_examples(
            count=arguments.physics_count,
            seed=arguments.seed,
        ),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
