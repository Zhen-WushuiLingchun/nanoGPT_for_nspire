"""Strict public parsers and exact Lesson 12 math/physics curricula."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import random
import re
from typing import Iterable, Mapping
import unicodedata


MAX_CONVERSATION_TOKENS = 256
CONVERSATION_SPECIAL_TOKENS = 5
_GSM8K_FINAL_PATTERN = re.compile(
    r"####\s*"
    r"([-+]?(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?)"
    r"\s*\Z"
)


class CurriculumError(ValueError):
    """Raised when a Lesson 12 record is unsafe, ambiguous, or unverifiable."""


def _stable_digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _non_negative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CurriculumError(f"{name} must be a non-negative integer")
    return value


def _normalize_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CurriculumError(f"{name} must be a non-empty string")
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(
        " ".join(line.split()) for line in normalized.split("\n")
    ).strip()
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    if "\ufffd" in normalized:
        raise CurriculumError(f"{name} contains a replacement character")
    for character in normalized:
        if (
            unicodedata.category(character) == "Cc"
            and character not in {"\n", "\t"}
        ):
            raise CurriculumError(f"{name} contains a control character")
    try:
        normalized.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise CurriculumError(f"{name} is not valid UTF-8") from error
    return normalized


def _format_decimal(value: Decimal) -> str:
    if not value.is_finite():
        raise CurriculumError("numeric value must be finite")
    text = format(value.normalize(), "f")
    return "0" if text == "-0" else text


def _positive_decimal(value: object, name: str) -> Decimal:
    if isinstance(value, bool):
        raise CurriculumError(f"{name} must be a positive finite Decimal")
    if isinstance(value, int):
        result = Decimal(value)
    elif isinstance(value, Decimal):
        result = value
    else:
        raise CurriculumError(f"{name} must be a positive finite Decimal")
    if not result.is_finite() or result <= 0:
        raise CurriculumError(f"{name} must be positive and finite")
    if len(result.as_tuple().digits) > 10:
        raise CurriculumError(f"{name} exceeds the digit limit")
    return result


def parse_gsm8k_final_answer(answer: object) -> str:
    """Extract GSM8K's final decimal answer without evaluating annotations."""

    normalized = _normalize_text(answer, "answer")
    match = _GSM8K_FINAL_PATTERN.search(normalized)
    if match is None:
        raise CurriculumError(
            "GSM8K final answer must end with '#### <decimal>'"
        )
    number = match.group(1).replace(",", "")
    try:
        value = Decimal(number)
    except InvalidOperation as error:
        raise CurriculumError("GSM8K final answer is not a decimal") from error
    if not value.is_finite():
        raise CurriculumError("GSM8K final answer must be finite")
    return _format_decimal(value)


@dataclass(frozen=True)
class GSM8KExample:
    """One official GSM8K row with a concise independently parsed target."""

    question: str
    exact_answer: str
    direct_answer: str
    source_split: str
    row_index: int
    family_id: str
    record_id: str

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, object],
        *,
        source_split: str,
        row_index: int,
    ) -> GSM8KExample:
        if not isinstance(record, Mapping):
            raise CurriculumError("GSM8K record must be a mapping")
        if source_split not in {"train", "test"}:
            raise CurriculumError("source_split must be 'train' or 'test'")
        row_index = _non_negative_integer(row_index, "row_index")
        if "question" not in record or "answer" not in record:
            raise CurriculumError("GSM8K record requires question and answer")
        question = _normalize_text(record["question"], "question")
        exact_answer = parse_gsm8k_final_answer(record["answer"])
        direct_answer = f"The answer is {exact_answer}."
        if (
            len(question.encode("utf-8"))
            + len(direct_answer.encode("utf-8"))
            + CONVERSATION_SPECIAL_TOKENS
            > MAX_CONVERSATION_TOKENS
        ):
            raise CurriculumError("GSM8K conversation exceeds context limit")
        family_id = f"gsm8k-{_stable_digest({'question': question})[:24]}"
        return cls(
            question=question,
            exact_answer=exact_answer,
            direct_answer=direct_answer,
            source_split=source_split,
            row_index=row_index,
            family_id=family_id,
            record_id=f"{family_id}:{source_split}:{row_index}",
        )


def _required_message_text(
    message: Mapping[str, object],
    field: str,
) -> str:
    if field not in message:
        raise CurriculumError(f"OASST1 message is missing {field}")
    return _normalize_text(message[field], field)


def _required_identifier(
    message: Mapping[str, object],
    field: str,
) -> str:
    value = message.get(field)
    if not isinstance(value, str) or not value:
        raise CurriculumError(f"OASST1 {field} must be a non-empty string")
    return value


@dataclass(frozen=True)
class OASSTPair:
    """One short English top-ranked direct prompt/assistant reply."""

    question: str
    answer: str
    message_tree_id: str
    prompt_message_id: str
    answer_message_id: str
    family_id: str
    record_id: str

    @classmethod
    def from_messages(
        cls,
        prompt: Mapping[str, object],
        assistant: Mapping[str, object],
    ) -> OASSTPair:
        if not isinstance(prompt, Mapping) or not isinstance(
            assistant,
            Mapping,
        ):
            raise CurriculumError("OASST1 messages must be mappings")
        if prompt.get("role") != "prompter":
            raise CurriculumError("OASST1 prompt role must be prompter")
        if assistant.get("role") != "assistant":
            raise CurriculumError("OASST1 reply role must be assistant")
        if prompt.get("lang") != "en" or assistant.get("lang") != "en":
            raise CurriculumError("OASST1 pair must be English")
        if prompt.get("deleted") is not False or assistant.get("deleted") is not False:
            raise CurriculumError("OASST1 deleted messages are ineligible")
        if assistant.get("rank") != 0:
            raise CurriculumError("OASST1 assistant rank must be zero")
        prompt_id = _required_identifier(prompt, "message_id")
        assistant_id = _required_identifier(assistant, "message_id")
        prompt_tree = _required_identifier(prompt, "message_tree_id")
        assistant_tree = _required_identifier(assistant, "message_tree_id")
        if prompt_tree != assistant_tree:
            raise CurriculumError("OASST1 messages must share one tree")
        if assistant.get("parent_id") != prompt_id:
            raise CurriculumError(
                "OASST1 assistant must be a direct reply to the prompt"
            )
        if prompt.get("parent_id") is not None:
            raise CurriculumError("OASST1 prompt must be a root message")
        question = _required_message_text(prompt, "text")
        answer = _required_message_text(assistant, "text")
        if (
            len(question.encode("utf-8"))
            + len(answer.encode("utf-8"))
            + CONVERSATION_SPECIAL_TOKENS
            > MAX_CONVERSATION_TOKENS
        ):
            raise CurriculumError("OASST1 conversation exceeds context limit")
        return cls(
            question=question,
            answer=answer,
            message_tree_id=prompt_tree,
            prompt_message_id=prompt_id,
            answer_message_id=assistant_id,
            family_id=f"oasst1-{prompt_tree}",
            record_id=f"oasst1-{assistant_id}",
        )


_PHYSICS_FORMULAS: dict[str, dict[str, object]] = {
    "density": {
        "left_name": "mass",
        "left_unit": "kg",
        "right_name": "volume",
        "right_unit": "m^3",
        "result_name": "density",
        "unit": "kg/m^3",
        "symbol": "rho = m / V",
        "operation": "divide",
    },
    "force": {
        "left_name": "mass",
        "left_unit": "kg",
        "right_name": "acceleration",
        "right_unit": "m/s^2",
        "result_name": "force",
        "unit": "N",
        "symbol": "F = m a",
        "operation": "multiply",
    },
    "kinetic-energy": {
        "left_name": "mass",
        "left_unit": "kg",
        "right_name": "speed",
        "right_unit": "m/s",
        "result_name": "kinetic energy",
        "unit": "J",
        "symbol": "K = 1/2 m v^2",
        "operation": "kinetic",
    },
    "momentum": {
        "left_name": "mass",
        "left_unit": "kg",
        "right_name": "velocity",
        "right_unit": "m/s",
        "result_name": "momentum",
        "unit": "kg*m/s",
        "symbol": "p = m v",
        "operation": "multiply",
    },
    "ohms-law": {
        "left_name": "current",
        "left_unit": "A",
        "right_name": "resistance",
        "right_unit": "ohm",
        "result_name": "voltage",
        "unit": "V",
        "symbol": "V = I R",
        "operation": "multiply",
    },
    "power": {
        "left_name": "energy",
        "left_unit": "J",
        "right_name": "time",
        "right_unit": "s",
        "result_name": "power",
        "unit": "W",
        "symbol": "P = E / t",
        "operation": "divide",
    },
    "pressure": {
        "left_name": "force",
        "left_unit": "N",
        "right_name": "area",
        "right_unit": "m^2",
        "result_name": "pressure",
        "unit": "Pa",
        "symbol": "p = F / A",
        "operation": "divide",
    },
    "speed": {
        "left_name": "distance",
        "left_unit": "m",
        "right_name": "time",
        "right_unit": "s",
        "result_name": "speed",
        "unit": "m/s",
        "symbol": "v = d / t",
        "operation": "divide",
    },
    "wave-speed": {
        "left_name": "frequency",
        "left_unit": "Hz",
        "right_name": "wavelength",
        "right_unit": "m",
        "result_name": "wave speed",
        "unit": "m/s",
        "symbol": "v = f lambda",
        "operation": "multiply",
    },
    "weight": {
        "left_name": "mass",
        "left_unit": "kg",
        "right_name": "gravitational field strength",
        "right_unit": "N/kg",
        "result_name": "weight",
        "unit": "N",
        "symbol": "W = m g",
        "operation": "multiply",
    },
}


def _evaluate_formula(formula_id: str, left: Decimal, right: Decimal) -> Decimal:
    definition = _PHYSICS_FORMULAS[formula_id]
    operation = definition["operation"]
    if operation == "multiply":
        return left * right
    if operation == "divide":
        return left / right
    if operation == "kinetic":
        return Decimal("0.5") * left * right * right
    raise AssertionError(f"unknown internal operation {operation}")


@dataclass(frozen=True)
class PhysicsExample:
    """One exactly computed two-quantity introductory-physics problem."""

    formula_id: str
    left: str
    right: str
    left_name: str
    left_unit: str
    right_name: str
    right_unit: str
    result_name: str
    formula: str
    exact_answer: str
    unit: str
    question: str
    direct_answer: str
    worked_answer: str
    family_id: str
    record_id: str

    @classmethod
    def create(
        cls,
        *,
        formula_id: str,
        left: Decimal,
        right: Decimal,
    ) -> PhysicsExample:
        if formula_id not in _PHYSICS_FORMULAS:
            raise CurriculumError(f"unsupported physics formula {formula_id!r}")
        left_value = _positive_decimal(left, "left")
        right_value = _positive_decimal(right, "right")
        result = _evaluate_formula(formula_id, left_value, right_value)
        definition = _PHYSICS_FORMULAS[formula_id]
        left_text = _format_decimal(left_value)
        right_text = _format_decimal(right_value)
        result_text = _format_decimal(result)
        payload = {
            "formula_id": formula_id,
            "left": left_text,
            "right": right_text,
        }
        family_id = f"physics-{_stable_digest(payload)[:24]}"
        left_name = str(definition["left_name"])
        right_name = str(definition["right_name"])
        left_unit = str(definition["left_unit"])
        right_unit = str(definition["right_unit"])
        result_name = str(definition["result_name"])
        formula = str(definition["symbol"])
        unit = str(definition["unit"])
        question = (
            f"A system has {left_name} {left_text} {left_unit} and "
            f"{right_name} {right_text} {right_unit}. What is its "
            f"{result_name}?"
        )
        direct_answer = f"The {result_name} is {result_text} {unit}."
        worked_answer = (
            f"Use {formula}. Substitution gives {result_text} {unit}."
        )
        if (
            len(question.encode("utf-8"))
            + len(worked_answer.encode("utf-8"))
            + CONVERSATION_SPECIAL_TOKENS
            > MAX_CONVERSATION_TOKENS
        ):
            raise CurriculumError("physics conversation exceeds context limit")
        return cls(
            formula_id=formula_id,
            left=left_text,
            right=right_text,
            left_name=left_name,
            left_unit=left_unit,
            right_name=right_name,
            right_unit=right_unit,
            result_name=result_name,
            formula=formula,
            exact_answer=result_text,
            unit=unit,
            question=question,
            direct_answer=direct_answer,
            worked_answer=worked_answer,
            family_id=family_id,
            record_id=f"{family_id}-canonical",
        )


def verify_physics_example(example: object) -> bool:
    """Recompute one physics record and compare all derived invariants."""

    if not isinstance(example, PhysicsExample):
        return False
    try:
        rebuilt = PhysicsExample.create(
            formula_id=example.formula_id,
            left=Decimal(example.left),
            right=Decimal(example.right),
        )
    except (CurriculumError, InvalidOperation):
        return False
    return rebuilt == example


def generate_physics_examples(
    *,
    count: int,
    seed: int,
) -> tuple[PhysicsExample, ...]:
    """Generate a balanced deterministic set of exact physics families."""

    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise CurriculumError("count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise CurriculumError("seed must be an integer")
    formula_ids = tuple(sorted(_PHYSICS_FORMULAS))
    generator = random.Random(seed)
    examples: list[PhysicsExample] = []
    seen: set[str] = set()
    attempts = 0
    while len(examples) < count:
        attempts += 1
        if attempts > count * 100:
            raise CurriculumError("could not generate unique physics examples")
        formula_id = formula_ids[len(examples) % len(formula_ids)]
        if formula_id == "weight":
            left = Decimal(generator.randint(1, 10_000)) / Decimal(10)
            right = Decimal("9.8")
        elif formula_id == "kinetic-energy":
            left = Decimal(generator.randint(1, 80)) / Decimal(2)
            right = Decimal(generator.randint(1, 30))
        elif formula_id in {"density", "power", "pressure", "speed"}:
            right = Decimal(generator.randint(1, 25))
            multiplier = Decimal(generator.randint(1, 80)) / Decimal(2)
            left = right * multiplier
        else:
            left = Decimal(generator.randint(1, 100)) / Decimal(2)
            right = Decimal(generator.randint(1, 60)) / Decimal(2)
        example = PhysicsExample.create(
            formula_id=formula_id,
            left=left,
            right=right,
        )
        if example.family_id in seen:
            continue
        seen.add(example.family_id)
        examples.append(example)
    return tuple(examples)


def canonical_jsonl_bytes(rows: Iterable[Mapping[str, object]]) -> bytes:
    """Serialize finite JSON mappings as deterministic UTF-8 JSONL."""

    payloads: list[bytes] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise CurriculumError("JSONL rows must be mappings")
        try:
            text = json.dumps(
                dict(row),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise CurriculumError("row is not finite canonical JSON") from error
        payloads.append((text + "\n").encode("utf-8"))
    return b"".join(payloads)
