"""Deterministic rollout curriculum and mode schedule for Lesson 17."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
import random
from typing import Iterable, Mapping, Sequence

from nanogpt_nspire.lesson12_curriculum import (
    PhysicsExample,
    generate_physics_examples,
    verify_physics_example,
)
from nanogpt_nspire.math_curriculum import (
    ArithmeticExample,
    verify_arithmetic_example,
)
from nanogpt_nspire.reasoning_format import (
    DIRECT_MODE,
    THINK_MODE,
    SUPPORTED_MODES,
)


LESSON17_DATA_SEED = 20260730
FORMAL_POLICY_SEEDS = (20260731, 20260732, 20260733)


class Lesson17DataError(ValueError):
    """Raised when rollout prompts violate the frozen curriculum contract."""


@dataclass(frozen=True)
class RLProblem:
    record_id: str
    family_id: str
    task: str
    prompt: str
    expected_answer: str
    expected_unit: str | None
    formula: str | None
    difficulty: str
    source_id: str

    def __post_init__(self) -> None:
        for name in (
            "record_id",
            "family_id",
            "prompt",
            "expected_answer",
            "difficulty",
            "source_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise Lesson17DataError(f"{name} must be non-empty")
        if self.task not in {"arithmetic", "physics_numeric"}:
            raise Lesson17DataError("task is unsupported")
        if self.task == "arithmetic":
            if self.expected_unit is not None or self.formula is not None:
                raise Lesson17DataError(
                    "arithmetic must not declare unit or formula"
                )
        elif (
            not isinstance(self.expected_unit, str)
            or not self.expected_unit
            or not isinstance(self.formula, str)
            or not self.formula
        ):
            raise Lesson17DataError(
                "physics requires unit and formula"
            )
        try:
            self.prompt.encode("ascii", errors="strict")
        except UnicodeEncodeError as error:
            raise Lesson17DataError(
                "rollout prompt must be ASCII"
            ) from error

    def evaluation_record(self) -> dict[str, object]:
        return {
            "expected_answer": self.expected_answer,
            "expected_unit": self.expected_unit,
            "family_id": self.family_id,
            "prompt": self.prompt,
            "source_id": self.source_id,
            "task": self.task,
        }


@dataclass(frozen=True)
class ScheduledPrompt:
    schedule_id: str
    update: int
    slot: int
    mode: str
    problem: RLProblem

    def __post_init__(self) -> None:
        if not isinstance(self.schedule_id, str) or not self.schedule_id:
            raise Lesson17DataError("schedule_id must be non-empty")
        if (
            isinstance(self.update, bool)
            or not isinstance(self.update, int)
            or self.update <= 0
        ):
            raise Lesson17DataError("update must be positive")
        if (
            isinstance(self.slot, bool)
            or not isinstance(self.slot, int)
            or self.slot < 0
        ):
            raise Lesson17DataError("slot must be non-negative")
        if self.mode not in SUPPORTED_MODES:
            raise Lesson17DataError("mode is unsupported")
        if not isinstance(self.problem, RLProblem):
            raise Lesson17DataError("problem must be RLProblem")


def _stable_rank(seed: int, family_id: str) -> bytes:
    return hashlib.sha256(
        f"lesson17:{seed}:{family_id}".encode("ascii")
    ).digest()


def _arithmetic_candidates(
    *,
    count: int,
    seed: int,
) -> tuple[ArithmeticExample, ...]:
    generator = random.Random(seed)
    candidates: list[ArithmeticExample] = []
    seen: set[str] = set()
    attempts = 0
    while len(candidates) < count:
        attempts += 1
        if attempts > count * 100:
            raise Lesson17DataError(
                "could not generate unique arithmetic candidates"
            )
        route = attempts % 8
        if route == 0:
            item = ArithmeticExample.create(
                left=generator.randint(-12, 30),
                operator="+",
                right=generator.randint(-12, 30),
            )
        elif route == 1:
            item = ArithmeticExample.create(
                left=generator.randint(-12, 30),
                operator="-",
                right=generator.randint(-12, 30),
            )
        elif route == 2:
            item = ArithmeticExample.create(
                left=generator.randint(-12, 12),
                operator="*",
                right=generator.randint(-12, 12),
            )
        elif route == 3:
            divisor = generator.randint(1, 12)
            quotient = generator.randint(-12, 12)
            item = ArithmeticExample.create(
                left=divisor * quotient,
                operator="/",
                right=divisor,
            )
        elif route == 4:
            item = ArithmeticExample.create(
                left=Decimal(generator.randint(-120, 300))
                / Decimal(10),
                operator="+",
                right=Decimal(generator.randint(-120, 300))
                / Decimal(10),
            )
        elif route == 5:
            item = ArithmeticExample.create(
                left=Decimal(generator.randint(-50, 50))
                / Decimal(10),
                operator="*",
                right=Decimal(generator.randint(-50, 50))
                / Decimal(10),
            )
        elif route == 6:
            item = ArithmeticExample.create(
                left=generator.randint(-10, 20),
                operator="+",
                right=generator.randint(-10, 20),
                outer_operator="*",
                outer_right=generator.randint(-5, 5),
            )
        else:
            item = ArithmeticExample.create(
                left=Decimal(generator.randint(-120, 300))
                / Decimal(10),
                operator="-",
                right=Decimal(generator.randint(-120, 300))
                / Decimal(10),
            )
        if item.family_id in seen:
            continue
        if not verify_arithmetic_example(item):
            raise Lesson17DataError("arithmetic recomputation failed")
        seen.add(item.family_id)
        candidates.append(item)
    return tuple(candidates)


def _from_arithmetic(item: ArithmeticExample) -> RLProblem:
    return RLProblem(
        record_id=f"{item.example_id}-rl17",
        family_id=item.family_id,
        task="arithmetic",
        prompt=item.question,
        expected_answer=item.exact_answer,
        expected_unit=None,
        formula=None,
        difficulty=item.category,
        source_id="project-arithmetic-v1",
    )


def _from_physics(item: PhysicsExample) -> RLProblem:
    if not verify_physics_example(item):
        raise Lesson17DataError("physics recomputation failed")
    return RLProblem(
        record_id=f"{item.record_id}-rl17",
        family_id=item.family_id,
        task="physics_numeric",
        prompt=item.question,
        expected_answer=item.exact_answer,
        expected_unit=item.unit,
        formula=item.formula,
        difficulty=item.formula_id,
        source_id="project-arithmetic-v1",
    )


def build_lesson17_problem_pool(
    *,
    count_per_task: int,
    seed: int,
    excluded_families: Iterable[str],
) -> tuple[RLProblem, ...]:
    """Build a deterministic balanced pool disjoint from every holdout."""

    if (
        isinstance(count_per_task, bool)
        or not isinstance(count_per_task, int)
        or count_per_task <= 0
    ):
        raise Lesson17DataError("count_per_task must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise Lesson17DataError("seed must be an integer")
    excluded = set(excluded_families)
    arithmetic = [
        _from_arithmetic(item)
        for item in _arithmetic_candidates(
            count=max(count_per_task * 4, 512),
            seed=seed,
        )
        if item.family_id not in excluded
    ]
    physics = [
        _from_physics(item)
        for item in generate_physics_examples(
            count=max(count_per_task * 4, 512),
            seed=seed,
        )
        if item.family_id not in excluded
    ]
    arithmetic.sort(key=lambda item: _stable_rank(seed, item.family_id))
    physics.sort(key=lambda item: _stable_rank(seed, item.family_id))
    if (
        len(arithmetic) < count_per_task
        or len(physics) < count_per_task
    ):
        raise Lesson17DataError("insufficient disjoint prompt families")
    selected = arithmetic[:count_per_task] + physics[:count_per_task]
    selected.sort(key=lambda item: (item.task, item.family_id))
    if len({item.family_id for item in selected}) != len(selected):
        raise Lesson17DataError("problem families must be unique")
    if {item.family_id for item in selected} & excluded:
        raise Lesson17DataError("problem pool overlaps a holdout")
    return tuple(selected)


def build_prompt_schedule(
    problems: Sequence[RLProblem],
    *,
    seed: int,
    updates: int,
    prompts_per_update: int,
) -> tuple[ScheduledPrompt, ...]:
    """Freeze two tasks and two modes in every formal update."""

    if prompts_per_update != 4:
        raise Lesson17DataError(
            "Lesson 17 freezes four prompts per update"
        )
    if (
        isinstance(updates, bool)
        or not isinstance(updates, int)
        or updates <= 0
    ):
        raise Lesson17DataError("updates must be positive")
    grouped = {
        task: [item for item in problems if item.task == task]
        for task in ("arithmetic", "physics_numeric")
    }
    required_per_task = updates * 2
    if any(len(items) < required_per_task for items in grouped.values()):
        raise Lesson17DataError("problem pool is too small for schedule")
    generator = random.Random(seed)
    for items in grouped.values():
        generator.shuffle(items)
    schedule: list[ScheduledPrompt] = []
    for update in range(1, updates + 1):
        rows: list[tuple[RLProblem, str]] = []
        offset = (update - 1) * 2
        for task_index, task in enumerate(
            ("arithmetic", "physics_numeric")
        ):
            first_mode, second_mode = (
                (DIRECT_MODE, THINK_MODE)
                if (update + task_index) % 2
                else (THINK_MODE, DIRECT_MODE)
            )
            rows.extend(
                (
                    (grouped[task][offset], first_mode),
                    (grouped[task][offset + 1], second_mode),
                )
            )
        generator.shuffle(rows)
        for slot, (problem, mode) in enumerate(rows):
            schedule.append(
                ScheduledPrompt(
                    schedule_id=(
                        f"policy-{seed}:update-{update:02d}:slot-{slot}"
                    ),
                    update=update,
                    slot=slot,
                    mode=mode,
                    problem=problem,
                )
            )
    return tuple(schedule)


def canonical_problem_pool_bytes(
    problems: Sequence[RLProblem],
) -> bytes:
    rows = [
        {
            "difficulty": item.difficulty,
            **item.evaluation_record(),
            "formula": item.formula,
            "record_id": item.record_id,
        }
        for item in problems
    ]
    return b"".join(
        (
            json.dumps(
                row,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
        for row in rows
    )
