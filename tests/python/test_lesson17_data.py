from __future__ import annotations

from nanogpt_nspire.lesson17_data import (
    FORMAL_POLICY_SEEDS,
    build_lesson17_problem_pool,
    build_prompt_schedule,
)


def test_problem_pool_is_deterministic_balanced_and_disjoint() -> None:
    excluded = {"blocked-family"}
    first = build_lesson17_problem_pool(
        count_per_task=128,
        seed=20260730,
        excluded_families=excluded,
    )
    second = build_lesson17_problem_pool(
        count_per_task=128,
        seed=20260730,
        excluded_families=excluded,
    )

    assert first == second
    assert len(first) == 256
    assert len({item.family_id for item in first}) == 256
    assert {item.task for item in first} == {
        "arithmetic",
        "physics_numeric",
    }
    assert sum(item.task == "arithmetic" for item in first) == 128
    assert sum(item.task == "physics_numeric" for item in first) == 128
    assert not ({item.family_id for item in first} & excluded)


def test_prompt_schedule_freezes_two_tasks_and_two_modes_per_update() -> None:
    pool = build_lesson17_problem_pool(
        count_per_task=128,
        seed=20260730,
        excluded_families=(),
    )

    schedule = build_prompt_schedule(
        pool,
        seed=FORMAL_POLICY_SEEDS[0],
        updates=16,
        prompts_per_update=4,
    )

    assert len(schedule) == 64
    assert len({item.schedule_id for item in schedule}) == 64
    for update in range(1, 17):
        rows = [item for item in schedule if item.update == update]
        assert len(rows) == 4
        assert sum(item.mode == "direct" for item in rows) == 2
        assert sum(item.mode == "think" for item in rows) == 2
        assert sum(item.problem.task == "arithmetic" for item in rows) == 2
        assert sum(item.problem.task == "physics_numeric" for item in rows) == 2


def test_policy_seeds_are_three_distinct_frozen_values() -> None:
    assert FORMAL_POLICY_SEEDS == (20260731, 20260732, 20260733)
