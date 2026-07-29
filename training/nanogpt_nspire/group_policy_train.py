"""Frozen end-to-end RLVR and direct-RLAIF training for Lesson 17."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import time

import torch

from nanogpt_nspire.assistant_eval import load_evaluation_records
from nanogpt_nspire.base_train import (
    _atomic_torch_save,
    _cpu_state_dict,
)
from nanogpt_nspire.byte_tokenizer import VOCAB_SIZE
from nanogpt_nspire.direct_small_train import configure_adamw
from nanogpt_nspire.efficient_context import (
    ARCHITECTURE_NAME,
    GQA_ALIBI_SFT_ROUTE,
    GQA_ALIBI_SFT_V2_ROUTE,
    lesson15_efficient_config,
    load_efficient_checkpoint,
    model_state_sha256,
)
from nanogpt_nspire.group_policy import (
    collate_trajectories,
    group_policy_loss,
    normalize_group_advantages,
    reference_token_log_probs,
)
from nanogpt_nspire.judge_cache import (
    JudgeCache,
    judge_with_cache,
    render_judge_response,
)
from nanogpt_nspire.lesson17_data import (
    FORMAL_POLICY_SEEDS,
    LESSON17_DATA_SEED,
    RLProblem,
    ScheduledPrompt,
    build_lesson17_problem_pool,
    build_prompt_schedule,
    canonical_problem_pool_bytes,
)
from nanogpt_nspire.lesson17_routes import (
    COMBINED_ROUTE,
    DIRECT_RLAIF_ROUTE,
    RLVR_ROUTE,
    TRAINABLE_ROUTES,
)
from nanogpt_nspire.models.efficient_long_context_gpt import (
    ALIBI_POSITIONS,
    EfficientLongContextGPT,
)
from nanogpt_nspire.preference_judge import (
    JudgeCandidate,
    JudgeProblem,
    PreferenceJudgeClient,
    ordered_candidates,
)
from nanogpt_nspire.reasoning_eval import score_mode_completion
from nanogpt_nspire.rl_rewards import (
    combined_reward,
    direct_rlaif_reward,
    verifier_reward,
)
from nanogpt_nspire.rl_rollout import (
    RolloutTrajectory,
    sample_mode_group,
)
from nanogpt_nspire.secret_safety import assert_secret_free
from nanogpt_nspire.training_support import (
    resolve_device,
    sha256_file,
    synchronize,
    write_json_atomic,
)


START_ROUTES = frozenset(
    {GQA_ALIBI_SFT_ROUTE, GQA_ALIBI_SFT_V2_ROUTE}
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class RouteReward:
    candidate_id: str
    total: float
    verifier_total: float
    verifier_numeric: float
    verifier_unit: float
    verifier_format: float
    ai_reward: float | None


def compose_route_rewards(
    *,
    route: str,
    candidate_ids: Sequence[str],
    local_scores: Sequence[Mapping[str, object]],
    ai_rewards: Mapping[str, float] | None,
) -> tuple[RouteReward, ...]:
    """Compose route-specific reward without allowing source leakage."""

    if route not in TRAINABLE_ROUTES:
        raise ValueError("route is not trainable")
    if (
        not candidate_ids
        or len(candidate_ids) != len(local_scores)
        or len(set(candidate_ids)) != len(candidate_ids)
    ):
        raise ValueError(
            "candidate IDs and local scores must align and be unique"
        )
    needs_ai = route in {DIRECT_RLAIF_ROUTE, COMBINED_ROUTE}
    if needs_ai:
        if ai_rewards is None or set(ai_rewards) != set(candidate_ids):
            raise ValueError(
                "AI reward must exist for every candidate"
            )
    elif ai_rewards is not None:
        raise ValueError("RLVR route must not receive AI rewards")
    rewards: list[RouteReward] = []
    for candidate_id, local_score in zip(
        candidate_ids,
        local_scores,
        strict=True,
    ):
        verifier = verifier_reward(local_score)
        ai_value = (
            None
            if ai_rewards is None
            else float(ai_rewards[candidate_id])
        )
        if route == RLVR_ROUTE:
            total = verifier.total
        elif route == DIRECT_RLAIF_ROUTE:
            assert ai_value is not None
            total = direct_rlaif_reward(
                local_score,
                ai_reward=ai_value,
            )
        else:
            assert ai_value is not None
            total = combined_reward(
                verifier,
                ai_reward=ai_value,
            )
        rewards.append(
            RouteReward(
                candidate_id=candidate_id,
                total=total,
                verifier_total=verifier.total,
                verifier_numeric=verifier.numeric,
                verifier_unit=verifier.unit,
                verifier_format=verifier.format,
                ai_reward=ai_value,
            )
        )
    return tuple(rewards)


def build_judge_problem(
    scheduled: ScheduledPrompt,
    trajectories: Sequence[RolloutTrajectory],
) -> JudgeProblem:
    """Build an answer-free judge input from one sampled candidate group."""

    if not isinstance(scheduled, ScheduledPrompt):
        raise ValueError("scheduled must be ScheduledPrompt")
    if len(trajectories) < 2:
        raise ValueError("judge group needs at least two trajectories")
    if any(
        item.schedule_id != scheduled.schedule_id
        for item in trajectories
    ):
        raise ValueError("judge trajectories do not share schedule ID")
    return JudgeProblem(
        schedule_id=scheduled.schedule_id,
        task=scheduled.problem.task,
        mode=scheduled.mode,
        prompt=scheduled.problem.prompt,
        candidates=tuple(
            JudgeCandidate(
                candidate_id=item.candidate_id,
                response=render_judge_response(item),
            )
            for item in trajectories
        ),
    )


@dataclass(frozen=True)
class PolicyTrainingConfig:
    route: str
    output_dir: Path
    start_checkpoint: Path
    start_checkpoint_sha256: str
    start_route: str
    source_commit: str
    seed: int
    device: str = "cuda"
    rollout_updates: int = 16
    prompt_groups_per_update: int = 4
    group_size: int = 8
    max_new_tokens: int = 256
    temperature: float = 0.8
    policy_epochs: int = 2
    policy_micro_batch_size: int = 4
    learning_rate: float = 5e-6
    clip_epsilon: float = 0.2
    kl_beta: float = 0.02
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    use_bfloat16: bool = True

    @property
    def profile(self) -> str:
        return "formal"

    @property
    def optimizer_steps(self) -> int:
        return self.rollout_updates * self.policy_epochs

    def validate(self) -> None:
        if self.route not in TRAINABLE_ROUTES:
            raise ValueError("training route is unsupported")
        if self.start_route not in START_ROUTES:
            raise ValueError("start route is unsupported")
        if (
            not isinstance(self.start_checkpoint_sha256, str)
            or _SHA256_PATTERN.fullmatch(
                self.start_checkpoint_sha256
            )
            is None
        ):
            raise ValueError("start checkpoint SHA-256 is invalid")
        if not isinstance(self.source_commit, str) or not self.source_commit:
            raise ValueError("source_commit must be non-empty")
        if self.seed not in FORMAL_POLICY_SEEDS:
            raise ValueError("seed is not a frozen formal policy seed")
        expected_integers = {
            "rollout_updates": 16,
            "prompt_groups_per_update": 4,
            "group_size": 8,
            "max_new_tokens": 256,
            "policy_epochs": 2,
            "policy_micro_batch_size": 4,
        }
        for name, expected in expected_integers.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} is frozen at {expected}")
        expected_scalars = {
            "temperature": 0.8,
            "learning_rate": 5e-6,
            "clip_epsilon": 0.2,
            "kl_beta": 0.02,
            "max_grad_norm": 1.0,
            "weight_decay": 0.0,
            "beta1": 0.9,
            "beta2": 0.95,
        }
        for name, expected in expected_scalars.items():
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isclose(
                    float(value),
                    expected,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            ):
                raise ValueError(f"{name} is frozen at {expected}")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be non-empty")
        if not isinstance(self.use_bfloat16, bool):
            raise ValueError("use_bfloat16 must be boolean")
        assert_secret_free(
            self.public_record(),
            context="policy training configuration",
        )

    def public_record(self) -> dict[str, object]:
        return {
            **{
                key: value
                for key, value in asdict(self).items()
                if key
                not in {
                    "output_dir",
                    "start_checkpoint",
                }
            },
            "optimizer_steps": self.optimizer_steps,
            "profile": self.profile,
            "output_dir": str(self.output_dir),
            "start_checkpoint": str(self.start_checkpoint),
        }


def frozen_policy_training_config(
    *,
    route: str,
    output_dir: str | Path,
    start_checkpoint: str | Path,
    start_checkpoint_sha256: str,
    start_route: str,
    source_commit: str,
    seed: int,
    **overrides: object,
) -> PolicyTrainingConfig:
    frozen = {
        "rollout_updates",
        "prompt_groups_per_update",
        "group_size",
        "max_new_tokens",
        "temperature",
        "policy_epochs",
        "policy_micro_batch_size",
        "learning_rate",
        "clip_epsilon",
        "kl_beta",
        "max_grad_norm",
        "weight_decay",
        "beta1",
        "beta2",
    } & set(overrides)
    if frozen:
        raise ValueError(
            "Lesson 17 optimization is frozen; remove overrides: "
            + ", ".join(sorted(frozen))
        )
    defaults: dict[str, object] = {
        "device": "cuda",
        "use_bfloat16": True,
    }
    defaults.update(overrides)
    config = PolicyTrainingConfig(
        route=route,
        output_dir=Path(output_dir),
        start_checkpoint=Path(start_checkpoint),
        start_checkpoint_sha256=start_checkpoint_sha256,
        start_route=start_route,
        source_commit=source_commit,
        seed=seed,
        **defaults,
    )
    config.validate()
    return config


@dataclass(frozen=True)
class SmokePolicyTrainingConfig(PolicyTrainingConfig):
    """Explicit one-update profile that cannot satisfy the formal contract."""

    @property
    def profile(self) -> str:
        return "smoke"

    def validate(self) -> None:
        if self.route not in TRAINABLE_ROUTES:
            raise ValueError("training route is unsupported")
        if self.start_route not in START_ROUTES:
            raise ValueError("start route is unsupported")
        if (
            not isinstance(self.start_checkpoint_sha256, str)
            or _SHA256_PATTERN.fullmatch(
                self.start_checkpoint_sha256
            )
            is None
        ):
            raise ValueError("start checkpoint SHA-256 is invalid")
        if not isinstance(self.source_commit, str) or not self.source_commit:
            raise ValueError("source_commit must be non-empty")
        if self.seed not in FORMAL_POLICY_SEEDS:
            raise ValueError("seed is not a frozen policy seed")
        expected = {
            "rollout_updates": 1,
            "prompt_groups_per_update": 4,
            "group_size": 2,
            "max_new_tokens": 32,
            "policy_epochs": 1,
            "policy_micro_batch_size": 2,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(
                    f"smoke {name} must equal {value}"
                )
        for name in (
            "temperature",
            "learning_rate",
            "clip_epsilon",
            "kl_beta",
            "max_grad_norm",
            "weight_decay",
            "beta1",
            "beta2",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"smoke {name} must be finite")
        if self.temperature <= 0 or self.learning_rate <= 0:
            raise ValueError(
                "smoke temperature and learning rate must be positive"
            )
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be non-empty")
        if not isinstance(self.use_bfloat16, bool):
            raise ValueError("use_bfloat16 must be boolean")
        assert_secret_free(
            self.public_record(),
            context="smoke policy training configuration",
        )


def smoke_policy_training_config(
    *,
    route: str,
    output_dir: str | Path,
    start_checkpoint: str | Path,
    start_checkpoint_sha256: str,
    start_route: str,
    source_commit: str,
    seed: int,
    device: str = "cuda",
    use_bfloat16: bool = True,
) -> SmokePolicyTrainingConfig:
    config = SmokePolicyTrainingConfig(
        route=route,
        output_dir=Path(output_dir),
        start_checkpoint=Path(start_checkpoint),
        start_checkpoint_sha256=start_checkpoint_sha256,
        start_route=start_route,
        source_commit=source_commit,
        seed=seed,
        device=device,
        rollout_updates=1,
        prompt_groups_per_update=4,
        group_size=2,
        max_new_tokens=32,
        temperature=0.8,
        policy_epochs=1,
        policy_micro_batch_size=2,
        learning_rate=5e-6,
        clip_epsilon=0.2,
        kl_beta=0.02,
        max_grad_norm=1.0,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.95,
        use_bfloat16=use_bfloat16,
    )
    config.validate()
    return config


def _validate_schedule(
    schedule: Sequence[ScheduledPrompt],
    config: PolicyTrainingConfig,
) -> None:
    expected = (
        config.rollout_updates * config.prompt_groups_per_update
    )
    if len(schedule) != expected:
        raise ValueError(f"schedule must contain exactly {expected} prompts")
    identifiers = [item.schedule_id for item in schedule]
    families = [item.problem.family_id for item in schedule]
    if (
        len(set(identifiers)) != len(identifiers)
        or len(set(families)) != len(families)
    ):
        raise ValueError("schedule IDs and families must be unique")
    for update in range(1, config.rollout_updates + 1):
        rows = [item for item in schedule if item.update == update]
        if len(rows) != config.prompt_groups_per_update:
            raise ValueError("schedule update size is invalid")
        if sum(item.mode == "direct" for item in rows) != 2:
            raise ValueError("schedule update must have two Direct prompts")
        if sum(item.mode == "think" for item in rows) != 2:
            raise ValueError("schedule update must have two Think prompts")
        if sum(item.problem.task == "arithmetic" for item in rows) != 2:
            raise ValueError("schedule update must have two arithmetic prompts")
        if sum(
            item.problem.task == "physics_numeric" for item in rows
        ) != 2:
            raise ValueError("schedule update must have two physics prompts")


def _append_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    payload = b"".join(
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
    assert_secret_free(payload, context="policy trajectory record")
    with path.open("ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _chunks(
    values: Sequence[int],
    size: int,
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(values[offset : offset + size])
        for offset in range(0, len(values), size)
    )


def _weighted_mean(
    values: Sequence[tuple[float, int]],
) -> float:
    total_weight = sum(weight for _, weight in values)
    return (
        sum(value * weight for value, weight in values)
        / total_weight
    )


def run_group_policy_training(
    config: PolicyTrainingConfig,
    *,
    schedule: Sequence[ScheduledPrompt],
    problem_pool: Sequence[RLProblem],
    judge_client: PreferenceJudgeClient | None = None,
    judge_cache: JudgeCache | None = None,
) -> dict[str, object]:
    """Train one formal route through 16 rollouts and 32 policy steps."""

    if not isinstance(config, PolicyTrainingConfig):
        raise ValueError("config must be PolicyTrainingConfig")
    config.validate()
    _validate_schedule(schedule, config)
    if config.output_dir.exists():
        raise ValueError(
            f"output directory already exists: {config.output_dir}"
        )
    if not config.start_checkpoint.is_file():
        raise FileNotFoundError(config.start_checkpoint)
    needs_ai = config.route in {
        DIRECT_RLAIF_ROUTE,
        COMBINED_ROUTE,
    }
    if needs_ai and (
        not isinstance(judge_client, PreferenceJudgeClient)
        or not isinstance(judge_cache, JudgeCache)
    ):
        raise ValueError(
            "RLAIF routes require a preference judge client and cache"
        )
    if not needs_ai and (
        judge_client is not None or judge_cache is not None
    ):
        raise ValueError("RLVR route must not receive a judge")

    device = resolve_device(config.device)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.reset_peak_memory_stats(device)
    model_config = lesson15_efficient_config(ALIBI_POSITIONS)
    model, parent = load_efficient_checkpoint(
        config.start_checkpoint,
        expected_sha256=config.start_checkpoint_sha256,
        expected_route=config.start_route,
        expected_model_config=model_config,
    )
    model.to(device)
    model.eval()
    reference = EfficientLongContextGPT(model_config)
    reference.load_state_dict(model.state_dict(), strict=True)
    reference.to(device)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    optimizer = configure_adamw(
        model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        beta1=config.beta1,
        beta2=config.beta2,
    )
    sampling_generator = torch.Generator(device="cpu").manual_seed(
        config.seed
    )

    config.output_dir.mkdir(parents=True)
    (config.output_dir / "problem_pool.jsonl").write_bytes(
        canonical_problem_pool_bytes(problem_pool)
    )
    trajectory_path = config.output_dir / "trajectories.jsonl"
    update_history: list[dict[str, object]] = []
    optimizer_step = 0
    started = time.perf_counter()
    for update in range(1, config.rollout_updates + 1):
        scheduled_rows = [
            item for item in schedule if item.update == update
        ]
        trajectories: list[RolloutTrajectory] = []
        local_scores: list[dict[str, object]] = []
        route_rewards: list[RouteReward] = []
        audit_rows: list[dict[str, object]] = []
        judge_records: dict[str, dict[str, object]] = {}
        sampled_groups: list[
            tuple[
                ScheduledPrompt,
                tuple[RolloutTrajectory, ...],
                tuple[dict[str, object], ...],
            ]
        ] = []
        model.eval()
        for scheduled in scheduled_rows:
            group = sample_mode_group(
                model,
                scheduled.problem.prompt,
                mode=scheduled.mode,
                schedule_id=scheduled.schedule_id,
                family_id=scheduled.problem.family_id,
                group_size=config.group_size,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                device=device,
                generator=sampling_generator,
                use_bfloat16=config.use_bfloat16,
            )
            group_scores = tuple(
                dict(
                    score_mode_completion(
                        scheduled.problem.evaluation_record(),
                        item.completion,
                    )
                )
                for item in group
            )
            sampled_groups.append((scheduled, group, group_scores))

        judgments: dict[
            str,
            tuple[Mapping[str, float], dict[str, object]],
        ] = {}
        if needs_ai:
            assert judge_client is not None
            assert judge_cache is not None

            def judge_group(
                item: tuple[
                    ScheduledPrompt,
                    tuple[RolloutTrajectory, ...],
                    tuple[dict[str, object], ...],
                ],
            ) -> tuple[
                str,
                Mapping[str, float],
                dict[str, object],
            ]:
                scheduled, group, _ = item
                judge_problem = build_judge_problem(scheduled, group)
                answer, cache_hit = judge_with_cache(
                    client=judge_client,
                    problem=judge_problem,
                    cache=judge_cache,
                )
                record = {
                    "answer": answer.public_record(),
                    "cache_hit": cache_hit,
                    "candidate_permutation": [
                        candidate.candidate_id
                        for candidate in ordered_candidates(
                            judge_problem
                        )
                    ],
                    "transport_attempts": (
                        0
                        if cache_hit
                        else answer.transport_attempts
                    ),
                }
                return (
                    scheduled.schedule_id,
                    answer.reward_by_candidate(),
                    record,
                )

            with ThreadPoolExecutor(
                max_workers=config.prompt_groups_per_update,
            ) as executor:
                for schedule_id, ai_rewards, record in executor.map(
                    judge_group,
                    sampled_groups,
                ):
                    judgments[schedule_id] = (ai_rewards, record)

        for scheduled, group, group_scores in sampled_groups:
            ai_rewards: Mapping[str, float] | None = None
            if needs_ai:
                assert judge_client is not None
                assert judge_cache is not None
                ai_rewards, record = judgments[scheduled.schedule_id]
                judge_records[scheduled.schedule_id] = record
            group_rewards = compose_route_rewards(
                route=config.route,
                candidate_ids=tuple(
                    item.candidate_id for item in group
                ),
                local_scores=group_scores,
                ai_rewards=ai_rewards,
            )
            trajectories.extend(group)
            local_scores.extend(group_scores)
            route_rewards.extend(group_rewards)
            for trajectory, local_score, reward in zip(
                group,
                group_scores,
                group_rewards,
                strict=True,
            ):
                audit_rows.append(
                    {
                        "ai_reward": reward.ai_reward,
                        "candidate_id": trajectory.candidate_id,
                        "completion": dict(trajectory.completion),
                        "family_id": scheduled.problem.family_id,
                        "generated_token_count": len(
                            trajectory.generated_tokens
                        ),
                        "local_score": dict(local_score),
                        "mode": scheduled.mode,
                        "reward": asdict(reward),
                        "schedule_id": scheduled.schedule_id,
                        "task": scheduled.problem.task,
                        "update": update,
                    }
                )
        reward_values = tuple(item.total for item in route_rewards)
        group_ids = tuple(item.schedule_id for item in trajectories)
        advantages = normalize_group_advantages(
            rewards=reward_values,
            group_ids=group_ids,
            device=device,
        )
        total_tokens = sum(
            len(item.generated_tokens) for item in trajectories
        )
        epoch_records: list[dict[str, object]] = []
        for epoch in range(1, config.policy_epochs + 1):
            indices = list(range(len(trajectories)))
            random.Random(
                config.seed * 10_000 + update * 10 + epoch
            ).shuffle(indices)
            optimizer.zero_grad(set_to_none=True)
            statistics_by_name: dict[
                str,
                list[tuple[float, int]],
            ] = defaultdict(list)
            for index_chunk in _chunks(
                indices,
                config.policy_micro_batch_size,
            ):
                micro_trajectories = tuple(
                    trajectories[index] for index in index_chunk
                )
                batch = collate_trajectories(
                    micro_trajectories,
                    device=device,
                )
                reference_log_probs = reference_token_log_probs(
                    reference,
                    batch,
                    temperature=config.temperature,
                    use_bfloat16=config.use_bfloat16,
                )
                result = group_policy_loss(
                    model,
                    batch,
                    advantages=advantages[list(index_chunk)],
                    reference_log_probs=reference_log_probs,
                    temperature=config.temperature,
                    clip_epsilon=config.clip_epsilon,
                    kl_beta=config.kl_beta,
                    use_bfloat16=config.use_bfloat16,
                )
                micro_tokens = int(
                    batch.generated_mask.sum().item()
                )
                scaled_loss = result.loss * (
                    micro_tokens / total_tokens
                )
                if not bool(torch.isfinite(scaled_loss).item()):
                    raise RuntimeError("policy loss became non-finite")
                scaled_loss.backward()
                for name, value in result.statistics.items():
                    statistics_by_name[name].append(
                        (value, micro_tokens)
                    )
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.max_grad_norm,
            )
            if not bool(torch.isfinite(gradient_norm).item()):
                raise RuntimeError("policy gradient became non-finite")
            optimizer.step()
            optimizer_step += 1
            epoch_records.append(
                {
                    "epoch": epoch,
                    "gradient_l2_norm_before_clip": float(
                        gradient_norm.item()
                    ),
                    "optimizer_step": optimizer_step,
                    **{
                        name: _weighted_mean(values)
                        for name, values in statistics_by_name.items()
                    },
                }
            )
        grouped_rewards: dict[str, list[float]] = defaultdict(list)
        for group_id, value in zip(
            group_ids,
            reward_values,
            strict=True,
        ):
            grouped_rewards[group_id].append(value)
        update_record = {
            "ai_judgments": judge_records,
            "exact_completions": sum(
                bool(score["task_correct"]) for score in local_scores
            ),
            "format_valid_completions": sum(
                bool(score["format_valid"]) for score in local_scores
            ),
            "generated_tokens": total_tokens,
            "mean_reward": statistics.fmean(reward_values),
            "mixed_exact_groups": sum(
                any(
                    bool(local_scores[index]["task_correct"])
                    for index, group_id in enumerate(group_ids)
                    if group_id == scheduled.schedule_id
                )
                and not all(
                    bool(local_scores[index]["task_correct"])
                    for index, group_id in enumerate(group_ids)
                    if group_id == scheduled.schedule_id
                )
                for scheduled in scheduled_rows
            ),
            "optimizer_epochs": epoch_records,
            "reward_population_std": statistics.pstdev(reward_values),
            "update": update,
            "zero_variance_groups": sum(
                statistics.pstdev(values) == 0.0
                for values in grouped_rewards.values()
            ),
        }
        update_history.append(update_record)
        _append_jsonl(trajectory_path, audit_rows)
        write_json_atomic(
            config.output_dir / "progress.json",
            {
                "completed_rollout_updates": update,
                "optimizer_steps": optimizer_step,
                "route": config.route,
                "seed": config.seed,
                "updates": update_history,
            },
        )
    synchronize(device)
    elapsed = time.perf_counter() - started
    state = _cpu_state_dict(model)
    checkpoint = {
        "architecture": ARCHITECTURE_NAME,
        "lesson17": {
            "optimizer_steps": optimizer_step,
            "reward_route": config.route,
            "rollout_updates": config.rollout_updates,
            "seed": config.seed,
        },
        "model_config": asdict(model_config),
        "model_state_dict": state,
        "model_state_sha256": model_state_sha256(state),
        "parent_checkpoint_sha256": config.start_checkpoint_sha256,
        "parent_route": config.start_route,
        "route": config.route,
        "schema_version": 1,
        "source_commit": config.source_commit,
        "tokenizer": {
            "kind": "byte_plus_fixed_special_tokens",
            "vocab_size": VOCAB_SIZE,
        },
    }
    filename = (
        config.route.lower()
        .replace("+", "plus")
        .replace(" ", "-")
        .replace("/", "-")
        + f"-seed-{config.seed}.pt"
    )
    checkpoint_path = config.output_dir / filename
    _atomic_torch_save(checkpoint, checkpoint_path)
    summary: dict[str, object] = {
        "checkpoint": {
            "bytes": checkpoint_path.stat().st_size,
            "model_state_sha256": checkpoint["model_state_sha256"],
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
        "configuration": config.public_record(),
        "elapsed_seconds": elapsed,
        "judge": (
            None
            if judge_client is None
            else {
                "logical_requests": judge_client.logical_requests,
                "public_configuration": (
                    judge_client.config.public_metadata()
                ),
                "transport_attempts": (
                    judge_client.transport_attempts
                ),
            }
        ),
        "parent": parent,
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "problem_pool_families": len(problem_pool),
        "route": config.route,
        "schema_version": 1,
        "updates": update_history,
    }
    assert_secret_free(summary, context="policy training summary")
    write_json_atomic(config.output_dir / "run.json", summary)
    return summary


def _holdout_families(paths: Sequence[Path]) -> frozenset[str]:
    families: set[str] = set()
    for path in paths:
        families.update(
            str(row["family_id"])
            for row in load_evaluation_records(path)
        )
    return frozenset(families)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--route",
        choices=("rlvr", "rlaif", "combined"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-checkpoint", type=Path, required=True)
    parser.add_argument("--start-checkpoint-sha256", required=True)
    parser.add_argument(
        "--start-route",
        choices=tuple(sorted(START_ROUTES)),
        required=True,
    )
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--seed",
        choices=FORMAL_POLICY_SEEDS,
        type=int,
        required=True,
    )
    parser.add_argument("--primary-evaluation", type=Path, required=True)
    parser.add_argument("--challenge-evaluation", type=Path, required=True)
    parser.add_argument("--judge-cache", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-bfloat16", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    route = {
        "rlvr": RLVR_ROUTE,
        "rlaif": DIRECT_RLAIF_ROUTE,
        "combined": COMBINED_ROUTE,
    }[arguments.route]
    needs_ai = route != RLVR_ROUTE
    if needs_ai and arguments.judge_cache is None:
        raise SystemExit("--judge-cache is required for RLAIF routes")
    if not needs_ai and arguments.judge_cache is not None:
        raise SystemExit("--judge-cache is not used by RLVR")
    excluded = _holdout_families(
        (
            arguments.primary_evaluation,
            arguments.challenge_evaluation,
        )
    )
    pool = build_lesson17_problem_pool(
        count_per_task=32,
        seed=LESSON17_DATA_SEED,
        excluded_families=excluded,
    )
    schedule = build_prompt_schedule(
        pool,
        seed=arguments.seed,
        updates=16,
        prompts_per_update=4,
    )
    config = frozen_policy_training_config(
        route=route,
        output_dir=arguments.output_dir,
        start_checkpoint=arguments.start_checkpoint,
        start_checkpoint_sha256=arguments.start_checkpoint_sha256,
        start_route=arguments.start_route,
        source_commit=arguments.source_commit,
        seed=arguments.seed,
        device=arguments.device,
        use_bfloat16=not arguments.no_bfloat16,
    )
    client = PreferenceJudgeClient() if needs_ai else None
    cache = (
        JudgeCache(arguments.judge_cache)
        if arguments.judge_cache is not None
        else None
    )
    result = run_group_policy_training(
        config,
        schedule=schedule,
        problem_pool=pool,
        judge_client=client,
        judge_cache=cache,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
