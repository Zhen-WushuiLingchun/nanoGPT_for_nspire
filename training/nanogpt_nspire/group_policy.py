"""Token-exact grouped policy objective for Lesson 17."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import nn

from nanogpt_nspire.base_train import _autocast_context
from nanogpt_nspire.byte_tokenizer import PAD_ID
from nanogpt_nspire.rl_rollout import RolloutTrajectory


@dataclass(frozen=True)
class PolicyBatch:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    generated_mask: torch.Tensor
    old_log_probs: torch.Tensor
    candidate_ids: tuple[str, ...]
    group_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        shape = self.input_ids.shape
        if self.input_ids.ndim != 2 or not shape[0] or not shape[1]:
            raise ValueError("policy input_ids must be a non-empty matrix")
        for name in (
            "target_ids",
            "generated_mask",
            "old_log_probs",
        ):
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} shape must match input_ids")
        if self.input_ids.dtype != torch.long:
            raise ValueError("input_ids must be torch.long")
        if self.target_ids.dtype != torch.long:
            raise ValueError("target_ids must be torch.long")
        if self.generated_mask.dtype != torch.bool:
            raise ValueError("generated_mask must be boolean")
        if not torch.is_floating_point(self.old_log_probs):
            raise ValueError("old_log_probs must be floating point")
        if len(self.candidate_ids) != shape[0]:
            raise ValueError("candidate_ids length must match batch")
        if len(self.group_ids) != shape[0]:
            raise ValueError("group_ids length must match batch")
        if not bool(self.generated_mask.any().item()):
            raise ValueError("policy batch has no generated targets")


@dataclass(frozen=True)
class PolicyLoss:
    loss: torch.Tensor
    statistics: dict[str, float]


def collate_trajectories(
    trajectories: Sequence[RolloutTrajectory],
    *,
    device: torch.device,
) -> PolicyBatch:
    """Right-pad trajectories while masking prompt and padding targets."""

    if not trajectories:
        raise ValueError("trajectories must be non-empty")
    if not isinstance(device, torch.device):
        raise ValueError("device must be torch.device")
    if any(
        not isinstance(item, RolloutTrajectory)
        for item in trajectories
    ):
        raise ValueError("every trajectory must be RolloutTrajectory")
    sequence_lengths = [len(item.full_tokens) - 1 for item in trajectories]
    if any(length <= 0 for length in sequence_lengths):
        raise ValueError("trajectory is too short for next-token training")
    maximum = max(sequence_lengths)
    rows = len(trajectories)
    input_ids = torch.full(
        (rows, maximum),
        PAD_ID,
        dtype=torch.long,
        device=device,
    )
    target_ids = torch.full_like(input_ids, PAD_ID)
    generated_mask = torch.zeros(
        (rows, maximum),
        dtype=torch.bool,
        device=device,
    )
    old_log_probs = torch.zeros(
        (rows, maximum),
        dtype=torch.float32,
        device=device,
    )
    for row, trajectory in enumerate(trajectories):
        length = sequence_lengths[row]
        input_ids[row, :length] = torch.tensor(
            trajectory.full_tokens[:-1],
            dtype=torch.long,
            device=device,
        )
        target_ids[row, :length] = torch.tensor(
            trajectory.full_tokens[1:],
            dtype=torch.long,
            device=device,
        )
        generated_start = len(trajectory.prompt_tokens) - 1
        generated_end = generated_start + len(
            trajectory.generated_tokens
        )
        generated_mask[row, generated_start:generated_end] = True
        old_log_probs[row, generated_start:generated_end] = torch.tensor(
            trajectory.old_log_probs,
            dtype=torch.float32,
            device=device,
        )
    return PolicyBatch(
        input_ids=input_ids,
        target_ids=target_ids,
        generated_mask=generated_mask,
        old_log_probs=old_log_probs,
        candidate_ids=tuple(
            item.candidate_id for item in trajectories
        ),
        group_ids=tuple(item.schedule_id for item in trajectories),
    )


def normalize_group_advantages(
    *,
    rewards: Sequence[float],
    group_ids: Sequence[str],
    device: torch.device,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Return population-standardized rewards independently per prompt."""

    if not rewards or len(rewards) != len(group_ids):
        raise ValueError("rewards and group_ids must have equal non-zero length")
    if not isinstance(device, torch.device):
        raise ValueError("device must be torch.device")
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not math.isfinite(float(epsilon))
        or float(epsilon) <= 0.0
    ):
        raise ValueError("epsilon must be finite and positive")
    values = torch.tensor(
        tuple(float(value) for value in rewards),
        dtype=torch.float32,
        device=device,
    )
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("rewards must be finite")
    advantages = torch.zeros_like(values)
    for group_id in dict.fromkeys(group_ids):
        indices = [
            index
            for index, value in enumerate(group_ids)
            if value == group_id
        ]
        group = values[indices]
        standard_deviation = group.std(unbiased=False)
        if float(standard_deviation.item()) <= float(epsilon):
            advantages[indices] = 0.0
        else:
            advantages[indices] = (
                group - group.mean()
            ) / (standard_deviation + float(epsilon))
    return advantages


def _validate_scalar(
    value: object,
    name: str,
    *,
    lower: float,
    inclusive: bool,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (
            float(value) < lower
            if inclusive
            else float(value) <= lower
        )
    ):
        relation = ">=" if inclusive else ">"
        raise ValueError(f"{name} must be finite and {relation} {lower}")
    return float(value)


def _selected_log_probs(
    model: nn.Module,
    batch: PolicyBatch,
    *,
    temperature: float,
    use_bfloat16: bool,
) -> torch.Tensor:
    device = batch.input_ids.device
    with _autocast_context(device, enabled=use_bfloat16):
        logits, _ = model(batch.input_ids)
    scaled = logits.float() / temperature
    all_log_probs = torch.log_softmax(scaled, dim=-1)
    return all_log_probs.gather(
        dim=-1,
        index=batch.target_ids.unsqueeze(-1),
    ).squeeze(-1)


def reference_token_log_probs(
    reference_model: nn.Module,
    batch: PolicyBatch,
    *,
    temperature: float,
    use_bfloat16: bool,
) -> torch.Tensor:
    """Compute immutable reference log-probs at all generated-token slots."""

    temperature = _validate_scalar(
        temperature,
        "temperature",
        lower=0.0,
        inclusive=False,
    )
    if not isinstance(use_bfloat16, bool):
        raise ValueError("use_bfloat16 must be boolean")
    with torch.inference_mode():
        selected = _selected_log_probs(
            reference_model,
            batch,
            temperature=temperature,
            use_bfloat16=use_bfloat16,
        )
    return torch.where(
        batch.generated_mask,
        selected,
        torch.zeros_like(selected),
    ).detach()


def group_policy_loss(
    model: nn.Module,
    batch: PolicyBatch,
    *,
    advantages: torch.Tensor,
    reference_log_probs: torch.Tensor,
    temperature: float,
    clip_epsilon: float,
    kl_beta: float,
    use_bfloat16: bool,
) -> PolicyLoss:
    """Compute clipped GRPO surrogate plus non-negative sampled KL."""

    if not isinstance(batch, PolicyBatch):
        raise ValueError("batch must be PolicyBatch")
    temperature = _validate_scalar(
        temperature,
        "temperature",
        lower=0.0,
        inclusive=False,
    )
    clip_epsilon = _validate_scalar(
        clip_epsilon,
        "clip_epsilon",
        lower=0.0,
        inclusive=False,
    )
    kl_beta = _validate_scalar(
        kl_beta,
        "kl_beta",
        lower=0.0,
        inclusive=True,
    )
    if not isinstance(use_bfloat16, bool):
        raise ValueError("use_bfloat16 must be boolean")
    if advantages.shape != (batch.input_ids.shape[0],):
        raise ValueError("advantages shape must match batch rows")
    if reference_log_probs.shape != batch.input_ids.shape:
        raise ValueError(
            "reference_log_probs shape must match input_ids"
        )
    if advantages.device != batch.input_ids.device:
        raise ValueError("advantages must be on the batch device")
    if reference_log_probs.device != batch.input_ids.device:
        raise ValueError("reference_log_probs must be on the batch device")

    current = _selected_log_probs(
        model,
        batch,
        temperature=temperature,
        use_bfloat16=use_bfloat16,
    )
    mask = batch.generated_mask
    ratios = torch.exp(current - batch.old_log_probs)
    token_advantages = advantages.unsqueeze(1).expand_as(current)
    unclipped = ratios * token_advantages
    clipped = torch.clamp(
        ratios,
        1.0 - clip_epsilon,
        1.0 + clip_epsilon,
    ) * token_advantages
    surrogate = torch.minimum(unclipped, clipped)
    delta = reference_log_probs - current
    sampled_kl = torch.exp(delta) - delta - 1.0
    token_objective = surrogate - kl_beta * sampled_kl
    loss = -token_objective[mask].mean()

    with torch.no_grad():
        masked_ratio = ratios[mask]
        masked_kl = sampled_kl[mask]
        clipped_fraction = (
            (torch.abs(masked_ratio - 1.0) > clip_epsilon)
            .float()
            .mean()
        )
        statistics = {
            "behavior_approx_kl": float(
                (batch.old_log_probs[mask] - current[mask]).mean().item()
            ),
            "clip_fraction": float(clipped_fraction.item()),
            "generated_tokens": float(mask.sum().item()),
            "mean_ratio": float(masked_ratio.mean().item()),
            "reference_kl": float(masked_kl.mean().item()),
            "surrogate": float(surrogate[mask].mean().item()),
        }
    return PolicyLoss(loss=loss, statistics=statistics)
