from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from nanogpt_nspire.byte_tokenizer import EOS_ID, PAD_ID, VOCAB_SIZE
from nanogpt_nspire.group_policy import (
    collate_trajectories,
    group_policy_loss,
    normalize_group_advantages,
    reference_token_log_probs,
)
from nanogpt_nspire.rl_rollout import RolloutTrajectory


class TrainableUniformModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block_size = 64
        self.bias = nn.Parameter(torch.zeros(VOCAB_SIZE))

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        batch, length = token_ids.shape
        logits = self.bias.view(1, 1, -1).expand(
            batch,
            length,
            VOCAB_SIZE,
        )
        return logits, None


def _trajectory(
    *,
    candidate_id: str,
    schedule_id: str,
    prompt_tokens: tuple[int, ...],
    generated_tokens: tuple[int, ...],
    old_log_prob: float,
) -> RolloutTrajectory:
    return RolloutTrajectory(
        candidate_id=candidate_id,
        schedule_id=schedule_id,
        family_id=f"family-{schedule_id}",
        mode="direct",
        prompt_tokens=prompt_tokens,
        full_tokens=prompt_tokens + generated_tokens,
        generated_tokens=generated_tokens,
        old_log_probs=tuple(old_log_prob for _ in generated_tokens),
        completion={
            "final_text": "",
            "reasoning_text": "",
        },
    )


def test_collation_masks_only_generated_targets_and_right_padding() -> None:
    uniform_log_prob = -math.log(VOCAB_SIZE)
    trajectories = (
        _trajectory(
            candidate_id="a",
            schedule_id="g1",
            prompt_tokens=(256, 258, 65, 259, 262),
            generated_tokens=(49, EOS_ID),
            old_log_prob=uniform_log_prob,
        ),
        _trajectory(
            candidate_id="b",
            schedule_id="g1",
            prompt_tokens=(256, 258, 66, 259, 262),
            generated_tokens=(50, 51, EOS_ID),
            old_log_prob=uniform_log_prob,
        ),
    )

    batch = collate_trajectories(
        trajectories,
        device=torch.device("cpu"),
    )

    assert batch.input_ids.shape == (2, 7)
    assert batch.target_ids.shape == (2, 7)
    assert batch.generated_mask.tolist() == [
        [False, False, False, False, True, True, False],
        [False, False, False, False, True, True, True],
    ]
    assert batch.target_ids[0, -1].item() == PAD_ID
    assert batch.old_log_probs[0, -1].item() == 0.0
    assert batch.generated_mask.sum().item() == 5


def test_group_advantages_are_normalized_inside_each_prompt_group() -> None:
    advantages = normalize_group_advantages(
        rewards=(0.0, 1.0, 4.0, 4.0),
        group_ids=("a", "a", "b", "b"),
        device=torch.device("cpu"),
    )

    assert advantages[:2].mean().item() == pytest.approx(0.0)
    assert advantages[:2].pow(2).mean().sqrt().item() == pytest.approx(1.0)
    assert advantages[2:].tolist() == [0.0, 0.0]


def test_identical_policy_reference_and_behavior_have_zero_kl() -> None:
    model = TrainableUniformModel()
    uniform_log_prob = -math.log(VOCAB_SIZE)
    trajectories = (
        _trajectory(
            candidate_id="a",
            schedule_id="g1",
            prompt_tokens=(256, 258, 65, 259, 262),
            generated_tokens=(49, EOS_ID),
            old_log_prob=uniform_log_prob,
        ),
        _trajectory(
            candidate_id="b",
            schedule_id="g1",
            prompt_tokens=(256, 258, 65, 259, 262),
            generated_tokens=(50, EOS_ID),
            old_log_prob=uniform_log_prob,
        ),
    )
    batch = collate_trajectories(
        trajectories,
        device=torch.device("cpu"),
    )
    reference = reference_token_log_probs(
        model,
        batch,
        temperature=1.0,
        use_bfloat16=False,
    )
    advantages = normalize_group_advantages(
        rewards=(0.0, 1.0),
        group_ids=batch.group_ids,
        device=torch.device("cpu"),
    )

    result = group_policy_loss(
        model,
        batch,
        advantages=advantages,
        reference_log_probs=reference,
        temperature=1.0,
        clip_epsilon=0.2,
        kl_beta=0.02,
        use_bfloat16=False,
    )

    assert result.statistics["mean_ratio"] == pytest.approx(1.0)
    assert result.statistics["reference_kl"] == pytest.approx(0.0)
    assert result.loss.item() == pytest.approx(0.0, abs=1e-7)
    result.loss.backward()
    assert model.bias.grad is not None
    assert torch.isfinite(model.bias.grad).all()


def test_policy_loss_rejects_reference_shape_mismatch() -> None:
    model = TrainableUniformModel()
    trajectory = _trajectory(
        candidate_id="a",
        schedule_id="g1",
        prompt_tokens=(256, 258, 65, 259, 262),
        generated_tokens=(49, EOS_ID),
        old_log_prob=-math.log(VOCAB_SIZE),
    )
    batch = collate_trajectories(
        (trajectory,),
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="reference_log_probs"):
        group_policy_loss(
            model,
            batch,
            advantages=torch.zeros(1),
            reference_log_probs=torch.zeros(1, 1),
            temperature=1.0,
            clip_epsilon=0.2,
            kl_beta=0.02,
            use_bfloat16=False,
        )
