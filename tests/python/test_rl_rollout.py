from __future__ import annotations

import torch
from torch import nn

from nanogpt_nspire.byte_tokenizer import (
    EOS_ID,
    FINAL_ID,
    PAD_ID,
    TOOL_ID,
    VOCAB_SIZE,
)
from nanogpt_nspire.reasoning_format import DIRECT_MODE, THINK_MODE
from nanogpt_nspire.rl_rollout import (
    RolloutError,
    sample_mode_group,
)


class ScriptedModel(nn.Module):
    def __init__(
        self,
        *,
        prefix_length: int,
        script: tuple[int, ...],
        block_size: int = 128,
    ) -> None:
        super().__init__()
        self.prefix_length = prefix_length
        self.script = script
        self.block_size = block_size
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        batch, length = token_ids.shape
        step = length - self.prefix_length
        token = self.script[min(step, len(self.script) - 1)]
        logits = torch.full(
            (batch, length, VOCAB_SIZE),
            -100.0,
            device=token_ids.device,
        )
        logits[:, -1, token] = 100.0 + self.anchor
        return logits, None


class UniformByteModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block_size = 128
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        batch, length = token_ids.shape
        logits = torch.full(
            (batch, length, VOCAB_SIZE),
            -100.0,
            device=token_ids.device,
        )
        logits[:, -1, 48:58] = self.anchor
        logits[:, -1, EOS_ID] = self.anchor
        return logits, None


def _prefix_length(prompt: str) -> int:
    return len(prompt.encode("utf-8")) + 4


def test_direct_rollout_includes_terminal_token_and_old_log_probs() -> None:
    prompt = "What is 2 + 3?"
    model = ScriptedModel(
        prefix_length=_prefix_length(prompt),
        script=(ord("5"), EOS_ID),
    )

    trajectories = sample_mode_group(
        model,
        prompt,
        mode=DIRECT_MODE,
        schedule_id="schedule-1",
        family_id="family-1",
        group_size=2,
        max_new_tokens=8,
        temperature=0.8,
        device=torch.device("cpu"),
        generator=torch.Generator(device="cpu").manual_seed(17),
        use_bfloat16=False,
    )

    assert len(trajectories) == 2
    for index, trajectory in enumerate(trajectories):
        assert trajectory.candidate_id == f"schedule-1:candidate-{index}"
        assert trajectory.generated_tokens == (ord("5"), EOS_ID)
        assert trajectory.full_tokens[-2:] == (ord("5"), EOS_ID)
        assert len(trajectory.old_log_probs) == 2
        assert trajectory.completion["final_text"] == "5"
        assert trajectory.completion["terminated"] is True
        assert trajectory.completion["special_token_leak"] is False


def test_think_rollout_records_final_transition_as_generated_token() -> None:
    prompt = "What is 4 * 2?"
    model = ScriptedModel(
        prefix_length=_prefix_length(prompt),
        script=(
            ord("4"),
            ord("*"),
            ord("2"),
            ord("="),
            ord("8"),
            FINAL_ID,
            ord("8"),
            EOS_ID,
        ),
    )

    (trajectory,) = sample_mode_group(
        model,
        prompt,
        mode=THINK_MODE,
        schedule_id="schedule-2",
        family_id="family-2",
        group_size=1,
        max_new_tokens=16,
        temperature=0.8,
        device=torch.device("cpu"),
        generator=torch.Generator(device="cpu").manual_seed(19),
        use_bfloat16=False,
    )

    assert FINAL_ID in trajectory.generated_tokens
    assert trajectory.completion["reasoning_text"] == "4*2=8"
    assert trajectory.completion["final_text"] == "8"
    assert trajectory.completion["final_transition"] is True
    assert trajectory.completion["terminated"] is True


def test_special_token_leak_is_terminal_and_auditable() -> None:
    prompt = "What is 1 + 1?"
    model = ScriptedModel(
        prefix_length=_prefix_length(prompt),
        script=(TOOL_ID,),
    )

    (trajectory,) = sample_mode_group(
        model,
        prompt,
        mode=DIRECT_MODE,
        schedule_id="schedule-3",
        family_id="family-3",
        group_size=1,
        max_new_tokens=4,
        temperature=1.0,
        device=torch.device("cpu"),
        generator=torch.Generator(device="cpu").manual_seed(23),
        use_bfloat16=False,
    )

    assert trajectory.generated_tokens == (TOOL_ID,)
    assert trajectory.completion["special_token_leak"] is True
    assert trajectory.completion["leaked_token"] == "<TOOL>"
    assert trajectory.completion["terminated"] is False


def test_sampling_is_reproducible_with_frozen_cpu_generator() -> None:
    prompt = "Give one digit."
    model = UniformByteModel()

    def run(seed: int) -> tuple[tuple[int, ...], ...]:
        return tuple(
            item.generated_tokens
            for item in sample_mode_group(
                model,
                prompt,
                mode=DIRECT_MODE,
                schedule_id="schedule-4",
                family_id="family-4",
                group_size=4,
                max_new_tokens=6,
                temperature=0.8,
                device=torch.device("cpu"),
                generator=torch.Generator(device="cpu").manual_seed(seed),
                use_bfloat16=False,
            )
        )

    assert run(29) == run(29)
    assert run(29) != run(31)


def test_invalid_rollout_arguments_are_rejected() -> None:
    model = UniformByteModel()
    try:
        sample_mode_group(
            model,
            "Prompt",
            mode=DIRECT_MODE,
            schedule_id="schedule",
            family_id="family",
            group_size=0,
            max_new_tokens=8,
            temperature=0.8,
            device=torch.device("cpu"),
            generator=torch.Generator(device="cpu"),
            use_bfloat16=False,
        )
    except RolloutError as error:
        assert "group_size" in str(error)
    else:
        raise AssertionError("invalid group size was accepted")

    assert PAD_ID >= 256
