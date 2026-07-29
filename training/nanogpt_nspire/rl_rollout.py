"""Auditable grouped stochastic rollouts for Lesson 17 policy training."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import nn

from nanogpt_nspire.base_train import _autocast_context
from nanogpt_nspire.byte_tokenizer import (
    BYTE_VOCAB_SIZE,
    EOS_ID,
    FINAL_ID,
    PAD_ID,
    SPECIAL_TOKEN_NAMES,
)
from nanogpt_nspire.reasoning_format import (
    THINK_MODE,
    SUPPORTED_MODES,
    ReasoningFormatError,
    encode_mode_prompt,
)


class RolloutError(ValueError):
    """Raised when a grouped policy rollout violates its frozen contract."""


@dataclass(frozen=True)
class RolloutTrajectory:
    """One sampled response plus the exact behavior-policy probabilities."""

    candidate_id: str
    schedule_id: str
    family_id: str
    mode: str
    prompt_tokens: tuple[int, ...]
    full_tokens: tuple[int, ...]
    generated_tokens: tuple[int, ...]
    old_log_probs: tuple[float, ...]
    completion: Mapping[str, object]

    def __post_init__(self) -> None:
        for name in ("candidate_id", "schedule_id", "family_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise RolloutError(f"{name} must be non-empty")
        if self.mode not in SUPPORTED_MODES:
            raise RolloutError("trajectory mode is unsupported")
        if not self.prompt_tokens:
            raise RolloutError("prompt_tokens must be non-empty")
        if not self.generated_tokens:
            raise RolloutError("generated_tokens must be non-empty")
        if self.full_tokens != (
            self.prompt_tokens + self.generated_tokens
        ):
            raise RolloutError(
                "full_tokens must exactly concatenate prompt and generation"
            )
        if len(self.old_log_probs) != len(self.generated_tokens):
            raise RolloutError(
                "each generated token needs one old log probability"
            )
        if any(not math.isfinite(value) for value in self.old_log_probs):
            raise RolloutError("old log probabilities must be finite")


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RolloutError(f"{name} must be a positive integer")
    return value


def _decode_bytes(values: list[int]) -> str:
    return bytes(values).decode("utf-8", errors="backslashreplace")


def sample_mode_group(
    model: nn.Module,
    prompt: str,
    *,
    mode: str,
    schedule_id: str,
    family_id: str,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
    generator: torch.Generator,
    use_bfloat16: bool,
) -> tuple[RolloutTrajectory, ...]:
    """Sample one same-prompt group using a deterministic CPU RNG stream.

    EOS, the first Think-mode FINAL transition, and leaked special tokens are
    retained in ``generated_tokens``. This makes the behavior-policy log
    probabilities align exactly with every token later optimized by GRPO.
    """

    if not isinstance(model, nn.Module):
        raise RolloutError("model must be a torch module")
    if mode not in SUPPORTED_MODES:
        raise RolloutError("mode is unsupported")
    for value, name in (
        (schedule_id, "schedule_id"),
        (family_id, "family_id"),
    ):
        if not isinstance(value, str) or not value:
            raise RolloutError(f"{name} must be non-empty")
    group_size = _positive_integer(group_size, "group_size")
    max_new_tokens = _positive_integer(
        max_new_tokens,
        "max_new_tokens",
    )
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or float(temperature) <= 0.0
    ):
        raise RolloutError("temperature must be finite and positive")
    if not isinstance(device, torch.device):
        raise RolloutError("device must be torch.device")
    if not isinstance(generator, torch.Generator):
        raise RolloutError("generator must be torch.Generator")
    if str(generator.device) != "cpu":
        raise RolloutError("rollout generator must be a CPU generator")
    if not isinstance(use_bfloat16, bool):
        raise RolloutError("use_bfloat16 must be boolean")
    block_size = getattr(model, "block_size", None)
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size <= 0
    ):
        raise RolloutError("model must expose a positive block_size")
    try:
        prompt_tokens = encode_mode_prompt(
            prompt,
            mode=mode,
            block_size=block_size,
        )
    except ReasoningFormatError as error:
        raise RolloutError(str(error)) from error

    sequences = [list(prompt_tokens) for _ in range(group_size)]
    generated = [[] for _ in range(group_size)]
    old_log_probs = [[] for _ in range(group_size)]
    reasoning_bytes = [[] for _ in range(group_size)]
    final_bytes = [[] for _ in range(group_size)]
    final_transition = [False] * group_size
    terminated = [False] * group_size
    special_token_leak = [False] * group_size
    leaked_token: list[str | None] = [None] * group_size
    context_exhausted = [False] * group_size
    active = [True] * group_size

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            if not any(active):
                break
            sequence_length = len(sequences[0])
            if sequence_length > block_size:
                for index in range(group_size):
                    if active[index]:
                        context_exhausted[index] = True
                        active[index] = False
                break
            inputs = torch.tensor(
                sequences,
                dtype=torch.long,
                device=device,
            )
            with _autocast_context(device, enabled=use_bfloat16):
                logits, _ = model(inputs)
            scaled = logits[:, -1, :].float() / float(temperature)
            log_probabilities = torch.log_softmax(scaled, dim=-1)
            probabilities = torch.exp(log_probabilities).cpu()
            sampled = torch.multinomial(
                probabilities,
                num_samples=1,
                replacement=True,
                generator=generator,
            ).squeeze(1)

            for index in range(group_size):
                if not active[index]:
                    sequences[index].append(PAD_ID)
                    continue
                token = int(sampled[index].item())
                generated[index].append(token)
                old_log_probs[index].append(
                    float(log_probabilities[index, token].item())
                )
                sequences[index].append(token)
                if token == EOS_ID:
                    terminated[index] = True
                    active[index] = False
                elif (
                    mode == THINK_MODE
                    and token == FINAL_ID
                    and not final_transition[index]
                ):
                    final_transition[index] = True
                elif token >= BYTE_VOCAB_SIZE:
                    special_token_leak[index] = True
                    leaked_token[index] = SPECIAL_TOKEN_NAMES[token]
                    active[index] = False
                elif mode == THINK_MODE and not final_transition[index]:
                    reasoning_bytes[index].append(token)
                else:
                    final_bytes[index].append(token)

            if len(sequences[0]) > block_size:
                for index in range(group_size):
                    if active[index]:
                        context_exhausted[index] = True
                        active[index] = False

    trajectories: list[RolloutTrajectory] = []
    for index in range(group_size):
        visible_generation = tuple(generated[index])
        budget_exhausted = (
            not terminated[index]
            and not special_token_leak[index]
            and not context_exhausted[index]
            and len(visible_generation) == max_new_tokens
        )
        completion: dict[str, object] = {
            "budget_exhausted": budget_exhausted,
            "context_exhausted": context_exhausted[index],
            "final_text": _decode_bytes(final_bytes[index]),
            "final_tokens": len(final_bytes[index]),
            "final_transition": final_transition[index],
            "generated_tokens": len(visible_generation),
            "leaked_token": leaked_token[index],
            "mode": mode,
            "reasoning_text": _decode_bytes(reasoning_bytes[index]),
            "reasoning_tokens": len(reasoning_bytes[index]),
            "special_token_leak": special_token_leak[index],
            "terminated": terminated[index],
            "truncated": (
                budget_exhausted or context_exhausted[index]
            ),
        }
        trajectories.append(
            RolloutTrajectory(
                candidate_id=(
                    f"{schedule_id}:candidate-{index}"
                ),
                schedule_id=schedule_id,
                family_id=family_id,
                mode=mode,
                prompt_tokens=prompt_tokens,
                full_tokens=prompt_tokens + visible_generation,
                generated_tokens=visible_generation,
                old_log_probs=tuple(old_log_probs[index]),
                completion=completion,
            )
        )
    return tuple(trajectories)
