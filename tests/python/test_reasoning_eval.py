from pathlib import Path

import torch

from nanogpt_nspire.byte_tokenizer import EOS_ID, FINAL_ID, VOCAB_SIZE
from nanogpt_nspire.reasoning_eval import (
    generate_mode_completion,
    score_mode_completion,
)
from nanogpt_nspire.reasoning_format import DIRECT_MODE, THINK_MODE


class ScriptedModel(torch.nn.Module):
    def __init__(self, script: list[int], *, block_size: int = 256) -> None:
        super().__init__()
        self.script = script
        self.block_size = block_size
        self.calls = 0

    def forward(self, tokens: torch.Tensor):
        logits = torch.full(
            (1, tokens.shape[1], VOCAB_SIZE),
            -1000.0,
            dtype=torch.float32,
        )
        logits[0, -1, self.script[self.calls]] = 1000.0
        self.calls += 1
        return logits, None


def _record() -> dict[str, object]:
    return {
        "task": "arithmetic",
        "expected_answer": "84",
        "expected_unit": None,
    }


def test_direct_generation_treats_final_cue_as_input() -> None:
    script = [*b"The answer is 84.", EOS_ID]
    completion = generate_mode_completion(
        ScriptedModel(script),
        "Calculate 12 * 7.",
        mode=DIRECT_MODE,
        max_new_tokens=48,
        device=torch.device("cpu"),
        use_bfloat16=False,
    )
    score = score_mode_completion(_record(), completion)

    assert completion["reasoning_text"] == ""
    assert completion["final_text"] == "The answer is 84."
    assert completion["final_transition"] is False
    assert completion["terminated"] is True
    assert score["mode_compliant"] is True
    assert score["task_correct"] is True


def test_think_generation_scores_only_text_after_generated_final() -> None:
    script = [
        *b"12 * 7 = 84.",
        FINAL_ID,
        *b"The answer is 84.",
        EOS_ID,
    ]
    completion = generate_mode_completion(
        ScriptedModel(script),
        "Calculate 12 * 7.",
        mode=THINK_MODE,
        max_new_tokens=48,
        device=torch.device("cpu"),
        use_bfloat16=False,
    )
    score = score_mode_completion(_record(), completion)

    assert completion["reasoning_text"] == "12 * 7 = 84."
    assert completion["final_text"] == "The answer is 84."
    assert completion["final_transition"] is True
    assert completion["reasoning_tokens"] == len(b"12 * 7 = 84.")
    assert score["mode_compliant"] is True
    assert score["task_correct"] is True


def test_number_in_reasoning_gets_no_credit_without_final_transition() -> None:
    completion = generate_mode_completion(
        ScriptedModel([*b"12 * 7 = 84.", EOS_ID]),
        "Calculate 12 * 7.",
        mode=THINK_MODE,
        max_new_tokens=48,
        device=torch.device("cpu"),
        use_bfloat16=False,
    )
    score = score_mode_completion(_record(), completion)

    assert completion["reasoning_text"] == "12 * 7 = 84."
    assert completion["final_text"] == ""
    assert completion["final_transition"] is False
    assert score["numeric_correct"] is False
    assert score["mode_compliant"] is False
    assert score["task_correct"] is False


def test_generation_separates_budget_and_context_truncation() -> None:
    budget = generate_mode_completion(
        ScriptedModel([*b"abcdef"]),
        "Q",
        mode=THINK_MODE,
        max_new_tokens=3,
        device=torch.device("cpu"),
        use_bfloat16=False,
    )
    context = generate_mode_completion(
        ScriptedModel([*b"abcdef"], block_size=8),
        "Q",
        mode=THINK_MODE,
        max_new_tokens=20,
        device=torch.device("cpu"),
        use_bfloat16=False,
    )

    assert budget["budget_exhausted"] is True
    assert budget["context_exhausted"] is False
    assert context["context_exhausted"] is True
    assert context["budget_exhausted"] is False


def test_generated_unexpected_special_token_is_a_leak() -> None:
    completion = generate_mode_completion(
        ScriptedModel([FINAL_ID]),
        "Q",
        mode=DIRECT_MODE,
        max_new_tokens=8,
        device=torch.device("cpu"),
        use_bfloat16=False,
    )
    score = score_mode_completion(_record(), completion)

    assert completion["special_token_leak"] is True
    assert completion["leaked_token"] == "<FINAL>"
    assert score["task_correct"] is False

