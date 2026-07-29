import pytest

from nanogpt_nspire.byte_tokenizer import (
    ASSISTANT_ID,
    BOS_ID,
    EOS_ID,
    FINAL_ID,
    THINK_ID,
    USER_ID,
    ByteTokenizer,
)
from nanogpt_nspire.reasoning_format import (
    DIRECT_MODE,
    THINK_MODE,
    ReasoningFormatError,
    encode_mode_prompt,
    format_supervised_response,
)


def test_direct_mode_uses_final_as_a_condition_not_a_target() -> None:
    tokenizer = ByteTokenizer()
    tokens, mask = format_supervised_response(
        prompt="Calculate 12 * 7.",
        final_answer="The answer is 84.",
        mode=DIRECT_MODE,
        context_limit=256,
    )

    assert tokens == (
        BOS_ID,
        USER_ID,
        *tokenizer.encode_text("Calculate 12 * 7."),
        ASSISTANT_ID,
        FINAL_ID,
        *tokenizer.encode_text("The answer is 84."),
        EOS_ID,
    )
    answer_start = tokens.index(FINAL_ID) + 1
    assert all(value == 0 for value in mask[:answer_start])
    assert all(value == 1 for value in mask[answer_start:])


def test_think_mode_targets_reasoning_transition_final_and_eos() -> None:
    tokenizer = ByteTokenizer()
    reasoning = "12 * 7 = 84."
    final = "The answer is 84."
    tokens, mask = format_supervised_response(
        prompt="Calculate 12 * 7.",
        final_answer=final,
        mode=THINK_MODE,
        reasoning=reasoning,
        context_limit=256,
    )

    think_position = tokens.index(THINK_ID)
    final_position = tokens.index(FINAL_ID)
    assert tokens[think_position + 1 : final_position] == (
        tokenizer.encode_text(reasoning)
    )
    assert mask[think_position] == 0
    assert all(value == 1 for value in mask[think_position + 1 :])
    assert mask[final_position] == 1
    assert tokens[-1] == EOS_ID
    assert mask[-1] == 1


def test_mode_prompt_supplies_exact_prefix_token() -> None:
    direct = encode_mode_prompt(
        "What is force?",
        mode=DIRECT_MODE,
        block_size=256,
    )
    think = encode_mode_prompt(
        "What is force?",
        mode=THINK_MODE,
        block_size=256,
    )

    assert direct[-2:] == (ASSISTANT_ID, FINAL_ID)
    assert think[-2:] == (ASSISTANT_ID, THINK_ID)
    assert direct[:-1] == think[:-1]


@pytest.mark.parametrize(
    "kwargs,message",
    (
        (
            {
                "prompt": "Q",
                "final_answer": "A",
                "mode": "maybe",
            },
            "mode",
        ),
        (
            {
                "prompt": "Q",
                "final_answer": "A",
                "mode": THINK_MODE,
            },
            "reasoning",
        ),
        (
            {
                "prompt": "Q",
                "final_answer": "A",
                "mode": DIRECT_MODE,
                "reasoning": "not allowed",
            },
            "reasoning",
        ),
    ),
)
def test_supervised_format_rejects_ambiguous_contracts(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ReasoningFormatError, match=message):
        format_supervised_response(**kwargs)


def test_supervised_format_rejects_over_context_without_truncation() -> None:
    with pytest.raises(ReasoningFormatError, match="context"):
        format_supervised_response(
            prompt="q" * 250,
            final_answer="answer",
            mode=THINK_MODE,
            reasoning="reason",
            context_limit=256,
        )

