import pytest

from nanogpt_nspire.byte_tokenizer import (
    ASSISTANT_ID,
    BOS_ID,
    EOS_ID,
    USER_ID,
    ByteTokenizer,
    ByteTokenizerError,
    ConversationTurn,
    format_conversation,
)


def test_single_turn_serialization_and_assistant_only_loss():
    tokenizer = ByteTokenizer()
    question = "What is 12 * 7?"
    answer = "12 times 7 is 84."
    turns = (
        ConversationTurn("user", question),
        ConversationTurn("assistant", answer),
    )

    tokens, loss_mask = format_conversation(turns)

    question_tokens = tokenizer.encode_text(question)
    answer_tokens = tokenizer.encode_text(answer)
    expected_tokens = (
        BOS_ID,
        USER_ID,
        *question_tokens,
        ASSISTANT_ID,
        *answer_tokens,
        EOS_ID,
    )
    expected_mask = (
        0,
        0,
        *(0 for _ in question_tokens),
        0,
        *(1 for _ in answer_tokens),
        1,
    )
    assert tokens == expected_tokens
    assert loss_mask == expected_mask
    assert len(tokens) == len(loss_mask)


def test_multi_turn_serialization_preserves_role_order():
    tokenizer = ByteTokenizer()
    turns = (
        ConversationTurn("user", "What is force?"),
        ConversationTurn("assistant", "Force is mass times acceleration."),
        ConversationTurn("user", "What is its SI unit?"),
        ConversationTurn("assistant", "The SI unit is the newton."),
    )

    tokens, loss_mask = format_conversation(turns)

    assert tokens.count(USER_ID) == 2
    assert tokens.count(ASSISTANT_ID) == 2
    assert tokens[0] == BOS_ID
    assert tokens[-1] == EOS_ID
    rendered = tokenizer.render_tokens(tokens)
    assert rendered == (
        "<BOS><USER>What is force?"
        "<ASSISTANT>Force is mass times acceleration."
        "<USER>What is its SI unit?"
        "<ASSISTANT>The SI unit is the newton.<EOS>"
    )
    first_answer = tokenizer.encode_text("Force is mass times acceleration.")
    second_answer = tokenizer.encode_text("The SI unit is the newton.")
    assert sum(loss_mask) == len(first_answer) + len(second_answer) + 1


@pytest.mark.parametrize(
    "turns, message",
    [
        ((), "at least one"),
        ((ConversationTurn("assistant", "Hello"),), "start with user"),
        ((ConversationTurn("user", "Hello"),), "end with assistant"),
        (
            (
                ConversationTurn("user", "One"),
                ConversationTurn("user", "Two"),
            ),
            "alternate",
        ),
        (
            (
                ConversationTurn("user", ""),
                ConversationTurn("assistant", "Answer"),
            ),
            "must not be empty",
        ),
    ],
)
def test_conversation_rejects_invalid_turn_sequences(turns, message):
    with pytest.raises(ByteTokenizerError, match=message):
        format_conversation(turns)


def test_conversation_turn_rejects_invalid_fields():
    with pytest.raises(ByteTokenizerError, match="role"):
        ConversationTurn("system", "text")

    with pytest.raises(ByteTokenizerError, match="content"):
        ConversationTurn("user", b"text")  # type: ignore[arg-type]


def test_embedded_nul_is_data_not_a_terminator():
    tokenizer = ByteTokenizer()
    turns = (
        ConversationTurn("user", "a\0b"),
        ConversationTurn("assistant", "c\0d"),
    )

    tokens, _ = format_conversation(turns)

    assert tokenizer.render_tokens(tokens) == (
        "<BOS><USER>a\0b<ASSISTANT>c\0d<EOS>"
    )


def test_context_limit_is_checked_after_utf8_encoding():
    turns = (
        ConversationTurn("user", "Δ"),
        ConversationTurn("assistant", "v"),
    )

    tokens, _ = format_conversation(turns)
    assert len(tokens) == 7

    with pytest.raises(ByteTokenizerError, match="context limit"):
        format_conversation(turns, context_limit=6)
