import pytest

from nanogpt_nspire.byte_tokenizer import (
    ASSISTANT_ID,
    BOS_ID,
    EOS_ID,
    FINAL_ID,
    PAD_ID,
    THINK_ID,
    TOOL_ID,
    USER_ID,
    VOCAB_SIZE,
    ByteTokenizer,
    ByteTokenizerError,
)


def test_all_bytes_round_trip():
    tokenizer = ByteTokenizer()
    payload = bytes(range(256))

    tokens = tokenizer.encode_bytes(payload)

    assert tokens == tuple(range(256))
    assert tokenizer.decode_bytes(tokens) == payload


def test_utf8_text_round_trip_is_strict():
    tokenizer = ByteTokenizer()
    text = "Force = mass * acceleration. Δv is useful."

    assert tokenizer.decode_text(tokenizer.encode_text(text)) == text

    with pytest.raises(ByteTokenizerError, match="valid UTF-8"):
        tokenizer.decode_text((0xFF,))


def test_special_ids_are_frozen():
    assert BOS_ID == 256
    assert EOS_ID == 257
    assert USER_ID == 258
    assert ASSISTANT_ID == 259
    assert TOOL_ID == 260
    assert THINK_ID == 261
    assert FINAL_ID == 262
    assert PAD_ID == 263
    assert VOCAB_SIZE == 264


@pytest.mark.parametrize(
    "tokens",
    [
        (True,),
        (-1,),
        (VOCAB_SIZE,),
        (BOS_ID,),
    ],
)
def test_decode_bytes_rejects_non_byte_tokens(tokens):
    tokenizer = ByteTokenizer()

    with pytest.raises(ByteTokenizerError):
        tokenizer.decode_bytes(tokens)


def test_render_tokens_names_special_tokens_without_making_them_text():
    tokenizer = ByteTokenizer()
    tokens = (BOS_ID, USER_ID, *tokenizer.encode_text("Hi"), EOS_ID)

    assert tokenizer.render_tokens(tokens) == "<BOS><USER>Hi<EOS>"

    with pytest.raises(ByteTokenizerError, match="special token"):
        tokenizer.decode_text(tokens)


def test_encode_rejects_non_bytes_and_non_text():
    tokenizer = ByteTokenizer()

    with pytest.raises(ByteTokenizerError, match="bytes-like"):
        tokenizer.encode_bytes("not bytes")  # type: ignore[arg-type]

    with pytest.raises(ByteTokenizerError, match="text must be a string"):
        tokenizer.encode_text(b"not text")  # type: ignore[arg-type]
