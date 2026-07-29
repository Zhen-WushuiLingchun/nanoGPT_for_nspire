"""Exact direct/thinking assistant-prefix protocol for Lesson 14."""

from __future__ import annotations

from nanogpt_nspire.byte_tokenizer import (
    ASSISTANT_ID,
    BOS_ID,
    EOS_ID,
    FINAL_ID,
    THINK_ID,
    USER_ID,
    ByteTokenizer,
    ByteTokenizerError,
)


DIRECT_MODE = "direct"
THINK_MODE = "think"
SUPPORTED_MODES = frozenset({DIRECT_MODE, THINK_MODE})


class ReasoningFormatError(ValueError):
    """Raised when a direct/CoT sequence violates the frozen protocol."""


def _nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReasoningFormatError(f"{name} must be a non-empty string")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise ReasoningFormatError(f"{name} must be valid UTF-8") from error
    return value


def _validated_mode(mode: object) -> str:
    if mode not in SUPPORTED_MODES:
        raise ReasoningFormatError("mode must be 'direct' or 'think'")
    return str(mode)


def _validated_context_limit(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ReasoningFormatError(f"{name} must be a positive integer")
    return value


def format_supervised_response(
    *,
    prompt: str,
    final_answer: str,
    mode: str,
    reasoning: str | None = None,
    context_limit: int | None = None,
    tokenizer: ByteTokenizer | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Serialize one direct or short-CoT target and its assistant loss mask."""

    prompt = _nonempty_text(prompt, "prompt")
    final_answer = _nonempty_text(final_answer, "final_answer")
    mode = _validated_mode(mode)
    if mode == DIRECT_MODE:
        if reasoning is not None:
            raise ReasoningFormatError(
                "reasoning must be null in direct mode"
            )
    else:
        reasoning = _nonempty_text(reasoning, "reasoning")

    codec = tokenizer or ByteTokenizer()
    try:
        prompt_tokens = codec.encode_text(prompt)
        final_tokens = codec.encode_text(final_answer)
        tokens: list[int] = [
            BOS_ID,
            USER_ID,
            *prompt_tokens,
            ASSISTANT_ID,
        ]
        mask: list[int] = [0] * len(tokens)
        if mode == DIRECT_MODE:
            tokens.append(FINAL_ID)
            mask.append(0)
        else:
            assert reasoning is not None
            reasoning_tokens = codec.encode_text(reasoning)
            tokens.append(THINK_ID)
            mask.append(0)
            tokens.extend(reasoning_tokens)
            mask.extend(1 for _ in reasoning_tokens)
            tokens.append(FINAL_ID)
            mask.append(1)
        tokens.extend(final_tokens)
        mask.extend(1 for _ in final_tokens)
        tokens.append(EOS_ID)
        mask.append(1)
    except (ByteTokenizerError, UnicodeEncodeError) as error:
        raise ReasoningFormatError("response text is not encodable") from error

    if context_limit is not None:
        context_limit = _validated_context_limit(
            context_limit,
            "context_limit",
        )
        if len(tokens) > context_limit:
            raise ReasoningFormatError(
                f"response length {len(tokens)} exceeds context limit "
                f"{context_limit}"
            )
    return tuple(tokens), tuple(mask)


def encode_mode_prompt(
    prompt: str,
    *,
    mode: str,
    block_size: int,
    tokenizer: ByteTokenizer | None = None,
) -> tuple[int, ...]:
    """Encode `<BOS><USER>...<ASSISTANT><mode-cue>` for inference."""

    prompt = _nonempty_text(prompt, "prompt")
    mode = _validated_mode(mode)
    block_size = _validated_context_limit(block_size, "block_size")
    codec = tokenizer or ByteTokenizer()
    try:
        prompt_tokens = codec.encode_text(prompt)
    except (ByteTokenizerError, UnicodeEncodeError) as error:
        raise ReasoningFormatError("prompt is not encodable") from error
    cue = FINAL_ID if mode == DIRECT_MODE else THINK_ID
    tokens = (
        BOS_ID,
        USER_ID,
        *prompt_tokens,
        ASSISTANT_ID,
        cue,
    )
    if len(tokens) > block_size:
        raise ReasoningFormatError("mode prompt exceeds model context")
    return tokens

