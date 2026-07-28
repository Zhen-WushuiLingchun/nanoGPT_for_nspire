"""Fixed byte and special-token protocol for the English model line."""

from __future__ import annotations

from collections.abc import Iterable


BYTE_VOCAB_SIZE = 256
BOS_ID = 256
EOS_ID = 257
USER_ID = 258
ASSISTANT_ID = 259
TOOL_ID = 260
THINK_ID = 261
FINAL_ID = 262
PAD_ID = 263
VOCAB_SIZE = 264
TOKENIZER_SCHEMA_VERSION = 1

SPECIAL_TOKEN_NAMES: dict[int, str] = {
    BOS_ID: "<BOS>",
    EOS_ID: "<EOS>",
    USER_ID: "<USER>",
    ASSISTANT_ID: "<ASSISTANT>",
    TOOL_ID: "<TOOL>",
    THINK_ID: "<THINK>",
    FINAL_ID: "<FINAL>",
    PAD_ID: "<PAD>",
}


class ByteTokenizerError(ValueError):
    """Raised when bytes, text, or token IDs violate the frozen protocol."""


def _validated_token(token: object, position: int) -> int:
    if isinstance(token, bool) or not isinstance(token, int):
        raise ByteTokenizerError(
            f"token at position {position} must be an integer token ID"
        )
    if token < 0 or token >= VOCAB_SIZE:
        raise ByteTokenizerError(
            f"token ID {token} at position {position} is outside [0, {VOCAB_SIZE})"
        )
    return token


class ByteTokenizer:
    """Encode UTF-8 as raw bytes while reserving stable model-only token IDs."""

    vocab_size = VOCAB_SIZE
    schema_version = TOKENIZER_SCHEMA_VERSION

    def encode_bytes(
        self,
        payload: bytes | bytearray | memoryview,
    ) -> tuple[int, ...]:
        """Return one token per raw byte."""

        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise ByteTokenizerError("payload must be a bytes-like object")
        return tuple(bytes(payload))

    def decode_bytes(self, tokens: Iterable[int]) -> bytes:
        """Decode byte tokens and reject special tokens."""

        payload = bytearray()
        for position, raw_token in enumerate(tokens):
            token = _validated_token(raw_token, position)
            if token >= BYTE_VOCAB_SIZE:
                name = SPECIAL_TOKEN_NAMES[token]
                raise ByteTokenizerError(
                    f"special token {name} at position {position} is not a raw byte"
                )
            payload.append(token)
        return bytes(payload)

    def encode_text(self, text: str) -> tuple[int, ...]:
        """Encode Unicode text through strict UTF-8."""

        if not isinstance(text, str):
            raise ByteTokenizerError("text must be a string")
        return self.encode_bytes(text.encode("utf-8", errors="strict"))

    def decode_text(self, tokens: Iterable[int]) -> str:
        """Decode raw byte tokens as strict UTF-8."""

        payload = self.decode_bytes(tokens)
        try:
            return payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ByteTokenizerError(
                f"byte tokens do not contain valid UTF-8: {error}"
            ) from error

    def render_tokens(self, tokens: Iterable[int]) -> str:
        """Render bytes as text and special IDs as unambiguous diagnostics."""

        output: list[str] = []
        pending = bytearray()

        def flush_pending() -> None:
            if pending:
                output.append(bytes(pending).decode("utf-8", errors="backslashreplace"))
                pending.clear()

        for position, raw_token in enumerate(tokens):
            token = _validated_token(raw_token, position)
            if token < BYTE_VOCAB_SIZE:
                pending.append(token)
            else:
                flush_pending()
                output.append(SPECIAL_TOKEN_NAMES[token])
        flush_pending()
        return "".join(output)
