"""Deterministic character-token data preparation for Lesson 01."""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Iterable, Sequence
import urllib.request


DATASET_SCHEMA_VERSION = 1
TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
TINY_SHAKESPEARE_SHA256 = (
    "86c4e6aa9db7c042ec79f339dcb96d42"
    "b0075e16b8fc2e86bf0ca57e2dc565ed"
)
MAX_SOURCE_BYTES = 8 * 1024 * 1024


class DatasetError(ValueError):
    """Raised when source text, tokens, or generated artifacts are invalid."""


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 digest for exact bytes."""

    return hashlib.sha256(data).hexdigest()


def build_vocabulary(text: str) -> tuple[str, ...]:
    """Build a deterministic character vocabulary sorted by code point."""

    if not text:
        raise DatasetError("source text is empty")
    return tuple(sorted(set(text)))


def _token_map(vocabulary: Sequence[str]) -> dict[str, int]:
    if not vocabulary:
        raise DatasetError("vocabulary is empty")
    if any(not isinstance(character, str) or len(character) != 1 for character in vocabulary):
        raise DatasetError("every vocabulary entry must be exactly one character")
    if len(set(vocabulary)) != len(vocabulary):
        raise DatasetError("vocabulary contains duplicate characters")
    return {character: token_id for token_id, character in enumerate(vocabulary)}


def encode_text(text: str, vocabulary: Sequence[str]) -> list[int]:
    """Map each character in text to its integer token ID."""

    mapping = _token_map(vocabulary)
    tokens: list[int] = []
    for position, character in enumerate(text):
        try:
            tokens.append(mapping[character])
        except KeyError as error:
            raise DatasetError(
                f"character {character!r} at position {position} is not in the vocabulary"
            ) from error
    return tokens


def decode_tokens(tokens: Iterable[int], vocabulary: Sequence[str]) -> str:
    """Map token IDs back to their characters."""

    _token_map(vocabulary)
    characters: list[str] = []
    for position, token_id in enumerate(tokens):
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            or token_id >= len(vocabulary)
        ):
            raise DatasetError(
                f"token ID {token_id!r} at position {position} is outside vocabulary"
            )
        characters.append(vocabulary[token_id])
    return "".join(characters)


def split_tokens(
    tokens: Sequence[int],
    train_fraction: float = 0.9,
) -> tuple[list[int], list[int]]:
    """Split a token sequence in order, using floor for the train boundary."""

    if not math.isfinite(train_fraction) or not 0.0 < train_fraction < 1.0:
        raise DatasetError("train_fraction must be strictly between 0 and 1")
    if len(tokens) < 2:
        raise DatasetError("train and validation splits each need at least one token")

    boundary = int(len(tokens) * train_fraction)
    if boundary == 0 or boundary == len(tokens):
        raise DatasetError("train and validation splits each need at least one token")
    return list(tokens[:boundary]), list(tokens[boundary:])


def pack_u16_le(tokens: Iterable[int]) -> bytes:
    """Pack token IDs as an explicitly little-endian uint16 byte stream."""

    packed = array("H")
    for position, token_id in enumerate(tokens):
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            or token_id > 0xFFFF
        ):
            raise DatasetError(
                f"token ID {token_id!r} at position {position} does not fit uint16"
            )
        packed.append(token_id)
    if sys.byteorder != "little":
        packed.byteswap()
    return packed.tobytes()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def fetch_tiny_shakespeare(
    output_path: str | Path,
    *,
    url: str = TINY_SHAKESPEARE_URL,
    expected_sha256: str = TINY_SHAKESPEARE_SHA256,
    timeout_seconds: float = 30.0,
) -> dict[str, object]:
    """Download the pinned Tiny Shakespeare source after validating its hash."""

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "nanoGPT-for-Nspire/0.1"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        source_bytes = response.read(MAX_SOURCE_BYTES + 1)

    if len(source_bytes) > MAX_SOURCE_BYTES:
        raise DatasetError(
            f"download exceeds the {MAX_SOURCE_BYTES}-byte source safety limit"
        )

    actual_sha256 = sha256_bytes(source_bytes)
    if actual_sha256 != expected_sha256.lower():
        raise DatasetError(
            "Tiny Shakespeare SHA-256 mismatch: "
            f"expected {expected_sha256.lower()}, got {actual_sha256}"
        )

    destination = Path(output_path)
    _atomic_write(destination, source_bytes)
    return {
        "bytes": len(source_bytes),
        "output": str(destination),
        "sha256": actual_sha256,
        "url": url,
    }


def prepare_dataset(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    train_fraction: float = 0.9,
) -> dict[str, object]:
    """Create deterministic train/validation token files and their manifest."""

    source = Path(source_path)
    source_bytes = source.read_bytes()
    if len(source_bytes) > MAX_SOURCE_BYTES:
        raise DatasetError(
            f"source exceeds the {MAX_SOURCE_BYTES}-byte source safety limit"
        )
    try:
        text = source_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise DatasetError(f"source is not valid UTF-8: {error}") from error

    vocabulary = build_vocabulary(text)
    tokens = encode_text(text, vocabulary)
    train_tokens, validation_tokens = split_tokens(tokens, train_fraction)
    train_bytes = pack_u16_le(train_tokens)
    validation_bytes = pack_u16_le(validation_tokens)

    manifest: dict[str, object] = {
        "dtype": "uint16-le",
        "files": {
            "train.bin": {
                "bytes": len(train_bytes),
                "sha256": sha256_bytes(train_bytes),
            },
            "val.bin": {
                "bytes": len(validation_bytes),
                "sha256": sha256_bytes(validation_bytes),
            },
        },
        "schema_version": DATASET_SCHEMA_VERSION,
        "source": {
            "bytes": len(source_bytes),
            "filename": source.name,
            "sha256": sha256_bytes(source_bytes),
        },
        "split": {
            "kind": "sequential",
            "train_fraction": train_fraction,
        },
        "tokens": {
            "total": len(tokens),
            "train": len(train_tokens),
            "validation": len(validation_tokens),
        },
        "vocab_size": len(vocabulary),
        "vocabulary": list(vocabulary),
    }
    manifest_bytes = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    destination = Path(output_dir)
    _atomic_write(destination / "train.bin", train_bytes)
    _atomic_write(destination / "val.bin", validation_bytes)
    _atomic_write(destination / "manifest.json", manifest_bytes)
    return manifest


def _summary(manifest: dict[str, object], output_dir: str | Path) -> dict[str, object]:
    source = manifest["source"]
    tokens = manifest["tokens"]
    assert isinstance(source, dict)
    assert isinstance(tokens, dict)
    return {
        "output": str(output_dir),
        "source_bytes": source["bytes"],
        "source_sha256": source["sha256"],
        "total_tokens": tokens["total"],
        "train_tokens": tokens["train"],
        "validation_tokens": tokens["validation"],
        "vocab_size": manifest["vocab_size"],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic character-token datasets."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser(
        "fetch",
        help="Download the pinned Tiny Shakespeare source.",
    )
    fetch_parser.add_argument("--output", type=Path, required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Encode a UTF-8 source file into train and validation token files.",
    )
    prepare_parser.add_argument("--input", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--train-fraction", type=float, default=0.9)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fetch or prepare command-line interface."""

    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "fetch":
            result = fetch_tiny_shakespeare(arguments.output)
        else:
            manifest = prepare_dataset(
                arguments.input,
                arguments.output,
                train_fraction=arguments.train_fraction,
            )
            result = _summary(manifest, arguments.output)
    except (DatasetError, OSError) as error:
        parser.exit(2, f"error: {error}\n")

    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
