"""Verified token loading and deterministic next-token batch sampling."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import torch

from nanogpt_nspire.data import DATASET_SCHEMA_VERSION, DatasetError, sha256_bytes


@dataclass(frozen=True)
class TokenDataset:
    """Validated train/validation tensors plus their shared vocabulary."""

    train: torch.Tensor
    validation: torch.Tensor
    vocabulary: tuple[str, ...]
    manifest: dict[str, object]


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DatasetError(f"{name} must be a JSON object")
    return value


def _require_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DatasetError(f"{name} must be a non-negative integer")
    return value


def _load_token_file(
    path: Path,
    file_manifest: dict[str, Any],
    *,
    expected_tokens: int,
    vocab_size: int,
) -> torch.Tensor:
    expected_bytes = _require_nonnegative_int(
        file_manifest.get("bytes"),
        f"{path.name} manifest bytes",
    )
    expected_sha256 = file_manifest.get("sha256")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise DatasetError(f"{path.name} manifest SHA-256 is invalid")

    token_bytes = path.read_bytes()
    if len(token_bytes) != expected_bytes:
        raise DatasetError(
            f"{path.name} byte length mismatch: expected {expected_bytes}, "
            f"got {len(token_bytes)}"
        )
    if len(token_bytes) % 2 != 0:
        raise DatasetError(f"{path.name} byte length is not divisible by two")
    if len(token_bytes) != expected_tokens * 2:
        raise DatasetError(
            f"{path.name} byte length does not match {expected_tokens} uint16 tokens"
        )

    actual_sha256 = sha256_bytes(token_bytes)
    if actual_sha256 != expected_sha256:
        raise DatasetError(
            f"{path.name} SHA-256 mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )

    unpacked = array("H")
    unpacked.frombytes(token_bytes)
    if sys.byteorder != "little":
        unpacked.byteswap()
    if any(token_id >= vocab_size for token_id in unpacked):
        raise DatasetError(f"{path.name} contains a token outside the vocabulary")
    return torch.tensor(unpacked, dtype=torch.long)


def load_token_dataset(data_dir: str | Path) -> TokenDataset:
    """Load Lesson 01 artifacts only after revalidating their manifest."""

    directory = Path(data_dir)
    manifest_path = directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DatasetError(f"manifest.json is not valid JSON: {error}") from error
    manifest = _require_dict(manifest, "manifest")

    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise DatasetError(
            f"unsupported dataset schema_version {manifest.get('schema_version')!r}"
        )
    if manifest.get("dtype") != "uint16-le":
        raise DatasetError(f"unsupported dataset dtype {manifest.get('dtype')!r}")

    vocabulary_value = manifest.get("vocabulary")
    if not isinstance(vocabulary_value, list):
        raise DatasetError("manifest vocabulary must be a JSON array")
    vocabulary = tuple(vocabulary_value)
    if (
        any(not isinstance(character, str) or len(character) != 1 for character in vocabulary)
        or len(set(vocabulary)) != len(vocabulary)
    ):
        raise DatasetError("manifest vocabulary entries must be unique characters")
    vocab_size = _require_nonnegative_int(manifest.get("vocab_size"), "vocab_size")
    if not vocabulary or len(vocabulary) != vocab_size:
        raise DatasetError(
            f"vocabulary length {len(vocabulary)} does not match vocab_size {vocab_size}"
        )

    token_manifest = _require_dict(manifest.get("tokens"), "tokens")
    train_token_count = _require_nonnegative_int(
        token_manifest.get("train"),
        "tokens.train",
    )
    validation_token_count = _require_nonnegative_int(
        token_manifest.get("validation"),
        "tokens.validation",
    )
    total_token_count = _require_nonnegative_int(
        token_manifest.get("total"),
        "tokens.total",
    )
    if train_token_count + validation_token_count != total_token_count:
        raise DatasetError("train and validation token counts do not equal total")

    file_manifests = _require_dict(manifest.get("files"), "files")
    train_file_manifest = _require_dict(
        file_manifests.get("train.bin"),
        "files.train.bin",
    )
    validation_file_manifest = _require_dict(
        file_manifests.get("val.bin"),
        "files.val.bin",
    )

    train = _load_token_file(
        directory / "train.bin",
        train_file_manifest,
        expected_tokens=train_token_count,
        vocab_size=vocab_size,
    )
    validation = _load_token_file(
        directory / "val.bin",
        validation_file_manifest,
        expected_tokens=validation_token_count,
        vocab_size=vocab_size,
    )
    return TokenDataset(
        train=train,
        validation=validation,
        vocabulary=vocabulary,
        manifest=manifest,
    )


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DatasetError(f"{name} must be a positive integer")
    return value


def make_batch(
    tokens: torch.Tensor,
    *,
    batch_size: int,
    block_size: int,
    generator: torch.Generator,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample shifted next-token windows and move the finished batch to device."""

    _positive_integer(batch_size, "batch_size")
    _positive_integer(block_size, "block_size")
    if tokens.ndim != 1 or tokens.dtype != torch.long:
        raise DatasetError("tokens must be a one-dimensional torch.long tensor")
    if tokens.numel() < block_size + 1:
        raise DatasetError(
            "token stream must contain at least block_size plus one token"
        )
    if not isinstance(generator, torch.Generator):
        raise DatasetError("generator must be a torch.Generator")

    start_positions = torch.randint(
        low=0,
        high=tokens.numel() - block_size,
        size=(batch_size,),
        generator=generator,
        device="cpu",
    )
    windows = torch.stack(
        [
            tokens[int(start) : int(start) + block_size + 1]
            for start in start_positions
        ]
    )
    inputs = windows[:, :-1]
    targets = windows[:, 1:]
    target_device = torch.device(device)
    return inputs.to(target_device), targets.to(target_device)
