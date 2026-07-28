"""Canonical records, family-level splits, and atomic Lesson 10 shards."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable
import unicodedata

from nanogpt_nspire.byte_tokenizer import (
    BOS_ID,
    EOS_ID,
    SPECIAL_TOKEN_NAMES,
    TOKENIZER_SCHEMA_VERSION,
    VOCAB_SIZE,
    ByteTokenizer,
    ByteTokenizerError,
    ConversationTurn,
    format_conversation,
)
from nanogpt_nspire.data import pack_u16_le
from nanogpt_nspire.source_registry import (
    SourceRecord,
    canonical_registry_bytes,
    license_is_eligible,
    load_source_registry,
)


CORPUS_SCHEMA_VERSION = 1
SPLIT_KIND = "sha256-family-90-5-5"
SPLIT_NAMES = ("train", "validation", "test")


class CorpusError(ValueError):
    """Raised when records cannot form a safe reproducible corpus."""


def _nonempty_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise CorpusError(f"{field} must be a non-empty string")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise CorpusError(f"{field} must contain valid UTF-8 text") from error
    return value


@dataclass(frozen=True)
class CorpusRecord:
    """One base document or role-aware conversation with provenance."""

    record_id: str
    family_id: str
    kind: str
    source_id: str
    license_id: str
    text: str | None
    turns: tuple[ConversationTurn, ...]

    def __post_init__(self) -> None:
        _nonempty_text(self.record_id, "record_id")
        _nonempty_text(self.family_id, "family_id")
        _nonempty_text(self.source_id, "source_id")
        _nonempty_text(self.license_id, "license_id")
        if self.kind == "base":
            if self.text is None:
                raise CorpusError("base record text must be provided")
            _nonempty_text(self.text, "text")
            if self.turns:
                raise CorpusError("base record must not contain conversation turns")
        elif self.kind == "conversation":
            if self.text is not None:
                raise CorpusError("conversation record must not contain base text")
            if not self.turns:
                raise CorpusError("conversation record requires turns")
            try:
                format_conversation(self.turns)
            except ByteTokenizerError as error:
                raise CorpusError(f"conversation record is invalid: {error}") from error
        else:
            raise CorpusError("kind must be 'base' or 'conversation'")

    @classmethod
    def base(
        cls,
        *,
        record_id: str,
        family_id: str,
        text: str,
        source_id: str,
        license_id: str,
    ) -> CorpusRecord:
        return cls(
            record_id=record_id,
            family_id=family_id,
            kind="base",
            source_id=source_id,
            license_id=license_id,
            text=text,
            turns=(),
        )

    @classmethod
    def conversation(
        cls,
        *,
        record_id: str,
        family_id: str,
        turns: Iterable[ConversationTurn],
        source_id: str,
        license_id: str,
    ) -> CorpusRecord:
        return cls(
            record_id=record_id,
            family_id=family_id,
            kind="conversation",
            source_id=source_id,
            license_id=license_id,
            text=None,
            turns=tuple(turns),
        )


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    return "\n".join(lines).strip()


def normalized_content_fingerprint(record: CorpusRecord) -> str:
    """Hash normalized semantic content without record or family identifiers."""

    if record.kind == "base":
        assert record.text is not None
        content = {"kind": "base", "text": _normalize_text(record.text)}
    else:
        content = {
            "kind": "conversation",
            "turns": [
                {
                    "content": _normalize_text(turn.content),
                    "role": turn.role,
                }
                for turn in record.turns
            ],
        }
    payload = json.dumps(
        content,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_family_split(
    family_id: str,
    *,
    split_seed: str,
) -> str:
    """Assign all variants of one family to a stable 90/5/5 split."""

    _nonempty_text(family_id, "family_id")
    _nonempty_text(split_seed, "split_seed")
    digest = hashlib.sha256(f"{split_seed}:{family_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10_000
    if bucket < 9_000:
        return "train"
    if bucket < 9_500:
        return "validation"
    return "test"


def encode_corpus_record(
    record: CorpusRecord,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Encode one record and its same-length target eligibility mask."""

    tokenizer = ByteTokenizer()
    if record.kind == "conversation":
        return format_conversation(record.turns, tokenizer=tokenizer)
    assert record.text is not None
    content = tokenizer.encode_text(record.text)
    tokens = (BOS_ID, *content, EOS_ID)
    mask = (0, *(1 for _ in content), 1)
    return tokens, mask


def _source_allows_record(source: SourceRecord, record: CorpusRecord) -> None:
    if source.policy == "excluded":
        raise CorpusError(
            f"source {source.source_id} is excluded: {source.exclusion_reason}"
        )
    if source.policy == "eligible":
        if record.license_id != source.license_id:
            raise CorpusError(
                f"record {record.record_id} license does not match source"
            )
        if not license_is_eligible(record.license_id):
            raise CorpusError(
                f"record {record.record_id} license is not eligible"
            )
        return
    if record.license_id not in source.allowed_row_licenses:
        raise CorpusError(
            f"record {record.record_id} row license is not allowed by source"
        )


def _file_metadata(payload: bytes) -> dict[str, object]:
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def build_corpus(
    records: Iterable[CorpusRecord],
    output_dir: str | Path,
    *,
    registry_path: str | Path,
    split_seed: str,
    require_all_splits: bool = True,
) -> dict[str, object]:
    """Validate, deduplicate, split, encode, and atomically publish shards."""

    destination = Path(output_dir)
    if destination.exists():
        raise CorpusError(f"output destination already exists: {destination}")
    if not isinstance(require_all_splits, bool):
        raise CorpusError("require_all_splits must be boolean")
    _nonempty_text(split_seed, "split_seed")

    registry_file = Path(registry_path)
    registry = load_source_registry(registry_file)
    registry_payload = canonical_registry_bytes(registry)
    source_by_id = {source.source_id: source for source in registry.sources}

    materialized = tuple(records)
    if not materialized:
        raise CorpusError("corpus requires at least one record")
    if any(not isinstance(record, CorpusRecord) for record in materialized):
        raise CorpusError("every corpus item must be a CorpusRecord")

    record_ids: set[str] = set()
    by_content: dict[str, CorpusRecord] = {}
    accepted: list[CorpusRecord] = []
    deduplicated = 0
    for record in materialized:
        if record.record_id in record_ids:
            raise CorpusError(f"duplicate record_id {record.record_id}")
        record_ids.add(record.record_id)
        source = source_by_id.get(record.source_id)
        if source is None:
            raise CorpusError(f"record {record.record_id} has unknown source")
        _source_allows_record(source, record)
        fingerprint = normalized_content_fingerprint(record)
        previous = by_content.get(fingerprint)
        if previous is not None:
            if previous.family_id != record.family_id:
                raise CorpusError(
                    "duplicate content appears in conflicting families "
                    f"{previous.family_id} and {record.family_id}"
                )
            deduplicated += 1
            continue
        by_content[fingerprint] = record
        accepted.append(record)

    accepted.sort(key=lambda record: record.record_id)
    split_records: dict[str, list[CorpusRecord]] = {
        split: [] for split in SPLIT_NAMES
    }
    for record in accepted:
        split_records[
            stable_family_split(record.family_id, split_seed=split_seed)
        ].append(record)
    if require_all_splits:
        empty = [split for split, items in split_records.items() if not items]
        if empty:
            raise CorpusError(
                f"required corpus splits are empty: {', '.join(empty)}"
            )

    file_payloads: dict[str, bytes] = {}
    record_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    token_counts: dict[str, int] = {}
    for split in SPLIT_NAMES:
        token_stream: list[int] = []
        loss_stream = bytearray()
        families: set[str] = set()
        for record in split_records[split]:
            tokens, loss_mask = encode_corpus_record(record)
            if len(tokens) != len(loss_mask):
                raise CorpusError(
                    f"record {record.record_id} token/mask length mismatch"
                )
            token_stream.extend(tokens)
            loss_stream.extend(loss_mask)
            families.add(record.family_id)
        token_payload = pack_u16_le(token_stream)
        token_filename = f"{split}.tokens.bin"
        loss_filename = f"{split}.loss.bin"
        file_payloads[token_filename] = token_payload
        file_payloads[loss_filename] = bytes(loss_stream)
        record_counts[split] = len(split_records[split])
        family_counts[split] = len(families)
        token_counts[split] = len(token_stream)

    files = {
        filename: _file_metadata(payload)
        for filename, payload in sorted(file_payloads.items())
    }
    manifest: dict[str, object] = {
        "families": {
            **family_counts,
            "total": len({record.family_id for record in accepted}),
        },
        "files": files,
        "records": {
            "deduplicated": deduplicated,
            **record_counts,
            "total": len(accepted),
        },
        "schema_version": CORPUS_SCHEMA_VERSION,
        "source_registry": {
            "schema_version": registry.schema_version,
            "sha256": hashlib.sha256(registry_payload).hexdigest(),
        },
        "split": {
            "buckets": {
                "test": [9500, 10000],
                "train": [0, 9000],
                "validation": [9000, 9500],
            },
            "kind": SPLIT_KIND,
            "seed": split_seed,
        },
        "tokenizer": {
            "byte_tokens": 256,
            "schema_version": TOKENIZER_SCHEMA_VERSION,
            "special_tokens": {
                name: token_id
                for token_id, name in sorted(SPECIAL_TOKEN_NAMES.items())
            },
            "vocab_size": VOCAB_SIZE,
        },
        "tokens": {
            **token_counts,
            "total": sum(token_counts.values()),
        },
    }
    manifest_payload = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )
    try:
        for filename, payload in file_payloads.items():
            _write_bytes(temporary / filename, payload)
        _write_bytes(temporary / "manifest.json", manifest_payload)
        if destination.exists():
            raise CorpusError(
                f"output destination appeared during build: {destination}"
            )
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return manifest
