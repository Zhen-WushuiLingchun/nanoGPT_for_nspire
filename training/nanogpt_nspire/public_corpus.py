"""Pinned public Parquet ingestion for the English base-model pilot."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Iterable, Iterator, Mapping
import unicodedata

from nanogpt_nspire.base_corpus import CorpusRecord, build_corpus


MIN_DOCUMENT_UTF8_BYTES = 256
MAX_DOCUMENT_UTF8_BYTES = 64 * 1024
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_SUPPORTED_KINDS = {"fineweb_edu", "openwebmath"}


class PublicCorpusError(ValueError):
    """Raised when public source data is ambiguous, unsafe, or untraceable."""


def _nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PublicCorpusError(f"{name} must be a non-empty string")
    return value.strip()


def _non_negative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PublicCorpusError(f"{name} must be a non-negative integer")
    return value


def _positive_integer(value: object, name: str) -> int:
    value = _non_negative_integer(value, name)
    if value == 0:
        raise PublicCorpusError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class PublicSourceSnapshot:
    """One immutable Parquet input and the row groups permitted for scanning."""

    source_id: str
    repository: str
    revision: str
    parquet_path: str
    row_groups: tuple[int, ...]
    license_id: str
    kind: str

    def validate(self) -> None:
        _nonempty_text(self.source_id, "source_id")
        repository = _nonempty_text(self.repository, "repository")
        if (
            repository.startswith(("/", "\\"))
            or repository.count("/") != 1
            or ".." in repository
        ):
            raise PublicCorpusError(
                "repository must be an owner/name dataset identifier"
            )
        if not isinstance(self.revision, str) or not _REVISION_PATTERN.fullmatch(
            self.revision
        ):
            raise PublicCorpusError(
                "revision must be an exact 40 lowercase hex commit"
            )
        parquet_path = _nonempty_text(self.parquet_path, "parquet_path")
        if (
            not parquet_path.endswith(".parquet")
            or "\\" in parquet_path
            or parquet_path.startswith("/")
            or ".." in parquet_path.split("/")
        ):
            raise PublicCorpusError(
                "parquet_path must be a safe relative .parquet path"
            )
        if (
            not isinstance(self.row_groups, tuple)
            or not self.row_groups
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                for index in self.row_groups
            )
        ):
            raise PublicCorpusError(
                "row_groups must be a non-empty tuple of non-negative integers"
            )
        if len(set(self.row_groups)) != len(self.row_groups):
            raise PublicCorpusError("row_groups must be unique")
        if tuple(sorted(self.row_groups)) != self.row_groups:
            raise PublicCorpusError("row_groups must be sorted")
        _nonempty_text(self.license_id, "license_id")
        if self.kind not in _SUPPORTED_KINDS:
            raise PublicCorpusError(f"unsupported source kind: {self.kind!r}")

    @property
    def hf_path(self) -> str:
        self.validate()
        return (
            f"datasets/{self.repository}@{self.revision}/"
            f"{self.parquet_path}"
        )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        value = asdict(self)
        value["row_groups"] = list(self.row_groups)
        return value


def normalize_public_text(text: object) -> str:
    """Normalize line endings and reject unsafe or unhelpfully small text."""

    if not isinstance(text, str):
        raise PublicCorpusError("text must be a string")
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(
        line.rstrip() for line in normalized.split("\n")
    ).strip()
    while "\n\n\n" in normalized:
        normalized = normalized.replace("\n\n\n", "\n\n")
    if "\ufffd" in normalized:
        raise PublicCorpusError("text contains a Unicode replacement character")
    for character in normalized:
        if (
            unicodedata.category(character) == "Cc"
            and character not in {"\n", "\t"}
        ):
            raise PublicCorpusError("text contains a forbidden control character")
    try:
        payload = normalized.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise PublicCorpusError("text is not valid UTF-8") from error
    if len(payload) < MIN_DOCUMENT_UTF8_BYTES:
        raise PublicCorpusError(
            f"text must contain at least {MIN_DOCUMENT_UTF8_BYTES} UTF-8 bytes"
        )
    if len(payload) > MAX_DOCUMENT_UTF8_BYTES:
        raise PublicCorpusError(
            f"text exceeds {MAX_DOCUMENT_UTF8_BYTES} UTF-8 bytes"
        )
    return normalized


@dataclass(frozen=True)
class PublicDocument:
    """A normalized base document plus complete source-row provenance."""

    record_id: str
    family_id: str
    snapshot: PublicSourceSnapshot
    source_document_id: str
    source_document_url: str
    text: str
    row_group: int
    row_index: int
    quality_json: str

    @classmethod
    def create(
        cls,
        *,
        snapshot: PublicSourceSnapshot,
        source_document_id: str,
        source_document_url: str,
        text: str,
        row_group: int,
        row_index: int,
        quality: Mapping[str, object],
    ) -> PublicDocument:
        if not isinstance(snapshot, PublicSourceSnapshot):
            raise PublicCorpusError(
                "snapshot must be a PublicSourceSnapshot"
            )
        snapshot.validate()
        source_document_id = _nonempty_text(
            source_document_id,
            "source_document_id",
        )
        source_document_url = _nonempty_text(
            source_document_url,
            "source_document_url",
        )
        if not source_document_url.startswith(("http://", "https://")):
            raise PublicCorpusError(
                "source_document_url must use HTTP or HTTPS"
            )
        normalized = normalize_public_text(text)
        row_group = _non_negative_integer(row_group, "row_group")
        row_index = _non_negative_integer(row_index, "row_index")
        if not isinstance(quality, Mapping):
            raise PublicCorpusError("quality must be a mapping")
        try:
            quality_json = json.dumps(
                dict(quality),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise PublicCorpusError(
                "quality must contain finite JSON values"
            ) from error
        identity = hashlib.sha256(
            (
                f"{snapshot.source_id}\0{snapshot.revision}\0"
                f"{snapshot.parquet_path}\0{source_document_id}"
            ).encode("utf-8")
        ).hexdigest()
        record_id = f"{snapshot.source_id}:{identity}"
        return cls(
            record_id=record_id,
            family_id=record_id,
            snapshot=snapshot,
            source_document_id=source_document_id,
            source_document_url=source_document_url,
            text=normalized,
            row_group=row_group,
            row_index=row_index,
            quality_json=quality_json,
        )

    def to_corpus_record(self) -> CorpusRecord:
        return CorpusRecord.base(
            record_id=self.record_id,
            family_id=self.family_id,
            text=self.text,
            source_id=self.snapshot.source_id,
            license_id=self.snapshot.license_id,
        )

    def provenance(self) -> dict[str, object]:
        payload = self.text.encode("utf-8")
        return {
            "family_id": self.family_id,
            "license_id": self.snapshot.license_id,
            "parquet_path": self.snapshot.parquet_path,
            "quality": json.loads(self.quality_json),
            "record_id": self.record_id,
            "row_group": self.row_group,
            "row_index": self.row_index,
            "source_document_id": self.source_document_id,
            "source_document_url": self.source_document_url,
            "source_id": self.snapshot.source_id,
            "source_repository": self.snapshot.repository,
            "source_revision": self.snapshot.revision,
            "text_sha256": hashlib.sha256(payload).hexdigest(),
            "text_utf8_bytes": len(payload),
        }


def _required_row_text(row: Mapping[str, object], name: str) -> str:
    if name not in row:
        raise PublicCorpusError(f"row is missing required {name}")
    return _nonempty_text(row[name], name)


def document_from_row(
    snapshot: PublicSourceSnapshot,
    row: Mapping[str, object],
    *,
    row_group: int,
    row_index: int,
) -> PublicDocument:
    """Validate one source-specific row and convert it to the common schema."""

    if not isinstance(snapshot, PublicSourceSnapshot):
        raise PublicCorpusError("snapshot must be a PublicSourceSnapshot")
    snapshot.validate()
    if not isinstance(row, Mapping):
        raise PublicCorpusError("row must be a mapping")
    text = _required_row_text(row, "text")
    url = _required_row_text(row, "url")

    if snapshot.kind == "fineweb_edu":
        source_document_id = _required_row_text(row, "id")
        if row.get("language") != "en":
            raise PublicCorpusError("FineWeb-Edu row language must be en")
        language_score = row.get("language_score")
        if (
            isinstance(language_score, bool)
            or not isinstance(language_score, (int, float))
            or not math_is_finite(language_score)
            or language_score < 0.9
        ):
            raise PublicCorpusError(
                "FineWeb-Edu language score must be at least 0.9"
            )
        educational_score = row.get("int_score")
        if (
            isinstance(educational_score, bool)
            or not isinstance(educational_score, int)
            or educational_score < 4
        ):
            raise PublicCorpusError(
                "FineWeb-Edu educational score must be at least 4"
            )
        quality = {
            "educational_score": educational_score,
            "language": "en",
            "language_score": float(language_score),
            "source_token_count": row.get("token_count"),
        }
    else:
        source_document_id = url
        quality = {
            "dataset_filter": "OpenWebMath English/math/quality pipeline",
        }
    return PublicDocument.create(
        snapshot=snapshot,
        source_document_id=source_document_id,
        source_document_url=url,
        text=text,
        row_group=row_group,
        row_index=row_index,
        quality=quality,
    )


def math_is_finite(value: int | float) -> bool:
    """Avoid accepting NaN/Inf without coercing arbitrary numeric objects."""

    return value == value and value not in {float("inf"), float("-inf")}


def select_public_documents(
    documents: Iterable[PublicDocument],
    *,
    seed: str,
    max_documents: int,
    max_utf8_bytes: int,
) -> tuple[PublicDocument, ...]:
    """Select hash-ranked unique text independently of input order."""

    seed = _nonempty_text(seed, "seed")
    max_documents = _positive_integer(max_documents, "max_documents")
    max_utf8_bytes = _positive_integer(max_utf8_bytes, "max_utf8_bytes")
    materialized = tuple(documents)
    if not materialized:
        raise PublicCorpusError("selection requires at least one document")
    if any(not isinstance(item, PublicDocument) for item in materialized):
        raise PublicCorpusError(
            "selection requires PublicDocument instances"
        )
    by_record_id: dict[str, PublicDocument] = {}
    for document in materialized:
        if document.record_id in by_record_id:
            raise PublicCorpusError(
                f"duplicate record_id {document.record_id}"
            )
        by_record_id[document.record_id] = document
    ranked = sorted(
        materialized,
        key=lambda item: (
            hashlib.sha256(
                f"{seed}:{item.record_id}".encode("utf-8")
            ).digest(),
            item.record_id,
        ),
    )
    selected: list[PublicDocument] = []
    selected_bytes = 0
    seen_content: set[str] = set()
    for document in ranked:
        payload = document.text.encode("utf-8")
        content_hash = hashlib.sha256(payload).hexdigest()
        if content_hash in seen_content:
            continue
        if selected_bytes + len(payload) > max_utf8_bytes:
            continue
        selected.append(document)
        selected_bytes += len(payload)
        seen_content.add(content_hash)
        if len(selected) >= max_documents:
            break
    if not selected:
        raise PublicCorpusError(
            "no document fits the requested selection limits"
        )
    return tuple(selected)


def scan_parquet_snapshot(
    snapshot: PublicSourceSnapshot,
) -> tuple[tuple[PublicDocument, ...], dict[str, object]]:
    """Read only the pinned row groups through Hugging Face range requests."""

    snapshot.validate()
    try:
        from huggingface_hub import HfFileSystem
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise PublicCorpusError(
            "public data requires huggingface-hub and pyarrow"
        ) from error

    filesystem = HfFileSystem()
    documents: list[PublicDocument] = []
    rejection_messages: Counter[str] = Counter()
    rows_scanned = 0
    with filesystem.open(snapshot.hf_path, "rb") as stream:
        parquet_file = parquet.ParquetFile(stream)
        row_group_count = parquet_file.metadata.num_row_groups
        invalid = [
            index
            for index in snapshot.row_groups
            if index >= row_group_count
        ]
        if invalid:
            raise PublicCorpusError(
                f"row groups exceed Parquet metadata: {invalid}"
            )
        for row_group in snapshot.row_groups:
            table = parquet_file.read_row_group(row_group)
            for row_index, row in enumerate(table.to_pylist()):
                rows_scanned += 1
                try:
                    document = document_from_row(
                        snapshot,
                        row,
                        row_group=row_group,
                        row_index=row_index,
                    )
                except PublicCorpusError as error:
                    rejection_messages[str(error)] += 1
                    continue
                documents.append(document)
    return tuple(documents), {
        "accepted_rows": len(documents),
        "parquet_row_groups": row_group_count,
        "parquet_rows": parquet_file.metadata.num_rows,
        "rejected": dict(sorted(rejection_messages.items())),
        "rejected_rows": rows_scanned - len(documents),
        "row_groups_scanned": list(snapshot.row_groups),
        "rows_scanned": rows_scanned,
        "snapshot": snapshot.to_dict(),
    }


def _stable_json_bytes(value: object, *, pretty: bool) -> bytes:
    if pretty:
        text = json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    else:
        text = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
    return (text + "\n").encode("utf-8")


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _file_metadata(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def build_public_pilot(
    documents: Iterable[PublicDocument],
    output_dir: str | Path,
    *,
    registry_path: str | Path,
    split_seed: str,
    require_all_splits: bool = True,
    acquisition: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Atomically publish shards plus text-free row-level provenance."""

    destination = Path(output_dir)
    if destination.exists():
        raise PublicCorpusError(
            f"output destination already exists: {destination}"
        )
    materialized = tuple(documents)
    if not materialized:
        raise PublicCorpusError("public pilot requires documents")
    if any(not isinstance(item, PublicDocument) for item in materialized):
        raise PublicCorpusError(
            "public pilot requires PublicDocument instances"
        )
    ordered = tuple(sorted(materialized, key=lambda item: item.record_id))
    record_ids = [item.record_id for item in ordered]
    if len(set(record_ids)) != len(record_ids):
        raise PublicCorpusError("public pilot record IDs must be unique")
    snapshots: dict[
        tuple[str, str, str],
        PublicSourceSnapshot,
    ] = {}
    for document in ordered:
        snapshot = document.snapshot
        snapshot.validate()
        key = (
            snapshot.source_id,
            snapshot.revision,
            snapshot.parquet_path,
        )
        snapshots[key] = snapshot

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )
    try:
        corpus_manifest = build_corpus(
            (item.to_corpus_record() for item in ordered),
            temporary / "shards",
            registry_path=registry_path,
            split_seed=split_seed,
            require_all_splits=require_all_splits,
        )
        provenance_payload = b"".join(
            _stable_json_bytes(item.provenance(), pretty=False)
            for item in ordered
        )
        _write_bytes(temporary / "provenance.jsonl", provenance_payload)
        if acquisition is not None:
            if not isinstance(acquisition, Mapping):
                raise PublicCorpusError("acquisition must be a mapping")
            try:
                acquisition_payload = _stable_json_bytes(
                    dict(acquisition),
                    pretty=True,
                )
            except (TypeError, ValueError) as error:
                raise PublicCorpusError(
                    "acquisition must contain finite JSON values"
                ) from error
            _write_bytes(
                temporary / "acquisition.json",
                acquisition_payload,
            )

        source_counts: Counter[str] = Counter()
        source_bytes: Counter[str] = Counter()
        for item in ordered:
            source_counts[item.snapshot.source_id] += 1
            source_bytes[item.snapshot.source_id] += len(
                item.text.encode("utf-8")
            )
        files = {
            path.relative_to(temporary).as_posix(): _file_metadata(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        }
        manifest: dict[str, object] = {
            "acquisition": (
                dict(acquisition) if acquisition is not None else None
            ),
            "corpus": corpus_manifest,
            "files": files,
            "records": {
                "source_counts": dict(sorted(source_counts.items())),
                "source_utf8_bytes": dict(sorted(source_bytes.items())),
                "total": len(ordered),
                "total_utf8_bytes": sum(source_bytes.values()),
            },
            "schema_version": 1,
            "source_snapshots": [
                snapshot.to_dict()
                for _, snapshot in sorted(snapshots.items())
            ],
        }
        _write_bytes(
            temporary / "manifest.json",
            _stable_json_bytes(manifest, pretty=True),
        )
        if destination.exists():
            raise PublicCorpusError(
                f"output destination appeared during build: {destination}"
            )
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return manifest
