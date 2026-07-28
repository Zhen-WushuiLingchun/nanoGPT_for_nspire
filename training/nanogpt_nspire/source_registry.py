"""Strict provenance and license eligibility for Lesson 10 data sources."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Mapping, Sequence


SOURCE_REGISTRY_SCHEMA_VERSION = 1
ELIGIBLE_LICENSES = frozenset(
    {
        "Apache-2.0",
        "MIT",
        "CC0-1.0",
        "CC-BY-4.0",
        "ODC-By-1.0",
        "Public-Domain",
        "DeepSeek-Output-Terms-2026-03-27",
    }
)
SOURCE_POLICIES = frozenset({"eligible", "row-filtered", "excluded"})
SOURCE_STAGES = frozenset(
    {
        "base_pretraining",
        "continued_pretraining",
        "sft",
        "distillation",
        "evaluation",
    }
)
SOURCE_FIELDS = frozenset(
    {
        "allowed_row_licenses",
        "exclusion_reason",
        "license_id",
        "name",
        "notes",
        "policy",
        "revision",
        "source_id",
        "stages",
        "subset",
        "url",
    }
)
SECRET_PATTERNS = (
    re.compile(rb"sk-[A-Za-z0-9]{16,}"),
    re.compile(rb"DEEPSEEK_API_KEY\s*=", re.IGNORECASE),
)


class SourceRegistryError(ValueError):
    """Raised when source provenance is incomplete or ineligible."""


@dataclass(frozen=True)
class SourceRecord:
    """One pinned public, generated, or explicitly excluded source."""

    allowed_row_licenses: tuple[str, ...]
    exclusion_reason: str | None
    license_id: str
    name: str
    notes: str
    policy: str
    revision: str
    source_id: str
    stages: tuple[str, ...]
    subset: str
    url: str


@dataclass(frozen=True)
class SourceRegistry:
    """Canonical versioned source collection."""

    schema_version: int
    sources: tuple[SourceRecord, ...]


def license_is_eligible(license_id: object) -> bool:
    """Return whether a row can enter the initial permissive training mix."""

    return isinstance(license_id, str) and license_id in ELIGIBLE_LICENSES


def registry_contains_secret(payload: bytes) -> bool:
    """Detect credential-shaped values without logging their contents."""

    return any(pattern.search(payload) is not None for pattern in SECRET_PATTERNS)


def _required_text(mapping: Mapping[str, object], field: str) -> str:
    value = mapping[field]
    if not isinstance(value, str) or not value.strip():
        raise SourceRegistryError(f"{field} must be a non-empty string")
    return value


def _text_tuple(value: object, field: str) -> tuple[str, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
    ):
        raise SourceRegistryError(f"{field} must be an array of strings")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise SourceRegistryError(f"{field} must contain non-empty strings")
    if len(set(result)) != len(result):
        raise SourceRegistryError(f"{field} contains duplicates")
    return result


def _parse_source(raw: object) -> SourceRecord:
    if not isinstance(raw, Mapping):
        raise SourceRegistryError("each source must be an object")
    fields = frozenset(raw)
    unknown = fields - SOURCE_FIELDS
    missing = SOURCE_FIELDS - fields
    if unknown:
        raise SourceRegistryError(
            f"source has unknown fields: {', '.join(sorted(unknown))}"
        )
    if missing:
        raise SourceRegistryError(
            f"source is missing fields: {', '.join(sorted(missing))}"
        )

    source_id = _required_text(raw, "source_id")
    name = _required_text(raw, "name")
    url = _required_text(raw, "url")
    revision = _required_text(raw, "revision")
    subset = _required_text(raw, "subset")
    license_id = _required_text(raw, "license_id")
    policy = _required_text(raw, "policy")
    notes = _required_text(raw, "notes")
    stages = _text_tuple(raw["stages"], "stages")
    allowed_row_licenses = _text_tuple(
        raw["allowed_row_licenses"],
        "allowed_row_licenses",
    )
    exclusion_reason = raw["exclusion_reason"]

    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", source_id):
        raise SourceRegistryError(f"source_id {source_id!r} is not canonical")
    if not url.startswith(("https://", "http://")):
        raise SourceRegistryError(f"source {source_id} URL must be HTTP(S)")
    if policy not in SOURCE_POLICIES:
        raise SourceRegistryError(f"source {source_id} policy is invalid")
    unknown_stages = set(stages) - SOURCE_STAGES
    if not stages or unknown_stages:
        raise SourceRegistryError(f"source {source_id} stages are invalid")
    if exclusion_reason is not None and (
        not isinstance(exclusion_reason, str) or not exclusion_reason.strip()
    ):
        raise SourceRegistryError(
            f"source {source_id} exclusion_reason must be null or non-empty"
        )

    if policy == "eligible":
        if not license_is_eligible(license_id):
            raise SourceRegistryError(
                f"source {source_id} license {license_id!r} is not eligible"
            )
        if allowed_row_licenses:
            raise SourceRegistryError(
                f"source {source_id} must not declare row licenses"
            )
        if exclusion_reason is not None:
            raise SourceRegistryError(
                f"source {source_id} eligible policy conflicts with exclusion_reason"
            )
    elif policy == "row-filtered":
        if license_id != "per-document":
            raise SourceRegistryError(
                f"source {source_id} row-filtered license must be per-document"
            )
        if not allowed_row_licenses:
            raise SourceRegistryError(
                f"source {source_id} requires at least one row license"
            )
        for row_license in allowed_row_licenses:
            if not license_is_eligible(row_license):
                raise SourceRegistryError(
                    f"source {source_id} row license {row_license!r} "
                    "is not eligible"
                )
        if exclusion_reason is not None:
            raise SourceRegistryError(
                f"source {source_id} row-filtered policy conflicts "
                "with exclusion_reason"
            )
    else:
        if not isinstance(exclusion_reason, str) or not exclusion_reason.strip():
            raise SourceRegistryError(
                f"source {source_id} excluded policy requires exclusion_reason"
            )
        if allowed_row_licenses:
            raise SourceRegistryError(
                f"excluded source {source_id} must not declare row licenses"
            )

    return SourceRecord(
        allowed_row_licenses=allowed_row_licenses,
        exclusion_reason=exclusion_reason,
        license_id=license_id,
        name=name,
        notes=notes,
        policy=policy,
        revision=revision,
        source_id=source_id,
        stages=stages,
        subset=subset,
        url=url,
    )


def parse_source_registry(raw: object) -> SourceRegistry:
    """Validate an in-memory source registry and canonicalize source order."""

    if not isinstance(raw, Mapping):
        raise SourceRegistryError("registry must be an object")
    unknown = frozenset(raw) - {"schema_version", "sources"}
    missing = {"schema_version", "sources"} - frozenset(raw)
    if unknown:
        raise SourceRegistryError(
            f"registry has unknown fields: {', '.join(sorted(unknown))}"
        )
    if missing:
        raise SourceRegistryError(
            f"registry is missing fields: {', '.join(sorted(missing))}"
        )
    schema_version = raw["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != SOURCE_REGISTRY_SCHEMA_VERSION
    ):
        raise SourceRegistryError(
            f"schema_version must be {SOURCE_REGISTRY_SCHEMA_VERSION}"
        )
    sources_raw = raw["sources"]
    if (
        not isinstance(sources_raw, Sequence)
        or isinstance(sources_raw, (str, bytes, bytearray))
        or not sources_raw
    ):
        raise SourceRegistryError("sources must be a non-empty array")
    sources = tuple(sorted((_parse_source(item) for item in sources_raw), key=lambda item: item.source_id))
    source_ids = [source.source_id for source in sources]
    duplicates = sorted(
        source_id for source_id in set(source_ids) if source_ids.count(source_id) > 1
    )
    if duplicates:
        raise SourceRegistryError(
            f"duplicate source_id: {', '.join(duplicates)}"
        )
    return SourceRegistry(schema_version=schema_version, sources=sources)


def canonical_registry_bytes(registry: SourceRegistry) -> bytes:
    """Serialize one registry in stable UTF-8 JSON form."""

    mapping = {
        "schema_version": registry.schema_version,
        "sources": [asdict(source) for source in registry.sources],
    }
    return (
        json.dumps(mapping, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def load_source_registry(path: str | Path) -> SourceRegistry:
    """Read, scan, parse, and validate one registry file."""

    payload = Path(path).read_bytes()
    if registry_contains_secret(payload):
        raise SourceRegistryError("registry contains credential-shaped data")
    try:
        raw = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceRegistryError(f"registry is not valid UTF-8 JSON: {error}") from error
    return parse_source_registry(raw)
