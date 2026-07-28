import json
from pathlib import Path

import pytest

from nanogpt_nspire.source_registry import (
    SourceRegistryError,
    canonical_registry_bytes,
    license_is_eligible,
    load_source_registry,
    parse_source_registry,
    registry_contains_secret,
)


REGISTRY_PATH = (
    Path(__file__).parents[2] / "experiments" / "lesson10-public-sources.json"
)


@pytest.mark.parametrize(
    "license_id",
    [
        "Apache-2.0",
        "MIT",
        "CC0-1.0",
        "CC-BY-4.0",
        "ODC-By-1.0",
        "Public-Domain",
        "DeepSeek-Output-Terms-2026-03-27",
    ],
)
def test_initial_mix_accepts_declared_permissive_licenses(license_id):
    assert license_is_eligible(license_id)


@pytest.mark.parametrize(
    "license_id",
    [
        "unknown",
        "CC-BY-NC-3.0",
        "CC-BY-NC-SA-4.0",
        "CC-BY-SA-4.0",
        "AI-TRAINING-PROHIBITED",
        "",
    ],
)
def test_initial_mix_fails_closed_on_disallowed_licenses(license_id):
    assert not license_is_eligible(license_id)


def test_committed_registry_is_canonical_and_contains_expected_sources():
    registry = load_source_registry(REGISTRY_PATH)
    source_ids = {source.source_id for source in registry.sources}

    assert {
        "fineweb-edu",
        "common-corpus",
        "openwebmath",
        "deepmind-mathematics",
        "gsm8k",
        "openmathinstruct-2",
        "oasst1",
        "deepseek-v4-pro-generated",
    } <= source_ids
    assert {
        "openstax-college-physics-2e",
        "sciq",
        "openbookqa",
    } <= source_ids
    assert canonical_registry_bytes(registry) == REGISTRY_PATH.read_bytes()
    assert not registry_contains_secret(REGISTRY_PATH.read_bytes())


def test_registry_rejects_duplicate_ids_and_unknown_fields():
    source = {
        "allowed_row_licenses": [],
        "exclusion_reason": None,
        "license_id": "MIT",
        "name": "Example",
        "notes": "Fixture",
        "policy": "eligible",
        "revision": "v1",
        "source_id": "example",
        "stages": ["sft"],
        "subset": "default",
        "url": "https://example.invalid/data",
    }
    duplicate = {"schema_version": 1, "sources": [source, dict(source)]}

    with pytest.raises(SourceRegistryError, match="duplicate source_id"):
        parse_source_registry(duplicate)

    unknown = json.loads(json.dumps({"schema_version": 1, "sources": [source]}))
    unknown["sources"][0]["api_key"] = "not-a-real-value"
    with pytest.raises(SourceRegistryError, match="unknown fields"):
        parse_source_registry(unknown)


def test_registry_rejects_inconsistent_policy_and_license():
    disallowed = {
        "allowed_row_licenses": [],
        "exclusion_reason": None,
        "license_id": "CC-BY-NC-3.0",
        "name": "Bad source",
        "notes": "Fixture",
        "policy": "eligible",
        "revision": "v1",
        "source_id": "bad-source",
        "stages": ["sft"],
        "subset": "default",
        "url": "https://example.invalid/data",
    }
    with pytest.raises(SourceRegistryError, match="not eligible"):
        parse_source_registry({"schema_version": 1, "sources": [disallowed]})

    missing_reason = dict(disallowed)
    missing_reason["policy"] = "excluded"
    missing_reason["exclusion_reason"] = None
    with pytest.raises(SourceRegistryError, match="exclusion_reason"):
        parse_source_registry({"schema_version": 1, "sources": [missing_reason]})


def test_row_filtered_source_requires_only_permissive_row_licenses():
    source = {
        "allowed_row_licenses": ["Public-Domain", "CC-BY-4.0"],
        "exclusion_reason": None,
        "license_id": "per-document",
        "name": "Mixed source",
        "notes": "Fixture",
        "policy": "row-filtered",
        "revision": "v1",
        "source_id": "mixed",
        "stages": ["base_pretraining"],
        "subset": "English",
        "url": "https://example.invalid/data",
    }

    registry = parse_source_registry({"schema_version": 1, "sources": [source]})

    assert registry.sources[0].allowed_row_licenses == (
        "Public-Domain",
        "CC-BY-4.0",
    )

    source["allowed_row_licenses"] = ["Public-Domain", "CC-BY-SA-4.0"]
    with pytest.raises(SourceRegistryError, match="row license"):
        parse_source_registry({"schema_version": 1, "sources": [source]})


def test_secret_scanner_detects_key_shapes_without_storing_a_real_key():
    prefix = "s" + "k-"
    fake_secret = (prefix + "x" * 24).encode("ascii")

    assert registry_contains_secret(fake_secret)
    assert registry_contains_secret(b"DEEPSEEK_API_KEY=placeholder")
