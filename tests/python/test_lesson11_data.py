import json
from pathlib import Path

from nanogpt_nspire.public_corpus import (
    PublicDocument,
    PublicSourceSnapshot,
    build_public_pilot,
)
from nanogpt_nspire.lesson11_data import (
    FINEWEB_EDU_SNAPSHOT,
    OPENWEBMATH_SNAPSHOT,
    run_pinned_public_pilot,
)


REGISTRY_PATH = (
    Path(__file__).parents[2] / "experiments" / "lesson10-public-sources.json"
)


def _documents(count: int) -> list[PublicDocument]:
    snapshot = PublicSourceSnapshot(
        source_id="fineweb-edu",
        repository="HuggingFaceFW/fineweb-edu",
        revision="9" * 40,
        parquet_path="sample-10BT/train/0000.parquet",
        row_groups=(0,),
        license_id="ODC-By-1.0",
        kind="fineweb_edu",
    )
    return [
        PublicDocument.create(
            snapshot=snapshot,
            source_document_id=f"synthetic-{index}",
            source_document_url=f"https://example.edu/{index}",
            text=(
                f"Document {index}. "
                + "This educational paragraph explains a scientific idea. " * 8
            ),
            row_group=0,
            row_index=index,
            quality={"fixture": True},
        )
        for index in range(count)
    ]


def test_public_pilot_is_atomic_deterministic_and_auditable(tmp_path) -> None:
    documents = _documents(200)
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_manifest = build_public_pilot(
        documents,
        first,
        registry_path=REGISTRY_PATH,
        split_seed="lesson11-test",
    )
    second_manifest = build_public_pilot(
        reversed(documents),
        second,
        registry_path=REGISTRY_PATH,
        split_seed="lesson11-test",
    )

    assert first_manifest == second_manifest
    assert first_manifest["schema_version"] == 1
    assert first_manifest["corpus"]["records"]["total"] == 200
    assert first_manifest["corpus"]["families"]["total"] == 200
    assert first_manifest["source_snapshots"][0]["revision"] == "9" * 40
    for relative_path, metadata in first_manifest["files"].items():
        first_payload = (first / relative_path).read_bytes()
        second_payload = (second / relative_path).read_bytes()
        assert first_payload == second_payload
        assert len(first_payload) == metadata["bytes"]

    provenance_lines = (
        first / "provenance.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(provenance_lines) == 200
    provenance = [json.loads(line) for line in provenance_lines]
    assert all("text" not in row for row in provenance)
    assert all("text_sha256" in row for row in provenance)
    assert all(row["source_document_url"].startswith("https://") for row in provenance)


def test_existing_destination_fails_without_modification(tmp_path) -> None:
    destination = tmp_path / "pilot"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    try:
        build_public_pilot(
            _documents(10),
            destination,
            registry_path=REGISTRY_PATH,
            split_seed="lesson11-test",
            require_all_splits=False,
        )
    except ValueError as error:
        assert "already exists" in str(error)
    else:
        raise AssertionError("existing destination should fail")

    assert marker.read_text(encoding="utf-8") == "keep"


def test_remote_snapshots_are_exact_commits_and_sparse_row_groups() -> None:
    assert FINEWEB_EDU_SNAPSHOT.revision == (
        "92cece42bcce787ee4af4619ab449fe48d86230d"
    )
    assert OPENWEBMATH_SNAPSHOT.revision == (
        "c5476cfea8186f9db20fe4b45f43fa2e231aa9ba"
    )
    assert FINEWEB_EDU_SNAPSHOT.row_groups == (0, 181, 363, 545)
    assert OPENWEBMATH_SNAPSHOT.row_groups == (0, 14, 28, 42)
    FINEWEB_EDU_SNAPSHOT.validate()
    OPENWEBMATH_SNAPSHOT.validate()


def test_pinned_runner_keeps_sources_separate_before_build(
    tmp_path,
    monkeypatch,
) -> None:
    fineweb = _documents(80)
    open_snapshot = PublicSourceSnapshot(
        source_id="openwebmath",
        repository="open-web-math/open-web-math",
        revision="a" * 40,
        parquet_path="default/train/0000.parquet",
        row_groups=(0,),
        license_id="ODC-By-1.0",
        kind="openwebmath",
    )
    openwebmath = [
        PublicDocument.create(
            snapshot=open_snapshot,
            source_document_id=f"open-{index}",
            source_document_url=f"https://math.example/{index}",
            text=(
                f"Math document {index}. "
                + "A variable and an equation form a mathematical example. " * 8
            ),
            row_group=0,
            row_index=index,
            quality={"fixture": True},
        )
        for index in range(80)
    ]

    def fake_scan(snapshot):
        documents = (
            fineweb
            if snapshot.source_id == "fineweb-edu"
            else openwebmath
        )
        return tuple(documents), {
            "accepted_rows": len(documents),
            "snapshot": snapshot.to_dict(),
        }

    monkeypatch.setattr(
        "nanogpt_nspire.lesson11_data.scan_parquet_snapshot",
        fake_scan,
    )
    output = tmp_path / "pilot"

    summary = run_pinned_public_pilot(
        output_dir=output,
        registry_path=REGISTRY_PATH,
        split_seed="runner-test",
        fineweb_max_utf8_bytes=20_000,
        openwebmath_max_utf8_bytes=20_000,
        max_documents_per_source=50,
    )

    assert summary["records"]["source_counts"]["fineweb-edu"] > 0
    assert summary["records"]["source_counts"]["openwebmath"] > 0
    acquisition = json.loads(
        (output / "acquisition.json").read_text(encoding="utf-8")
    )
    assert set(acquisition["selection"]) == {
        "fineweb-edu",
        "openwebmath",
    }
