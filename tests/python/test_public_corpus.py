import json

import pytest

from nanogpt_nspire.public_corpus import (
    PublicCorpusError,
    PublicSourceSnapshot,
    document_from_row,
    select_public_documents,
)


FINEWEB = PublicSourceSnapshot(
    source_id="fineweb-edu",
    repository="HuggingFaceFW/fineweb-edu",
    revision="9" * 40,
    parquet_path="sample-10BT/train/0000.parquet",
    row_groups=(0, 3),
    license_id="ODC-By-1.0",
    kind="fineweb_edu",
)

OPENWEBMATH = PublicSourceSnapshot(
    source_id="openwebmath",
    repository="open-web-math/open-web-math",
    revision="a" * 40,
    parquet_path="default/train/0000.parquet",
    row_groups=(1, 4),
    license_id="ODC-By-1.0",
    kind="openwebmath",
)


def _fineweb_row(index: int, *, score: int = 4) -> dict[str, object]:
    return {
        "dump": "CC-MAIN-2024-10",
        "file_path": "s3://example/file.json.gz",
        "id": f"doc-{index}",
        "int_score": score,
        "language": "en",
        "language_score": 0.99,
        "score": float(score),
        "text": (
            f"Educational document {index}. "
            + "A clear explanation of force, motion, energy, and algebra. " * 8
        ),
        "token_count": 128,
        "url": f"https://example.edu/{index}",
    }


def _openwebmath_row(index: int) -> dict[str, object]:
    return {
        "date": "2023-01-01",
        "metadata": json.dumps({"language": "en"}),
        "text": (
            f"Mathematics document {index}. "
            + "Let x be a real number and solve the equation step by step. " * 8
        ),
        "url": f"https://math.example/{index}",
    }


def test_snapshot_requires_exact_revision_and_bounded_row_groups() -> None:
    FINEWEB.validate()

    with pytest.raises(PublicCorpusError, match="40 lowercase hex"):
        PublicSourceSnapshot(
            source_id="fineweb-edu",
            repository="HuggingFaceFW/fineweb-edu",
            revision="main",
            parquet_path="sample-10BT/train/0000.parquet",
            row_groups=(0,),
            license_id="ODC-By-1.0",
            kind="fineweb_edu",
        ).validate()
    with pytest.raises(PublicCorpusError, match="unique"):
        PublicSourceSnapshot(
            source_id="fineweb-edu",
            repository="HuggingFaceFW/fineweb-edu",
            revision="9" * 40,
            parquet_path="sample-10BT/train/0000.parquet",
            row_groups=(0, 0),
            license_id="ODC-By-1.0",
            kind="fineweb_edu",
        ).validate()


def test_fineweb_row_preserves_provenance_and_filters_quality() -> None:
    document = document_from_row(
        FINEWEB,
        _fineweb_row(7),
        row_group=3,
        row_index=7,
    )

    assert document.source_document_id == "doc-7"
    assert document.source_document_url == "https://example.edu/7"
    assert document.snapshot == FINEWEB
    assert document.record_id.startswith("fineweb-edu:")
    assert document.family_id == document.record_id
    provenance = document.provenance()
    assert "text" not in provenance
    assert provenance["source_revision"] == "9" * 40
    assert provenance["text_utf8_bytes"] == len(
        document.text.encode("utf-8")
    )

    with pytest.raises(PublicCorpusError, match="educational score"):
        document_from_row(
            FINEWEB,
            _fineweb_row(8, score=3),
            row_group=3,
            row_index=8,
        )
    missing_url = _fineweb_row(9)
    del missing_url["url"]
    with pytest.raises(PublicCorpusError, match="url"):
        document_from_row(
            FINEWEB,
            missing_url,
            row_group=3,
            row_index=9,
        )


def test_openwebmath_row_is_real_base_text_not_a_conversation() -> None:
    document = document_from_row(
        OPENWEBMATH,
        _openwebmath_row(5),
        row_group=1,
        row_index=5,
    )

    record = document.to_corpus_record()
    assert record.kind == "base"
    assert record.text == document.text
    assert record.turns == ()
    assert record.source_id == "openwebmath"


def test_selection_is_order_independent_and_obeys_byte_budget() -> None:
    documents = [
        document_from_row(
            FINEWEB,
            _fineweb_row(index),
            row_group=0,
            row_index=index,
        )
        for index in range(20)
    ]
    one_size = len(documents[0].text.encode("utf-8"))
    budget = one_size * 5 + 20

    first = select_public_documents(
        documents,
        seed="selection-test",
        max_documents=8,
        max_utf8_bytes=budget,
    )
    second = select_public_documents(
        reversed(documents),
        seed="selection-test",
        max_documents=8,
        max_utf8_bytes=budget,
    )

    assert [item.record_id for item in first] == [
        item.record_id for item in second
    ]
    assert sum(len(item.text.encode("utf-8")) for item in first) <= budget
    assert 1 <= len(first) <= 5


def test_text_normalization_rejects_controls_and_short_documents() -> None:
    short = _fineweb_row(1)
    short["text"] = "too short"
    with pytest.raises(PublicCorpusError, match="at least"):
        document_from_row(
            FINEWEB,
            short,
            row_group=0,
            row_index=1,
        )

    controlled = _fineweb_row(2)
    controlled["text"] = str(controlled["text"]) + "\x00"
    with pytest.raises(PublicCorpusError, match="control"):
        document_from_row(
            FINEWEB,
            controlled,
            row_group=0,
            row_index=2,
        )
