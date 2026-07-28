import hashlib
from pathlib import Path
import struct

import pytest

from nanogpt_nspire.base_corpus import (
    CorpusError,
    CorpusRecord,
    build_corpus,
    encode_corpus_record,
    normalized_content_fingerprint,
    stable_family_split,
)
from nanogpt_nspire.byte_tokenizer import (
    ASSISTANT_ID,
    BOS_ID,
    EOS_ID,
    USER_ID,
    ConversationTurn,
)


REGISTRY_PATH = (
    Path(__file__).parents[2] / "experiments" / "lesson10-public-sources.json"
)


def _base_record(record_id: str, family_id: str, text: str) -> CorpusRecord:
    return CorpusRecord.base(
        record_id=record_id,
        family_id=family_id,
        text=text,
        source_id="project-arithmetic-v1",
        license_id="MIT",
    )


def _family_for_split(split: str, split_seed: str) -> str:
    for index in range(100_000):
        family_id = f"family-{split}-{index}"
        if stable_family_split(family_id, split_seed=split_seed) == split:
            return family_id
    raise AssertionError(f"could not find a family assigned to {split}")


def test_base_and_conversation_records_encode_with_explicit_loss_masks():
    base = _base_record("base-1", "family-base", "A")
    conversation = CorpusRecord.conversation(
        record_id="chat-1",
        family_id="family-chat",
        turns=(
            ConversationTurn("user", "Q"),
            ConversationTurn("assistant", "A"),
        ),
        source_id="project-arithmetic-v1",
        license_id="MIT",
    )

    base_tokens, base_mask = encode_corpus_record(base)
    chat_tokens, chat_mask = encode_corpus_record(conversation)

    assert base_tokens == (BOS_ID, ord("A"), EOS_ID)
    assert base_mask == (0, 1, 1)
    assert chat_tokens == (
        BOS_ID,
        USER_ID,
        ord("Q"),
        ASSISTANT_ID,
        ord("A"),
        EOS_ID,
    )
    assert chat_mask == (0, 0, 0, 0, 1, 1)


def test_family_split_is_stable_and_variants_cannot_cross_splits():
    split_seed = "lesson10-test"
    family_id = "arith-fixed-family"

    first = stable_family_split(family_id, split_seed=split_seed)
    second = stable_family_split(family_id, split_seed=split_seed)

    assert first == second
    assert first in {"train", "validation", "test"}
    assert stable_family_split(
        family_id + "-different",
        split_seed=split_seed,
    ) in {"train", "validation", "test"}


def test_normalized_duplicates_with_conflicting_families_fail_before_output(
    tmp_path,
):
    records = (
        _base_record("one", "family-one", "Same\r\ntext  "),
        _base_record("two", "family-two", "Same\ntext"),
    )
    output = tmp_path / "corpus"

    with pytest.raises(CorpusError, match="duplicate content.*families"):
        build_corpus(
            records,
            output,
            registry_path=REGISTRY_PATH,
            split_seed="test",
            require_all_splits=False,
        )

    assert not output.exists()


def test_normalized_fingerprint_is_stable_across_line_endings():
    first = _base_record("one", "family-one", "A\r\nB  ")
    second = _base_record("two", "family-one", "A\nB")

    assert normalized_content_fingerprint(first) == (
        normalized_content_fingerprint(second)
    )


def test_build_is_input_order_independent_and_byte_identical(tmp_path):
    split_seed = "lesson10-build"
    records = []
    for split in ("train", "validation", "test"):
        family_id = _family_for_split(split, split_seed)
        records.append(
            _base_record(
                f"record-{split}",
                family_id,
                f"English {split} example.",
            )
        )
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    first_manifest = build_corpus(
        records,
        first_output,
        registry_path=REGISTRY_PATH,
        split_seed=split_seed,
    )
    second_manifest = build_corpus(
        reversed(records),
        second_output,
        registry_path=REGISTRY_PATH,
        split_seed=split_seed,
    )

    assert first_manifest == second_manifest
    assert first_manifest["schema_version"] == 1
    assert first_manifest["tokenizer"]["vocab_size"] == 264
    assert first_manifest["split"]["kind"] == "sha256-family-90-5-5"
    assert first_manifest["records"] == {
        "deduplicated": 0,
        "test": 1,
        "total": 3,
        "train": 1,
        "validation": 1,
    }
    for filename in (
        "train.tokens.bin",
        "train.loss.bin",
        "validation.tokens.bin",
        "validation.loss.bin",
        "test.tokens.bin",
        "test.loss.bin",
        "manifest.json",
    ):
        first_bytes = (first_output / filename).read_bytes()
        second_bytes = (second_output / filename).read_bytes()
        assert first_bytes == second_bytes
        if filename != "manifest.json":
            assert first_manifest["files"][filename]["sha256"] == hashlib.sha256(
                first_bytes
            ).hexdigest()


def test_binary_token_stream_is_explicit_little_endian(tmp_path):
    split_seed = "single-split"
    family = _family_for_split("train", split_seed)
    record = _base_record("record", family, "A")
    output = tmp_path / "corpus"

    build_corpus(
        (record,),
        output,
        registry_path=REGISTRY_PATH,
        split_seed=split_seed,
        require_all_splits=False,
    )

    assert (output / "train.tokens.bin").read_bytes() == struct.pack(
        "<3H",
        BOS_ID,
        ord("A"),
        EOS_ID,
    )
    assert (output / "train.loss.bin").read_bytes() == b"\x00\x01\x01"


def test_build_rejects_unknown_source_and_disallowed_license(tmp_path):
    unknown = CorpusRecord.base(
        record_id="unknown",
        family_id="family-unknown",
        text="Text",
        source_id="does-not-exist",
        license_id="MIT",
    )
    with pytest.raises(CorpusError, match="unknown source"):
        build_corpus(
            (unknown,),
            tmp_path / "unknown",
            registry_path=REGISTRY_PATH,
            split_seed="test",
            require_all_splits=False,
        )

    disallowed = CorpusRecord.base(
        record_id="disallowed",
        family_id="family-disallowed",
        text="Text",
        source_id="sciq",
        license_id="CC-BY-NC-3.0",
    )
    with pytest.raises(CorpusError, match="excluded"):
        build_corpus(
            (disallowed,),
            tmp_path / "disallowed",
            registry_path=REGISTRY_PATH,
            split_seed="test",
            require_all_splits=False,
        )


@pytest.mark.parametrize(
    "factory, message",
    [
        (
            lambda: CorpusRecord.base(
                record_id="",
                family_id="family",
                text="Text",
                source_id="project-arithmetic-v1",
                license_id="MIT",
            ),
            "record_id",
        ),
        (
            lambda: CorpusRecord.base(
                record_id="record",
                family_id="family",
                text="",
                source_id="project-arithmetic-v1",
                license_id="MIT",
            ),
            "text",
        ),
        (
            lambda: CorpusRecord.base(
                record_id="record",
                family_id="family",
                text="\ud800",
                source_id="project-arithmetic-v1",
                license_id="MIT",
            ),
            "UTF-8",
        ),
    ],
)
def test_record_schema_fails_closed(factory, message):
    with pytest.raises(CorpusError, match=message):
        factory()


def test_build_rejects_existing_destination(tmp_path):
    output = tmp_path / "corpus"
    output.mkdir()
    record = _base_record("record", "family", "Text")

    with pytest.raises(CorpusError, match="already exists"):
        build_corpus(
            (record,),
            output,
            registry_path=REGISTRY_PATH,
            split_seed="test",
            require_all_splits=False,
        )
