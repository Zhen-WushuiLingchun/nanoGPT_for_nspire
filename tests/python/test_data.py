import hashlib
import io
import json
import struct

import pytest

from nanogpt_nspire import data as data_module
from nanogpt_nspire.data import (
    DatasetError,
    build_vocabulary,
    decode_tokens,
    encode_text,
    fetch_tiny_shakespeare,
    pack_u16_le,
    prepare_dataset,
    split_tokens,
)


def test_vocabulary_is_sorted_and_round_trips():
    text = "cab\n"

    vocabulary = build_vocabulary(text)

    assert vocabulary == ("\n", "a", "b", "c")
    assert decode_tokens(encode_text(text, vocabulary), vocabulary) == text


def test_tokenizer_rejects_unknown_characters_and_token_ids():
    vocabulary = ("\n", "a", "b")

    with pytest.raises(DatasetError, match="not in the vocabulary"):
        encode_text("abc", vocabulary)

    with pytest.raises(DatasetError, match="outside vocabulary"):
        decode_tokens([0, 3], vocabulary)


def test_tokenizer_rejects_empty_or_duplicate_vocabulary():
    with pytest.raises(DatasetError, match="source text is empty"):
        build_vocabulary("")

    with pytest.raises(DatasetError, match="duplicate"):
        encode_text("a", ("a", "a"))


def test_split_uses_floor_boundary_and_rejects_unusable_splits():
    train, validation = split_tokens([0, 1, 2, 3, 4], train_fraction=0.8)

    assert train == [0, 1, 2, 3]
    assert validation == [4]

    with pytest.raises(DatasetError, match="strictly between"):
        split_tokens([0, 1], train_fraction=1.0)

    with pytest.raises(DatasetError, match="at least one token"):
        split_tokens([0], train_fraction=0.9)


def test_pack_u16_is_little_endian_and_range_checked():
    assert pack_u16_le([1, 0x0203]) == b"\x01\x00\x03\x02"

    with pytest.raises(DatasetError, match="uint16"):
        pack_u16_le([65536])


def test_prepare_dataset_writes_deterministic_artifacts(tmp_path):
    source_path = tmp_path / "tiny.txt"
    source_bytes = "ab\nab".encode("utf-8")
    source_path.write_bytes(source_bytes)
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    first_manifest = prepare_dataset(source_path, first_output)
    second_manifest = prepare_dataset(source_path, second_output)

    assert first_manifest == second_manifest
    assert first_manifest["source"]["sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert first_manifest["source"]["bytes"] == len(source_bytes)
    assert first_manifest["vocabulary"] == ["\n", "a", "b"]
    assert first_manifest["tokens"] == {
        "total": 5,
        "train": 4,
        "validation": 1,
    }
    assert first_manifest["dtype"] == "uint16-le"

    expected_train = struct.pack("<4H", 1, 2, 0, 1)
    expected_validation = struct.pack("<1H", 2)
    assert (first_output / "train.bin").read_bytes() == expected_train
    assert (first_output / "val.bin").read_bytes() == expected_validation

    for filename in ("train.bin", "val.bin", "manifest.json"):
        assert (first_output / filename).read_bytes() == (
            second_output / filename
        ).read_bytes()

    on_disk_manifest = json.loads(
        (first_output / "manifest.json").read_text(encoding="utf-8")
    )
    assert on_disk_manifest == first_manifest
    assert on_disk_manifest["files"]["train.bin"]["sha256"] == hashlib.sha256(
        expected_train
    ).hexdigest()
    assert on_disk_manifest["files"]["val.bin"]["sha256"] == hashlib.sha256(
        expected_validation
    ).hexdigest()


def test_prepare_rejects_invalid_utf8_without_creating_output(tmp_path):
    source_path = tmp_path / "invalid.txt"
    source_path.write_bytes(b"\xff")
    output_path = tmp_path / "output"

    with pytest.raises(DatasetError, match="not valid UTF-8"):
        prepare_dataset(source_path, output_path)

    assert not output_path.exists()


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def test_fetch_writes_payload_after_hash_validation(monkeypatch, tmp_path):
    payload = b"known source"
    monkeypatch.setattr(
        data_module.urllib.request,
        "urlopen",
        lambda url, timeout: _FakeResponse(payload),
    )
    output_path = tmp_path / "source.txt"

    result = fetch_tiny_shakespeare(
        output_path,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )

    assert output_path.read_bytes() == payload
    assert result["bytes"] == len(payload)
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()


def test_fetch_rejects_hash_mismatch(monkeypatch, tmp_path):
    monkeypatch.setattr(
        data_module.urllib.request,
        "urlopen",
        lambda url, timeout: _FakeResponse(b"unexpected"),
    )
    output_path = tmp_path / "source.txt"

    with pytest.raises(DatasetError, match="SHA-256 mismatch"):
        fetch_tiny_shakespeare(
            output_path,
            expected_sha256="0" * 64,
        )

    assert not output_path.exists()
