from dataclasses import replace
import struct
import zlib

import pytest

from nanogpt_nspire.export_format import (
    ACTIVATION_DYNAMIC_INT8_GROUPWISE,
    ENDIAN_MARKER,
    FILE_HEADER_BYTES,
    FILE_MAGIC,
    FORMAT_VERSION,
    MODEL_STORAGE_FP32,
    MODEL_STORAGE_W4A8,
    POSITION_ALIBI,
    STORAGE_FP32,
    STORAGE_INT4_GROUPWISE,
    TENSOR_ENTRY_BYTES,
    TOKENIZER_BYTE_SPECIAL,
    ModelFormatError,
    ModelSpec,
    TensorPayload,
    build_model_file,
    parse_model_file,
)


def _spec(**overrides) -> ModelSpec:
    values = {
        "vocab_size": 3,
        "block_size": 4,
        "n_layer": 1,
        "n_head": 2,
        "n_embd": 4,
        "mlp_ratio": 2,
        "tie_embeddings": True,
        "bias": False,
        "model_storage": MODEL_STORAGE_W4A8,
        "weight_group_size": 4,
        "activation_quantization": ACTIVATION_DYNAMIC_INT8_GROUPWISE,
        "activation_group_size": 4,
    }
    values.update(overrides)
    return ModelSpec(**values)


def _tensors() -> list[TensorPayload]:
    return [
        TensorPayload(
            tensor_id=1,
            storage=STORAGE_FP32,
            shape=(2, 2),
            data=struct.pack("<4f", 1.0, -2.0, 3.5, 4.0),
        ),
        TensorPayload(
            tensor_id=2,
            storage=STORAGE_INT4_GROUPWISE,
            shape=(2, 3),
            group_size=4,
            padded_last_dim=4,
            data=bytes((0x21, 0xF7, 0x80, 0x34)),
            auxiliary=struct.pack("<2f", 0.25, 0.5),
        ),
    ]


def _valid_file() -> bytes:
    return build_model_file(
        spec=_spec(),
        vocabulary=("\n", "a", "é"),
        tensors=_tensors(),
    )


def _repair_checksums(data: bytearray) -> None:
    header = list(struct.unpack("<8s30I", data[:FILE_HEADER_BYTES]))
    header_crc_index = 7
    payload_crc_index = 6
    header[payload_crc_index] = zlib.crc32(data[FILE_HEADER_BYTES:])
    header[header_crc_index] = 0
    packed = struct.pack("<8s30I", *header)
    header[header_crc_index] = zlib.crc32(packed)
    data[:FILE_HEADER_BYTES] = struct.pack("<8s30I", *header)


def _set_tensor_field(
    data: bytearray,
    *,
    tensor_index: int,
    field_index: int,
    value: int,
) -> None:
    offset = (
        FILE_HEADER_BYTES
        + tensor_index * TENSOR_ENTRY_BYTES
        + field_index * 4
    )
    struct.pack_into("<I", data, offset, value)
    _repair_checksums(data)


def test_format_layout_sizes_and_markers_are_frozen() -> None:
    assert FILE_MAGIC == b"NGNSP001"
    assert FORMAT_VERSION == 2
    assert ENDIAN_MARKER == 0x01020304
    assert FILE_HEADER_BYTES == 128
    assert TENSOR_ENTRY_BYTES == 64
    assert struct.calcsize("<8s30I") == FILE_HEADER_BYTES
    assert struct.calcsize("<16I") == TENSOR_ENTRY_BYTES


def test_build_and_parse_mixed_storage_file() -> None:
    data = _valid_file()
    parsed = parse_model_file(data)

    assert len(data) % 64 == 0
    assert parsed.spec == _spec()
    assert parsed.vocabulary == ("\n", "a", "é")
    assert tuple(parsed.tensors) == (1, 2)


def test_format_v2_byte_special_gqa_metadata_round_trips() -> None:
    spec = _spec(
        vocab_size=264,
        block_size=512,
        n_head=6,
        n_embd=384,
        n_kv_head=2,
        position_mode=POSITION_ALIBI,
        tokenizer_type=TOKENIZER_BYTE_SPECIAL,
    )
    data = build_model_file(
        spec=spec,
        vocabulary=(),
        tensors=_tensors(),
    )
    parsed = parse_model_file(data)

    assert parsed.spec == spec
    assert parsed.vocabulary[65] == "A"
    assert parsed.vocabulary[256] == "<BOS>"
    assert parsed.vocabulary[261] == "<THINK>"
    assert parsed.vocabulary[262] == "<FINAL>"
    fp32 = parsed.tensors[1]
    assert fp32.storage == STORAGE_FP32
    assert fp32.shape == (2, 2)
    assert bytes(fp32.data) == _tensors()[0].data
    assert not fp32.auxiliary
    int4 = parsed.tensors[2]
    assert int4.storage == STORAGE_INT4_GROUPWISE
    assert int4.shape == (2, 3)
    assert int4.group_size == 4
    assert int4.padded_last_dim == 4
    assert bytes(int4.data) == _tensors()[1].data
    assert bytes(int4.auxiliary) == _tensors()[1].auxiliary
    assert fp32.data_offset % 64 == 0
    assert int4.data_offset % 64 == 0
    assert int4.auxiliary_offset % 64 == 0


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda data: data.__setitem__(0, ord("X")), "magic"),
        (lambda data: data.__setitem__(8, 3), "version"),
        (lambda data: data.__setitem__(16, 5), "endian"),
    ],
)
def test_parser_rejects_bad_header_fields(mutation, message) -> None:
    data = bytearray(_valid_file())
    mutation(data)

    with pytest.raises(ModelFormatError, match=message):
        parse_model_file(data)


def test_parser_rejects_header_and_payload_corruption() -> None:
    bad_header = bytearray(_valid_file())
    bad_header[40] ^= 0x01
    with pytest.raises(ModelFormatError, match="header CRC32"):
        parse_model_file(bad_header)

    bad_payload = bytearray(_valid_file())
    bad_payload[-1] ^= 0x01
    with pytest.raises(ModelFormatError, match="payload CRC32"):
        parse_model_file(bad_payload)


def test_parser_rejects_truncation_and_declared_size_mismatch() -> None:
    data = _valid_file()
    with pytest.raises(ModelFormatError, match="file size"):
        parse_model_file(data[:-1])

    mismatched = bytearray(data)
    struct.pack_into("<I", mismatched, 24, len(data) + 64)
    _repair_checksums(mismatched)
    with pytest.raises(ModelFormatError, match="file size"):
        parse_model_file(mismatched)


def test_parser_rejects_tensor_offset_outside_data_region() -> None:
    data = bytearray(_valid_file())
    tensor_table_offset = FILE_HEADER_BYTES
    data_offset_field = tensor_table_offset + 10 * 4
    struct.pack_into("<I", data, data_offset_field, len(data) + 64)
    _repair_checksums(data)

    with pytest.raises(ModelFormatError, match="tensor data"):
        parse_model_file(data)


def test_parser_rejects_duplicate_tensor_id_and_overlapping_payloads() -> None:
    duplicate = bytearray(_valid_file())
    _set_tensor_field(
        duplicate,
        tensor_index=1,
        field_index=0,
        value=1,
    )
    with pytest.raises(ModelFormatError, match="tensor IDs"):
        parse_model_file(duplicate)

    overlapping = bytearray(_valid_file())
    first_data_offset = struct.unpack_from(
        "<I",
        overlapping,
        FILE_HEADER_BYTES + 10 * 4,
    )[0]
    _set_tensor_field(
        overlapping,
        tensor_index=1,
        field_index=10,
        value=first_data_offset,
    )
    with pytest.raises(ModelFormatError, match="overlaps"):
        parse_model_file(overlapping)


def test_parser_rejects_uint32_length_that_exceeds_file() -> None:
    data = bytearray(_valid_file())
    _set_tensor_field(
        data,
        tensor_index=0,
        field_index=11,
        value=0xFFFFFFFF,
    )

    with pytest.raises(ModelFormatError, match="tensor data"):
        parse_model_file(data)


def test_parser_rejects_invalid_utf8_vocabulary_after_valid_crc() -> None:
    data = bytearray(_valid_file())
    tensor_count = 2
    vocabulary_offset = FILE_HEADER_BYTES + tensor_count * TENSOR_ENTRY_BYTES
    first_token_byte = vocabulary_offset + 2
    data[first_token_byte] = 0xFF
    _repair_checksums(data)

    with pytest.raises(ModelFormatError, match="UTF-8"):
        parse_model_file(data)


def test_builder_rejects_duplicate_tensor_ids_and_bad_payload_sizes() -> None:
    tensors = _tensors()
    with pytest.raises(ModelFormatError, match="duplicate tensor"):
        build_model_file(
            spec=_spec(),
            vocabulary=("\n", "a", "é"),
            tensors=(tensors[0], tensors[0]),
        )

    bad_fp32 = replace(tensors[0], data=b"\0" * 12)
    with pytest.raises(ModelFormatError, match="FP32 byte count"):
        build_model_file(
            spec=_spec(),
            vocabulary=("\n", "a", "é"),
            tensors=(bad_fp32,),
        )

    bad_int4 = replace(tensors[1], auxiliary=b"\0" * 4)
    with pytest.raises(ModelFormatError, match="scale byte count"):
        build_model_file(
            spec=_spec(),
            vocabulary=("\n", "a", "é"),
            tensors=(bad_int4,),
        )


@pytest.mark.parametrize(
    "spec, vocabulary, message",
    [
        (_spec(vocab_size=2), ("\n", "a", "é"), "vocabulary count"),
        (_spec(), ("\n", "", "é"), "empty"),
        (
            _spec(
                model_storage=MODEL_STORAGE_FP32,
                weight_group_size=4,
                activation_quantization=ACTIVATION_DYNAMIC_INT8_GROUPWISE,
                activation_group_size=4,
            ),
            ("\n", "a", "é"),
            "FP32 model",
        ),
    ],
)
def test_builder_rejects_inconsistent_model_metadata(
    spec,
    vocabulary,
    message,
) -> None:
    with pytest.raises(ModelFormatError, match=message):
        build_model_file(
            spec=spec,
            vocabulary=vocabulary,
            tensors=_tensors(),
        )
