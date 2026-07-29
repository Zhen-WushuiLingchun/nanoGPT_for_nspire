"""Portable little-endian model container shared by Python and C."""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Iterable, Sequence
import zlib


FILE_MAGIC = b"NGNSP001"
LEGACY_FORMAT_VERSION = 1
FORMAT_VERSION = 2
ENDIAN_MARKER = 0x01020304
FILE_HEADER_BYTES = 128
TENSOR_ENTRY_BYTES = 64
ALIGNMENT_BYTES = 64
FILE_LIMIT_BYTES = 6 * 1024 * 1024

MODEL_STORAGE_FP32 = 1
MODEL_STORAGE_W4A8 = 2

STORAGE_FP32 = 1
STORAGE_INT4_GROUPWISE = 2

ACTIVATION_NONE = 0
ACTIVATION_DYNAMIC_INT8_GROUPWISE = 1

TOKENIZER_CHARACTER_UTF8 = 1
TOKENIZER_BYTE_SPECIAL = 2

POSITION_LEARNED = 1
POSITION_ALIBI = 2

BYTE_SPECIAL_VOCAB_SIZE = 264
BYTE_SPECIAL_NAMES = (
    "<BOS>",
    "<EOS>",
    "<USER>",
    "<ASSISTANT>",
    "<TOOL>",
    "<THINK>",
    "<FINAL>",
    "<PAD>",
)

FLAG_LITTLE_ENDIAN = 1 << 0
FLAG_TIED_EMBEDDING = 1 << 1
FLAG_BIAS = 1 << 2
FLAG_TANH_GELU = 1 << 3

_HEADER_STRUCT = struct.Struct("<8s30I")
_TENSOR_STRUCT = struct.Struct("<16I")
_U16_STRUCT = struct.Struct("<H")
_U32_STRUCT = struct.Struct("<I")
_HEADER_CRC_OFFSET = 32


class ModelFormatError(ValueError):
    """The model cannot be represented or safely parsed as format v1."""


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelFormatError(f"{name} must be a positive integer")
    if value > 0xFFFFFFFF:
        raise ModelFormatError(f"{name} exceeds uint32")
    return value


def _checked_product(values: Iterable[int], name: str) -> int:
    result = 1
    for value in values:
        _positive_int(value, f"{name} dimension")
        result *= value
        if result > 0xFFFFFFFF:
            raise ModelFormatError(f"{name} element count exceeds uint32")
    return result


def _align(value: int, alignment: int = ALIGNMENT_BYTES) -> int:
    if value < 0:
        raise ModelFormatError("offset must be non-negative")
    return (value + alignment - 1) // alignment * alignment


def _as_bytes(value: bytes | bytearray | memoryview, name: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise ModelFormatError(f"{name} must be bytes-like")
    return bytes(value)


@dataclass(frozen=True)
class ModelSpec:
    """Architecture and numeric route encoded in the fixed header."""

    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    mlp_ratio: int
    tie_embeddings: bool
    bias: bool
    model_storage: int
    weight_group_size: int = 0
    activation_quantization: int = ACTIVATION_NONE
    activation_group_size: int = 0
    n_kv_head: int = 0
    position_mode: int = POSITION_LEARNED
    tokenizer_type: int = TOKENIZER_CHARACTER_UTF8

    def __post_init__(self) -> None:
        if self.n_kv_head == 0:
            object.__setattr__(self, "n_kv_head", self.n_head)

    @property
    def effective_n_kv_head(self) -> int:
        """Return the stored KV-head count, defaulting legacy specs to MHA."""

        return self.n_kv_head

    def validate(self) -> None:
        for name in (
            "vocab_size",
            "block_size",
            "n_layer",
            "n_head",
            "n_embd",
            "mlp_ratio",
        ):
            _positive_int(getattr(self, name), name)
        if self.n_embd % self.n_head:
            raise ModelFormatError("n_embd must be divisible by n_head")
        n_kv_head = self.effective_n_kv_head
        _positive_int(n_kv_head, "n_kv_head")
        if self.n_head % n_kv_head:
            raise ModelFormatError("n_head must be divisible by n_kv_head")
        if self.position_mode not in {POSITION_LEARNED, POSITION_ALIBI}:
            raise ModelFormatError("position_mode is unsupported")
        if self.tokenizer_type not in {
            TOKENIZER_CHARACTER_UTF8,
            TOKENIZER_BYTE_SPECIAL,
        }:
            raise ModelFormatError("tokenizer_type is unsupported")
        if (
            self.tokenizer_type == TOKENIZER_BYTE_SPECIAL
            and self.vocab_size != BYTE_SPECIAL_VOCAB_SIZE
        ):
            raise ModelFormatError(
                "byte-special tokenizer requires the frozen 264-token vocabulary"
            )
        if not isinstance(self.tie_embeddings, bool):
            raise ModelFormatError("tie_embeddings must be boolean")
        if not isinstance(self.bias, bool):
            raise ModelFormatError("bias must be boolean")
        if self.model_storage == MODEL_STORAGE_FP32:
            if (
                self.weight_group_size != 0
                or self.activation_quantization != ACTIVATION_NONE
                or self.activation_group_size != 0
            ):
                raise ModelFormatError(
                    "FP32 model must not declare quantization groups"
                )
        elif self.model_storage == MODEL_STORAGE_W4A8:
            _positive_int(self.weight_group_size, "weight_group_size")
            _positive_int(
                self.activation_group_size,
                "activation_group_size",
            )
            if self.weight_group_size % 2:
                raise ModelFormatError("weight_group_size must be even")
            if (
                self.activation_quantization
                != ACTIVATION_DYNAMIC_INT8_GROUPWISE
            ):
                raise ModelFormatError(
                    "W4A8 model requires dynamic groupwise INT8 activation"
                )
            if self.activation_group_size != self.weight_group_size:
                raise ModelFormatError(
                    "W4A8 activation and weight group sizes must match"
                )
        else:
            raise ModelFormatError(
                f"unknown model_storage {self.model_storage!r}"
            )


@dataclass(frozen=True)
class TensorPayload:
    """One canonical tensor before its offsets are assigned."""

    tensor_id: int
    storage: int
    shape: tuple[int, ...]
    data: bytes | bytearray | memoryview
    auxiliary: bytes | bytearray | memoryview = b""
    group_size: int = 0
    padded_last_dim: int = 0


@dataclass(frozen=True)
class TensorView:
    """One validated zero-copy view into an immutable model blob."""

    tensor_id: int
    storage: int
    shape: tuple[int, ...]
    group_size: int
    padded_last_dim: int
    data_offset: int
    auxiliary_offset: int
    data: memoryview
    auxiliary: memoryview

    @property
    def element_count(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True)
class ParsedModel:
    """A fully validated model file and its tensor views."""

    spec: ModelSpec
    vocabulary: tuple[str, ...]
    tensors: dict[int, TensorView]
    blob: bytes
    payload_crc32: int
    header_crc32: int
    data_offset: int
    data_bytes: int


@dataclass(frozen=True)
class _TensorLayout:
    payload: TensorPayload
    shape: tuple[int, ...]
    data: bytes
    auxiliary: bytes
    element_count: int
    data_offset: int
    auxiliary_offset: int


def _validate_tensor_payload(
    tensor: TensorPayload,
) -> tuple[tuple[int, ...], bytes, bytes, int]:
    _positive_int(tensor.tensor_id, "tensor_id")
    if (
        not isinstance(tensor.shape, tuple)
        or not 1 <= len(tensor.shape) <= 4
    ):
        raise ModelFormatError("tensor shape must be a tuple of rank 1 to 4")
    element_count = _checked_product(tensor.shape, "tensor")
    data = _as_bytes(tensor.data, "tensor data")
    auxiliary = _as_bytes(tensor.auxiliary, "tensor auxiliary")

    if tensor.storage == STORAGE_FP32:
        if len(data) != element_count * 4:
            raise ModelFormatError("FP32 byte count does not match tensor shape")
        if auxiliary:
            raise ModelFormatError("FP32 tensor must not have auxiliary bytes")
        if tensor.group_size != 0 or tensor.padded_last_dim != 0:
            raise ModelFormatError("FP32 tensor must not have group metadata")
    elif tensor.storage == STORAGE_INT4_GROUPWISE:
        group_size = _positive_int(tensor.group_size, "tensor group_size")
        if group_size % 2:
            raise ModelFormatError("INT4 tensor group_size must be even")
        expected_padded = (
            math.ceil(tensor.shape[-1] / group_size) * group_size
        )
        if tensor.padded_last_dim != expected_padded:
            raise ModelFormatError(
                "INT4 padded_last_dim does not match shape and group_size"
            )
        rows = math.prod(tensor.shape[:-1]) if len(tensor.shape) > 1 else 1
        value_count = rows * tensor.padded_last_dim
        if len(data) != (value_count + 1) // 2:
            raise ModelFormatError(
                "INT4 packed byte count does not match tensor shape"
            )
        scale_count = rows * tensor.padded_last_dim // group_size
        if len(auxiliary) != scale_count * 4:
            raise ModelFormatError(
                "INT4 scale byte count does not match tensor shape"
            )
    else:
        raise ModelFormatError(f"unknown tensor storage {tensor.storage!r}")
    return tensor.shape, data, auxiliary, element_count


def _byte_special_vocabulary() -> tuple[str, ...]:
    return tuple(chr(index) for index in range(256)) + BYTE_SPECIAL_NAMES


def _encode_vocabulary(
    vocabulary: Sequence[str],
    *,
    tokenizer_type: int,
) -> bytes:
    if tokenizer_type == TOKENIZER_BYTE_SPECIAL:
        if len(vocabulary) not in {0, BYTE_SPECIAL_VOCAB_SIZE}:
            raise ModelFormatError(
                "byte-special vocabulary must be empty or canonical"
            )
        if (
            len(vocabulary) == BYTE_SPECIAL_VOCAB_SIZE
            and tuple(vocabulary) != _byte_special_vocabulary()
        ):
            raise ModelFormatError(
                "byte-special vocabulary does not match the frozen protocol"
            )
        return b""
    output = bytearray()
    for index, token in enumerate(vocabulary):
        if not isinstance(token, str):
            raise ModelFormatError(f"vocabulary token {index} is not text")
        if not token:
            raise ModelFormatError(f"vocabulary token {index} is empty")
        if len(token) != 1:
            raise ModelFormatError(
                f"vocabulary token {index} is not one Unicode character"
            )
        encoded = token.encode("utf-8")
        if len(encoded) > 0xFFFF:
            raise ModelFormatError("UTF-8 vocabulary token is too long")
        output.extend(_U16_STRUCT.pack(len(encoded)))
        output.extend(encoded)
    return bytes(output)


def _flags_for_spec(spec: ModelSpec) -> int:
    flags = FLAG_LITTLE_ENDIAN | FLAG_TANH_GELU
    if spec.tie_embeddings:
        flags |= FLAG_TIED_EMBEDDING
    if spec.bias:
        flags |= FLAG_BIAS
    return flags


def build_model_file(
    *,
    spec: ModelSpec,
    vocabulary: Sequence[str],
    tensors: Sequence[TensorPayload],
) -> bytes:
    """Build one deterministic, aligned and checksum-protected model file."""

    if not isinstance(spec, ModelSpec):
        raise ModelFormatError("spec must be a ModelSpec")
    spec.validate()
    if not isinstance(vocabulary, Sequence):
        raise ModelFormatError("vocabulary must be a sequence")
    if (
        spec.tokenizer_type == TOKENIZER_CHARACTER_UTF8
        and len(vocabulary) != spec.vocab_size
    ):
        raise ModelFormatError("vocabulary count does not match vocab_size")
    vocabulary_bytes = _encode_vocabulary(
        vocabulary,
        tokenizer_type=spec.tokenizer_type,
    )
    if not isinstance(tensors, Sequence) or not tensors:
        raise ModelFormatError("tensors must be a non-empty sequence")

    validated: list[
        tuple[TensorPayload, tuple[int, ...], bytes, bytes, int]
    ] = []
    seen_ids: set[int] = set()
    for tensor in tensors:
        if not isinstance(tensor, TensorPayload):
            raise ModelFormatError("every tensor must be a TensorPayload")
        if tensor.tensor_id in seen_ids:
            raise ModelFormatError(
                f"duplicate tensor id {tensor.tensor_id}"
            )
        seen_ids.add(tensor.tensor_id)
        shape, data, auxiliary, element_count = _validate_tensor_payload(
            tensor
        )
        if (
            spec.model_storage == MODEL_STORAGE_FP32
            and tensor.storage != STORAGE_FP32
        ):
            raise ModelFormatError("FP32 model contains a non-FP32 tensor")
        if (
            tensor.storage == STORAGE_INT4_GROUPWISE
            and tensor.group_size != spec.weight_group_size
        ):
            raise ModelFormatError(
                "INT4 tensor group_size disagrees with model"
            )
        validated.append(
            (tensor, shape, data, auxiliary, element_count)
        )
    validated.sort(key=lambda item: item[0].tensor_id)

    tensor_count = len(validated)
    tensor_table_offset = FILE_HEADER_BYTES
    tensor_table_bytes = tensor_count * TENSOR_ENTRY_BYTES
    vocabulary_offset = tensor_table_offset + tensor_table_bytes
    data_offset = _align(vocabulary_offset + len(vocabulary_bytes))
    blob = bytearray(data_offset)
    blob[vocabulary_offset : vocabulary_offset + len(vocabulary_bytes)] = (
        vocabulary_bytes
    )

    layouts: list[_TensorLayout] = []
    for tensor, shape, data, auxiliary, element_count in validated:
        aligned = _align(len(blob))
        blob.extend(b"\0" * (aligned - len(blob)))
        tensor_data_offset = len(blob)
        blob.extend(data)
        auxiliary_offset = 0
        if auxiliary:
            aligned = _align(len(blob))
            blob.extend(b"\0" * (aligned - len(blob)))
            auxiliary_offset = len(blob)
            blob.extend(auxiliary)
        layouts.append(
            _TensorLayout(
                payload=tensor,
                shape=shape,
                data=data,
                auxiliary=auxiliary,
                element_count=element_count,
                data_offset=tensor_data_offset,
                auxiliary_offset=auxiliary_offset,
            )
        )
    final_size = _align(len(blob))
    blob.extend(b"\0" * (final_size - len(blob)))
    if len(blob) > FILE_LIMIT_BYTES:
        raise ModelFormatError(
            f"model file exceeds {FILE_LIMIT_BYTES} byte limit"
        )

    for index, layout in enumerate(layouts):
        dimensions = (*layout.shape, *(0 for _ in range(4 - len(layout.shape))))
        entry = _TENSOR_STRUCT.pack(
            layout.payload.tensor_id,
            layout.payload.storage,
            len(layout.shape),
            0,
            *dimensions,
            layout.payload.group_size,
            layout.payload.padded_last_dim,
            layout.data_offset,
            len(layout.data),
            layout.auxiliary_offset,
            len(layout.auxiliary),
            layout.element_count,
            0,
        )
        start = tensor_table_offset + index * TENSOR_ENTRY_BYTES
        blob[start : start + TENSOR_ENTRY_BYTES] = entry

    quantized_min = (
        (-7) & 0xFFFFFFFF
        if spec.model_storage == MODEL_STORAGE_W4A8
        else 0
    )
    quantized_max = 7 if spec.model_storage == MODEL_STORAGE_W4A8 else 0
    header_values: list[int | bytes] = [
        FILE_MAGIC,
        FORMAT_VERSION,
        FILE_HEADER_BYTES,
        ENDIAN_MARKER,
        _flags_for_spec(spec),
        len(blob),
        0,
        0,
        tensor_count,
        TENSOR_ENTRY_BYTES,
        tensor_table_offset,
        tensor_table_bytes,
        vocabulary_offset,
        len(vocabulary_bytes),
        data_offset,
        len(blob) - data_offset,
        spec.vocab_size,
        spec.block_size,
        spec.n_layer,
        spec.n_head,
        spec.n_embd,
        spec.mlp_ratio,
        spec.model_storage,
        spec.weight_group_size,
        quantized_min,
        quantized_max,
        spec.tokenizer_type,
        spec.activation_quantization,
        spec.activation_group_size,
        spec.effective_n_kv_head,
        spec.position_mode,
    ]
    payload_crc32 = zlib.crc32(blob[FILE_HEADER_BYTES:]) & 0xFFFFFFFF
    header_values[6] = payload_crc32
    header = _HEADER_STRUCT.pack(*header_values)
    header_crc32 = zlib.crc32(header) & 0xFFFFFFFF
    header_values[7] = header_crc32
    blob[:FILE_HEADER_BYTES] = _HEADER_STRUCT.pack(*header_values)
    return bytes(blob)


def _decode_signed_u32(value: int) -> int:
    return value if value < 0x80000000 else value - 0x100000000


def _checked_region(
    *,
    offset: int,
    length: int,
    lower: int,
    upper: int,
    name: str,
) -> tuple[int, int]:
    if offset < lower or length < 0 or offset > upper:
        raise ModelFormatError(f"{name} is outside its allowed region")
    end = offset + length
    if end < offset or end > upper:
        raise ModelFormatError(f"{name} is outside its allowed region")
    return offset, end


def _decode_vocabulary(
    blob: bytes,
    *,
    offset: int,
    length: int,
    count: int,
) -> tuple[str, ...]:
    cursor = offset
    end = offset + length
    tokens: list[str] = []
    for index in range(count):
        if cursor + _U16_STRUCT.size > end:
            raise ModelFormatError("vocabulary is truncated")
        token_bytes = _U16_STRUCT.unpack_from(blob, cursor)[0]
        cursor += _U16_STRUCT.size
        if token_bytes == 0 or cursor + token_bytes > end:
            raise ModelFormatError("vocabulary token length is invalid")
        try:
            token = blob[cursor : cursor + token_bytes].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ModelFormatError(
                f"vocabulary token {index} is not valid UTF-8"
            ) from error
        cursor += token_bytes
        if len(token) != 1:
            raise ModelFormatError(
                f"vocabulary token {index} is not one Unicode character"
            )
        tokens.append(token)
    if cursor != end:
        raise ModelFormatError("vocabulary byte count has trailing data")
    return tuple(tokens)


def parse_model_file(
    data: bytes | bytearray | memoryview,
) -> ParsedModel:
    """Validate the complete file before returning immutable tensor views."""

    blob = _as_bytes(data, "model file")
    if len(blob) < FILE_HEADER_BYTES:
        raise ModelFormatError("file size is smaller than the header")
    if len(blob) > FILE_LIMIT_BYTES:
        raise ModelFormatError("file size exceeds the deployment limit")
    unpacked = _HEADER_STRUCT.unpack_from(blob)
    magic = unpacked[0]
    fields = unpacked[1:]
    if magic != FILE_MAGIC:
        raise ModelFormatError("model magic does not match")
    version = fields[0]
    if version not in {LEGACY_FORMAT_VERSION, FORMAT_VERSION}:
        raise ModelFormatError("unsupported format version")
    if fields[1] != FILE_HEADER_BYTES:
        raise ModelFormatError("header size does not match format v1")
    if fields[2] != ENDIAN_MARKER:
        raise ModelFormatError("endian marker does not match little-endian")
    if fields[4] != len(blob):
        raise ModelFormatError("declared file size does not match actual file size")

    payload_crc32 = fields[5]
    header_crc32 = fields[6]
    header_for_crc = bytearray(blob[:FILE_HEADER_BYTES])
    _U32_STRUCT.pack_into(header_for_crc, _HEADER_CRC_OFFSET, 0)
    if (zlib.crc32(header_for_crc) & 0xFFFFFFFF) != header_crc32:
        raise ModelFormatError("header CRC32 mismatch")
    if (
        zlib.crc32(blob[FILE_HEADER_BYTES:]) & 0xFFFFFFFF
    ) != payload_crc32:
        raise ModelFormatError("payload CRC32 mismatch")

    flags = fields[3]
    known_flags = (
        FLAG_LITTLE_ENDIAN
        | FLAG_TIED_EMBEDDING
        | FLAG_BIAS
        | FLAG_TANH_GELU
    )
    if flags & ~known_flags or not flags & FLAG_LITTLE_ENDIAN:
        raise ModelFormatError("header flags are invalid")
    if not flags & FLAG_TANH_GELU:
        raise ModelFormatError("format v1 requires tanh GELU")

    (
        tensor_count,
        tensor_entry_bytes,
        tensor_table_offset,
        tensor_table_bytes,
        vocabulary_offset,
        vocabulary_bytes,
        data_offset,
        data_bytes,
        vocab_size,
        block_size,
        n_layer,
        n_head,
        n_embd,
        mlp_ratio,
        model_storage,
        weight_group_size,
        raw_quantized_min,
        quantized_max,
        tokenizer_type,
        activation_quantization,
        activation_group_size,
        raw_n_kv_head,
        raw_position_mode,
    ) = fields[7:]
    if version == LEGACY_FORMAT_VERSION:
        if raw_n_kv_head != 0 or raw_position_mode != 0:
            raise ModelFormatError(
                "legacy reserved header fields must be zero"
            )
        n_kv_head = n_head
        position_mode = POSITION_LEARNED
    else:
        n_kv_head = raw_n_kv_head
        position_mode = raw_position_mode
    if tensor_count == 0:
        raise ModelFormatError("tensor table must not be empty")
    if tensor_entry_bytes != TENSOR_ENTRY_BYTES:
        raise ModelFormatError("tensor entry size does not match format v1")
    if tensor_table_offset != FILE_HEADER_BYTES:
        raise ModelFormatError("tensor table offset is invalid")
    if tensor_table_bytes != tensor_count * TENSOR_ENTRY_BYTES:
        raise ModelFormatError("tensor table byte count is invalid")
    if vocabulary_offset != tensor_table_offset + tensor_table_bytes:
        raise ModelFormatError("vocabulary offset is invalid")
    if data_offset != _align(vocabulary_offset + vocabulary_bytes):
        raise ModelFormatError("data offset is invalid")
    if data_offset % ALIGNMENT_BYTES:
        raise ModelFormatError("data offset is not aligned")
    if data_offset + data_bytes != len(blob):
        raise ModelFormatError("data byte count is invalid")
    if tokenizer_type not in {
        TOKENIZER_CHARACTER_UTF8,
        TOKENIZER_BYTE_SPECIAL,
    }:
        raise ModelFormatError("unsupported tokenizer type")

    quantized_min = _decode_signed_u32(raw_quantized_min)
    if model_storage == MODEL_STORAGE_FP32:
        if quantized_min != 0 or quantized_max != 0:
            raise ModelFormatError("FP32 quantized range must be zero")
    elif model_storage == MODEL_STORAGE_W4A8:
        if quantized_min != -7 or quantized_max != 7:
            raise ModelFormatError("W4A8 quantized range must be [-7, 7]")
    spec = ModelSpec(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        mlp_ratio=mlp_ratio,
        tie_embeddings=bool(flags & FLAG_TIED_EMBEDDING),
        bias=bool(flags & FLAG_BIAS),
        model_storage=model_storage,
        weight_group_size=weight_group_size,
        activation_quantization=activation_quantization,
        activation_group_size=activation_group_size,
        n_kv_head=n_kv_head,
        position_mode=position_mode,
        tokenizer_type=tokenizer_type,
    )
    spec.validate()
    if tokenizer_type == TOKENIZER_BYTE_SPECIAL:
        if vocabulary_bytes != 0:
            raise ModelFormatError(
                "byte-special tokenizer must not store a vocabulary payload"
            )
        vocabulary = _byte_special_vocabulary()
    else:
        vocabulary = _decode_vocabulary(
            blob,
            offset=vocabulary_offset,
            length=vocabulary_bytes,
            count=vocab_size,
        )

    views: dict[int, TensorView] = {}
    occupied: list[tuple[int, int, str]] = []
    previous_tensor_id = 0
    for index in range(tensor_count):
        entry_offset = tensor_table_offset + index * TENSOR_ENTRY_BYTES
        values = _TENSOR_STRUCT.unpack_from(blob, entry_offset)
        (
            tensor_id,
            storage,
            rank,
            entry_flags,
            dim0,
            dim1,
            dim2,
            dim3,
            group_size,
            padded_last_dim,
            tensor_data_offset,
            tensor_data_bytes,
            auxiliary_offset,
            auxiliary_bytes,
            element_count,
            entry_reserved,
        ) = values
        if tensor_id <= previous_tensor_id:
            raise ModelFormatError(
                "tensor IDs must be unique and strictly increasing"
            )
        previous_tensor_id = tensor_id
        if entry_flags != 0 or entry_reserved != 0:
            raise ModelFormatError("reserved tensor fields must be zero")
        if not 1 <= rank <= 4:
            raise ModelFormatError("tensor rank must be in [1, 4]")
        raw_dimensions = (dim0, dim1, dim2, dim3)
        shape = raw_dimensions[:rank]
        if any(value <= 0 for value in shape) or any(
            value != 0 for value in raw_dimensions[rank:]
        ):
            raise ModelFormatError("tensor dimensions are invalid")
        if _checked_product(shape, "tensor") != element_count:
            raise ModelFormatError("tensor element count is invalid")
        data_start, data_end = _checked_region(
            offset=tensor_data_offset,
            length=tensor_data_bytes,
            lower=data_offset,
            upper=len(blob),
            name="tensor data",
        )
        if tensor_data_offset % ALIGNMENT_BYTES:
            raise ModelFormatError("tensor data offset is not aligned")
        auxiliary = b""
        auxiliary_start = 0
        auxiliary_end = 0
        if auxiliary_bytes:
            auxiliary_start, auxiliary_end = _checked_region(
                offset=auxiliary_offset,
                length=auxiliary_bytes,
                lower=data_offset,
                upper=len(blob),
                name="tensor auxiliary",
            )
            if auxiliary_offset % ALIGNMENT_BYTES:
                raise ModelFormatError(
                    "tensor auxiliary offset is not aligned"
                )
            auxiliary = blob[auxiliary_start:auxiliary_end]
        elif auxiliary_offset != 0:
            raise ModelFormatError(
                "zero-length tensor auxiliary must have zero offset"
            )
        payload = TensorPayload(
            tensor_id=tensor_id,
            storage=storage,
            shape=tuple(shape),
            data=blob[data_start:data_end],
            auxiliary=auxiliary,
            group_size=group_size,
            padded_last_dim=padded_last_dim,
        )
        _validate_tensor_payload(payload)
        if (
            storage == STORAGE_INT4_GROUPWISE
            and group_size != spec.weight_group_size
        ):
            raise ModelFormatError(
                "INT4 tensor group_size disagrees with model"
            )
        if (
            spec.model_storage == MODEL_STORAGE_FP32
            and storage != STORAGE_FP32
        ):
            raise ModelFormatError("FP32 model contains a non-FP32 tensor")
        occupied.append((data_start, data_end, "tensor data"))
        if auxiliary_bytes:
            occupied.append(
                (auxiliary_start, auxiliary_end, "tensor auxiliary")
            )
        views[tensor_id] = TensorView(
            tensor_id=tensor_id,
            storage=storage,
            shape=tuple(shape),
            group_size=group_size,
            padded_last_dim=padded_last_dim,
            data_offset=tensor_data_offset,
            auxiliary_offset=auxiliary_offset,
            data=memoryview(blob)[data_start:data_end],
            auxiliary=memoryview(blob)[auxiliary_start:auxiliary_end],
        )

    occupied.sort()
    for left, right in zip(occupied, occupied[1:]):
        if left[1] > right[0]:
            raise ModelFormatError(
                f"{left[2]} overlaps {right[2]}"
            )
    return ParsedModel(
        spec=spec,
        vocabulary=vocabulary,
        tensors=views,
        blob=blob,
        payload_crc32=payload_crc32,
        header_crc32=header_crc32,
        data_offset=data_offset,
        data_bytes=data_bytes,
    )
