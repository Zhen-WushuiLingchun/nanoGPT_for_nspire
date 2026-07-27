"""Validate frozen checkpoints and export portable ``.ngm`` model files."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from nanogpt_nspire.export_format import (
    ACTIVATION_DYNAMIC_INT8_GROUPWISE,
    ACTIVATION_NONE,
    MODEL_STORAGE_FP32,
    MODEL_STORAGE_W4A8,
    STORAGE_FP32,
    STORAGE_INT4_GROUPWISE,
    ModelFormatError,
    ModelSpec,
    TensorPayload,
    build_model_file,
    parse_model_file,
)
from nanogpt_nspire.models.direct_small_gpt import (
    DirectSmallConfig,
)
from nanogpt_nspire.quantization import GroupwiseInt4Tensor
from nanogpt_nspire.quantization.model_state import (
    SCHEMA_VERSION as QUANTIZED_STATE_SCHEMA_VERSION,
    SCHEME as QUANTIZED_STATE_SCHEME,
)
from nanogpt_nspire.training_support import (
    sha256_file,
    write_json_atomic,
)


TOKEN_EMBEDDING_TENSOR_ID = 1
POSITION_EMBEDDING_TENSOR_ID = 2
BLOCK_TENSOR_ID_BASE = 100
BLOCK_TENSOR_ID_STRIDE = 10
FINAL_NORM_TENSOR_ID = 1000

_SUPPORTED_FP32_ROUTES = {
    "Direct-Small",
    "Distilled-Small",
    "Distilled-Small-Extended",
}
_QUANTIZED_ROUTE = "Quantized-Small"


@dataclass(frozen=True)
class TensorDescriptor:
    """Stable ID, state-dict name and shape for one canonical GPT tensor."""

    tensor_id: int
    name: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class ExportedModel:
    """In-memory export plus its deterministic reader-facing manifest."""

    data: bytes
    route: str
    manifest: dict[str, object]


def expected_tensor_descriptors(
    config: DirectSmallConfig,
) -> tuple[TensorDescriptor, ...]:
    """Return the fixed tensor table, omitting the tied vocabulary alias."""

    if not isinstance(config, DirectSmallConfig):
        raise ModelFormatError("config must be a DirectSmallConfig")
    try:
        config.validate()
    except ValueError as error:
        raise ModelFormatError(str(error)) from error
    descriptors = [
        TensorDescriptor(
            tensor_id=TOKEN_EMBEDDING_TENSOR_ID,
            name="token_embedding.weight",
            shape=(config.vocab_size, config.n_embd),
        ),
        TensorDescriptor(
            tensor_id=POSITION_EMBEDDING_TENSOR_ID,
            name="position_embedding.weight",
            shape=(config.block_size, config.n_embd),
        ),
    ]
    block_shapes = (
        ("attention_norm.weight", (config.n_embd,)),
        ("attention.qkv.weight", (3 * config.n_embd, config.n_embd)),
        ("attention.output.weight", (config.n_embd, config.n_embd)),
        ("mlp_norm.weight", (config.n_embd,)),
        (
            "mlp.input.weight",
            (config.mlp_ratio * config.n_embd, config.n_embd),
        ),
        (
            "mlp.output.weight",
            (config.n_embd, config.mlp_ratio * config.n_embd),
        ),
    )
    for block_index in range(config.n_layer):
        for slot, (suffix, shape) in enumerate(block_shapes):
            descriptors.append(
                TensorDescriptor(
                    tensor_id=(
                        BLOCK_TENSOR_ID_BASE
                        + block_index * BLOCK_TENSOR_ID_STRIDE
                        + slot
                    ),
                    name=f"blocks.{block_index}.{suffix}",
                    shape=shape,
                )
            )
    descriptors.append(
        TensorDescriptor(
            tensor_id=FINAL_NORM_TENSOR_ID,
            name="final_norm.weight",
            shape=(config.n_embd,),
        )
    )
    return tuple(descriptors)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelFormatError(f"{name} must be a mapping")
    return value


def _checkpoint_config(checkpoint: Mapping[str, Any]) -> DirectSmallConfig:
    raw_config = _mapping(
        checkpoint.get("model_config"),
        "checkpoint model_config",
    )
    try:
        config = DirectSmallConfig(**raw_config)
        config.validate()
    except (TypeError, ValueError) as error:
        raise ModelFormatError(
            f"checkpoint model_config is invalid: {error}"
        ) from error
    if config.bias:
        raise ModelFormatError("format v1 supports only bias-free GPT")
    if not config.tie_embeddings:
        raise ModelFormatError("format v1 requires tied embeddings")
    if config.block_size > 128:
        raise ModelFormatError("format v1 deployment context exceeds 128")
    return config


def _checkpoint_route(
    checkpoint: Mapping[str, Any],
    run_metadata: Mapping[str, Any] | None,
) -> str:
    checkpoint_route = checkpoint.get("route")
    run_route = None if run_metadata is None else run_metadata.get("route")
    if checkpoint_route is None:
        checkpoint_route = run_route
    elif run_route is not None and run_route != checkpoint_route:
        raise ModelFormatError("checkpoint and run route disagree")
    if not isinstance(checkpoint_route, str) or not checkpoint_route:
        raise ModelFormatError("checkpoint route is missing")
    if checkpoint_route not in _SUPPORTED_FP32_ROUTES | {_QUANTIZED_ROUTE}:
        if checkpoint_route == "Quantized-Small-Diagnostic":
            raise ModelFormatError(
                "only the formal Quantized-Small route can be exported"
            )
        raise ModelFormatError(
            f"unsupported checkpoint route {checkpoint_route!r}"
        )
    checkpoint_source = checkpoint.get("source_commit")
    if not isinstance(checkpoint_source, str) or not checkpoint_source:
        raise ModelFormatError("checkpoint source_commit is missing")
    if run_metadata is not None:
        run_source = run_metadata.get("source_commit")
        if run_source is not None and run_source != checkpoint_source:
            raise ModelFormatError(
                "checkpoint and run source_commit disagree"
            )
    return checkpoint_route


def _vocabulary(
    checkpoint: Mapping[str, Any],
    config: DirectSmallConfig,
) -> tuple[str, ...]:
    raw = checkpoint.get("vocabulary")
    if not isinstance(raw, (list, tuple)):
        raise ModelFormatError("checkpoint vocabulary must be a sequence")
    vocabulary = tuple(raw)
    if len(vocabulary) != config.vocab_size:
        raise ModelFormatError(
            "checkpoint vocabulary count does not match model_config"
        )
    return vocabulary


def _fp32_bytes(tensor: torch.Tensor, name: str) -> bytes:
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.dtype != torch.float32
    ):
        raise ModelFormatError(f"tensor {name!r} must be float32")
    value = tensor.detach().to(device="cpu").contiguous()
    if not bool(torch.isfinite(value).all().item()):
        raise ModelFormatError(f"tensor {name!r} contains non-finite values")
    array = value.numpy()
    return array.astype("<f4", copy=False).tobytes(order="C")


def _validate_shape(
    tensor: torch.Tensor,
    descriptor: TensorDescriptor,
) -> None:
    if tuple(tensor.shape) != descriptor.shape:
        raise ModelFormatError(
            f"tensor {descriptor.name!r} shape {tuple(tensor.shape)} "
            f"does not match {descriptor.shape}"
        )


def _fp32_payloads(
    checkpoint: Mapping[str, Any],
    descriptors: Sequence[TensorDescriptor],
) -> list[TensorPayload]:
    state = _mapping(
        checkpoint.get("model_state_dict"),
        "checkpoint model_state_dict",
    )
    canonical_names = {descriptor.name for descriptor in descriptors}
    expected_names = canonical_names | {"lm_head.weight"}
    if set(state) != expected_names:
        missing = sorted(expected_names - set(state))
        extra = sorted(set(state) - expected_names)
        raise ModelFormatError(
            f"FP32 tensor names do not match; missing={missing}, extra={extra}"
        )
    token_embedding = state["token_embedding.weight"]
    lm_head = state["lm_head.weight"]
    if (
        not isinstance(token_embedding, torch.Tensor)
        or not isinstance(lm_head, torch.Tensor)
        or not torch.equal(token_embedding, lm_head)
    ):
        raise ModelFormatError(
            "tied token embedding and lm_head values disagree"
        )
    payloads: list[TensorPayload] = []
    for descriptor in descriptors:
        tensor = state[descriptor.name]
        if not isinstance(tensor, torch.Tensor):
            raise ModelFormatError(
                f"tensor {descriptor.name!r} is not a torch.Tensor"
            )
        _validate_shape(tensor, descriptor)
        payloads.append(
            TensorPayload(
                tensor_id=descriptor.tensor_id,
                storage=STORAGE_FP32,
                shape=descriptor.shape,
                data=_fp32_bytes(tensor, descriptor.name),
            )
        )
    return payloads


def _int4_payloads(
    checkpoint: Mapping[str, Any],
    descriptors: Sequence[TensorDescriptor],
) -> tuple[list[TensorPayload], int]:
    package = _mapping(
        checkpoint.get("quantized_model_state"),
        "checkpoint quantized_model_state",
    )
    if package.get("schema_version") != QUANTIZED_STATE_SCHEMA_VERSION:
        raise ModelFormatError("unsupported quantized state schema")
    quantization = _mapping(
        package.get("quantization"),
        "quantized state quantization",
    )
    if (
        quantization.get("scheme") != QUANTIZED_STATE_SCHEME
        or quantization.get("quantized_range") != [-7, 7]
        or quantization.get("nibble_order") != "low_first"
    ):
        raise ModelFormatError("quantized state scheme is unsupported")
    group_size = quantization.get("group_size")
    if (
        isinstance(group_size, bool)
        or not isinstance(group_size, int)
        or group_size <= 0
        or group_size % 2
    ):
        raise ModelFormatError("quantized group_size must be positive and even")
    aliases = _mapping(package.get("aliases"), "quantized state aliases")
    if dict(aliases) != {
        "lm_head.weight": "token_embedding.weight"
    }:
        raise ModelFormatError("quantized tied alias metadata is invalid")
    tensors = _mapping(package.get("tensors"), "quantized state tensors")
    expected_names = {descriptor.name for descriptor in descriptors}
    if set(tensors) != expected_names:
        missing = sorted(expected_names - set(tensors))
        extra = sorted(set(tensors) - expected_names)
        raise ModelFormatError(
            f"INT4 tensor names do not match; missing={missing}, extra={extra}"
        )

    payloads: list[TensorPayload] = []
    for descriptor in descriptors:
        raw_payload = _mapping(
            tensors[descriptor.name],
            f"tensor {descriptor.name!r}",
        )
        storage = raw_payload.get("storage")
        if len(descriptor.shape) == 2:
            if storage != "int4_groupwise":
                raise ModelFormatError(
                    f"matrix {descriptor.name!r} is not INT4"
                )
            try:
                quantized = GroupwiseInt4Tensor.from_payload(
                    {
                        key: value
                        for key, value in raw_payload.items()
                        if key != "storage"
                    }
                )
            except ValueError as error:
                raise ModelFormatError(
                    f"tensor {descriptor.name!r} is invalid: {error}"
                ) from error
            if quantized.shape != descriptor.shape:
                raise ModelFormatError(
                    f"tensor {descriptor.name!r} shape "
                    f"{quantized.shape} does not match {descriptor.shape}"
                )
            if quantized.group_size != group_size:
                raise ModelFormatError(
                    f"tensor {descriptor.name!r} group_size disagrees"
                )
            payloads.append(
                TensorPayload(
                    tensor_id=descriptor.tensor_id,
                    storage=STORAGE_INT4_GROUPWISE,
                    shape=descriptor.shape,
                    data=quantized.packed.numpy().tobytes(order="C"),
                    auxiliary=(
                        quantized.scales.numpy()
                        .astype("<f4", copy=False)
                        .tobytes(order="C")
                    ),
                    group_size=group_size,
                    padded_last_dim=quantized.padded_last_dim,
                )
            )
        else:
            if storage != "fp32":
                raise ModelFormatError(
                    f"vector {descriptor.name!r} is not FP32"
                )
            if set(raw_payload) != {"storage", "value"}:
                raise ModelFormatError(
                    f"FP32 tensor {descriptor.name!r} has unexpected fields"
                )
            value = raw_payload.get("value")
            if not isinstance(value, torch.Tensor):
                raise ModelFormatError(
                    f"tensor {descriptor.name!r} value is not a tensor"
                )
            _validate_shape(value, descriptor)
            payloads.append(
                TensorPayload(
                    tensor_id=descriptor.tensor_id,
                    storage=STORAGE_FP32,
                    shape=descriptor.shape,
                    data=_fp32_bytes(value, descriptor.name),
                )
            )
    return payloads, group_size


def _tensor_manifest(
    descriptors: Sequence[TensorDescriptor],
    data: bytes,
) -> list[dict[str, object]]:
    parsed = parse_model_file(data)
    descriptor_by_id = {
        descriptor.tensor_id: descriptor
        for descriptor in descriptors
    }
    result: list[dict[str, object]] = []
    for tensor_id, view in parsed.tensors.items():
        descriptor = descriptor_by_id[tensor_id]
        result.append(
            {
                "auxiliary_bytes": len(view.auxiliary),
                "auxiliary_offset": view.auxiliary_offset,
                "auxiliary_sha256": (
                    hashlib.sha256(view.auxiliary).hexdigest()
                    if view.auxiliary
                    else None
                ),
                "data_bytes": len(view.data),
                "data_offset": view.data_offset,
                "data_sha256": hashlib.sha256(view.data).hexdigest(),
                "group_size": view.group_size,
                "name": descriptor.name,
                "padded_last_dim": view.padded_last_dim,
                "shape": list(view.shape),
                "storage": (
                    "fp32"
                    if view.storage == STORAGE_FP32
                    else "int4_groupwise"
                ),
                "tensor_id": tensor_id,
            }
        )
    return result


def build_export(
    checkpoint: Mapping[str, Any],
    *,
    run_metadata: Mapping[str, Any] | None = None,
) -> ExportedModel:
    """Build and reparse a complete export entirely in memory."""

    checkpoint = _mapping(checkpoint, "checkpoint")
    if checkpoint.get("schema_version") != 1:
        raise ModelFormatError("unsupported checkpoint schema_version")
    if run_metadata is not None:
        run_metadata = _mapping(run_metadata, "run metadata")
    route = _checkpoint_route(checkpoint, run_metadata)
    config = _checkpoint_config(checkpoint)
    vocabulary = _vocabulary(checkpoint, config)
    descriptors = expected_tensor_descriptors(config)

    if route == _QUANTIZED_ROUTE:
        if checkpoint.get("model_type") != "direct_small_gpt_int4":
            raise ModelFormatError("Quantized-Small model_type is invalid")
        payloads, group_size = _int4_payloads(
            checkpoint,
            descriptors,
        )
        spec = ModelSpec(
            vocab_size=config.vocab_size,
            block_size=config.block_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            mlp_ratio=config.mlp_ratio,
            tie_embeddings=config.tie_embeddings,
            bias=config.bias,
            model_storage=MODEL_STORAGE_W4A8,
            weight_group_size=group_size,
            activation_quantization=(
                ACTIVATION_DYNAMIC_INT8_GROUPWISE
            ),
            activation_group_size=group_size,
        )
    else:
        if checkpoint.get("model_type") != "direct_small_gpt":
            raise ModelFormatError("FP32 checkpoint model_type is invalid")
        payloads = _fp32_payloads(checkpoint, descriptors)
        spec = ModelSpec(
            vocab_size=config.vocab_size,
            block_size=config.block_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            mlp_ratio=config.mlp_ratio,
            tie_embeddings=config.tie_embeddings,
            bias=config.bias,
            model_storage=MODEL_STORAGE_FP32,
            weight_group_size=0,
            activation_quantization=ACTIVATION_NONE,
            activation_group_size=0,
        )
    data = build_model_file(
        spec=spec,
        vocabulary=vocabulary,
        tensors=payloads,
    )
    parsed = parse_model_file(data)
    aliases = {
        "lm_head.weight": "token_embedding.weight"
    }
    manifest: dict[str, object] = {
        "aliases": aliases,
        "architecture": asdict(config),
        "format": {
            "activation_group_size": spec.activation_group_size,
            "activation_quantization": spec.activation_quantization,
            "data_bytes": parsed.data_bytes,
            "data_offset": parsed.data_offset,
            "header_crc32": parsed.header_crc32,
            "magic": "NGNSP001",
            "model_storage": spec.model_storage,
            "payload_crc32": parsed.payload_crc32,
            "schema_version": 1,
            "weight_group_size": spec.weight_group_size,
        },
        "route": route,
        "schema_version": 1,
        "source_commit": checkpoint["source_commit"],
        "tensors": _tensor_manifest(descriptors, data),
        "vocabulary": list(vocabulary),
    }
    return ExportedModel(data=data, route=route, manifest=manifest)


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def export_checkpoint(
    *,
    checkpoint_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """Load, validate, atomically export and describe one checkpoint."""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if output_path.suffix.lower() != ".ngm":
        raise ModelFormatError("output path must end in .ngm")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    run_path = checkpoint_path.with_name("run.json")
    run_metadata = None
    if run_path.is_file():
        with run_path.open("r", encoding="utf-8") as stream:
            run_metadata = json.load(stream)
    exported = build_export(
        checkpoint,
        run_metadata=run_metadata,
    )
    _write_bytes_atomic(output_path, exported.data)
    manifest = {
        **exported.manifest,
        "output": {
            "bytes": output_path.stat().st_size,
            "path": str(output_path),
            "sha256": sha256_file(output_path),
        },
        "source": {
            "bytes": checkpoint_path.stat().st_size,
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
    }
    manifest_path = output_path.with_suffix(".ngm.json")
    write_json_atomic(manifest_path, manifest)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a frozen Direct, Distilled or formal Quantized checkpoint "
            "to the portable CRC-protected .ngm format."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        manifest = export_checkpoint(
            checkpoint_path=arguments.checkpoint,
            output_path=arguments.output,
        )
    except (
        FileNotFoundError,
        ModelFormatError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        raise SystemExit(f"error: {error}") from error
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
