"""Export Lesson 15/16 byte-token GQA checkpoints for Host C and Ndless."""

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
    MODEL_STORAGE_W4A8,
    POSITION_ALIBI,
    POSITION_LEARNED,
    STORAGE_FP32,
    STORAGE_INT4_GROUPWISE,
    TOKENIZER_BYTE_SPECIAL,
    ModelFormatError,
    ModelSpec,
    TensorPayload,
    build_model_file,
    parse_model_file,
)
from nanogpt_nspire.export_model import (
    BLOCK_TENSOR_ID_BASE,
    BLOCK_TENSOR_ID_STRIDE,
    FINAL_NORM_TENSOR_ID,
    POSITION_EMBEDDING_TENSOR_ID,
    TOKEN_EMBEDDING_TENSOR_ID,
)
from nanogpt_nspire.models.efficient_long_context_gpt import (
    ALIBI_POSITIONS,
    LEARNED_POSITIONS,
    EfficientLongContextConfig,
)
from nanogpt_nspire.quantization import quantize_groupwise_int4
from nanogpt_nspire.training_support import sha256_file, write_json_atomic


@dataclass(frozen=True)
class EfficientTensorDescriptor:
    tensor_id: int
    name: str
    shape: tuple[int, ...]


def expected_efficient_tensor_descriptors(
    config: EfficientLongContextConfig,
) -> tuple[EfficientTensorDescriptor, ...]:
    """Return the stable tensor order understood by the format-v2 C loader."""

    config.validate()
    head_dim = config.n_embd // config.n_head
    kv_width = config.n_kv_head * head_dim
    descriptors = [
        EfficientTensorDescriptor(
            TOKEN_EMBEDDING_TENSOR_ID,
            "token_embedding.weight",
            (config.vocab_size, config.n_embd),
        )
    ]
    if config.position_mode == LEARNED_POSITIONS:
        descriptors.append(
            EfficientTensorDescriptor(
                POSITION_EMBEDDING_TENSOR_ID,
                "position_embedding.weight",
                (config.block_size, config.n_embd),
            )
        )
    block_shapes = (
        ("attention_norm.weight", (config.n_embd,)),
        (
            "attention.qkv.weight",
            (config.n_embd + 2 * kv_width, config.n_embd),
        ),
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
                EfficientTensorDescriptor(
                    BLOCK_TENSOR_ID_BASE
                    + block_index * BLOCK_TENSOR_ID_STRIDE
                    + slot,
                    f"blocks.{block_index}.{suffix}",
                    shape,
                )
            )
    descriptors.append(
        EfficientTensorDescriptor(
            FINAL_NORM_TENSOR_ID,
            "final_norm.weight",
            (config.n_embd,),
        )
    )
    return tuple(descriptors)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelFormatError(f"{name} must be a mapping")
    return value


def _fp32_bytes(tensor: torch.Tensor, name: str) -> bytes:
    if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
        raise ModelFormatError(f"tensor {name!r} must be floating point")
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(value).all().item()):
        raise ModelFormatError(f"tensor {name!r} contains non-finite values")
    return value.numpy().astype("<f4", copy=False).tobytes(order="C")


def _validated_checkpoint(
    checkpoint: Mapping[str, Any],
) -> tuple[
    EfficientLongContextConfig,
    Mapping[str, torch.Tensor],
    tuple[EfficientTensorDescriptor, ...],
]:
    if checkpoint.get("schema_version") != 1:
        raise ModelFormatError("unsupported checkpoint schema_version")
    if checkpoint.get("architecture") != "efficient_long_context_gpt":
        raise ModelFormatError(
            "deployment export requires efficient_long_context_gpt"
        )
    tokenizer = _mapping(checkpoint.get("tokenizer"), "checkpoint tokenizer")
    if dict(tokenizer) != {
        "kind": "byte_plus_fixed_special_tokens",
        "vocab_size": 264,
    }:
        raise ModelFormatError("checkpoint tokenizer protocol is unsupported")
    raw_config = _mapping(
        checkpoint.get("model_config"),
        "checkpoint model_config",
    )
    try:
        config = EfficientLongContextConfig(**raw_config)
        config.validate()
    except (TypeError, ValueError) as error:
        raise ModelFormatError(f"checkpoint model_config is invalid: {error}") from error
    if config.bias or not config.tie_embeddings:
        raise ModelFormatError(
            "deployment requires bias-free tied-embedding GPT"
        )
    if config.block_size > 512:
        raise ModelFormatError("deployment context exceeds 512 tokens")
    state = _mapping(
        checkpoint.get("model_state_dict"),
        "checkpoint model_state_dict",
    )
    descriptors = expected_efficient_tensor_descriptors(config)
    canonical_names = {descriptor.name for descriptor in descriptors}
    expected_names = canonical_names | {"lm_head.weight"}
    if set(state) != expected_names:
        missing = sorted(expected_names - set(state))
        extra = sorted(set(state) - expected_names)
        raise ModelFormatError(
            f"checkpoint tensors disagree; missing={missing}, extra={extra}"
        )
    token_embedding = state["token_embedding.weight"]
    lm_head = state["lm_head.weight"]
    if (
        not isinstance(token_embedding, torch.Tensor)
        or not isinstance(lm_head, torch.Tensor)
        or not torch.equal(token_embedding, lm_head)
    ):
        raise ModelFormatError("tied token embedding and lm_head disagree")
    for descriptor in descriptors:
        tensor = state[descriptor.name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(tensor.shape) != descriptor.shape
        ):
            raise ModelFormatError(
                f"tensor {descriptor.name!r} has an invalid shape"
            )
    return config, state, descriptors


def build_deployment_export(
    checkpoint: Mapping[str, Any],
    *,
    group_size: int = 64,
) -> tuple[bytes, dict[str, object]]:
    """Quantize one efficient checkpoint and build a deterministic NGM blob."""

    checkpoint = _mapping(checkpoint, "checkpoint")
    config, state, descriptors = _validated_checkpoint(checkpoint)
    payloads: list[TensorPayload] = []
    for descriptor in descriptors:
        tensor = state[descriptor.name]
        if tensor.ndim == 2:
            quantized = quantize_groupwise_int4(
                tensor,
                group_size=group_size,
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
        elif tensor.ndim == 1:
            payloads.append(
                TensorPayload(
                    tensor_id=descriptor.tensor_id,
                    storage=STORAGE_FP32,
                    shape=descriptor.shape,
                    data=_fp32_bytes(tensor, descriptor.name),
                )
            )
        else:
            raise ModelFormatError(
                f"tensor {descriptor.name!r} has unsupported rank"
            )
    position_mode = (
        POSITION_ALIBI
        if config.position_mode == ALIBI_POSITIONS
        else POSITION_LEARNED
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
        activation_quantization=ACTIVATION_DYNAMIC_INT8_GROUPWISE,
        activation_group_size=group_size,
        n_kv_head=config.n_kv_head,
        position_mode=position_mode,
        tokenizer_type=TOKENIZER_BYTE_SPECIAL,
    )
    data = build_model_file(spec=spec, vocabulary=(), tensors=payloads)
    parsed = parse_model_file(data)
    descriptor_by_id = {
        descriptor.tensor_id: descriptor for descriptor in descriptors
    }
    tensors = []
    for tensor_id, view in parsed.tensors.items():
        descriptor = descriptor_by_id[tensor_id]
        tensors.append(
            {
                "auxiliary_bytes": len(view.auxiliary),
                "data_bytes": len(view.data),
                "data_sha256": hashlib.sha256(view.data).hexdigest(),
                "name": descriptor.name,
                "shape": list(view.shape),
                "storage": (
                    "fp32"
                    if view.storage == STORAGE_FP32
                    else "int4_groupwise"
                ),
                "tensor_id": tensor_id,
            }
        )
    manifest: dict[str, object] = {
        "architecture": asdict(config),
        "format": {
            "activation": "dynamic_int8_groupwise",
            "format_version": 2,
            "model_storage": "packed_int4",
            "position_mode": config.position_mode,
            "tokenizer": "byte_plus_fixed_special_tokens",
            "weight_group_size": group_size,
        },
        "route": checkpoint.get("route"),
        "schema_version": 1,
        "source_commit": checkpoint.get("source_commit"),
        "tensors": tensors,
    }
    return data, manifest


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


def export_deployment_checkpoint(
    *,
    checkpoint_path: Path,
    output_path: Path,
    group_size: int = 64,
) -> dict[str, object]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if output_path.suffix.lower() != ".ngm":
        raise ModelFormatError("output path must end in .ngm")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    data, manifest = build_deployment_export(
        checkpoint,
        group_size=group_size,
    )
    _write_bytes_atomic(output_path, data)
    complete = {
        **manifest,
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
    write_json_atomic(output_path.with_suffix(".ngm.json"), complete)
    return complete


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Quantize and export a byte-token GQA Lesson 15/16 checkpoint "
            "for Host C and Ndless."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=64)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        manifest = export_deployment_checkpoint(
            checkpoint_path=arguments.checkpoint,
            output_path=arguments.output,
            group_size=arguments.group_size,
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
