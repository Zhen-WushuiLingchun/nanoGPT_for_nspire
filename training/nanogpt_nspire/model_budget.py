"""Estimate GPT parameters, packed model bytes, and C inference memory."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import math
from pathlib import Path

from nanogpt_nspire.models.direct_small_gpt import DirectSmallConfig
from nanogpt_nspire.training_support import write_json_atomic


MIB = 1024 * 1024
DEPLOYMENT_FILE_TARGET_MINIMUM_BYTES = 4 * MIB
DEPLOYMENT_FILE_LIMIT_BYTES = 6 * MIB
DEPLOYMENT_RAM_LIMIT_BYTES = 24 * MIB
_FILE_HEADER_BYTES = 128
_TENSOR_ENTRY_BYTES = 64
_ALIGNMENT_BYTES = 64
_SUPPORTED_STORAGE = {"fp32", "fp16", "int8", "w4a8"}


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _align(value: int, alignment: int = _ALIGNMENT_BYTES) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class ParameterTensor:
    """One unique physical parameter tensor."""

    name: str
    shape: tuple[int, ...]

    @property
    def element_count(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True)
class ModelBudgetPolicy:
    """Frozen assumptions shared by all Lesson 11 deployment estimates."""

    weight_group_size: int = 64
    file_target_minimum_bytes: int = DEPLOYMENT_FILE_TARGET_MINIMUM_BYTES
    file_limit_bytes: int = DEPLOYMENT_FILE_LIMIT_BYTES
    ram_limit_bytes: int = DEPLOYMENT_RAM_LIMIT_BYTES
    format_v2_safety_reserve_bytes: int = 64 * 1024
    safety_reserve_bytes: int = 2 * MIB

    def validate(self) -> None:
        for name in (
            "weight_group_size",
            "file_target_minimum_bytes",
            "file_limit_bytes",
            "ram_limit_bytes",
            "format_v2_safety_reserve_bytes",
            "safety_reserve_bytes",
        ):
            _positive_integer(getattr(self, name), name)
        if self.weight_group_size % 2:
            raise ValueError("weight_group_size must be even")
        if self.file_target_minimum_bytes >= self.file_limit_bytes:
            raise ValueError(
                "file_target_minimum_bytes must be smaller than file_limit_bytes"
            )


def parameter_tensors(
    config: DirectSmallConfig,
) -> tuple[ParameterTensor, ...]:
    """Describe every unique parameter using the PyTorch model's naming."""

    if not isinstance(config, DirectSmallConfig):
        raise ValueError("config must be a DirectSmallConfig")
    config.validate()
    width = config.n_embd
    mlp_width = config.mlp_ratio * width
    tensors = [
        ParameterTensor(
            "token_embedding.weight",
            (config.vocab_size, width),
        ),
        ParameterTensor(
            "position_embedding.weight",
            (config.block_size, width),
        ),
    ]
    for block_index in range(config.n_layer):
        prefix = f"blocks.{block_index}"
        tensors.extend(
            (
                ParameterTensor(
                    f"{prefix}.attention_norm.weight",
                    (width,),
                ),
                ParameterTensor(
                    f"{prefix}.attention.qkv.weight",
                    (3 * width, width),
                ),
                ParameterTensor(
                    f"{prefix}.attention.output.weight",
                    (width, width),
                ),
                ParameterTensor(
                    f"{prefix}.mlp_norm.weight",
                    (width,),
                ),
                ParameterTensor(
                    f"{prefix}.mlp.input.weight",
                    (mlp_width, width),
                ),
                ParameterTensor(
                    f"{prefix}.mlp.output.weight",
                    (width, mlp_width),
                ),
            )
        )
        if config.bias:
            tensors.extend(
                (
                    ParameterTensor(
                        f"{prefix}.attention_norm.bias",
                        (width,),
                    ),
                    ParameterTensor(
                        f"{prefix}.attention.qkv.bias",
                        (3 * width,),
                    ),
                    ParameterTensor(
                        f"{prefix}.attention.output.bias",
                        (width,),
                    ),
                    ParameterTensor(
                        f"{prefix}.mlp_norm.bias",
                        (width,),
                    ),
                    ParameterTensor(
                        f"{prefix}.mlp.input.bias",
                        (mlp_width,),
                    ),
                    ParameterTensor(
                        f"{prefix}.mlp.output.bias",
                        (width,),
                    ),
                )
            )
    tensors.append(ParameterTensor("final_norm.weight", (width,)))
    if config.bias:
        tensors.append(ParameterTensor("final_norm.bias", (width,)))
    if not config.tie_embeddings:
        tensors.append(
            ParameterTensor(
                "lm_head.weight",
                (config.vocab_size, width),
            )
        )
    return tuple(tensors)


def _tensor_storage(
    tensor: ParameterTensor,
    *,
    storage: str,
    group_size: int,
) -> tuple[int, int, str]:
    """Return primary bytes, scale bytes, and the primary storage class."""

    elements = tensor.element_count
    if storage == "fp32":
        return elements * 4, 0, "fp32"
    if storage == "fp16":
        return elements * 2, 0, "fp16"
    if len(tensor.shape) == 1:
        return elements * 4, 0, "fp32_passthrough"

    rows = math.prod(tensor.shape[:-1])
    groups_per_row = math.ceil(tensor.shape[-1] / group_size)
    padded_values = rows * groups_per_row * group_size
    scales = rows * groups_per_row * 4
    if storage == "int8":
        return padded_values, scales, "int8_groupwise"
    if storage == "w4a8":
        return (padded_values + 1) // 2, scales, "int4_groupwise"
    raise AssertionError("storage policy was not validated")


def _file_layout(
    tensors: tuple[ParameterTensor, ...],
    payloads: tuple[tuple[int, int, str], ...],
    *,
    vocab_size: int,
    format_v2_safety_reserve_bytes: int,
) -> dict[str, int]:
    tokenizer_metadata_bytes = vocab_size * 2
    cursor = (
        _FILE_HEADER_BYTES
        + len(tensors) * _TENSOR_ENTRY_BYTES
        + tokenizer_metadata_bytes
    )
    alignment_padding = _align(cursor) - cursor
    cursor = _align(cursor)
    for primary_bytes, auxiliary_bytes, _ in payloads:
        aligned = _align(cursor)
        alignment_padding += aligned - cursor
        cursor = aligned + primary_bytes
        if auxiliary_bytes:
            aligned = _align(cursor)
            alignment_padding += aligned - cursor
            cursor = aligned + auxiliary_bytes
    aligned = _align(cursor)
    alignment_padding += aligned - cursor

    payload_bytes = sum(
        primary + auxiliary
        for primary, auxiliary, _ in payloads
    )
    return {
        "alignment_padding_bytes": alignment_padding,
        "container_header_bytes": _FILE_HEADER_BYTES,
        "format_v2_safety_reserve_bytes": (
            format_v2_safety_reserve_bytes
        ),
        "payload_bytes": payload_bytes,
        "tensor_table_bytes": len(tensors) * _TENSOR_ENTRY_BYTES,
        "tokenizer_metadata_bytes": tokenizer_metadata_bytes,
    }


def _inference_arena(
    config: DirectSmallConfig,
    *,
    storage: str,
    group_size: int,
) -> dict[str, int]:
    kv_cache_bytes = (
        config.n_layer
        * config.block_size
        * config.n_embd
        * 2
        * 4
    )
    mlp_width = config.mlp_ratio * config.n_embd
    float_values = (
        7 * config.n_embd
        + mlp_width
        + config.vocab_size
        + config.block_size
    )
    quantized_activation_workspace_bytes = 0
    if storage == "w4a8":
        padded_mlp_width = _align(mlp_width, group_size)
        float_values += padded_mlp_width // group_size
        quantized_activation_workspace_bytes = padded_mlp_width
    float_workspace_bytes = float_values * 4

    cursor = 0
    for size in (
        kv_cache_bytes,
        float_workspace_bytes,
        quantized_activation_workspace_bytes,
    ):
        if size == 0:
            continue
        cursor = _align(cursor)
        cursor += size
    arena_bytes = _align(cursor)
    arena_alignment_padding_bytes = arena_bytes - (
        kv_cache_bytes
        + float_workspace_bytes
        + quantized_activation_workspace_bytes
    )
    return {
        "arena_alignment_padding_bytes": arena_alignment_padding_bytes,
        "float_workspace_bytes": float_workspace_bytes,
        "kv_cache_bytes": kv_cache_bytes,
        "quantized_activation_workspace_bytes": (
            quantized_activation_workspace_bytes
        ),
    }


def estimate_model_budget(
    config: DirectSmallConfig,
    *,
    storage: str = "w4a8",
    policy: ModelBudgetPolicy | None = None,
) -> dict[str, object]:
    """Return a complete static estimate under one named storage policy."""

    if not isinstance(config, DirectSmallConfig):
        raise ValueError("config must be a DirectSmallConfig")
    config.validate()
    if storage not in _SUPPORTED_STORAGE:
        raise ValueError(f"unsupported storage policy: {storage!r}")
    policy = policy or ModelBudgetPolicy()
    if not isinstance(policy, ModelBudgetPolicy):
        raise ValueError("policy must be a ModelBudgetPolicy")
    policy.validate()

    tensors = parameter_tensors(config)
    payloads = tuple(
        _tensor_storage(
            tensor,
            storage=storage,
            group_size=policy.weight_group_size,
        )
        for tensor in tensors
    )
    parameter_count = sum(tensor.element_count for tensor in tensors)
    packed_weight_bytes = sum(
        primary
        for primary, _, storage_class in payloads
        if storage_class in {"int4_groupwise", "int8_groupwise"}
    )
    fp32_scale_bytes = sum(auxiliary for _, auxiliary, _ in payloads)
    fp32_passthrough_bytes = sum(
        primary
        for primary, _, storage_class in payloads
        if storage_class == "fp32_passthrough"
    )
    floating_weight_bytes = sum(
        primary
        for primary, _, storage_class in payloads
        if storage_class in {"fp32", "fp16"}
    )

    file_components = _file_layout(
        tensors,
        payloads,
        vocab_size=config.vocab_size,
        format_v2_safety_reserve_bytes=(
            policy.format_v2_safety_reserve_bytes
        ),
    )
    estimated_file_bytes = sum(file_components.values())
    arena_components = _inference_arena(
        config,
        storage=storage,
        group_size=policy.weight_group_size,
    )
    inference_components = {
        **arena_components,
        "model_blob_bytes": estimated_file_bytes,
        "ui_allocator_safety_reserve_bytes": policy.safety_reserve_bytes,
    }
    estimated_peak_bytes = sum(inference_components.values())

    return {
        "architecture": asdict(config),
        "file": {
            "components": file_components,
            "estimated_bytes": estimated_file_bytes,
            "limit_bytes": policy.file_limit_bytes,
            "margin_bytes": policy.file_limit_bytes - estimated_file_bytes,
            "target_band_minimum_bytes": (
                policy.file_target_minimum_bytes
            ),
            "target_band_passed": (
                policy.file_target_minimum_bytes
                <= estimated_file_bytes
                <= policy.file_limit_bytes
            ),
        },
        "inference_ram": {
            "components": inference_components,
            "estimated_peak_bytes": estimated_peak_bytes,
            "limit_bytes": policy.ram_limit_bytes,
            "limit_passed": estimated_peak_bytes <= policy.ram_limit_bytes,
            "margin_bytes": policy.ram_limit_bytes - estimated_peak_bytes,
        },
        "parameter_count": parameter_count,
        "storage": {
            "floating_weight_bytes": floating_weight_bytes,
            "fp32_passthrough_bytes": fp32_passthrough_bytes,
            "fp32_scale_bytes": fp32_scale_bytes,
            "group_size": (
                policy.weight_group_size
                if storage in {"int8", "w4a8"}
                else None
            ),
            "logical_payload_bytes": sum(
                primary + auxiliary
                for primary, auxiliary, _ in payloads
            ),
            "packed_weight_bytes": packed_weight_bytes,
            "policy": storage,
        },
        "tensor_count": len(tensors),
        "training_memory_lower_bound": {
            "bytes": parameter_count * 12,
            "includes": "FP32 parameters, gradients, and Adam first/second moments",
            "excludes": "activations, temporary kernels, batches, and allocator overhead",
        },
    }


def lesson11_budget_report() -> dict[str, object]:
    """Compare the bounded architecture grid and freeze Lesson 11 shapes."""

    common = {
        "vocab_size": 264,
        "block_size": 256,
        "mlp_ratio": 4,
        "dropout": 0.0,
        "bias": False,
        "tie_embeddings": True,
    }
    grid = {
        "student-6x320": DirectSmallConfig(
            **common,
            n_layer=6,
            n_head=5,
            n_embd=320,
        ),
        "student-6x384-selected": DirectSmallConfig(
            **common,
            n_layer=6,
            n_head=6,
            n_embd=384,
        ),
        "student-8x320": DirectSmallConfig(
            **common,
            n_layer=8,
            n_head=5,
            n_embd=320,
        ),
        "student-6x448": DirectSmallConfig(
            **common,
            n_layer=6,
            n_head=7,
            n_embd=448,
        ),
    }
    teacher = DirectSmallConfig(
        **common,
        n_layer=12,
        n_head=10,
        n_embd=640,
    )
    return {
        "claim_boundary": (
            "Static format-v2 and arena estimate; actual .ngm v2 bytes, Host "
            "peak RAM, Nspire peak RAM, and speed remain unmeasured."
        ),
        "frozen": {
            "student": "student-6x384-selected",
            "student_reason": (
                "uses more parameters than 6x320 and 8x320 while retaining "
                "six serial blocks, 64-wide heads, and positive file/RAM margins"
            ),
            "teacher": "teacher-12x640",
            "teacher_deployment": "computer_only",
        },
        "policy": asdict(ModelBudgetPolicy()),
        "schema_version": 1,
        "student_grid": {
            name: estimate_model_budget(config, storage="w4a8")
            for name, config in grid.items()
        },
        "teacher": {
            "fp32": estimate_model_budget(teacher, storage="fp32"),
            "name": "teacher-12x640",
            "w4a8_reference": estimate_model_budget(
                teacher,
                storage="w4a8",
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    report = lesson11_budget_report()
    write_json_atomic(arguments.output, report)
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
