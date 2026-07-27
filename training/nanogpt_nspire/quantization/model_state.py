"""Quantize unique GPT parameters and rebuild a dequantized reference model."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from nanogpt_nspire.models.direct_small_gpt import (
    DirectSmallConfig,
    DirectSmallGPT,
)
from nanogpt_nspire.quantization.int4 import (
    GroupwiseInt4Tensor,
    dequantize_groupwise_int4,
    quantize_groupwise_int4,
)


SCHEMA_VERSION = 1
SCHEME = "symmetric_signed_int4_groupwise_last_dimension"


def quantize_model_state(
    model: nn.Module,
    *,
    group_size: int = 64,
) -> dict[str, Any]:
    """Store each physical Parameter once under the frozen INT4 policy."""

    if not isinstance(model, nn.Module):
        raise ValueError("model must be a torch.nn.Module")
    if (
        isinstance(group_size, bool)
        or not isinstance(group_size, int)
        or group_size <= 0
    ):
        raise ValueError("group_size must be a positive integer")

    canonical_by_parameter: dict[int, str] = {}
    tensors: dict[str, dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    packed_nibble_bytes = 0
    fp32_scale_bytes = 0
    fp32_passthrough_bytes = 0

    for name, parameter in model.named_parameters(remove_duplicate=False):
        identity = id(parameter)
        if identity in canonical_by_parameter:
            aliases[name] = canonical_by_parameter[identity]
            continue
        canonical_by_parameter[identity] = name
        if parameter.ndim == 2:
            quantized = quantize_groupwise_int4(
                parameter,
                group_size=group_size,
            )
            payload = quantized.to_payload()
            tensors[name] = {
                "storage": "int4_groupwise",
                **payload,
            }
            packed_nibble_bytes += quantized.packed_bytes
            fp32_scale_bytes += quantized.scale_bytes
        elif parameter.ndim == 1:
            value = (
                parameter.detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
                .clone()
            )
            if not torch.isfinite(value).all():
                raise ValueError(f"parameter {name!r} contains non-finite values")
            tensors[name] = {
                "storage": "fp32",
                "value": value,
            }
            fp32_passthrough_bytes += value.numel() * value.element_size()
        else:
            raise ValueError(
                f"parameter {name!r} has unsupported rank {parameter.ndim}"
            )

    logical_payload_bytes = (
        packed_nibble_bytes
        + fp32_scale_bytes
        + fp32_passthrough_bytes
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "quantization": {
            "scheme": SCHEME,
            "group_size": group_size,
            "quantized_range": [-7, 7],
            "nibble_order": "low_first",
        },
        "tensors": tensors,
        "aliases": aliases,
        "storage": {
            "packed_nibble_bytes": packed_nibble_bytes,
            "fp32_scale_bytes": fp32_scale_bytes,
            "fp32_passthrough_bytes": fp32_passthrough_bytes,
            "logical_payload_bytes": logical_payload_bytes,
            "canonical_tensor_count": len(tensors),
            "alias_count": len(aliases),
        },
    }


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def dequantize_model_state(
    package: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Validate a package and expand aliases into an FP32 state dict."""

    package = _require_mapping(package, "package")
    if package.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {package.get('schema_version')!r}")
    quantization = _require_mapping(
        package.get("quantization"),
        "quantization",
    )
    if (
        quantization.get("scheme") != SCHEME
        or quantization.get("quantized_range") != [-7, 7]
        or quantization.get("nibble_order") != "low_first"
    ):
        raise ValueError("quantization metadata does not match the frozen scheme")
    group_size = quantization.get("group_size")
    if (
        isinstance(group_size, bool)
        or not isinstance(group_size, int)
        or group_size <= 0
    ):
        raise ValueError("quantization group_size must be a positive integer")

    tensors = _require_mapping(package.get("tensors"), "tensors")
    aliases = _require_mapping(package.get("aliases"), "aliases")
    state: dict[str, torch.Tensor] = {}
    for name, raw_payload in tensors.items():
        if not isinstance(name, str) or not name:
            raise ValueError("canonical tensor names must be non-empty strings")
        payload = _require_mapping(raw_payload, f"tensor {name!r}")
        storage = payload.get("storage")
        if storage == "int4_groupwise":
            int4_payload = {
                key: value
                for key, value in payload.items()
                if key != "storage"
            }
            quantized = GroupwiseInt4Tensor.from_payload(int4_payload)
            if quantized.group_size != group_size:
                raise ValueError(
                    f"tensor {name!r} group_size disagrees with package"
                )
            state[name] = dequantize_groupwise_int4(quantized)
        elif storage == "fp32":
            if set(payload) != {"storage", "value"}:
                raise ValueError(f"FP32 tensor {name!r} has unexpected fields")
            value = payload.get("value")
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != torch.float32
                or value.device.type != "cpu"
                or value.ndim != 1
            ):
                raise ValueError(
                    f"FP32 tensor {name!r} must be a one-dimensional CPU float32 tensor"
                )
            if not torch.isfinite(value).all():
                raise ValueError(f"FP32 tensor {name!r} contains non-finite values")
            state[name] = value.clone()
        else:
            raise ValueError(f"tensor {name!r} has unknown storage {storage!r}")

    for alias, target in aliases.items():
        if not isinstance(alias, str) or not alias or alias in state:
            raise ValueError("alias names must be unique non-empty strings")
        if not isinstance(target, str) or target not in state:
            raise ValueError(f"alias target {target!r} does not exist")
        state[alias] = state[target]
    return state


def reconstruct_dequantized_reference(
    config: DirectSmallConfig,
    package: Mapping[str, Any],
) -> DirectSmallGPT:
    """Strictly load dequantized weights into the original PyTorch GPT."""

    if not isinstance(config, DirectSmallConfig):
        raise ValueError("config must be a DirectSmallConfig")
    model = DirectSmallGPT(config)
    state = dequantize_model_state(package)
    model.load_state_dict(state, strict=True)
    return model
