"""Signed nibble packing and symmetric groupwise INT4 quantization."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch


_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def pack_signed_int4(values: torch.Tensor) -> torch.Tensor:
    """Pack signed values in low-nibble-first two's-complement order."""

    if not isinstance(values, torch.Tensor):
        raise ValueError("values must be a torch.Tensor")
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if values.dtype not in _INTEGER_DTYPES:
        raise ValueError("values must use an integer dtype")
    values_cpu = values.detach().to(device="cpu", dtype=torch.int16)
    if values_cpu.numel():
        minimum = int(values_cpu.min().item())
        maximum = int(values_cpu.max().item())
        if minimum < -8 or maximum > 7:
            raise ValueError("signed INT4 values must lie in [-8, 7]")
    nibbles = torch.bitwise_and(values_cpu, 0x0F).to(torch.uint8)
    if nibbles.numel() % 2:
        nibbles = torch.cat((nibbles, torch.zeros(1, dtype=torch.uint8)))
    low = nibbles[0::2]
    high = torch.bitwise_left_shift(nibbles[1::2], 4)
    return torch.bitwise_or(low, high).contiguous()


def unpack_signed_int4(packed: torch.Tensor, *, count: int) -> torch.Tensor:
    """Unpack exactly ``count`` signed values from low-first nibbles."""

    if not isinstance(packed, torch.Tensor) or packed.dtype != torch.uint8:
        raise ValueError("packed must be a torch.uint8 tensor")
    if packed.ndim != 1:
        raise ValueError("packed must be one-dimensional")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a non-negative integer")
    expected_bytes = (count + 1) // 2
    if packed.numel() != expected_bytes:
        raise ValueError(
            "packed byte count does not match the declared value count"
        )
    packed_cpu = packed.detach().cpu()
    nibbles = torch.empty(expected_bytes * 2, dtype=torch.uint8)
    nibbles[0::2] = torch.bitwise_and(packed_cpu, 0x0F)
    nibbles[1::2] = torch.bitwise_right_shift(packed_cpu, 4)
    signed = nibbles.to(torch.int8)
    signed = torch.where(signed >= 8, signed - 16, signed)
    return signed[:count].contiguous()


@dataclass(frozen=True)
class GroupwiseInt4Tensor:
    """Portable tensors and metadata for one groupwise-quantized weight."""

    packed: torch.Tensor
    scales: torch.Tensor
    shape: tuple[int, ...]
    group_size: int
    padded_last_dim: int

    def validate(self) -> None:
        if not isinstance(self.shape, tuple) or not self.shape:
            raise ValueError("shape must be a non-empty tuple")
        for dimension in self.shape:
            _positive_integer(dimension, "shape dimension")
        group_size = _positive_integer(self.group_size, "group_size")
        padded_last_dim = _positive_integer(
            self.padded_last_dim,
            "padded_last_dim",
        )
        expected_padded = (
            math.ceil(self.shape[-1] / group_size) * group_size
        )
        if padded_last_dim != expected_padded:
            raise ValueError(
                "padded_last_dim does not match shape and group_size"
            )
        if (
            not isinstance(self.packed, torch.Tensor)
            or self.packed.dtype != torch.uint8
            or self.packed.ndim != 1
            or self.packed.device.type != "cpu"
        ):
            raise ValueError("packed must be a one-dimensional CPU uint8 tensor")
        row_count = math.prod(self.shape[:-1]) if len(self.shape) > 1 else 1
        value_count = row_count * padded_last_dim
        if self.packed.numel() != (value_count + 1) // 2:
            raise ValueError("packed byte count does not match quantized shape")
        expected_scale_shape = (
            *self.shape[:-1],
            padded_last_dim // group_size,
        )
        if (
            not isinstance(self.scales, torch.Tensor)
            or self.scales.dtype != torch.float32
            or self.scales.device.type != "cpu"
            or tuple(self.scales.shape) != expected_scale_shape
        ):
            raise ValueError(
                "scales must be a CPU float32 tensor with one value per group"
            )
        if not torch.isfinite(self.scales).all() or torch.any(self.scales <= 0):
            raise ValueError("scales must be finite and positive")

    @property
    def packed_bytes(self) -> int:
        return self.packed.numel()

    @property
    def scale_bytes(self) -> int:
        return self.scales.numel() * self.scales.element_size()

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "packed": self.packed.clone(),
            "scales": self.scales.clone(),
            "shape": list(self.shape),
            "group_size": self.group_size,
            "padded_last_dim": self.padded_last_dim,
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> GroupwiseInt4Tensor:
        if not isinstance(payload, Mapping):
            raise ValueError("INT4 payload must be a mapping")
        required = {
            "packed",
            "scales",
            "shape",
            "group_size",
            "padded_last_dim",
        }
        if set(payload) != required:
            raise ValueError("INT4 payload fields do not match the schema")
        raw_shape = payload["shape"]
        if (
            not isinstance(raw_shape, (list, tuple))
            or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_shape)
        ):
            raise ValueError("shape must contain integers")
        instance = cls(
            packed=payload["packed"],
            scales=payload["scales"],
            shape=tuple(raw_shape),
            group_size=payload["group_size"],
            padded_last_dim=payload["padded_last_dim"],
        )
        instance.validate()
        return instance


def quantize_groupwise_int4(
    tensor: torch.Tensor,
    *,
    group_size: int = 64,
) -> GroupwiseInt4Tensor:
    """Quantize final-axis groups symmetrically to the range ``[-7, 7]``."""

    group_size = _positive_integer(group_size, "group_size")
    if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
        raise ValueError("tensor must be a floating-point torch.Tensor")
    if tensor.ndim < 1:
        raise ValueError("tensor must have at least one dimension")
    if tensor.numel() == 0:
        raise ValueError("tensor must be non-empty")
    source = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not torch.isfinite(source).all():
        raise ValueError("tensor values must be finite")

    shape = tuple(source.shape)
    last_dim = shape[-1]
    group_count = math.ceil(last_dim / group_size)
    padded_last_dim = group_count * group_size
    rows = source.reshape(-1, last_dim)
    if padded_last_dim != last_dim:
        padding = torch.zeros(
            (rows.shape[0], padded_last_dim - last_dim),
            dtype=torch.float32,
        )
        rows = torch.cat((rows, padding), dim=1)
    groups = rows.reshape(rows.shape[0], group_count, group_size)
    max_abs = groups.abs().amax(dim=-1)
    scales = torch.where(
        max_abs == 0,
        torch.ones_like(max_abs),
        max_abs / 7.0,
    )
    quantized = torch.round(groups / scales.unsqueeze(-1))
    quantized = quantized.clamp(-7, 7).to(torch.int8)
    scale_shape = (*shape[:-1], group_count)
    result = GroupwiseInt4Tensor(
        packed=pack_signed_int4(quantized.reshape(-1)),
        scales=scales.reshape(scale_shape).contiguous(),
        shape=shape,
        group_size=group_size,
        padded_last_dim=padded_last_dim,
    )
    result.validate()
    return result


def dequantize_groupwise_int4(
    quantized: GroupwiseInt4Tensor,
) -> torch.Tensor:
    """Reconstruct an FP32 tensor for reference quality evaluation."""

    if not isinstance(quantized, GroupwiseInt4Tensor):
        raise ValueError("quantized must be a GroupwiseInt4Tensor")
    quantized.validate()
    row_count = (
        math.prod(quantized.shape[:-1])
        if len(quantized.shape) > 1
        else 1
    )
    group_count = quantized.padded_last_dim // quantized.group_size
    value_count = row_count * quantized.padded_last_dim
    values = unpack_signed_int4(
        quantized.packed,
        count=value_count,
    ).to(torch.float32)
    groups = values.reshape(
        row_count,
        group_count,
        quantized.group_size,
    )
    scales = quantized.scales.reshape(row_count, group_count, 1)
    rows = (groups * scales).reshape(row_count, quantized.padded_last_dim)
    return rows[:, : quantized.shape[-1]].reshape(quantized.shape).contiguous()
