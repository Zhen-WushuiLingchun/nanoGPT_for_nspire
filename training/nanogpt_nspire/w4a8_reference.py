"""Executable PyTorch reference for direct packed-W4/dynamic-A8 inference."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.nn import functional as F

from nanogpt_nspire.alignment import ProbeResult
from nanogpt_nspire.export_format import (
    MODEL_STORAGE_W4A8,
    STORAGE_FP32,
    STORAGE_INT4_GROUPWISE,
    ModelFormatError,
    ParsedModel,
    TensorView,
    parse_model_file,
)
from nanogpt_nspire.export_model import (
    BLOCK_TENSOR_ID_BASE,
    BLOCK_TENSOR_ID_STRIDE,
    FINAL_NORM_TENSOR_ID,
    POSITION_EMBEDDING_TENSOR_ID,
    TOKEN_EMBEDDING_TENSOR_ID,
)
from nanogpt_nspire.quantization import unpack_signed_int4


@dataclass(frozen=True)
class W4Tensor:
    """Unpacked signed nibbles plus FP32 scales for reference arithmetic."""

    values: torch.Tensor
    scales: torch.Tensor
    rows: int
    columns: int
    padded_columns: int
    group_size: int


def dynamic_quantize_int8(
    inputs: torch.Tensor,
    *,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize every final-axis group to symmetric signed ``[-127, 127]``."""

    if (
        not isinstance(group_size, int)
        or isinstance(group_size, bool)
        or group_size <= 0
    ):
        raise ValueError("group_size must be a positive integer")
    if not isinstance(inputs, torch.Tensor) or not inputs.is_floating_point():
        raise ValueError("inputs must be a floating-point tensor")
    if inputs.ndim < 1 or inputs.shape[-1] == 0:
        raise ValueError("inputs must have a non-empty final dimension")
    source = inputs.to(dtype=torch.float32)
    if not bool(torch.isfinite(source).all().item()):
        raise ValueError("inputs must be finite")
    columns = source.shape[-1]
    group_count = math.ceil(columns / group_size)
    padded_columns = group_count * group_size
    if padded_columns != columns:
        source = F.pad(source, (0, padded_columns - columns))
    groups = source.reshape(*source.shape[:-1], group_count, group_size)
    maximum = groups.abs().amax(dim=-1)
    scales = torch.where(
        maximum == 0.0,
        torch.ones_like(maximum),
        maximum / 127.0,
    )
    quantized = torch.round(groups / scales.unsqueeze(-1))
    quantized = quantized.clamp(-127, 127).to(torch.int8)
    return quantized, scales


def w4a8_matvec(inputs: torch.Tensor, weight: W4Tensor) -> torch.Tensor:
    """Apply a row-major W4 matrix without reconstructing FP32 weights."""

    if inputs.shape[-1] != weight.columns:
        raise ValueError("input width does not match W4 matrix")
    quantized, activation_scales = dynamic_quantize_int8(
        inputs,
        group_size=weight.group_size,
    )
    leading_shape = quantized.shape[:-2]
    group_count = weight.padded_columns // weight.group_size
    flat_count = math.prod(leading_shape) if leading_shape else 1
    activation_groups = quantized.reshape(
        flat_count,
        group_count,
        weight.group_size,
    )
    weight_groups = weight.values.reshape(
        weight.rows,
        group_count,
        weight.group_size,
    )
    flat_scales = activation_scales.reshape(flat_count, group_count)
    output = torch.zeros(
        (flat_count, weight.rows),
        dtype=torch.float32,
        device=inputs.device,
    )
    weight_values = weight_groups.to(device=inputs.device)
    weight_scales = weight.scales.to(device=inputs.device)
    for group in range(group_count):
        # All integer dot products are below 2^24 for the frozen group size,
        # so this FP32 matmul represents the INT32 result exactly.
        integer_dot = (
            activation_groups[:, group, :].to(torch.float32)
            @ weight_values[:, group, :].to(torch.float32).T
        )
        output += (
            integer_dot
            * weight_scales[None, :, group]
            * flat_scales[:, group, None]
        )
    return output.reshape(*leading_shape, weight.rows)


def _w4_tensor(view: TensorView) -> W4Tensor:
    if view.storage != STORAGE_INT4_GROUPWISE or len(view.shape) != 2:
        raise ModelFormatError("expected a rank-2 INT4 tensor")
    rows, columns = view.shape
    value_count = rows * view.padded_last_dim
    packed = torch.from_numpy(
        np.frombuffer(view.data, dtype=np.uint8).copy()
    )
    values = unpack_signed_int4(
        packed,
        count=value_count,
    ).reshape(rows, view.padded_last_dim)
    group_count = view.padded_last_dim // view.group_size
    scales = torch.from_numpy(
        np.frombuffer(view.auxiliary, dtype="<f4")
        .reshape(rows, group_count)
        .copy()
    )
    return W4Tensor(
        values=values,
        scales=scales,
        rows=rows,
        columns=columns,
        padded_columns=view.padded_last_dim,
        group_size=view.group_size,
    )


def _fp32_vector(view: TensorView) -> torch.Tensor:
    if view.storage != STORAGE_FP32 or len(view.shape) != 1:
        raise ModelFormatError("expected a rank-1 FP32 tensor")
    return torch.from_numpy(
        np.frombuffer(view.data, dtype="<f4").copy()
    )


class W4A8Reference:
    """Full-prefix and incremental reference sharing the frozen W4A8 policy."""

    def __init__(self, parsed: ParsedModel) -> None:
        if parsed.spec.model_storage != MODEL_STORAGE_W4A8:
            raise ModelFormatError("W4A8 reference requires a W4A8 export")
        self.parsed = parsed
        self.vocab_size = parsed.spec.vocab_size
        self.block_size = parsed.spec.block_size
        self.n_layer = parsed.spec.n_layer
        self.n_head = parsed.spec.n_head
        self.n_embd = parsed.spec.n_embd
        self.mlp_ratio = parsed.spec.mlp_ratio
        self.head_dim = self.n_embd // self.n_head
        self.matrices = {
            tensor_id: _w4_tensor(view)
            for tensor_id, view in parsed.tensors.items()
            if view.storage == STORAGE_INT4_GROUPWISE
        }
        self.vectors = {
            tensor_id: _fp32_vector(view)
            for tensor_id, view in parsed.tensors.items()
            if view.storage == STORAGE_FP32
        }
        self.key_cache = torch.zeros(
            self.n_layer,
            self.block_size,
            self.n_embd,
            dtype=torch.float32,
        )
        self.value_cache = torch.zeros_like(self.key_cache)
        self.position = 0
        self.training = False

    @classmethod
    def from_file(cls, path: Path) -> W4A8Reference:
        return cls(parse_model_file(path.read_bytes()))

    @staticmethod
    def _block_id(layer: int, slot: int) -> int:
        return (
            BLOCK_TENSOR_ID_BASE
            + layer * BLOCK_TENSOR_ID_STRIDE
            + slot
        )

    def _embedding(self, tensor_id: int, indices: torch.Tensor) -> torch.Tensor:
        weight = self.matrices[tensor_id]
        values = weight.values[indices, : weight.columns].to(torch.float32)
        groups = (
            torch.arange(weight.columns, device=indices.device)
            // weight.group_size
        )
        scales = weight.scales.to(indices.device)[indices]
        return values.to(indices.device) * scales[..., groups]

    def _linear(self, tensor_id: int, inputs: torch.Tensor) -> torch.Tensor:
        return w4a8_matvec(inputs, self.matrices[tensor_id])

    def __call__(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if token_ids.ndim != 2 or token_ids.dtype != torch.long:
            raise ValueError("token_ids must be a two-dimensional long tensor")
        _, sequence_length = token_ids.shape
        if sequence_length == 0 or sequence_length > self.block_size:
            raise ValueError("token sequence length is outside the context")
        positions = torch.arange(sequence_length, device=token_ids.device)
        hidden = (
            self._embedding(TOKEN_EMBEDDING_TENSOR_ID, token_ids)
            + self._embedding(POSITION_EMBEDDING_TENSOR_ID, positions)
        )
        for layer in range(self.n_layer):
            attention_norm = self.vectors[
                self._block_id(layer, 0)
            ].to(hidden.device)
            normalized = F.layer_norm(
                hidden,
                (self.n_embd,),
                attention_norm,
                None,
                1.0e-5,
            )
            qkv = self._linear(self._block_id(layer, 1), normalized)
            query, key, value = qkv.split(self.n_embd, dim=-1)

            def split_heads(tensor: torch.Tensor) -> torch.Tensor:
                return tensor.view(
                    tensor.shape[0],
                    sequence_length,
                    self.n_head,
                    self.head_dim,
                ).transpose(1, 2)

            query = split_heads(query)
            key = split_heads(key)
            value = split_heads(value)
            scores = query @ key.transpose(-2, -1)
            scores *= 1.0 / math.sqrt(self.head_dim)
            causal = torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=hidden.device,
            ).tril()
            scores = scores.masked_fill(~causal, float("-inf"))
            context = torch.softmax(scores, dim=-1) @ value
            context = (
                context.transpose(1, 2)
                .contiguous()
                .view(hidden.shape[0], sequence_length, self.n_embd)
            )
            hidden = hidden + self._linear(
                self._block_id(layer, 2),
                context,
            )
            mlp_norm = self.vectors[
                self._block_id(layer, 3)
            ].to(hidden.device)
            normalized = F.layer_norm(
                hidden,
                (self.n_embd,),
                mlp_norm,
                None,
                1.0e-5,
            )
            mlp = self._linear(self._block_id(layer, 4), normalized)
            mlp = F.gelu(mlp, approximate="tanh")
            hidden = hidden + self._linear(
                self._block_id(layer, 5),
                mlp,
            )
        final_weight = self.vectors[FINAL_NORM_TENSOR_ID].to(hidden.device)
        hidden = F.layer_norm(
            hidden,
            (self.n_embd,),
            final_weight,
            None,
            1.0e-5,
        )
        logits = self._linear(TOKEN_EMBEDDING_TENSOR_ID, hidden)
        loss = None
        if targets is not None:
            if targets.shape != token_ids.shape:
                raise ValueError("targets must match token_ids")
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
            )
        return logits, loss

    def to(self, device: str | torch.device) -> W4A8Reference:
        target = torch.device(device)
        self.matrices = {
            tensor_id: W4Tensor(
                values=weight.values.to(target),
                scales=weight.scales.to(target),
                rows=weight.rows,
                columns=weight.columns,
                padded_columns=weight.padded_columns,
                group_size=weight.group_size,
            )
            for tensor_id, weight in self.matrices.items()
        }
        self.vectors = {
            tensor_id: value.to(target)
            for tensor_id, value in self.vectors.items()
        }
        self.key_cache = self.key_cache.to(target)
        self.value_cache = self.value_cache.to(target)
        return self

    def eval(self) -> W4A8Reference:
        self.training = False
        return self

    def train(self, mode: bool = True) -> W4A8Reference:
        self.training = bool(mode)
        return self

    def reset(self) -> None:
        self.key_cache.zero_()
        self.value_cache.zero_()
        self.position = 0

    def forward_token(self, token_id: int) -> torch.Tensor:
        if not 0 <= token_id < self.vocab_size:
            raise ValueError("token ID is outside the vocabulary")
        if self.position >= self.block_size:
            raise ValueError("runtime context is full")
        token = torch.tensor(token_id, dtype=torch.long)
        position = torch.tensor(self.position, dtype=torch.long)
        hidden = (
            self._embedding(TOKEN_EMBEDDING_TENSOR_ID, token)
            + self._embedding(POSITION_EMBEDDING_TENSOR_ID, position)
        )
        for layer in range(self.n_layer):
            normalized = F.layer_norm(
                hidden,
                (self.n_embd,),
                self.vectors[self._block_id(layer, 0)],
                None,
                1.0e-5,
            )
            qkv = self._linear(self._block_id(layer, 1), normalized)
            query, key, value = qkv.split(self.n_embd)
            self.key_cache[layer, self.position].copy_(key)
            self.value_cache[layer, self.position].copy_(value)
            context_parts = []
            for head in range(self.n_head):
                begin = head * self.head_dim
                end = begin + self.head_dim
                keys = self.key_cache[
                    layer,
                    : self.position + 1,
                    begin:end,
                ]
                values = self.value_cache[
                    layer,
                    : self.position + 1,
                    begin:end,
                ]
                scores = (
                    keys @ query[begin:end]
                    * (1.0 / math.sqrt(self.head_dim))
                )
                context_parts.append(torch.softmax(scores, dim=0) @ values)
            context = torch.cat(context_parts)
            hidden = hidden + self._linear(
                self._block_id(layer, 2),
                context,
            )
            normalized = F.layer_norm(
                hidden,
                (self.n_embd,),
                self.vectors[self._block_id(layer, 3)],
                None,
                1.0e-5,
            )
            mlp = self._linear(self._block_id(layer, 4), normalized)
            mlp = F.gelu(mlp, approximate="tanh")
            hidden = hidden + self._linear(
                self._block_id(layer, 5),
                mlp,
            )
        hidden = F.layer_norm(
            hidden,
            (self.n_embd,),
            self.vectors[FINAL_NORM_TENSOR_ID],
            None,
            1.0e-5,
        )
        logits = self._linear(TOKEN_EMBEDDING_TENSOR_ID, hidden)
        self.position += 1
        return logits


def w4a8_greedy_probe(
    model: W4A8Reference,
    prompt_tokens: Sequence[int],
    generate_count: int,
) -> ProbeResult:
    """Run the incremental Python reference used to align the C kernel."""

    if not prompt_tokens:
        raise ValueError("prompt_tokens must not be empty")
    if len(prompt_tokens) + generate_count > model.block_size:
        raise ValueError("prompt plus generation exceeds block_size")
    model.reset()
    logits = torch.empty(0)
    for token in prompt_tokens:
        logits = model.forward_token(int(token))
    prompt_logits = (
        logits.detach().cpu().numpy().astype(np.float32, copy=True)
    )
    generated: list[int] = []
    for _ in range(generate_count):
        next_token = int(torch.argmax(logits).item())
        generated.append(next_token)
        logits = model.forward_token(next_token)
    return ProbeResult(
        logits=prompt_logits,
        generated_tokens=tuple(generated),
        metrics={
            "forward_tokens": len(prompt_tokens) + generate_count,
            "integer_accumulator": "int32",
            "matrix_storage": "packed_int4_no_fp32_expansion",
            "logits_checkpoint": "after_prompt",
        },
    )
