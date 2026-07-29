"""C-oriented grouped-query GPT with learned or ALiBi positions."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from nanogpt_nspire.models.direct_small_gpt import (
    DirectSmallGPT,
    FeedForward,
    OptionalBiasLayerNorm,
)
from nanogpt_nspire.models.embedding_lm import ModelInputError


LEARNED_POSITIONS = "learned"
ALIBI_POSITIONS = "alibi"
POSITION_MODES = frozenset({LEARNED_POSITIONS, ALIBI_POSITIONS})


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def alibi_slopes(n_head: int) -> tuple[float, ...]:
    """Return the deterministic slopes used by the reference ALiBi code."""

    _positive_integer(n_head, "n_head")

    def power_of_two(count: int) -> tuple[float, ...]:
        start = 2.0 ** (-2.0 ** -(math.log2(count) - 3.0))
        return tuple(start ** (index + 1) for index in range(count))

    if n_head & (n_head - 1) == 0:
        return power_of_two(n_head)
    closest = 2 ** math.floor(math.log2(n_head))
    base = power_of_two(closest)
    expanded = power_of_two(2 * closest)
    extras = expanded[0::2][: n_head - closest]
    return (*base, *extras)


@dataclass(frozen=True)
class EfficientLongContextConfig:
    """Architecture contract for the Lesson 15 GQA variants."""

    vocab_size: int = 264
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 6
    n_kv_head: int = 2
    n_embd: int = 384
    mlp_ratio: int = 4
    dropout: float = 0.1
    bias: bool = False
    tie_embeddings: bool = True
    position_mode: str = LEARNED_POSITIONS

    def validate(self) -> None:
        for name in (
            "vocab_size",
            "block_size",
            "n_layer",
            "n_head",
            "n_kv_head",
            "n_embd",
            "mlp_ratio",
        ):
            _positive_integer(getattr(self, name), name)
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not math.isfinite(self.dropout)
            or not 0.0 <= self.dropout < 1.0
        ):
            raise ValueError("dropout must be finite and in [0, 1)")
        if not isinstance(self.bias, bool):
            raise ValueError("bias must be boolean")
        if not isinstance(self.tie_embeddings, bool):
            raise ValueError("tie_embeddings must be boolean")
        if self.position_mode not in POSITION_MODES:
            raise ValueError(
                "position_mode must be 'learned' or 'alibi'"
            )


class GroupedQueryCausalSelfAttention(nn.Module):
    """Fused Q/K/V projection with fewer K/V heads than query heads."""

    def __init__(self, config: EfficientLongContextConfig) -> None:
        super().__init__()
        config.validate()
        self.block_size = config.block_size
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.kv_width = config.n_kv_head * self.head_dim
        self.query_heads_per_kv = config.n_head // config.n_kv_head
        self.position_mode = config.position_mode
        self.qkv = nn.Linear(
            config.n_embd,
            config.n_embd + 2 * self.kv_width,
            bias=config.bias,
        )
        self.output = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=config.bias,
        )
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(
                    config.block_size,
                    config.block_size,
                    dtype=torch.bool,
                )
            ),
            persistent=False,
        )
        slopes = torch.tensor(alibi_slopes(config.n_head)).view(
            1,
            config.n_head,
            1,
            1,
        )
        self.register_buffer(
            "alibi_slopes",
            slopes,
            persistent=False,
        )

    def _expand_kv(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, kv_heads, length, width = tensor.shape
        return (
            tensor[:, :, None, :, :]
            .expand(
                batch,
                kv_heads,
                self.query_heads_per_kv,
                length,
                width,
            )
            .reshape(batch, self.n_head, length, width)
        )

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3:
            raise ModelInputError(
                "attention inputs must have shape (B, T, C)"
            )
        batch_size, sequence_length, width = inputs.shape
        if width != self.n_embd:
            raise ModelInputError(
                f"attention input width must be {self.n_embd}"
            )
        if sequence_length == 0:
            raise ModelInputError("attention sequence must not be empty")
        if sequence_length > self.block_size:
            raise ModelInputError(
                "attention sequence length exceeds "
                f"block_size {self.block_size}"
            )
        query, key, value = self.qkv(inputs).split(
            (self.n_embd, self.kv_width, self.kv_width),
            dim=-1,
        )
        query = query.view(
            batch_size,
            sequence_length,
            self.n_head,
            self.head_dim,
        ).transpose(1, 2)

        def split_kv(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                batch_size,
                sequence_length,
                self.n_kv_head,
                self.head_dim,
            ).transpose(1, 2)

        key = self._expand_kv(split_kv(key))
        value = self._expand_kv(split_kv(value))
        scores = query @ key.transpose(-2, -1)
        scores = scores * (1.0 / math.sqrt(self.head_dim))
        if self.position_mode == ALIBI_POSITIONS:
            positions = torch.arange(
                sequence_length,
                device=inputs.device,
            )
            distances = (
                positions[:, None] - positions[None, :]
            ).clamp_min(0)
            scores = scores - (
                self.alibi_slopes
                * distances.to(dtype=scores.dtype).view(
                    1,
                    1,
                    sequence_length,
                    sequence_length,
                )
            )
        visible = self.causal_mask[:sequence_length, :sequence_length]
        scores = scores.masked_fill(~visible, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        context = self.attention_dropout(weights) @ value
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, self.n_embd)
        )
        output = self.residual_dropout(self.output(context))
        if return_weights:
            return output, weights
        return output


class EfficientTransformerBlock(nn.Module):
    def __init__(self, config: EfficientLongContextConfig) -> None:
        super().__init__()
        self.attention_norm = OptionalBiasLayerNorm(
            config.n_embd,
            bias=config.bias,
        )
        self.attention = GroupedQueryCausalSelfAttention(config)
        self.mlp_norm = OptionalBiasLayerNorm(
            config.n_embd,
            bias=config.bias,
        )
        self.mlp = FeedForward(config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs + self.attention(self.attention_norm(inputs))
        assert isinstance(hidden, torch.Tensor)
        return hidden + self.mlp(self.mlp_norm(hidden))


class EfficientLongContextGPT(DirectSmallGPT):
    """Grouped-query GPT kept as a DirectSmallGPT subtype for shared tools."""

    def __init__(
        self,
        config: EfficientLongContextConfig | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.config = config or EfficientLongContextConfig()
        self.config.validate()
        self.vocab_size = self.config.vocab_size
        self.block_size = self.config.block_size
        self.token_embedding = nn.Embedding(
            self.config.vocab_size,
            self.config.n_embd,
        )
        self.position_embedding = (
            nn.Embedding(
                self.config.block_size,
                self.config.n_embd,
            )
            if self.config.position_mode == LEARNED_POSITIONS
            else None
        )
        self.embedding_dropout = nn.Dropout(self.config.dropout)
        self.blocks = nn.ModuleList(
            [
                EfficientTransformerBlock(self.config)
                for _ in range(self.config.n_layer)
            ]
        )
        self.final_norm = OptionalBiasLayerNorm(
            self.config.n_embd,
            bias=self.config.bias,
        )
        self.lm_head = nn.Linear(
            self.config.n_embd,
            self.config.vocab_size,
            bias=False,
        )
        self.apply(self._initialize_module)
        if self.config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        residual_standard_deviation = 0.02 / math.sqrt(
            2 * self.config.n_layer
        )
        for block in self.blocks:
            nn.init.normal_(
                block.attention.output.weight,
                mean=0.0,
                std=residual_standard_deviation,
            )
            nn.init.normal_(
                block.mlp.output.weight,
                mean=0.0,
                std=residual_standard_deviation,
            )
        if self.parameter_count != self.expected_parameter_count:
            raise RuntimeError(
                "parameter formula does not match efficient model"
            )

    @property
    def expected_parameter_count(self) -> int:
        config = self.config
        head_dim = config.n_embd // config.n_head
        kv_width = config.n_kv_head * head_dim
        embeddings = config.vocab_size * config.n_embd
        if config.position_mode == LEARNED_POSITIONS:
            embeddings += config.block_size * config.n_embd
        matrix_weights_per_block = (
            config.n_embd * (config.n_embd + 2 * kv_width)
            + config.n_embd**2
            + 2 * config.mlp_ratio * config.n_embd**2
        )
        linear_biases_per_block = 0
        if config.bias:
            linear_biases_per_block = (
                (config.mlp_ratio + 3) * config.n_embd
                + 2 * kv_width
            )
        layer_norms_per_block = 2 * config.n_embd * (
            2 if config.bias else 1
        )
        final_norm = config.n_embd * (2 if config.bias else 1)
        untied_head = (
            0
            if config.tie_embeddings
            else config.vocab_size * config.n_embd
        )
        return (
            embeddings
            + config.n_layer
            * (
                matrix_weights_per_block
                + linear_biases_per_block
                + layer_norms_per_block
            )
            + final_norm
            + untied_head
        )

    @property
    def kv_cache_bytes_fp32(self) -> int:
        head_dim = self.config.n_embd // self.config.n_head
        return (
            2
            * self.config.n_layer
            * self.config.block_size
            * self.config.n_kv_head
            * head_dim
            * 4
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self._validate_tokens(token_ids, "token_ids")
        if targets is not None:
            self._validate_tokens(targets, "targets")
            if targets.shape != token_ids.shape:
                raise ModelInputError(
                    "targets must have the same shape as token_ids"
                )
        sequence_length = token_ids.shape[1]
        hidden = self.token_embedding(token_ids)
        if self.position_embedding is not None:
            positions = torch.arange(
                sequence_length,
                device=token_ids.device,
            )
            hidden = hidden + self.position_embedding(positions)
        hidden = self.embedding_dropout(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
            )
        return logits, loss
