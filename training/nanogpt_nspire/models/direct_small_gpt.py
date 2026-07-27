"""Reference implementation of the budgeted Direct-Small GPT."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from nanogpt_nspire.models.embedding_lm import ModelInputError


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class DirectSmallConfig:
    """Architecture shared by Direct-Small and the future distilled student."""

    vocab_size: int = 65
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 5
    n_embd: int = 160
    mlp_ratio: int = 4
    dropout: float = 0.1
    bias: bool = False
    tie_embeddings: bool = True

    def validate(self) -> None:
        for name in (
            "vocab_size",
            "block_size",
            "n_layer",
            "n_head",
            "n_embd",
            "mlp_ratio",
        ):
            _positive_integer(getattr(self, name), name)
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
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


class OptionalBiasLayerNorm(nn.Module):
    """LayerNorm with independently selectable affine bias."""

    def __init__(self, width: int, *, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width)) if bias else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            inputs,
            self.weight.shape,
            self.weight,
            self.bias,
            1e-5,
        )


class MultiHeadCausalSelfAttention(nn.Module):
    """Manual fused-QKV multi-head attention with a derived causal mask."""

    def __init__(self, config: DirectSmallConfig) -> None:
        super().__init__()
        config.validate()
        self.block_size = config.block_size
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(
            config.n_embd,
            3 * config.n_embd,
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

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3:
            raise ModelInputError("attention inputs must have shape (B, T, C)")
        batch_size, sequence_length, width = inputs.shape
        if width != self.n_embd:
            raise ModelInputError(f"attention input width must be {self.n_embd}")
        if sequence_length == 0:
            raise ModelInputError("attention sequence must not be empty")
        if sequence_length > self.block_size:
            raise ModelInputError(
                f"attention sequence length exceeds block_size {self.block_size}"
            )

        query, key, value = self.qkv(inputs).split(self.n_embd, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                batch_size,
                sequence_length,
                self.n_head,
                self.head_dim,
            ).transpose(1, 2)

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)
        scores = query @ key.transpose(-2, -1)
        scores = scores * (1.0 / math.sqrt(self.head_dim))
        visible = self.causal_mask[:sequence_length, :sequence_length]
        scores = scores.masked_fill(~visible, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        dropped_weights = self.attention_dropout(weights)
        context = dropped_weights @ value
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, self.n_embd)
        )
        output = self.residual_dropout(self.output(context))
        if return_weights:
            return output, weights
        return output


class FeedForward(nn.Module):
    """Four-times expansion MLP with a C-portable tanh GELU approximation."""

    def __init__(self, config: DirectSmallConfig) -> None:
        super().__init__()
        hidden_width = config.mlp_ratio * config.n_embd
        self.input = nn.Linear(
            config.n_embd,
            hidden_width,
            bias=config.bias,
        )
        self.activation = nn.GELU(approximate="tanh")
        self.output = nn.Linear(
            hidden_width,
            config.n_embd,
            bias=config.bias,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.dropout(
            self.output(
                self.activation(
                    self.input(inputs)
                )
            )
        )


class TransformerBlock(nn.Module):
    """Pre-norm attention and MLP residual branches."""

    def __init__(self, config: DirectSmallConfig) -> None:
        super().__init__()
        self.attention_norm = OptionalBiasLayerNorm(
            config.n_embd,
            bias=config.bias,
        )
        self.attention = MultiHeadCausalSelfAttention(config)
        self.mlp_norm = OptionalBiasLayerNorm(
            config.n_embd,
            bias=config.bias,
        )
        self.mlp = FeedForward(config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs + self.attention(self.attention_norm(inputs))
        assert isinstance(hidden, torch.Tensor)
        return hidden + self.mlp(self.mlp_norm(hidden))


class DirectSmallGPT(nn.Module):
    """Complete small decoder-only GPT constrained by the deployment budget."""

    def __init__(self, config: DirectSmallConfig | None = None) -> None:
        super().__init__()
        self.config = config or DirectSmallConfig()
        self.config.validate()
        self.vocab_size = self.config.vocab_size
        self.block_size = self.config.block_size
        self.token_embedding = nn.Embedding(
            self.config.vocab_size,
            self.config.n_embd,
        )
        self.position_embedding = nn.Embedding(
            self.config.block_size,
            self.config.n_embd,
        )
        self.embedding_dropout = nn.Dropout(self.config.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(self.config)
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
                "parameter formula does not match constructed Direct-Small model"
            )

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def expected_parameter_count(self) -> int:
        config = self.config
        embeddings = (
            config.vocab_size * config.n_embd
            + config.block_size * config.n_embd
        )
        matrix_weights_per_block = (
            4 + 2 * config.mlp_ratio
        ) * config.n_embd**2
        linear_biases_per_block = 0
        if config.bias:
            linear_biases_per_block = (
                5 + config.mlp_ratio
            ) * config.n_embd
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
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def raw_fp32_parameter_bytes(self) -> int:
        return self.parameter_count * 4

    def _validate_tokens(self, token_ids: torch.Tensor, name: str) -> None:
        if token_ids.ndim != 2:
            raise ModelInputError(f"{name} must be a two-dimensional (B, T) tensor")
        if token_ids.dtype != torch.long:
            raise ModelInputError(f"{name} must have torch.long dtype")
        if token_ids.numel() == 0:
            raise ModelInputError(f"{name} must contain at least one token")
        if token_ids.shape[1] > self.block_size:
            raise ModelInputError(
                f"{name} sequence length exceeds block_size {self.block_size}"
            )
        minimum = int(token_ids.min().item())
        maximum = int(token_ids.max().item())
        if minimum < 0 or maximum >= self.vocab_size:
            raise ModelInputError(
                f"{name} contains a token outside [0, {self.vocab_size})"
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
                raise ModelInputError("targets must have the same shape as token_ids")

        sequence_length = token_ids.shape[1]
        positions = torch.arange(sequence_length, device=token_ids.device)
        hidden = self.embedding_dropout(
            self.token_embedding(token_ids)
            + self.position_embedding(positions)
        )
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
