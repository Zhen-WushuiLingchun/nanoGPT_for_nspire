"""A transparent single-head causal self-attention language model."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from nanogpt_nspire.models.embedding_lm import ModelInputError


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class SingleHeadCausalSelfAttention(nn.Module):
    """Manual scaled dot-product attention with a strict causal mask."""

    def __init__(self, *, embedding_dim: int, block_size: int) -> None:
        super().__init__()
        self.embedding_dim = _positive_integer(embedding_dim, "embedding_dim")
        self.block_size = _positive_integer(block_size, "block_size")
        self.query = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.key = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.value = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.output = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(
                    self.block_size,
                    self.block_size,
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
        """Mix visible values and optionally return the `(B, T, T)` weights."""

        if inputs.ndim != 3:
            raise ModelInputError("attention inputs must have shape (B, T, C)")
        if inputs.shape[-1] != self.embedding_dim:
            raise ModelInputError(
                f"attention input width must be {self.embedding_dim}"
            )
        sequence_length = inputs.shape[1]
        if sequence_length == 0:
            raise ModelInputError("attention sequence must not be empty")
        if sequence_length > self.block_size:
            raise ModelInputError(
                f"attention sequence length exceeds block_size {self.block_size}"
            )

        queries = self.query(inputs)
        keys = self.key(inputs)
        values = self.value(inputs)
        scores = queries @ keys.transpose(-2, -1)
        scores = scores * (1.0 / math.sqrt(self.embedding_dim))
        visible = self.causal_mask[:sequence_length, :sequence_length]
        scores = scores.masked_fill(~visible, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        context = weights @ values
        output = self.output(context)
        if return_weights:
            return output, weights
        return output


class SingleHeadCausalLanguageModel(nn.Module):
    """Use one causal attention head to predict the next token from context."""

    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        block_size: int,
    ) -> None:
        super().__init__()
        self.vocab_size = _positive_integer(vocab_size, "vocab_size")
        self.embedding_dim = _positive_integer(embedding_dim, "embedding_dim")
        self.block_size = _positive_integer(block_size, "block_size")
        self.token_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.position_embedding = nn.Embedding(
            self.block_size,
            self.embedding_dim,
        )
        self.attention = SingleHeadCausalSelfAttention(
            embedding_dim=self.embedding_dim,
            block_size=self.block_size,
        )
        self.lm_head = nn.Linear(
            self.embedding_dim,
            self.vocab_size,
            bias=False,
        )

    @property
    def parameter_count(self) -> int:
        """Return the number of trainable scalar parameters."""

        return sum(parameter.numel() for parameter in self.parameters())

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
        """Return context-dependent `(B, T, V)` logits and optional loss."""

        self._validate_tokens(token_ids, "token_ids")
        if targets is not None:
            self._validate_tokens(targets, "targets")
            if targets.shape != token_ids.shape:
                raise ModelInputError("targets must have the same shape as token_ids")

        sequence_length = token_ids.shape[1]
        positions = torch.arange(sequence_length, device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        attention_output = self.attention(hidden)
        assert isinstance(attention_output, torch.Tensor)
        hidden = hidden + attention_output
        logits = self.lm_head(hidden)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
            )
        return logits, loss
