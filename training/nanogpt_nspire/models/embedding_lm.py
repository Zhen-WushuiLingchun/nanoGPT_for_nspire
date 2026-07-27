"""A no-attention next-token baseline: embedding followed by a linear head."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ModelInputError(ValueError):
    """Raised when token or target tensors cannot be processed by the model."""


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class EmbeddingLanguageModel(nn.Module):
    """Predict the next token from only the current token."""

    def __init__(self, *, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.vocab_size = _positive_integer(vocab_size, "vocab_size")
        self.embedding_dim = _positive_integer(embedding_dim, "embedding_dim")
        self.token_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
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
        """Return `(B, T, V)` logits and optional next-token cross-entropy."""

        self._validate_tokens(token_ids, "token_ids")
        if targets is not None:
            self._validate_tokens(targets, "targets")
            if targets.shape != token_ids.shape:
                raise ModelInputError("targets must have the same shape as token_ids")

        embeddings = self.token_embedding(token_ids)
        logits = self.lm_head(embeddings)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
            )
        return logits, loss
