"""Small learning models used before the full Transformer."""

from nanogpt_nspire.models.causal_attention_lm import (
    SingleHeadCausalLanguageModel,
    SingleHeadCausalSelfAttention,
)
from nanogpt_nspire.models.embedding_lm import EmbeddingLanguageModel, ModelInputError

__all__ = [
    "EmbeddingLanguageModel",
    "ModelInputError",
    "SingleHeadCausalLanguageModel",
    "SingleHeadCausalSelfAttention",
]
