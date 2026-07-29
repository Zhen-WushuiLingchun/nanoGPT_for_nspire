from __future__ import annotations

import torch

from nanogpt_nspire.deployment_export import build_deployment_export
from nanogpt_nspire.export_format import (
    MODEL_STORAGE_W4A8,
    POSITION_ALIBI,
    TOKENIZER_BYTE_SPECIAL,
    parse_model_file,
)
from nanogpt_nspire.models.efficient_long_context_gpt import (
    ALIBI_POSITIONS,
    EfficientLongContextConfig,
    EfficientLongContextGPT,
)
from nanogpt_nspire.w4a8_reference import W4A8Reference


def test_efficient_byte_gqa_checkpoint_exports_to_w4a8() -> None:
    torch.manual_seed(7)
    config = EfficientLongContextConfig(
        vocab_size=264,
        block_size=8,
        n_layer=1,
        n_head=2,
        n_kv_head=1,
        n_embd=4,
        mlp_ratio=2,
        dropout=0.0,
        bias=False,
        tie_embeddings=True,
        position_mode=ALIBI_POSITIONS,
    )
    model = EfficientLongContextGPT(config)
    checkpoint = {
        "architecture": "efficient_long_context_gpt",
        "model_config": {
            **config.__dict__,
        },
        "model_state_dict": model.state_dict(),
        "route": "test-efficient",
        "schema_version": 1,
        "source_commit": "test",
        "tokenizer": {
            "kind": "byte_plus_fixed_special_tokens",
            "vocab_size": 264,
        },
    }

    data, manifest = build_deployment_export(
        checkpoint,
        group_size=4,
    )
    parsed = parse_model_file(data)
    reference = W4A8Reference(parsed)
    logits = reference.forward_token(256)

    assert parsed.spec.model_storage == MODEL_STORAGE_W4A8
    assert parsed.spec.n_kv_head == 1
    assert parsed.spec.position_mode == POSITION_ALIBI
    assert parsed.spec.tokenizer_type == TOKENIZER_BYTE_SPECIAL
    assert len(parsed.tensors) == 8
    assert logits.shape == (264,)
    assert manifest["route"] == "test-efficient"
