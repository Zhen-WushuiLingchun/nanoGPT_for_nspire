from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from nanogpt_nspire.lesson13_distill_train import (
    COMBINED_ROUTE,
    LOCAL_LOGIT_ROUTE,
    SEQUENCE_ROUTE,
    frozen_lesson13_student_config,
    load_local_teacher_checkpoint,
    local_teacher_model_config,
)
from nanogpt_nspire.local_teacher_train import LOCAL_TEACHER_SFT_ROUTE
from nanogpt_nspire.models.direct_small_gpt import (
    DirectSmallConfig,
    DirectSmallGPT,
)
from nanogpt_nspire.training_support import sha256_file


def _student_config(tmp_path: Path, kind: str):
    return frozen_lesson13_student_config(
        kind=kind,
        data_dir=tmp_path / f"{kind}-data",
        output_dir=tmp_path / f"{kind}-out",
        parent_checkpoint=tmp_path / "cpt.pt",
        parent_checkpoint_sha256="a" * 64,
        source_commit="source",
    )


def test_three_student_routes_keep_fair_training_contract(
    tmp_path: Path,
) -> None:
    configs = {
        kind: _student_config(tmp_path, kind)
        for kind in ("sequence", "logit", "combined")
    }

    assert configs["sequence"].route == SEQUENCE_ROUTE
    assert configs["logit"].route == LOCAL_LOGIT_ROUTE
    assert configs["combined"].route == COMBINED_ROUTE
    assert {
        config.effective_batch_tokens * config.steps
        for config in configs.values()
    } == {4_096_000}
    assert {
        (
            config.n_layer,
            config.n_head,
            config.n_embd,
            config.block_size,
            config.vocab_size,
        )
        for config in configs.values()
    } == {(6, 6, 384, 256, 264)}
    assert {
        (
            config.learning_rate,
            config.min_learning_rate,
            config.warmup_steps,
            config.eval_interval,
            config.eval_batches,
        )
        for config in configs.values()
    } == {(0.0001, 0.00001, 50, 100, 20)}
    assert {
        config.expected_parent_route for config in configs.values()
    } == {"Math-Physics-CPT"}


def test_route_specific_fields_are_the_only_non_path_config_difference(
    tmp_path: Path,
) -> None:
    sequence = asdict(_student_config(tmp_path, "sequence"))
    logit = asdict(_student_config(tmp_path, "logit"))
    ignored = {
        "data_dir",
        "output_dir",
        "route_override",
        "checkpoint_filename_override",
    }

    assert {
        key: value for key, value in sequence.items() if key not in ignored
    } == {
        key: value for key, value in logit.items() if key not in ignored
    }


def test_local_teacher_model_contract_is_shared_tokenizer_and_larger() -> None:
    config = local_teacher_model_config()
    teacher = DirectSmallGPT(config)

    assert config == DirectSmallConfig(
        vocab_size=264,
        block_size=256,
        n_layer=12,
        n_head=10,
        n_embd=640,
        mlp_ratio=4,
        dropout=0.1,
        bias=False,
        tie_embeddings=True,
    )
    assert teacher.parameter_count == 59_331_200


def test_local_teacher_loader_checks_hash_route_tokenizer_and_shapes(
    tmp_path: Path,
) -> None:
    tiny = DirectSmallConfig(
        vocab_size=264,
        block_size=16,
        n_layer=1,
        n_head=2,
        n_embd=16,
        mlp_ratio=2,
        dropout=0.0,
        bias=False,
        tie_embeddings=True,
    )
    model = DirectSmallGPT(tiny)
    path = tmp_path / "teacher.pt"
    torch.save(
        {
            "model_config": asdict(tiny),
            "model_state_dict": model.state_dict(),
            "route": LOCAL_TEACHER_SFT_ROUTE,
            "schema_version": 1,
            "selected_validation_loss": 0.5,
            "source_commit": "teacher-source",
            "tokenizer": {
                "kind": "byte_plus_fixed_special_tokens",
                "vocab_size": 264,
            },
        },
        path,
    )
    digest = sha256_file(path)

    loaded, provenance = load_local_teacher_checkpoint(
        path,
        expected_sha256=digest,
        expected_model_config=tiny,
    )

    assert loaded.config == tiny
    assert provenance["checkpoint_sha256"] == digest
    assert provenance["route"] == LOCAL_TEACHER_SFT_ROUTE

    with pytest.raises(ValueError, match="SHA-256"):
        load_local_teacher_checkpoint(
            path,
            expected_sha256="0" * 64,
            expected_model_config=tiny,
        )
