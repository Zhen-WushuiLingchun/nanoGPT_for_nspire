from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from nanogpt_nspire.lesson14_train import (
    DIRECT_CONTROL_ROUTE,
    HYBRID_CONTROL_ROUTE,
    SHORT_COT_ROUTE,
    frozen_lesson14_student_config,
)


def _config(tmp_path: Path, kind: str):
    return frozen_lesson14_student_config(
        kind=kind,
        data_dir=tmp_path / f"{kind}-data",
        output_dir=tmp_path / f"{kind}-output",
        parent_checkpoint=tmp_path / "cpt.pt",
        parent_checkpoint_sha256="a" * 64,
        source_commit="source",
    )


def test_three_routes_keep_identical_training_contract(
    tmp_path: Path,
) -> None:
    configs = {
        kind: _config(tmp_path, kind)
        for kind in ("direct", "cot", "hybrid")
    }

    assert configs["direct"].route == DIRECT_CONTROL_ROUTE
    assert configs["cot"].route == SHORT_COT_ROUTE
    assert configs["hybrid"].route == HYBRID_CONTROL_ROUTE
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
            config.seed,
        )
        for config in configs.values()
    } == {(0.0001, 0.00001, 50, 100, 20, 20260728)}


def test_only_route_paths_and_checkpoint_name_differ(
    tmp_path: Path,
) -> None:
    direct = asdict(_config(tmp_path, "direct"))
    cot = asdict(_config(tmp_path, "cot"))
    ignored = {
        "data_dir",
        "output_dir",
        "route_override",
        "checkpoint_filename_override",
    }

    assert {
        key: value for key, value in direct.items() if key not in ignored
    } == {
        key: value for key, value in cot.items() if key not in ignored
    }


def test_architecture_and_unknown_route_cannot_be_overridden(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="architecture is frozen"):
        frozen_lesson14_student_config(
            kind="direct",
            data_dir=tmp_path / "data",
            output_dir=tmp_path / "output",
            parent_checkpoint=tmp_path / "cpt.pt",
            parent_checkpoint_sha256="a" * 64,
            source_commit="source",
            n_layer=4,
        )
    with pytest.raises(ValueError, match="kind must be"):
        _config(tmp_path, "unknown")
