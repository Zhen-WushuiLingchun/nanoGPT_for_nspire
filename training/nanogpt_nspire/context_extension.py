"""Prefix-preserving 256-to-512 context continuation for Lesson 14."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from nanogpt_nspire.base_train import _atomic_torch_save
from nanogpt_nspire.byte_tokenizer import VOCAB_SIZE
from nanogpt_nspire.models.direct_small_gpt import (
    DirectSmallConfig,
    DirectSmallGPT,
)
from nanogpt_nspire.stage_train import (
    StageTrainingConfig,
    load_parent_checkpoint,
    run_stage_training,
)
from nanogpt_nspire.training_support import sha256_file


SOURCE_ROUTE = "Math-Physics-CPT"
CONTEXT512_INIT_ROUTE = "Math-Physics-CPT-Context512-Init"
CONTEXT512_CPT_ROUTE = "Math-Physics-CPT-Context512"
CONTEXT512_HYBRID_ROUTE = "Hybrid-Control-SFT-Context512"
POSITION_KEY = "position_embedding.weight"
BASE_ARCHITECTURE = {
    "vocab_size": VOCAB_SIZE,
    "block_size": 256,
    "n_layer": 6,
    "n_head": 6,
    "n_embd": 384,
    "mlp_ratio": 4,
    "dropout": 0.1,
    "bias": False,
    "tie_embeddings": True,
}
EXTENDED_ARCHITECTURE = {
    **BASE_ARCHITECTURE,
    "block_size": 512,
}
_ARCHITECTURE_FIELDS = frozenset(EXTENDED_ARCHITECTURE)


def _reject_architecture_overrides(overrides: dict[str, object]) -> None:
    changed = sorted(_ARCHITECTURE_FIELDS & frozenset(overrides))
    if changed:
        raise ValueError(
            "Lesson 14 context architecture is frozen; remove overrides: "
            + ", ".join(changed)
        )


def kv_cache_bytes(
    *,
    block_size: int,
    n_layer: int = 6,
    n_embd: int = 384,
    element_bytes: int = 4,
) -> int:
    """Return incremental K+V cache bytes for the current C runtime layout."""

    for name, value in (
        ("block_size", block_size),
        ("n_layer", n_layer),
        ("n_embd", n_embd),
        ("element_bytes", element_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    return 2 * n_layer * block_size * n_embd * element_bytes


def create_context512_initial_checkpoint(
    *,
    parent_checkpoint: str | Path,
    parent_checkpoint_sha256: str,
    output_path: str | Path,
    source_commit: str,
) -> dict[str, object]:
    """Copy every learned tensor and repeat positions 0..255 into 256..511."""

    destination = Path(output_path)
    if destination.exists():
        raise ValueError(f"output already exists: {destination}")
    if not isinstance(source_commit, str) or not source_commit:
        raise ValueError("source_commit must be non-empty")
    base_config = DirectSmallConfig(**BASE_ARCHITECTURE)
    extended_config = DirectSmallConfig(**EXTENDED_ARCHITECTURE)
    parent = load_parent_checkpoint(
        parent_checkpoint,
        expected_sha256=parent_checkpoint_sha256,
        expected_route=SOURCE_ROUTE,
        expected_model_config=base_config,
    )
    source_state = parent["model_state_dict"]
    assert isinstance(source_state, dict)
    model = DirectSmallGPT(extended_config)
    target_state = model.state_dict()
    for name, target in target_state.items():
        source = source_state[name]
        assert isinstance(source, torch.Tensor)
        if name == POSITION_KEY:
            if source.shape != (256, 384) or target.shape != (512, 384):
                raise ValueError("position embedding shapes are unexpected")
            target[:256].copy_(source)
            target[256:].copy_(source)
        else:
            if source.shape != target.shape or source.dtype != target.dtype:
                raise ValueError(
                    f"non-position tensor {name} changed shape or dtype"
                )
            target.copy_(source)
    model.load_state_dict(target_state, strict=True)
    checkpoint = {
        "initialization": {
            "method": "prefix_preserving_repeat",
            "new_position_rows": 256,
            "preserved_position_rows": 256,
        },
        "model_config": asdict(extended_config),
        "model_state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        },
        "parent_checkpoint_sha256": parent["sha256"],
        "parent_route": parent["route"],
        "route": CONTEXT512_INIT_ROUTE,
        "schema_version": 1,
        "source_commit": source_commit,
        "tokenizer": {
            "kind": "byte_plus_fixed_special_tokens",
            "vocab_size": VOCAB_SIZE,
        },
    }
    _atomic_torch_save(checkpoint, destination)
    return {
        "checkpoint_bytes": destination.stat().st_size,
        "checkpoint_path": str(destination),
        "checkpoint_sha256": sha256_file(destination),
        "kv_cache_bytes_256": kv_cache_bytes(block_size=256),
        "kv_cache_bytes_512": kv_cache_bytes(block_size=512),
        "new_parameters": 256 * 384,
        "route": CONTEXT512_INIT_ROUTE,
    }


def frozen_context512_cpt_config(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    parent_checkpoint: str | Path,
    parent_checkpoint_sha256: str,
    source_commit: str,
    **overrides: object,
) -> StageTrainingConfig:
    """Bounded continued-pretraining pilot at 512 tokens."""

    _reject_architecture_overrides(overrides)
    defaults: dict[str, object] = {
        **{
            key: value
            for key, value in EXTENDED_ARCHITECTURE.items()
            if key != "vocab_size"
        },
        "device": "auto",
        "steps": 250,
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 0.0001,
        "min_learning_rate": 0.00001,
        "warmup_steps": 25,
        "eval_interval": 50,
        "eval_batches": 20,
        "log_interval": 25,
        "overfit_gate_steps": 20,
        "route_override": CONTEXT512_CPT_ROUTE,
        "checkpoint_filename_override": "context512_cpt.pt",
        "required_parent_route_override": CONTEXT512_INIT_ROUTE,
    }
    defaults.update(overrides)
    config = StageTrainingConfig(
        stage="cpt",
        data_dir=Path(data_dir),
        output_dir=Path(output_dir),
        parent_checkpoint=Path(parent_checkpoint),
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        expected_parent_route=CONTEXT512_INIT_ROUTE,
        source_commit=source_commit,
        **defaults,
    )
    config.validate()
    return config


def frozen_context512_hybrid_config(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    parent_checkpoint: str | Path,
    parent_checkpoint_sha256: str,
    source_commit: str,
    **overrides: object,
) -> StageTrainingConfig:
    """Hybrid-control SFT on the independently extended parent."""

    _reject_architecture_overrides(overrides)
    defaults: dict[str, object] = {
        **{
            key: value
            for key, value in EXTENDED_ARCHITECTURE.items()
            if key != "vocab_size"
        },
        "device": "auto",
        "steps": 1000,
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 0.0001,
        "min_learning_rate": 0.00001,
        "warmup_steps": 50,
        "eval_interval": 100,
        "eval_batches": 20,
        "log_interval": 25,
        "overfit_gate_steps": 20,
        "route_override": CONTEXT512_HYBRID_ROUTE,
        "checkpoint_filename_override": "hybrid_control_sft_context512.pt",
        "required_parent_route_override": CONTEXT512_CPT_ROUTE,
    }
    defaults.update(overrides)
    config = StageTrainingConfig(
        stage="sft",
        data_dir=Path(data_dir),
        output_dir=Path(output_dir),
        parent_checkpoint=Path(parent_checkpoint),
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        expected_parent_route=CONTEXT512_CPT_ROUTE,
        source_commit=source_commit,
        **defaults,
    )
    config.validate()
    return config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("initialize")
    initialize.add_argument("--parent-checkpoint", type=Path, required=True)
    initialize.add_argument("--parent-checkpoint-sha256", required=True)
    initialize.add_argument("--output", type=Path, required=True)
    initialize.add_argument("--source-commit", required=True)
    for command in ("cpt", "hybrid"):
        stage = subparsers.add_parser(command)
        stage.add_argument("--data-dir", type=Path, required=True)
        stage.add_argument("--output-dir", type=Path, required=True)
        stage.add_argument("--parent-checkpoint", type=Path, required=True)
        stage.add_argument("--parent-checkpoint-sha256", required=True)
        stage.add_argument("--source-commit", required=True)
        stage.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    if arguments.command == "initialize":
        result = create_context512_initial_checkpoint(
            parent_checkpoint=arguments.parent_checkpoint,
            parent_checkpoint_sha256=arguments.parent_checkpoint_sha256,
            output_path=arguments.output,
            source_commit=arguments.source_commit,
        )
    else:
        builder = (
            frozen_context512_cpt_config
            if arguments.command == "cpt"
            else frozen_context512_hybrid_config
        )
        config = builder(
            data_dir=arguments.data_dir,
            output_dir=arguments.output_dir,
            parent_checkpoint=arguments.parent_checkpoint,
            parent_checkpoint_sha256=arguments.parent_checkpoint_sha256,
            source_commit=arguments.source_commit,
            device=arguments.device,
        )
        result = run_stage_training(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
