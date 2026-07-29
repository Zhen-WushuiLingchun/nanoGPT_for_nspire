"""Deterministic MHA-to-GQA checkpoint conversion for Lesson 15."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

import torch

from nanogpt_nspire.base_train import _atomic_torch_save
from nanogpt_nspire.byte_tokenizer import VOCAB_SIZE
from nanogpt_nspire.context_extension import (
    CONTEXT512_CPT_ROUTE,
    EXTENDED_ARCHITECTURE,
)
from nanogpt_nspire.lesson17_routes import ALL_LESSON17_ROUTES
from nanogpt_nspire.models.direct_small_gpt import (
    DirectSmallConfig,
)
from nanogpt_nspire.models.efficient_long_context_gpt import (
    ALIBI_POSITIONS,
    LEARNED_POSITIONS,
    POSITION_MODES,
    EfficientLongContextConfig,
    EfficientLongContextGPT,
)
from nanogpt_nspire.stage_train import load_parent_checkpoint
from nanogpt_nspire.training_support import sha256_file


GQA_LEARNED_INIT_ROUTE = "GQA-Learned-Context512-Init"
GQA_ALIBI_INIT_ROUTE = "GQA-ALiBi-Context512-Init"
GQA_LEARNED_CPT_ROUTE = "GQA-Learned-Context512-CPT"
GQA_ALIBI_CPT_ROUTE = "GQA-ALiBi-Context512-CPT"
GQA_LEARNED_SFT_ROUTE = "GQA-Learned-Hybrid-SFT-Context512"
GQA_ALIBI_SFT_ROUTE = "GQA-ALiBi-Hybrid-SFT-Context512"
GQA_ALIBI_SFT_V2_ROUTE = "GQA-ALiBi-SFT-v2-Context512"
INIT_ROUTES = {
    LEARNED_POSITIONS: GQA_LEARNED_INIT_ROUTE,
    ALIBI_POSITIONS: GQA_ALIBI_INIT_ROUTE,
}
CPT_ROUTES = {
    LEARNED_POSITIONS: GQA_LEARNED_CPT_ROUTE,
    ALIBI_POSITIONS: GQA_ALIBI_CPT_ROUTE,
}
SFT_ROUTES = {
    LEARNED_POSITIONS: GQA_LEARNED_SFT_ROUTE,
    ALIBI_POSITIONS: GQA_ALIBI_SFT_ROUTE,
}
ALL_EFFICIENT_ROUTES = frozenset(
    {
        *INIT_ROUTES.values(),
        *CPT_ROUTES.values(),
        *SFT_ROUTES.values(),
        GQA_ALIBI_SFT_V2_ROUTE,
        *ALL_LESSON17_ROUTES,
    }
)
ARCHITECTURE_NAME = "efficient_long_context_gpt"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def model_state_sha256(state: Mapping[str, object]) -> str:
    """Hash tensor names, dtypes, shapes, and values canonically."""

    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("model state must map names to tensors")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(
            json.dumps(list(tensor.shape), separators=(",", ":")).encode(
                "ascii"
            )
        )
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def lesson15_efficient_config(
    position_mode: str,
) -> EfficientLongContextConfig:
    if position_mode not in POSITION_MODES:
        raise ValueError("position_mode must be 'learned' or 'alibi'")
    return EfficientLongContextConfig(
        vocab_size=VOCAB_SIZE,
        block_size=512,
        n_layer=6,
        n_head=6,
        n_kv_head=2,
        n_embd=384,
        mlp_ratio=4,
        dropout=0.1,
        bias=False,
        tie_embeddings=True,
        position_mode=position_mode,
    )


def convert_mha_qkv_tensor(
    tensor: torch.Tensor,
    *,
    n_head: int,
    n_kv_head: int,
    n_embd: int,
) -> torch.Tensor:
    """Copy Q and average source K/V heads within each target group."""

    for name, value in (
        ("n_head", n_head),
        ("n_kv_head", n_kv_head),
        ("n_embd", n_embd),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if n_embd % n_head != 0 or n_head % n_kv_head != 0:
        raise ValueError("attention head dimensions are incompatible")
    if tensor.ndim not in (1, 2) or tensor.shape[0] != 3 * n_embd:
        raise ValueError("source QKV tensor has an unexpected shape")
    head_dim = n_embd // n_head
    group_size = n_head // n_kv_head
    query, key, value = tensor.split(n_embd, dim=0)
    trailing = tensor.shape[1:]

    def compress(part: torch.Tensor) -> torch.Tensor:
        return (
            part.reshape(
                n_kv_head,
                group_size,
                head_dim,
                *trailing,
            )
            .mean(dim=1)
            .reshape(n_kv_head * head_dim, *trailing)
        )

    return torch.cat((query, compress(key), compress(value)), dim=0)


def create_efficient_init_checkpoint(
    *,
    parent_checkpoint: str | Path,
    parent_checkpoint_sha256: str,
    output_path: str | Path,
    position_mode: str,
    source_commit: str,
    source_model_config: DirectSmallConfig | None = None,
    target_model_config: EfficientLongContextConfig | None = None,
) -> dict[str, object]:
    """Create a complete GQA checkpoint without leaving random target tensors."""

    destination = Path(output_path)
    if destination.exists():
        raise ValueError(f"output already exists: {destination}")
    if not isinstance(source_commit, str) or not source_commit:
        raise ValueError("source_commit must be non-empty")
    source_config = source_model_config or DirectSmallConfig(
        **EXTENDED_ARCHITECTURE
    )
    target_config = target_model_config or lesson15_efficient_config(
        position_mode
    )
    source_config.validate()
    target_config.validate()
    if target_config.position_mode != position_mode:
        raise ValueError("target config and position_mode disagree")
    shared_fields = (
        "vocab_size",
        "block_size",
        "n_layer",
        "n_head",
        "n_embd",
        "mlp_ratio",
        "dropout",
        "bias",
        "tie_embeddings",
    )
    if any(
        getattr(source_config, name) != getattr(target_config, name)
        for name in shared_fields
    ):
        raise ValueError("source and target shared architecture fields differ")
    parent = load_parent_checkpoint(
        parent_checkpoint,
        expected_sha256=parent_checkpoint_sha256,
        expected_route=CONTEXT512_CPT_ROUTE,
        expected_model_config=source_config,
    )
    source_state = parent["model_state_dict"]
    if not isinstance(source_state, Mapping):
        raise ValueError("source state is invalid")
    model = EfficientLongContextGPT(target_config)
    target_state = model.state_dict()
    conversion_records: list[dict[str, object]] = []
    for name, target in target_state.items():
        if name.endswith(".attention.qkv.weight") or name.endswith(
            ".attention.qkv.bias"
        ):
            source = source_state.get(name)
            if not isinstance(source, torch.Tensor):
                raise ValueError(f"source tensor {name} is missing")
            converted = convert_mha_qkv_tensor(
                source,
                n_head=target_config.n_head,
                n_kv_head=target_config.n_kv_head,
                n_embd=target_config.n_embd,
            )
            if converted.shape != target.shape or converted.dtype != target.dtype:
                raise ValueError(f"converted tensor {name} is incompatible")
            target.copy_(converted)
            conversion_records.append(
                {
                    "source_shape": list(source.shape),
                    "target_shape": list(target.shape),
                    "tensor": name,
                }
            )
            continue
        source = source_state.get(name)
        if (
            not isinstance(source, torch.Tensor)
            or source.shape != target.shape
            or source.dtype != target.dtype
        ):
            raise ValueError(
                f"compatible source tensor {name} is missing or changed"
            )
        target.copy_(source)
    model.load_state_dict(target_state, strict=True)
    checkpoint = {
        "architecture": ARCHITECTURE_NAME,
        "conversion": {
            "k_v_reduction": "mean_within_contiguous_query_head_group",
            "query_heads": target_config.n_head,
            "source_route": parent["route"],
            "target_kv_heads": target_config.n_kv_head,
            "tensors": conversion_records,
        },
        "model_config": asdict(target_config),
        "model_state_dict": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        "parent_checkpoint_sha256": parent["sha256"],
        "parent_route": parent["route"],
        "route": INIT_ROUTES[position_mode],
        "schema_version": 1,
        "source_commit": source_commit,
        "tokenizer": {
            "kind": "byte_plus_fixed_special_tokens",
            "vocab_size": VOCAB_SIZE,
        },
    }
    checkpoint["model_state_sha256"] = model_state_sha256(
        checkpoint["model_state_dict"]
    )
    _atomic_torch_save(checkpoint, destination)
    return {
        "checkpoint_bytes": destination.stat().st_size,
        "checkpoint_path": str(destination),
        "checkpoint_sha256": sha256_file(destination),
        "kv_cache_bytes_fp32": model.kv_cache_bytes_fp32,
        "model_state_sha256": checkpoint["model_state_sha256"],
        "parameter_count": model.parameter_count,
        "position_mode": position_mode,
        "route": INIT_ROUTES[position_mode],
    }


def load_efficient_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_route: str,
    expected_model_config: EfficientLongContextConfig,
) -> tuple[EfficientLongContextGPT, dict[str, object]]:
    """Strictly load one Lesson 15 efficient checkpoint."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if (
        not isinstance(expected_sha256, str)
        or _SHA256_PATTERN.fullmatch(expected_sha256) is None
        or sha256_file(checkpoint_path) != expected_sha256
    ):
        raise ValueError("efficient checkpoint SHA-256 mismatch")
    if expected_route not in ALL_EFFICIENT_ROUTES:
        raise ValueError("expected efficient route is unsupported")
    try:
        raw = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise ValueError(
            "efficient checkpoint could not be loaded safely"
        ) from error
    if (
        not isinstance(raw, Mapping)
        or raw.get("schema_version") != 1
        or raw.get("architecture") != ARCHITECTURE_NAME
        or raw.get("route") != expected_route
        or raw.get("model_config") != asdict(expected_model_config)
    ):
        raise ValueError(
            "efficient checkpoint schema, route, or config mismatch"
        )
    tokenizer = raw.get("tokenizer")
    if (
        not isinstance(tokenizer, Mapping)
        or tokenizer.get("kind")
        != "byte_plus_fixed_special_tokens"
        or tokenizer.get("vocab_size") != VOCAB_SIZE
    ):
        raise ValueError("efficient checkpoint tokenizer is invalid")
    state = raw.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("efficient checkpoint state is missing")
    declared_state_hash = raw.get("model_state_sha256")
    if (
        not isinstance(declared_state_hash, str)
        or _SHA256_PATTERN.fullmatch(declared_state_hash) is None
        or model_state_sha256(state) != declared_state_hash
    ):
        raise ValueError("efficient checkpoint model-state hash mismatch")
    model = EfficientLongContextGPT(expected_model_config)
    reference = model.state_dict()
    if set(state) != set(reference):
        raise ValueError("efficient checkpoint tensor keys mismatch")
    checked: dict[str, torch.Tensor] = {}
    for name, expected in reference.items():
        value = state[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != expected.shape
            or value.dtype != expected.dtype
        ):
            raise ValueError(
                f"efficient tensor {name} shape or dtype mismatch"
            )
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"efficient tensor {name} is non-finite")
        checked[name] = value.detach().cpu()
    model.load_state_dict(checked, strict=True)
    if (
        expected_model_config.tie_embeddings
        and model.token_embedding.weight is not model.lm_head.weight
    ):
        raise ValueError("efficient tied embedding identity was lost")
    return model, {
        "model_config": asdict(expected_model_config),
        "route": expected_route,
        "sha256": expected_sha256,
        "source_commit": raw.get("source_commit"),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--position-mode",
        choices=tuple(sorted(POSITION_MODES)),
        required=True,
    )
    parser.add_argument("--source-commit", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    result = create_efficient_init_checkpoint(
        parent_checkpoint=arguments.parent_checkpoint,
        parent_checkpoint_sha256=arguments.parent_checkpoint_sha256,
        output_path=arguments.output,
        position_mode=arguments.position_mode,
        source_commit=arguments.source_commit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
