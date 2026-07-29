"""Frozen compact verified SFT-v2 training for Lesson 16."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import pathlib

import torch
import torch.nn.functional as F

from nanogpt_nspire.byte_tokenizer import EOS_ID, FINAL_ID
from nanogpt_nspire.efficient_context import (
    GQA_ALIBI_SFT_ROUTE,
    GQA_ALIBI_SFT_V2_ROUTE,
)
from nanogpt_nspire.efficient_train import (
    EfficientTrainingConfig,
    run_efficient_training,
)
from nanogpt_nspire.models.efficient_long_context_gpt import ALIBI_POSITIONS


LESSON16_TRAINING_SEED = 20260729
LESSON16_BOUNDARY_TOKEN_WEIGHT = 4.0


def boundary_weighted_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    boundary_token_weight: float,
) -> torch.Tensor:
    """Apply extra weight only to eligible ``FINAL`` and ``EOS`` targets."""

    if logits.ndim != 3 or targets.shape != logits.shape[:2]:
        raise ValueError("logits and targets have incompatible shapes")
    if target_mask.shape != targets.shape:
        raise ValueError("target_mask and targets have incompatible shapes")
    if (
        isinstance(boundary_token_weight, bool)
        or not isinstance(boundary_token_weight, (int, float))
        or not math.isfinite(float(boundary_token_weight))
        or float(boundary_token_weight) <= 0
    ):
        raise ValueError("boundary_token_weight must be finite and positive")
    eligible = target_mask.to(dtype=torch.bool)
    if not bool(eligible.any().item()):
        raise ValueError("target_mask must contain an eligible target")
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    boundary = targets.eq(FINAL_ID) | targets.eq(EOS_ID)
    weights = torch.ones_like(token_loss)
    weights = torch.where(
        boundary,
        weights * float(boundary_token_weight),
        weights,
    )
    weights = weights * eligible.to(dtype=token_loss.dtype)
    return (token_loss * weights).sum() / weights.sum()


@dataclass(frozen=True)
class SFTV2TrainingConfig(EfficientTrainingConfig):
    """Lesson 16 specialization with a frozen parent and weighted boundary."""

    boundary_token_weight: float = LESSON16_BOUNDARY_TOKEN_WEIGHT

    @property
    def expected_parent_route(self) -> str:
        return GQA_ALIBI_SFT_ROUTE

    @property
    def route(self) -> str:
        return GQA_ALIBI_SFT_V2_ROUTE

    @property
    def checkpoint_filename(self) -> str:
        return "gqa_alibi_sft_v2_context512.pt"

    def training_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        return boundary_weighted_cross_entropy(
            logits,
            targets,
            target_mask,
            boundary_token_weight=self.boundary_token_weight,
        )

    def validate(self) -> None:
        super().validate()
        if self.stage != "sft" or self.position_mode != ALIBI_POSITIONS:
            raise ValueError("Lesson 16 requires the ALiBi SFT route")
        if self.boundary_token_weight != LESSON16_BOUNDARY_TOKEN_WEIGHT:
            raise ValueError("Lesson 16 boundary token weight is frozen")


def frozen_sft_v2_config(
    *,
    data_dir: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    parent_checkpoint: str | pathlib.Path,
    parent_checkpoint_sha256: str,
    source_commit: str,
    **overrides: object,
) -> SFTV2TrainingConfig:
    """Create the auditable 4.096M-token Lesson 16 training contract."""

    frozen_names = {
        "block_size",
        "boundary_token_weight",
        "effective_batch_tokens",
        "eval_interval",
        "gradient_accumulation_steps",
        "learning_rate",
        "min_learning_rate",
        "micro_batch_size",
        "mlp_ratio",
        "n_embd",
        "n_head",
        "n_kv_head",
        "n_layer",
        "position_mode",
        "seed",
        "stage",
        "steps",
        "vocab_size",
        "warmup_steps",
    } & set(overrides)
    if frozen_names:
        raise ValueError(
            "Lesson 16 architecture and compute are frozen; remove overrides: "
            + ", ".join(sorted(frozen_names))
        )
    defaults: dict[str, object] = {
        "device": "auto",
        "eval_batches": 20,
        "log_interval": 25,
        "overfit_gate_steps": 20,
        "use_bfloat16": True,
    }
    defaults.update(overrides)
    config = SFTV2TrainingConfig(
        stage="sft",
        position_mode=ALIBI_POSITIONS,
        data_dir=pathlib.Path(data_dir),
        output_dir=pathlib.Path(output_dir),
        parent_checkpoint=pathlib.Path(parent_checkpoint),
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        source_commit=source_commit,
        seed=LESSON16_TRAINING_SEED,
        steps=1000,
        micro_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=0.0001,
        min_learning_rate=0.00001,
        warmup_steps=50,
        eval_interval=100,
        boundary_token_weight=LESSON16_BOUNDARY_TOKEN_WEIGHT,
        **defaults,
    )
    config.validate()
    return config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--parent-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--parent-checkpoint-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    config = frozen_sft_v2_config(
        data_dir=arguments.data_dir,
        output_dir=arguments.output_dir,
        parent_checkpoint=arguments.parent_checkpoint,
        parent_checkpoint_sha256=arguments.parent_checkpoint_sha256,
        source_commit=arguments.source_commit,
        device=arguments.device,
    )
    result = run_efficient_training(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
