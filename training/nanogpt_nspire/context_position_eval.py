"""Measure 512-token language-model loss before and after position 256."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from nanogpt_nspire.base_train import (
    _autocast_context,
    load_packed_dataset,
    make_packed_batch,
)
from nanogpt_nspire.context_extension import (
    CONTEXT512_CPT_ROUTE,
    CONTEXT512_INIT_ROUTE,
    EXTENDED_ARCHITECTURE,
)
from nanogpt_nspire.models.direct_small_gpt import (
    DirectSmallConfig,
    DirectSmallGPT,
)
from nanogpt_nspire.stage_train import load_parent_checkpoint
from nanogpt_nspire.training_support import (
    resolve_device,
    sha256_file,
    write_json_atomic,
)


SUPPORTED_ROUTES = frozenset(
    {
        CONTEXT512_INIT_ROUTE,
        CONTEXT512_CPT_ROUTE,
    }
)


def _loss_metrics(loss: float, targets: int) -> dict[str, object]:
    return {
        "bits_per_token": loss / math.log(2.0),
        "eligible_targets": targets,
        "loss": loss,
        "perplexity": math.exp(loss),
    }


def summarize_position_losses(
    losses: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    boundary: int,
) -> dict[str, dict[str, float | int]]:
    """Summarize one batch into prefix and extended-position buckets."""

    if losses.ndim != 2:
        raise ValueError("losses must have shape (B, T)")
    if target_mask.shape != losses.shape or target_mask.dtype != torch.bool:
        raise ValueError("target_mask must be boolean and match losses")
    if (
        isinstance(boundary, bool)
        or not isinstance(boundary, int)
        or boundary <= 0
        or boundary >= losses.shape[1]
    ):
        raise ValueError("boundary must split the sequence dimension")
    results: dict[str, dict[str, float | int]] = {}
    for name, start, stop in (
        ("positions_0_255", 0, boundary),
        ("positions_256_511", boundary, losses.shape[1]),
    ):
        mask = target_mask[:, start:stop]
        eligible = int(mask.sum().item())
        results[name] = {
            "eligible_targets": eligible,
            "loss_sum": (
                float(losses[:, start:stop].masked_select(mask).sum().item())
                if eligible
                else 0.0
            ),
        }
    return results


def run_context_position_evaluation(
    *,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    expected_route: str,
    data_dir: str | Path,
    output_path: str | Path,
    device_name: str = "auto",
    batch_size: int = 2,
    use_bfloat16: bool = True,
) -> dict[str, object]:
    """Evaluate every validation target once in non-overlapping 512 windows."""

    if expected_route not in SUPPORTED_ROUTES:
        raise ValueError("expected route is unsupported")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    output = Path(output_path)
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    config = DirectSmallConfig(**EXTENDED_ARCHITECTURE)
    parent = load_parent_checkpoint(
        checkpoint_path,
        expected_sha256=checkpoint_sha256,
        expected_route=expected_route,
        expected_model_config=config,
    )
    dataset = load_packed_dataset(data_dir)
    split = dataset.validation
    device = resolve_device(device_name)
    model = DirectSmallGPT(config).to(device)
    model.load_state_dict(parent["model_state_dict"], strict=True)
    model.eval()
    prediction_positions = split.token_count - 1
    starts = list(range(0, prediction_positions - 511, 512))
    if not starts:
        raise ValueError("validation split has no complete 512-token window")
    totals = {
        "positions_0_255": {"eligible_targets": 0, "loss_sum": 0.0},
        "positions_256_511": {"eligible_targets": 0, "loss_sum": 0.0},
    }
    generator = torch.Generator(device="cpu").manual_seed(0)
    evaluated_positions = 0
    with torch.inference_mode():
        for offset in range(0, len(starts), batch_size):
            selected = torch.tensor(
                starts[offset : offset + batch_size],
                dtype=torch.long,
            )
            batch = make_packed_batch(
                split,
                batch_size=len(selected),
                block_size=512,
                generator=generator,
                device=device,
                starts=selected,
            )
            with _autocast_context(
                device,
                enabled=use_bfloat16,
            ):
                logits, _ = model(batch.inputs)
                losses = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    batch.targets.reshape(-1),
                    reduction="none",
                ).reshape_as(batch.targets)
            buckets = summarize_position_losses(
                losses.float(),
                batch.target_mask,
                boundary=256,
            )
            evaluated_positions += batch.targets.numel()
            for name, values in buckets.items():
                totals[name]["eligible_targets"] += int(
                    values["eligible_targets"]
                )
                totals[name]["loss_sum"] += float(values["loss_sum"])
    bucket_metrics: dict[str, dict[str, object]] = {}
    total_loss_sum = 0.0
    total_targets = 0
    for name, values in totals.items():
        eligible = int(values["eligible_targets"])
        if eligible == 0:
            raise RuntimeError(f"{name} has no eligible targets")
        loss_sum = float(values["loss_sum"])
        bucket_metrics[name] = _loss_metrics(
            loss_sum / eligible,
            eligible,
        )
        total_loss_sum += loss_sum
        total_targets += eligible
    summary: dict[str, object] = {
        "checkpoint": {
            "route": expected_route,
            "sha256": checkpoint_sha256,
        },
        "configuration": {
            "batch_size": batch_size,
            "block_size": 512,
            "data_dir": str(data_dir),
            "device": str(device),
            "full_windows_only": True,
            "use_bfloat16": use_bfloat16 and device.type == "cuda",
        },
        "dataset_manifest_sha256": sha256_file(dataset.manifest_path),
        "evaluated_prediction_positions": evaluated_positions,
        "metrics": {
            "all_positions": _loss_metrics(
                total_loss_sum / total_targets,
                total_targets,
            ),
            **bucket_metrics,
        },
        "schema_version": 1,
    }
    write_json_atomic(output, summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument(
        "--expected-route",
        choices=tuple(sorted(SUPPORTED_ROUTES)),
        required=True,
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--no-bfloat16", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    summary = run_context_position_evaluation(
        checkpoint_path=arguments.checkpoint,
        checkpoint_sha256=arguments.checkpoint_sha256,
        expected_route=arguments.expected_route,
        data_dir=arguments.data_dir,
        output_path=arguments.output,
        device_name=arguments.device,
        batch_size=arguments.batch_size,
        use_bfloat16=not arguments.no_bfloat16,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
