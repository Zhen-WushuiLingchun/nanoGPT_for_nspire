"""Cross-check portable Host C inference against the PyTorch reference."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Sequence

import numpy as np
import torch

from nanogpt_nspire.export_format import (
    MODEL_STORAGE_FP32,
    MODEL_STORAGE_W4A8,
    STORAGE_FP32,
    ModelFormatError,
    ParsedModel,
    parse_model_file,
)
from nanogpt_nspire.export_model import expected_tensor_descriptors
from nanogpt_nspire.models.direct_small_gpt import (
    DirectSmallConfig,
    DirectSmallGPT,
)
from nanogpt_nspire.training_support import write_json_atomic


FP32_MAX_ABS_GATE = 2.0e-4
FP32_RMSE_GATE = 5.0e-5


@dataclass(frozen=True)
class ProbeResult:
    """Final logits and greedy continuation from one inference engine."""

    logits: np.ndarray
    generated_tokens: tuple[int, ...]
    metrics: dict[str, object]


def load_fp32_export(
    path: Path,
) -> tuple[DirectSmallGPT, ParsedModel]:
    """Reconstruct the exact evaluation-mode PyTorch model from an NGM file."""

    parsed = parse_model_file(path.read_bytes())
    if parsed.spec.model_storage != MODEL_STORAGE_FP32:
        raise ModelFormatError("PyTorch FP32 loader requires an FP32 export")
    config = DirectSmallConfig(
        vocab_size=parsed.spec.vocab_size,
        block_size=parsed.spec.block_size,
        n_layer=parsed.spec.n_layer,
        n_head=parsed.spec.n_head,
        n_embd=parsed.spec.n_embd,
        mlp_ratio=parsed.spec.mlp_ratio,
        dropout=0.0,
        bias=parsed.spec.bias,
        tie_embeddings=parsed.spec.tie_embeddings,
    )
    model = DirectSmallGPT(config)
    state: dict[str, torch.Tensor] = {}
    for descriptor in expected_tensor_descriptors(config):
        view = parsed.tensors[descriptor.tensor_id]
        if view.storage != STORAGE_FP32:
            raise ModelFormatError(
                f"tensor {descriptor.tensor_id} is not FP32"
            )
        array = (
            np.frombuffer(view.data, dtype="<f4")
            .reshape(descriptor.shape)
            .copy()
        )
        state[descriptor.name] = torch.from_numpy(array)
    state["lm_head.weight"] = state["token_embedding.weight"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, parsed


def torch_greedy_probe(
    model: DirectSmallGPT,
    prompt_tokens: Sequence[int],
    generate_count: int,
) -> ProbeResult:
    """Run full-prefix PyTorch inference after every greedy token."""

    if not prompt_tokens:
        raise ValueError("prompt_tokens must not be empty")
    if generate_count < 0:
        raise ValueError("generate_count must be non-negative")
    all_tokens = [int(token) for token in prompt_tokens]
    if len(all_tokens) + generate_count > model.block_size:
        raise ValueError("prompt plus generation exceeds block_size")
    generated: list[int] = []
    started = time.perf_counter()
    with torch.inference_mode():
        token_tensor = torch.tensor([all_tokens], dtype=torch.long)
        logits, _ = model(token_tensor)
        prompt_logits = (
            logits[0, -1]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=True)
        )
        for _ in range(generate_count):
            next_token = int(torch.argmax(logits[0, -1]).item())
            generated.append(next_token)
            all_tokens.append(next_token)
            token_tensor = torch.tensor([all_tokens], dtype=torch.long)
            logits, _ = model(token_tensor)
    elapsed = time.perf_counter() - started
    return ProbeResult(
        logits=prompt_logits,
        generated_tokens=tuple(generated),
        metrics={
            "elapsed_seconds": elapsed,
            "full_prefix_evaluations": generate_count + 1,
            "logits_checkpoint": "after_prompt",
        },
    )


def host_greedy_probe(
    runner: Path,
    model_path: Path,
    prompt_tokens: Sequence[int],
    generate_count: int,
) -> ProbeResult:
    """Run the C probe and validate its bounded binary outputs."""

    if not prompt_tokens:
        raise ValueError("prompt_tokens must not be empty")
    with tempfile.TemporaryDirectory(prefix="nanogpt-host-probe-") as directory:
        root = Path(directory)
        logits_path = root / "logits.f32"
        tokens_path = root / "tokens.u32"
        completed = subprocess.run(
            [
                str(runner),
                "--model",
                str(model_path),
                "--tokens",
                ",".join(str(int(token)) for token in prompt_tokens),
                "--logits-out",
                str(logits_path),
                "--tokens-out",
                str(tokens_path),
                "--generate",
                str(generate_count),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Host C probe failed "
                f"({completed.returncode}): {completed.stderr.strip()}"
            )
        try:
            metrics = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError("Host C probe emitted invalid JSON") from error
        logits = np.fromfile(logits_path, dtype="<f4")
        generated = np.fromfile(tokens_path, dtype="<u4")
    expected_vocab = int(metrics.get("vocab_size", logits.size))
    if logits.size != expected_vocab:
        raise RuntimeError("Host C logits output has an invalid length")
    if generated.size != generate_count:
        raise RuntimeError("Host C token output has an invalid length")
    return ProbeResult(
        logits=logits.astype(np.float32, copy=False),
        generated_tokens=tuple(int(token) for token in generated),
        metrics=metrics,
    )


def compare_probes(
    reference: ProbeResult,
    candidate: ProbeResult,
) -> dict[str, object]:
    """Calculate the pre-registered FP32 alignment gates."""

    if reference.logits.shape != candidate.logits.shape:
        raise ValueError("probe logits shapes differ")
    difference = (
        candidate.logits.astype(np.float64)
        - reference.logits.astype(np.float64)
    )
    max_absolute_error = float(np.max(np.abs(difference)))
    rmse = float(math.sqrt(float(np.mean(difference * difference))))
    greedy_exact = (
        candidate.generated_tokens == reference.generated_tokens
    )
    return {
        "max_absolute_error": max_absolute_error,
        "max_absolute_error_gate": FP32_MAX_ABS_GATE,
        "max_absolute_error_pass": (
            max_absolute_error <= FP32_MAX_ABS_GATE
        ),
        "rmse": rmse,
        "rmse_gate": FP32_RMSE_GATE,
        "rmse_pass": rmse <= FP32_RMSE_GATE,
        "greedy_exact": greedy_exact,
        "pass": (
            max_absolute_error <= FP32_MAX_ABS_GATE
            and rmse <= FP32_RMSE_GATE
            and greedy_exact
        ),
    }


def encode_text(text: str, vocabulary: Sequence[str]) -> tuple[int, ...]:
    """Encode a character prompt using the export's exact vocabulary."""

    token_by_character = {
        character: index
        for index, character in enumerate(vocabulary)
    }
    try:
        return tuple(token_by_character[character] for character in text)
    except KeyError as error:
        raise ValueError(
            f"prompt character {error.args[0]!r} is outside the vocabulary"
        ) from error


def _parse_token_list(text: str) -> tuple[int, ...]:
    try:
        result = tuple(int(part) for part in text.split(","))
    except ValueError as error:
        raise ValueError("prompt token list is invalid") from error
    if not result or any(token < 0 for token in result):
        raise ValueError("prompt token list must contain non-negative IDs")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--prompt-tokens")
    parser.add_argument("--generate", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    parsed = parse_model_file(arguments.model.read_bytes())
    if parsed.spec.model_storage == MODEL_STORAGE_FP32:
        model, parsed = load_fp32_export(arguments.model)
        reference_engine = "pytorch_fp32_full_prefix"
    elif parsed.spec.model_storage == MODEL_STORAGE_W4A8:
        from nanogpt_nspire.w4a8_reference import (
            W4A8Reference,
            w4a8_greedy_probe,
        )

        model = W4A8Reference(parsed)
        reference_engine = "pytorch_packed_w4a8_incremental"
    else:
        raise ModelFormatError("alignment model storage is unsupported")
    if arguments.prompt is not None:
        prompt_tokens = encode_text(
            arguments.prompt,
            parsed.vocabulary,
        )
    else:
        assert arguments.prompt_tokens is not None
        prompt_tokens = _parse_token_list(arguments.prompt_tokens)
    if parsed.spec.model_storage == MODEL_STORAGE_FP32:
        assert isinstance(model, DirectSmallGPT)
        reference = torch_greedy_probe(
            model,
            prompt_tokens,
            arguments.generate,
        )
    else:
        reference = w4a8_greedy_probe(
            model,
            prompt_tokens,
            arguments.generate,
        )
    candidate = host_greedy_probe(
        arguments.runner,
        arguments.model,
        prompt_tokens,
        arguments.generate,
    )
    comparison = compare_probes(reference, candidate)
    result = {
        "schema_version": 1,
        "model": str(arguments.model),
        "reference_engine": reference_engine,
        "prompt_tokens": list(prompt_tokens),
        "generated_tokens": list(candidate.generated_tokens),
        "generated_text": "".join(
            parsed.vocabulary[token]
            for token in candidate.generated_tokens
        ),
        "pytorch": reference.metrics,
        "host_c": candidate.metrics,
        "alignment": comparison,
    }
    write_json_atomic(arguments.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if comparison["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
