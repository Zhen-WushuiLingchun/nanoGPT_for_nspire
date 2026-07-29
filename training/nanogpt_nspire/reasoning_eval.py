"""Mode-aware frozen exact-answer evaluation for Lesson 14."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time
from typing import Mapping

import torch

from nanogpt_nspire.assistant_eval import (
    SUPPORTED_TASKS,
    EvaluationError,
    load_evaluation_model,
    load_evaluation_records,
    score_completion,
    select_evaluation_records,
)
from nanogpt_nspire.base_train import _autocast_context
from nanogpt_nspire.byte_tokenizer import (
    BYTE_VOCAB_SIZE,
    EOS_ID,
    FINAL_ID,
    SPECIAL_TOKEN_NAMES,
)
from nanogpt_nspire.efficient_context import (
    GQA_ALIBI_SFT_ROUTE,
    GQA_ALIBI_SFT_V2_ROUTE,
    GQA_LEARNED_SFT_ROUTE,
)
from nanogpt_nspire.models.direct_small_gpt import DirectSmallGPT
from nanogpt_nspire.reasoning_format import (
    DIRECT_MODE,
    THINK_MODE,
    SUPPORTED_MODES,
    ReasoningFormatError,
    encode_mode_prompt,
)
from nanogpt_nspire.training_support import (
    resolve_device,
    sha256_file,
    synchronize,
    write_json_atomic,
)


LESSON14_ROUTES = frozenset(
    {
        "Direct-Control-SFT",
        GQA_ALIBI_SFT_ROUTE,
        GQA_ALIBI_SFT_V2_ROUTE,
        GQA_LEARNED_SFT_ROUTE,
        "Short-CoT-SFT",
        "Hybrid-Control-SFT",
        "Hybrid-Control-SFT-Context512",
    }
)


def _decode_bytes(values: list[int]) -> str:
    return bytes(values).decode("utf-8", errors="backslashreplace")


def generate_mode_completion(
    model: DirectSmallGPT,
    prompt: str,
    *,
    mode: str,
    max_new_tokens: int,
    device: torch.device,
    use_bfloat16: bool,
) -> dict[str, object]:
    """Greedily generate direct or `<THINK>...<FINAL>...` output."""

    if mode not in SUPPORTED_MODES:
        raise EvaluationError("mode is unsupported")
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens <= 0
    ):
        raise EvaluationError("max_new_tokens must be positive")
    if not isinstance(device, torch.device):
        raise EvaluationError("device must be torch.device")
    try:
        tokens = list(
            encode_mode_prompt(
                prompt,
                mode=mode,
                block_size=model.block_size,
            )
        )
    except ReasoningFormatError as error:
        raise EvaluationError(str(error)) from error
    reasoning_bytes: list[int] = []
    final_bytes: list[int] = []
    final_transition = False
    generated_tokens = 0
    terminated = False
    special_token_leak = False
    leaked_token: str | None = None
    context_exhausted = False
    started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            inputs = torch.tensor(
                [tokens],
                dtype=torch.long,
                device=device,
            )
            with _autocast_context(device, enabled=use_bfloat16):
                logits, _ = model(inputs)
            next_token = int(torch.argmax(logits[0, -1]).item())
            generated_tokens += 1
            if next_token == EOS_ID:
                terminated = True
                break
            if mode == THINK_MODE and (
                next_token == FINAL_ID and not final_transition
            ):
                final_transition = True
                if len(tokens) >= model.block_size:
                    context_exhausted = True
                    break
                tokens.append(next_token)
                continue
            if next_token >= BYTE_VOCAB_SIZE:
                special_token_leak = True
                leaked_token = SPECIAL_TOKEN_NAMES[next_token]
                break
            if mode == THINK_MODE and not final_transition:
                reasoning_bytes.append(next_token)
            else:
                final_bytes.append(next_token)
            if len(tokens) >= model.block_size:
                context_exhausted = True
                break
            tokens.append(next_token)
    synchronize(device)
    elapsed = time.perf_counter() - started
    budget_exhausted = (
        not terminated
        and not special_token_leak
        and not context_exhausted
        and generated_tokens == max_new_tokens
    )
    return {
        "budget_exhausted": budget_exhausted,
        "context_exhausted": context_exhausted,
        "elapsed_seconds": elapsed,
        "final_text": _decode_bytes(final_bytes),
        "final_tokens": len(final_bytes),
        "final_transition": final_transition,
        "generated_tokens": generated_tokens,
        "leaked_token": leaked_token,
        "mode": mode,
        "reasoning_text": _decode_bytes(reasoning_bytes),
        "reasoning_tokens": len(reasoning_bytes),
        "special_token_leak": special_token_leak,
        "terminated": terminated,
        "tokens_per_second": (
            generated_tokens / elapsed if elapsed else None
        ),
        "truncated": budget_exhausted or context_exhausted,
    }


def score_mode_completion(
    record: Mapping[str, object],
    completion: Mapping[str, object],
) -> dict[str, object]:
    """Score the final segment while enforcing the requested mode grammar."""

    mode = completion.get("mode")
    if mode not in SUPPORTED_MODES:
        raise EvaluationError("completion mode is invalid")
    final_text = completion.get("final_text")
    reasoning_text = completion.get("reasoning_text")
    if not isinstance(final_text, str) or not isinstance(
        reasoning_text,
        str,
    ):
        raise EvaluationError("completion text fields are invalid")
    mode_compliant = (
        not bool(completion.get("special_token_leak"))
        and (
            (
                mode == DIRECT_MODE
                and bool(final_text.strip())
                and not bool(completion.get("final_transition"))
            )
            or (
                mode == THINK_MODE
                and bool(reasoning_text.strip())
                and bool(completion.get("final_transition"))
                and bool(final_text.strip())
            )
        )
    )
    base = score_completion(
        record,
        text=final_text,
        terminated=(
            bool(completion.get("terminated"))
            and mode_compliant
        ),
        special_token_leak=bool(
            completion.get("special_token_leak")
        ),
    )
    base["mode_compliant"] = mode_compliant
    base["task_correct"] = bool(base["task_correct"]) and mode_compliant
    return base


def _rate(rows: list[dict[str, object]], field: str) -> float:
    return sum(bool(row["score"][field]) for row in rows) / len(rows)


def run_reasoning_evaluation(
    *,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    expected_route: str,
    evaluation_path: str | Path,
    output_path: str | Path,
    mode: str,
    device_name: str = "auto",
    max_per_task: int = 32,
    max_new_tokens: int = 48,
    use_bfloat16: bool = True,
) -> dict[str, object]:
    """Evaluate one route/cue/budget on the frozen Lesson 12 families."""

    if expected_route not in LESSON14_ROUTES:
        raise EvaluationError("expected route is not a reasoning route")
    if mode not in SUPPORTED_MODES:
        raise EvaluationError("mode is unsupported")
    output = Path(output_path)
    if output.exists():
        raise EvaluationError(f"output already exists: {output}")
    device = resolve_device(device_name)
    selected = select_evaluation_records(
        load_evaluation_records(evaluation_path),
        max_per_task=max_per_task,
    )
    model, checkpoint = load_evaluation_model(
        checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        expected_route=expected_route,
        device=device,
    )
    results: list[dict[str, object]] = []
    for record in selected:
        completion = generate_mode_completion(
            model,
            str(record["prompt"]),
            mode=mode,
            max_new_tokens=max_new_tokens,
            device=device,
            use_bfloat16=use_bfloat16,
        )
        score = score_mode_completion(record, completion)
        results.append(
            {
                "completion": completion,
                "expected_answer": record["expected_answer"],
                "expected_unit": record["expected_unit"],
                "family_id": record["family_id"],
                "prompt": record["prompt"],
                "score": score,
                "source_id": record["source_id"],
                "task": record["task"],
            }
        )
    per_task: dict[str, dict[str, object]] = {}
    for task in sorted(SUPPORTED_TASKS):
        task_rows = [row for row in results if row["task"] == task]
        per_task[task] = {
            "count": len(task_rows),
            "format_valid_rate": _rate(task_rows, "format_valid"),
            "mode_compliance_rate": _rate(
                task_rows,
                "mode_compliant",
            ),
            "numeric_accuracy": _rate(
                task_rows,
                "numeric_correct",
            ),
            "task_accuracy": _rate(task_rows, "task_correct"),
        }
    total_generated = sum(
        int(row["completion"]["generated_tokens"]) for row in results
    )
    total_seconds = sum(
        float(row["completion"]["elapsed_seconds"]) for row in results
    )
    completions = Counter(
        str(row["completion"]["final_text"]) for row in results
    )
    summary: dict[str, object] = {
        "checkpoint": checkpoint,
        "configuration": {
            "device": str(device),
            "evaluation_path": str(evaluation_path),
            "max_new_tokens": max_new_tokens,
            "max_per_task": max_per_task,
            "mode": mode,
            "use_bfloat16": use_bfloat16 and device.type == "cuda",
        },
        "evaluation_file_sha256": sha256_file(Path(evaluation_path)),
        "metrics": {
            "budget_truncation_rate": sum(
                bool(row["completion"]["budget_exhausted"])
                for row in results
            )
            / len(results),
            "context_truncation_rate": sum(
                bool(row["completion"]["context_exhausted"])
                for row in results
            )
            / len(results),
            "examples": len(results),
            "final_transition_rate": sum(
                bool(row["completion"]["final_transition"])
                for row in results
            )
            / len(results),
            "format_valid_rate": _rate(results, "format_valid"),
            "generated_tokens": total_generated,
            "mean_final_tokens": sum(
                int(row["completion"]["final_tokens"])
                for row in results
            )
            / len(results),
            "mean_reasoning_tokens": sum(
                int(row["completion"]["reasoning_tokens"])
                for row in results
            )
            / len(results),
            "mode_compliance_rate": _rate(
                results,
                "mode_compliant",
            ),
            "most_common_final": (
                {
                    "count": completions.most_common(1)[0][1],
                    "text": completions.most_common(1)[0][0],
                }
                if completions
                else None
            ),
            "role_leak_rate": _rate(
                results,
                "special_token_leak",
            ),
            "task_accuracy": _rate(results, "task_correct"),
            "tokens_per_second": (
                total_generated / total_seconds if total_seconds else None
            ),
            "unique_finals": len(completions),
        },
        "per_task": per_task,
        "results": results,
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
        choices=tuple(sorted(LESSON14_ROUTES)),
        required=True,
    )
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=tuple(sorted(SUPPORTED_MODES)),
        required=True,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-per-task", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--no-bfloat16", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    summary = run_reasoning_evaluation(
        checkpoint_path=arguments.checkpoint,
        checkpoint_sha256=arguments.checkpoint_sha256,
        expected_route=arguments.expected_route,
        evaluation_path=arguments.evaluation,
        output_path=arguments.output,
        mode=arguments.mode,
        device_name=arguments.device,
        max_per_task=arguments.max_per_task,
        max_new_tokens=arguments.max_new_tokens,
        use_bfloat16=not arguments.no_bfloat16,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

