"""Frozen role-aware generation and exact-answer scoring for Lesson 12."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Iterable, Mapping

import torch

from nanogpt_nspire.base_train import _autocast_context
from nanogpt_nspire.byte_tokenizer import (
    ASSISTANT_ID,
    BOS_ID,
    BYTE_VOCAB_SIZE,
    EOS_ID,
    SPECIAL_TOKEN_NAMES,
    USER_ID,
    VOCAB_SIZE,
    ByteTokenizer,
)
from nanogpt_nspire.efficient_context import (
    ARCHITECTURE_NAME,
    GQA_ALIBI_SFT_ROUTE,
    GQA_ALIBI_SFT_V2_ROUTE,
    GQA_LEARNED_SFT_ROUTE,
    load_efficient_checkpoint,
)
from nanogpt_nspire.models.direct_small_gpt import (
    DirectSmallConfig,
    DirectSmallGPT,
)
from nanogpt_nspire.models.efficient_long_context_gpt import (
    EfficientLongContextConfig,
)
from nanogpt_nspire.training_support import (
    resolve_device,
    sha256_file,
    synchronize,
    write_json_atomic,
)


SUPPORTED_TASKS = frozenset(
    {"arithmetic", "arithmetic_easy", "physics_numeric", "gsm8k"}
)
SUPPORTED_ROUTES = frozenset(
    {
        "Combined-Sequence-Logit-SFT",
        "Direct-Control-SFT",
        "English-Base-Pilot",
        GQA_ALIBI_SFT_ROUTE,
        GQA_ALIBI_SFT_V2_ROUTE,
        GQA_LEARNED_SFT_ROUTE,
        "Hybrid-Control-SFT",
        "Hybrid-Control-SFT-Context512",
        "Local-Logit-Distilled-SFT",
        "Local-Teacher-SFT",
        "Math-Physics-CPT",
        "Role-Aware-SFT",
        "Short-CoT-SFT",
        "Verified-Sequence-SFT",
    }
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_DECIMAL_PATTERN = re.compile(
    r"(?<![\w/])"
    r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?"
    r"(?![\w/])"
)
_FRACTION_PATTERN = re.compile(r"[-+]?\d+\s*/\s*[-+]?\d+")


class EvaluationError(ValueError):
    """Raised when evaluation input or generated output is ambiguous."""


def _normalize_decimal(value: Decimal) -> str:
    if not value.is_finite():
        raise EvaluationError("decimal must be finite")
    text = format(value.normalize(), "f")
    return "0" if text == "-0" else text


def encode_assistant_prompt(
    prompt: str,
    *,
    block_size: int,
) -> tuple[int, ...]:
    """Encode `<BOS><USER>prompt<ASSISTANT>` with no UI-only metadata."""

    if not isinstance(prompt, str) or not prompt:
        raise EvaluationError("prompt must be a non-empty string")
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size <= 0
    ):
        raise EvaluationError("block_size must be a positive integer")
    tokenizer = ByteTokenizer()
    try:
        prompt_tokens = tokenizer.encode_text(prompt)
    except (UnicodeEncodeError, ValueError) as error:
        raise EvaluationError("prompt is not valid UTF-8") from error
    tokens = (BOS_ID, USER_ID, *prompt_tokens, ASSISTANT_ID)
    if len(tokens) > block_size:
        raise EvaluationError("assistant prompt exceeds model context")
    return tokens


def parse_last_decimal(text: str) -> str:
    """Return the final unambiguous decimal in an answer."""

    if not isinstance(text, str):
        raise EvaluationError("answer text must be a string")
    if _FRACTION_PATTERN.search(text):
        raise EvaluationError("answer does not contain a parseable decimal")
    matches = list(_DECIMAL_PATTERN.finditer(text))
    if not matches:
        raise EvaluationError("answer does not contain a parseable decimal")
    raw = matches[-1].group(0).replace(",", "")
    try:
        return _normalize_decimal(Decimal(raw))
    except InvalidOperation as error:
        raise EvaluationError(
            "answer does not contain a parseable decimal"
        ) from error


def repeated_phrase_detected(text: str) -> bool:
    """Flag a repeated three-word phrase appearing at least three times."""

    if not isinstance(text, str):
        raise EvaluationError("answer text must be a string")
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    if len(words) < 9:
        return False
    counts = Counter(
        tuple(words[index : index + 3])
        for index in range(len(words) - 2)
    )
    return any(count >= 3 for count in counts.values())


def score_completion(
    record: Mapping[str, object],
    *,
    text: str,
    terminated: bool,
    special_token_leak: bool,
) -> dict[str, object]:
    """Score numeric value, unit, role safety, repetition, and EOS separately."""

    if not isinstance(record, Mapping):
        raise EvaluationError("record must be a mapping")
    task = record.get("task")
    if task not in SUPPORTED_TASKS:
        raise EvaluationError("record task is unsupported")
    expected = record.get("expected_answer")
    if not isinstance(expected, str) or not expected:
        raise EvaluationError("expected_answer must be non-empty")
    try:
        parsed = parse_last_decimal(text)
        numeric_correct = Decimal(parsed) == Decimal(expected)
    except (EvaluationError, InvalidOperation):
        parsed = None
        numeric_correct = False
    expected_unit = record.get("expected_unit")
    if expected_unit is None:
        unit_correct = True
    elif isinstance(expected_unit, str) and expected_unit:
        unit_correct = re.search(
            rf"(?<![A-Za-z0-9]){re.escape(expected_unit)}"
            rf"(?![A-Za-z0-9])",
            text,
            flags=re.IGNORECASE,
        ) is not None
    else:
        raise EvaluationError("expected_unit must be null or non-empty")
    repeated = repeated_phrase_detected(text)
    format_valid = (
        bool(text.strip())
        and terminated
        and not special_token_leak
        and not repeated
        and parsed is not None
    )
    task_correct = (
        numeric_correct
        and unit_correct
        and format_valid
    )
    return {
        "format_valid": format_valid,
        "numeric_correct": numeric_correct,
        "parsed_answer": parsed,
        "repeated_phrase": repeated,
        "special_token_leak": special_token_leak,
        "task_correct": task_correct,
        "terminated": terminated,
        "unit_correct": unit_correct,
    }


def load_evaluation_records(
    path: str | Path,
) -> tuple[dict[str, object], ...]:
    """Load strict canonical-like JSONL and reject duplicate families."""

    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError) as error:
        raise EvaluationError(f"invalid evaluation file: {source}") from error
    rows: list[dict[str, object]] = []
    families: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise EvaluationError(
                f"invalid evaluation JSON at line {line_number}"
            ) from error
        if not isinstance(raw, dict):
            raise EvaluationError("evaluation row must be a JSON object")
        required = {
            "expected_answer",
            "expected_unit",
            "family_id",
            "prompt",
            "source_id",
            "task",
        }
        missing = required - set(raw)
        if missing:
            raise EvaluationError(
                f"evaluation row is missing {', '.join(sorted(missing))}"
            )
        task = raw["task"]
        if task not in SUPPORTED_TASKS:
            raise EvaluationError("evaluation row task is unsupported")
        for name in (
            "expected_answer",
            "family_id",
            "prompt",
            "source_id",
        ):
            if not isinstance(raw[name], str) or not raw[name]:
                raise EvaluationError(
                    f"evaluation {name} must be non-empty"
                )
        family = raw["family_id"]
        if family in families:
            raise EvaluationError(
                f"duplicate family in evaluation file: {family}"
            )
        families.add(family)
        expected_unit = raw["expected_unit"]
        if expected_unit is not None and (
            not isinstance(expected_unit, str) or not expected_unit
        ):
            raise EvaluationError(
                "evaluation expected_unit must be null or non-empty"
            )
        rows.append(raw)
    if not rows:
        raise EvaluationError("evaluation file contains no rows")
    return tuple(rows)


def select_evaluation_records(
    records: Iterable[Mapping[str, object]],
    *,
    max_per_task: int,
) -> tuple[dict[str, object], ...]:
    """Select one deterministic hash-ranked subset per frozen task."""

    if (
        isinstance(max_per_task, bool)
        or not isinstance(max_per_task, int)
        or max_per_task <= 0
    ):
        raise EvaluationError("max_per_task must be a positive integer")
    grouped: dict[str, list[dict[str, object]]] = {
        task: [] for task in sorted(SUPPORTED_TASKS)
    }
    for raw in records:
        if not isinstance(raw, Mapping):
            raise EvaluationError("evaluation record must be a mapping")
        row = dict(raw)
        task = row.get("task")
        family = row.get("family_id")
        if task not in SUPPORTED_TASKS or not isinstance(family, str):
            raise EvaluationError("evaluation task or family is invalid")
        grouped[str(task)].append(row)
    selected: list[dict[str, object]] = []
    for task, rows in grouped.items():
        if not rows:
            raise EvaluationError(f"evaluation task {task} is empty")
        ranked = sorted(
            rows,
            key=lambda row: (
                hashlib.sha256(
                    f"lesson12-pilot:{row['family_id']}".encode("utf-8")
                ).digest(),
                str(row["family_id"]),
            ),
        )
        selected.extend(ranked[:max_per_task])
    return tuple(
        sorted(
            selected,
            key=lambda row: (str(row["task"]), str(row["family_id"])),
        )
    )


def load_evaluation_model(
    checkpoint_path: str | Path,
    *,
    checkpoint_sha256: str,
    expected_route: str,
    device: torch.device,
) -> tuple[DirectSmallGPT, dict[str, object]]:
    """Load one supported checkpoint with exact hash and route checks."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if (
        not isinstance(checkpoint_sha256, str)
        or _SHA256_PATTERN.fullmatch(checkpoint_sha256) is None
        or sha256_file(path) != checkpoint_sha256
    ):
        raise EvaluationError("checkpoint SHA-256 mismatch")
    if expected_route not in SUPPORTED_ROUTES:
        raise EvaluationError("expected route is unsupported")
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise EvaluationError("checkpoint could not be loaded safely") from error
    if (
        not isinstance(raw, Mapping)
        or raw.get("schema_version") != 1
        or raw.get("route") != expected_route
    ):
        raise EvaluationError("checkpoint route or schema mismatch")
    tokenizer = raw.get("tokenizer")
    if (
        not isinstance(tokenizer, Mapping)
        or tokenizer.get("vocab_size") != VOCAB_SIZE
        or tokenizer.get("kind")
        != "byte_plus_fixed_special_tokens"
    ):
        raise EvaluationError("checkpoint tokenizer contract mismatch")
    configuration = raw.get("model_config")
    if not isinstance(configuration, Mapping):
        raise EvaluationError("checkpoint model configuration is missing")
    if raw.get("architecture") == ARCHITECTURE_NAME:
        try:
            efficient_config = EfficientLongContextConfig(
                **dict(configuration)
            )
            efficient_config.validate()
            model, _ = load_efficient_checkpoint(
                path,
                expected_sha256=checkpoint_sha256,
                expected_route=expected_route,
                expected_model_config=efficient_config,
            )
        except (TypeError, ValueError) as error:
            raise EvaluationError(
                "efficient checkpoint model configuration is invalid"
            ) from error
        model.to(device)
        model.eval()
        return model, {
            "best_step": raw.get("best_step"),
            "model_config": asdict(efficient_config),
            "route": expected_route,
            "sha256": checkpoint_sha256,
            "source_commit": raw.get("source_commit"),
        }
    try:
        model_config = DirectSmallConfig(**dict(configuration))
        model_config.validate()
    except (TypeError, ValueError) as error:
        raise EvaluationError("checkpoint model configuration is invalid") from error
    state = raw.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise EvaluationError("checkpoint state is missing")
    model = DirectSmallGPT(model_config)
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as error:
        raise EvaluationError("checkpoint state does not match model") from error
    model.to(device)
    model.eval()
    return model, {
        "best_step": raw.get("best_step"),
        "model_config": asdict(model_config),
        "route": expected_route,
        "sha256": checkpoint_sha256,
        "source_commit": raw.get("source_commit"),
    }


def generate_assistant_completion(
    model: DirectSmallGPT,
    prompt: str,
    *,
    max_new_tokens: int,
    device: torch.device,
    use_bfloat16: bool,
) -> dict[str, object]:
    """Greedily generate raw answer bytes until EOS or a leaked special token."""

    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens <= 0
    ):
        raise EvaluationError("max_new_tokens must be a positive integer")
    tokens = list(
        encode_assistant_prompt(prompt, block_size=model.block_size)
    )
    answer_bytes: list[int] = []
    generated_tokens = 0
    terminated = False
    special_token_leak = False
    leaked_token: str | None = None
    started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            window = tokens[-model.block_size :]
            inputs = torch.tensor(
                [window],
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
            if next_token >= BYTE_VOCAB_SIZE:
                special_token_leak = True
                leaked_token = SPECIAL_TOKEN_NAMES[next_token]
                break
            tokens.append(next_token)
            answer_bytes.append(next_token)
    synchronize(device)
    elapsed = time.perf_counter() - started
    text = bytes(answer_bytes).decode(
        "utf-8",
        errors="backslashreplace",
    )
    return {
        "elapsed_seconds": elapsed,
        "generated_tokens": generated_tokens,
        "leaked_token": leaked_token,
        "special_token_leak": special_token_leak,
        "terminated": terminated,
        "text": text,
        "tokens_per_second": (
            generated_tokens / elapsed if elapsed else None
        ),
    }


def run_checkpoint_evaluation(
    *,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    expected_route: str,
    evaluation_path: str | Path,
    output_path: str | Path,
    device_name: str = "auto",
    max_per_task: int = 64,
    max_new_tokens: int = 48,
    use_bfloat16: bool = True,
) -> dict[str, object]:
    """Evaluate one checkpoint on the exact same selected prompt families."""

    output = Path(output_path)
    if output.exists():
        raise EvaluationError(f"output already exists: {output}")
    device = resolve_device(device_name)
    records = load_evaluation_records(evaluation_path)
    selected = select_evaluation_records(
        records,
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
        completion = generate_assistant_completion(
            model,
            str(record["prompt"]),
            max_new_tokens=max_new_tokens,
            device=device,
            use_bfloat16=use_bfloat16,
        )
        score = score_completion(
            record,
            text=str(completion["text"]),
            terminated=bool(completion["terminated"]),
            special_token_leak=bool(
                completion["special_token_leak"]
            ),
        )
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
        task_results = [row for row in results if row["task"] == task]
        count = len(task_results)
        per_task[task] = {
            "count": count,
            "format_valid_rate": sum(
                bool(row["score"]["format_valid"]) for row in task_results
            )
            / count,
            "numeric_accuracy": sum(
                bool(row["score"]["numeric_correct"]) for row in task_results
            )
            / count,
            "task_accuracy": sum(
                bool(row["score"]["task_correct"]) for row in task_results
            )
            / count,
            "termination_rate": sum(
                bool(row["score"]["terminated"]) for row in task_results
            )
            / count,
        }
    total_generated = sum(
        int(row["completion"]["generated_tokens"]) for row in results
    )
    total_seconds = sum(
        float(row["completion"]["elapsed_seconds"]) for row in results
    )
    summary: dict[str, object] = {
        "checkpoint": checkpoint,
        "configuration": {
            "device": str(device),
            "evaluation_path": str(evaluation_path),
            "max_new_tokens": max_new_tokens,
            "max_per_task": max_per_task,
            "use_bfloat16": use_bfloat16 and device.type == "cuda",
        },
        "evaluation_file_sha256": sha256_file(Path(evaluation_path)),
        "metrics": {
            "examples": len(results),
            "format_valid_rate": sum(
                bool(row["score"]["format_valid"]) for row in results
            )
            / len(results),
            "generated_tokens": total_generated,
            "repeated_phrase_rate": sum(
                bool(row["score"]["repeated_phrase"]) for row in results
            )
            / len(results),
            "role_leak_rate": sum(
                bool(row["score"]["special_token_leak"]) for row in results
            )
            / len(results),
            "task_accuracy": sum(
                bool(row["score"]["task_correct"]) for row in results
            )
            / len(results),
            "tokens_per_second": (
                total_generated / total_seconds if total_seconds else None
            ),
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
        choices=tuple(sorted(SUPPORTED_ROUTES)),
        required=True,
    )
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-per-task", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--no-bfloat16", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    result = run_checkpoint_evaluation(
        checkpoint_path=arguments.checkpoint,
        checkpoint_sha256=arguments.checkpoint_sha256,
        expected_route=arguments.expected_route,
        evaluation_path=arguments.evaluation,
        output_path=arguments.output,
        device_name=arguments.device,
        max_per_task=arguments.max_per_task,
        max_new_tokens=arguments.max_new_tokens,
        use_bfloat16=not arguments.no_bfloat16,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
