"""Deterministic direct/short-CoT paired corpora for Lesson 14."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import unicodedata

from nanogpt_nspire.assistant_eval import load_evaluation_records
from nanogpt_nspire.base_corpus import stable_family_split
from nanogpt_nspire.byte_tokenizer import (
    SPECIAL_TOKEN_NAMES,
    TOKENIZER_SCHEMA_VERSION,
    VOCAB_SIZE,
)
from nanogpt_nspire.data import pack_u16_le
from nanogpt_nspire.lesson12_curriculum import (
    GSM8KExample,
    PhysicsExample,
)
from nanogpt_nspire.lesson12_data import PINNED_INPUTS
from nanogpt_nspire.math_curriculum import ArithmeticExample
from nanogpt_nspire.reasoning_format import (
    DIRECT_MODE,
    THINK_MODE,
    SUPPORTED_MODES,
    ReasoningFormatError,
    format_supervised_response,
)
from nanogpt_nspire.source_registry import (
    canonical_registry_bytes,
    license_is_eligible,
    load_source_registry,
)


LESSON14_SPLIT_SEED = "lesson12-domain-split-v1"
LESSON14_SCHEMA_VERSION = 1
_GSM_ANNOTATION = re.compile(r"<<[^<>]*>>")


class Lesson14DataError(ValueError):
    """Raised when a reasoning example or packed corpus is unsafe."""


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise Lesson14DataError(f"{name} must be non-empty")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise Lesson14DataError(f"{name} must be valid UTF-8") from error
    return value


def _normalize_text(value: object, name: str) -> str:
    value = _nonempty(value, name)
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = " ".join(normalized.split())
    if not normalized:
        raise Lesson14DataError(f"{name} must remain non-empty")
    return normalized


def _file_metadata(payload: bytes) -> dict[str, object]:
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _stable_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


@dataclass(frozen=True)
class Lesson14Example:
    """One problem with a verified final response and public short rationale."""

    record_id: str
    family_id: str
    task: str
    prompt: str
    reasoning: str
    final_answer: str
    exact_answer: str
    expected_unit: str | None
    source_id: str
    license_id: str

    def __post_init__(self) -> None:
        for name in (
            "record_id",
            "family_id",
            "task",
            "prompt",
            "reasoning",
            "final_answer",
            "exact_answer",
            "source_id",
            "license_id",
        ):
            _nonempty(getattr(self, name), name)
        if self.expected_unit is not None:
            _nonempty(self.expected_unit, "expected_unit")
        if self.exact_answer not in self.final_answer:
            raise Lesson14DataError(
                "final_answer must contain exact_answer verbatim"
            )
        upper = f"{self.prompt} {self.reasoning} {self.final_answer}".upper()
        if any(name.upper() in upper for name in SPECIAL_TOKEN_NAMES.values()):
            raise Lesson14DataError("text must not contain special-token names")


@dataclass(frozen=True)
class GSM8KReasoningExample:
    """One eligible GSM8K-train row with public worked text."""

    record_id: str
    family_id: str
    question: str
    reasoning: str
    exact_answer: str
    final_answer: str
    source_id: str = "gsm8k"
    license_id: str = "MIT"

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, object],
        *,
        row_index: int,
    ) -> GSM8KReasoningExample:
        if not isinstance(record, Mapping):
            raise Lesson14DataError("GSM8K row must be a mapping")
        if (
            isinstance(row_index, bool)
            or not isinstance(row_index, int)
            or row_index < 0
        ):
            raise Lesson14DataError("row_index must be non-negative")
        try:
            base = GSM8KExample.from_record(
                record,
                source_split="train",
                row_index=row_index,
            )
        except ValueError as error:
            raise Lesson14DataError(str(error)) from error
        raw_answer = _normalize_text(record.get("answer"), "answer")
        marker = raw_answer.rfind("####")
        if marker < 0:
            raise Lesson14DataError("GSM8K answer is missing final marker")
        reasoning = _GSM_ANNOTATION.sub("", raw_answer[:marker])
        reasoning = " ".join(reasoning.split()).strip()
        if not reasoning:
            raise Lesson14DataError("GSM8K rationale is empty")
        return cls(
            record_id=f"{base.record_id}-reasoning",
            family_id=base.family_id,
            question=base.question,
            reasoning=reasoning,
            exact_answer=base.exact_answer,
            final_answer=base.direct_answer,
        )

    def as_lesson14(self) -> Lesson14Example:
        return Lesson14Example(
            record_id=self.record_id,
            family_id=self.family_id,
            task="gsm8k",
            prompt=self.question,
            reasoning=self.reasoning,
            final_answer=self.final_answer,
            exact_answer=self.exact_answer,
            expected_unit=None,
            source_id=self.source_id,
            license_id=self.license_id,
        )


def _physics_reasoning(example: PhysicsExample) -> str:
    if example.formula_id == "kinetic-energy":
        substitution = (
            f"0.5 * {example.left} * {example.right}^2"
        )
    elif example.formula_id in {
        "density",
        "power",
        "pressure",
        "speed",
    }:
        substitution = f"{example.left} / {example.right}"
    else:
        substitution = f"{example.left} * {example.right}"
    return (
        f"Use {example.formula}. Substitute {substitution} = "
        f"{example.exact_answer} {example.unit}."
    )


def build_reasoning_examples(
    *,
    arithmetic: Iterable[ArithmeticExample],
    physics: Iterable[PhysicsExample],
    gsm8k_rows: Iterable[Mapping[str, object]],
) -> tuple[tuple[Lesson14Example, ...], dict[str, object]]:
    """Create verified project and GSM8K reasoning examples."""

    examples: list[Lesson14Example] = []
    for item in arithmetic:
        if not isinstance(item, ArithmeticExample):
            raise Lesson14DataError(
                "arithmetic must contain ArithmeticExample values"
            )
        examples.append(
            Lesson14Example(
                record_id=f"{item.example_id}-reasoning",
                family_id=item.family_id,
                task="arithmetic",
                prompt=item.question,
                reasoning=item.worked_answer,
                final_answer=item.direct_answer,
                exact_answer=item.exact_answer,
                expected_unit=None,
                source_id="project-arithmetic-v1",
                license_id="MIT",
            )
        )
    for item in physics:
        if not isinstance(item, PhysicsExample):
            raise Lesson14DataError(
                "physics must contain PhysicsExample values"
            )
        examples.append(
            Lesson14Example(
                record_id=f"{item.record_id}-reasoning",
                family_id=item.family_id,
                task="physics_numeric",
                prompt=item.question,
                reasoning=_physics_reasoning(item),
                final_answer=item.direct_answer,
                exact_answer=item.exact_answer,
                expected_unit=item.unit,
                source_id="project-arithmetic-v1",
                license_id="MIT",
            )
        )
    rejection_reasons: Counter[str] = Counter()
    gsm_rows = tuple(gsm8k_rows)
    for row_index, record in enumerate(gsm_rows):
        try:
            gsm = GSM8KReasoningExample.from_record(
                record,
                row_index=row_index,
            )
        except Lesson14DataError as error:
            rejection_reasons[str(error)] += 1
            continue
        examples.append(gsm.as_lesson14())
    examples.sort(key=lambda item: item.record_id)
    if not examples:
        raise Lesson14DataError("reasoning example set is empty")
    record_ids = {item.record_id for item in examples}
    if len(record_ids) != len(examples):
        raise Lesson14DataError("reasoning record IDs must be unique")
    return tuple(examples), {
        "accepted": len(examples),
        "accepted_by_task": dict(
            sorted(Counter(item.task for item in examples).items())
        ),
        "gsm8k_rows": len(gsm_rows),
        "rejected": sum(rejection_reasons.values()),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
    }


def select_paired_examples(
    examples: Iterable[Lesson14Example],
    *,
    context_limit: int,
    excluded_families: Iterable[str],
) -> tuple[tuple[Lesson14Example, ...], dict[str, object]]:
    """Keep only families fitting both modes and absent from frozen eval."""

    if (
        isinstance(context_limit, bool)
        or not isinstance(context_limit, int)
        or context_limit <= 0
    ):
        raise Lesson14DataError("context_limit must be positive")
    materialized = tuple(examples)
    if not materialized or any(
        not isinstance(item, Lesson14Example) for item in materialized
    ):
        raise Lesson14DataError(
            "examples must contain Lesson14Example values"
        )
    excluded = set(excluded_families)
    if any(not isinstance(item, str) or not item for item in excluded):
        raise Lesson14DataError("excluded families must be non-empty strings")
    selected: list[Lesson14Example] = []
    context_rejections: Counter[str] = Counter()
    frozen = 0
    for example in materialized:
        if example.family_id in excluded:
            frozen += 1
            continue
        failed = False
        for mode in (DIRECT_MODE, THINK_MODE):
            try:
                format_supervised_response(
                    prompt=example.prompt,
                    final_answer=example.final_answer,
                    mode=mode,
                    reasoning=(
                        example.reasoning if mode == THINK_MODE else None
                    ),
                    context_limit=context_limit,
                )
            except ReasoningFormatError:
                context_rejections[mode] += 1
                failed = True
        if not failed:
            selected.append(example)
    selected.sort(key=lambda item: item.record_id)
    if not selected:
        raise Lesson14DataError("no examples fit the paired mode contract")
    return tuple(selected), {
        "context_limit": context_limit,
        "context_rejections": dict(sorted(context_rejections.items())),
        "eligible": len(selected),
        "excluded_frozen_families": frozen,
        "input": len(materialized),
    }


def _validate_sources(
    examples: Sequence[Lesson14Example],
    registry_path: Path,
) -> str:
    registry = load_source_registry(registry_path)
    sources = {item.source_id: item for item in registry.sources}
    for example in examples:
        source = sources.get(example.source_id)
        if source is None:
            raise Lesson14DataError(
                f"unknown source {example.source_id}"
            )
        if (
            source.policy != "eligible"
            or source.license_id != example.license_id
            or not license_is_eligible(example.license_id)
        ):
            raise Lesson14DataError(
                f"source {example.source_id} is not eligible"
            )
    return hashlib.sha256(canonical_registry_bytes(registry)).hexdigest()


def build_mode_corpus(
    examples: Iterable[Lesson14Example],
    output_dir: str | Path,
    *,
    registry_path: str | Path,
    split_seed: str,
    modes: Sequence[str],
    context_limit: int,
) -> dict[str, object]:
    """Pack one or two mode variants with family-isolated splits."""

    destination = Path(output_dir)
    if destination.exists():
        raise Lesson14DataError(
            f"output destination already exists: {destination}"
        )
    materialized = tuple(examples)
    if not materialized or any(
        not isinstance(item, Lesson14Example) for item in materialized
    ):
        raise Lesson14DataError(
            "examples must contain Lesson14Example values"
        )
    selected_modes = tuple(modes)
    if (
        not selected_modes
        or len(set(selected_modes)) != len(selected_modes)
        or any(mode not in SUPPORTED_MODES for mode in selected_modes)
    ):
        raise Lesson14DataError("modes must be unique supported values")
    canonical_modes = tuple(
        mode
        for mode in (DIRECT_MODE, THINK_MODE)
        if mode in selected_modes
    )
    if not isinstance(split_seed, str) or not split_seed:
        raise Lesson14DataError("split_seed must be non-empty")
    registry = Path(registry_path)
    registry_sha256 = _validate_sources(materialized, registry)
    sorted_examples = tuple(
        sorted(materialized, key=lambda item: item.record_id)
    )
    if len({item.record_id for item in sorted_examples}) != len(
        sorted_examples
    ):
        raise Lesson14DataError("record IDs must be unique")

    split_rows: dict[
        str,
        list[tuple[Lesson14Example, str, tuple[int, ...], tuple[int, ...]]],
    ] = {name: [] for name in ("train", "validation", "test")}
    for example in sorted_examples:
        split = stable_family_split(
            example.family_id,
            split_seed=split_seed,
        )
        for mode in canonical_modes:
            try:
                tokens, mask = format_supervised_response(
                    prompt=example.prompt,
                    final_answer=example.final_answer,
                    mode=mode,
                    reasoning=(
                        example.reasoning if mode == THINK_MODE else None
                    ),
                    context_limit=context_limit,
                )
            except ReasoningFormatError as error:
                raise Lesson14DataError(
                    "preselected example exceeds context contract"
                ) from error
            split_rows[split].append((example, mode, tokens, mask))
    empty = [name for name, rows in split_rows.items() if not rows]
    if empty:
        raise Lesson14DataError(
            f"required corpus splits are empty: {', '.join(empty)}"
        )

    payloads: dict[str, bytes] = {}
    tokens_summary: dict[str, int] = {}
    target_summary: dict[str, int] = {}
    family_summary: dict[str, int] = {}
    record_summary: dict[str, int] = {}
    for split, rows in split_rows.items():
        token_stream: list[int] = []
        mask_stream = bytearray()
        families: set[str] = set()
        for example, _mode, tokens, mask in rows:
            token_stream.extend(tokens)
            mask_stream.extend(mask)
            families.add(example.family_id)
        payloads[f"{split}.tokens.bin"] = pack_u16_le(token_stream)
        payloads[f"{split}.loss.bin"] = bytes(mask_stream)
        tokens_summary[split] = len(token_stream)
        target_summary[split] = sum(mask_stream)
        family_summary[split] = len(families)
        record_summary[split] = len(rows)

    manifest: dict[str, object] = {
        "context_limit": context_limit,
        "eligible_targets": {
            **target_summary,
            "total": sum(target_summary.values()),
        },
        "families": {
            **family_summary,
            "total": len(
                {item.family_id for item in sorted_examples}
            ),
        },
        "files": {
            filename: _file_metadata(payload)
            for filename, payload in sorted(payloads.items())
        },
        "modes": list(canonical_modes),
        "records": {
            **record_summary,
            "total": sum(record_summary.values()),
        },
        "schema_version": LESSON14_SCHEMA_VERSION,
        "source_registry": {
            "path": registry.name,
            "sha256": registry_sha256,
        },
        "sources": dict(
            sorted(
                Counter(
                    item.source_id for item in sorted_examples
                ).items()
            )
        ),
        "split": {
            "kind": "sha256-family-90-5-5",
            "seed": split_seed,
        },
        "tokenizer": {
            "schema_version": TOKENIZER_SCHEMA_VERSION,
            "vocab_size": VOCAB_SIZE,
        },
        "tokens": {
            **tokens_summary,
            "total": sum(tokens_summary.values()),
        },
    }
    payloads["manifest.json"] = _stable_json_bytes(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )
    try:
        for filename, payload in payloads.items():
            _write_bytes(temporary / filename, payload)
        if destination.exists():
            raise Lesson14DataError(
                f"output destination appeared during build: {destination}"
            )
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return manifest


def _load_raw_jsonl(path: Path) -> tuple[Mapping[str, object], ...]:
    try:
        payload = path.read_bytes()
        text = payload.decode("utf-8", errors="strict")
    except (OSError, UnicodeError) as error:
        raise Lesson14DataError(f"invalid JSONL file: {path}") from error
    rows: list[Mapping[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise Lesson14DataError(
                f"invalid JSON at line {line_number}"
            ) from error
        if not isinstance(raw, Mapping):
            raise Lesson14DataError(
                f"JSONL row {line_number} must be an object"
            )
        rows.append(raw)
    if not rows:
        raise Lesson14DataError("JSONL file is empty")
    return tuple(rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_lesson14_data(
    *,
    gsm8k_train_path: str | Path,
    evaluation_path: str | Path,
    output_dir: str | Path,
    registry_path: str | Path,
    arithmetic: Iterable[ArithmeticExample],
    physics: Iterable[PhysicsExample],
) -> dict[str, object]:
    """Build all primary 256 and extension 512 SFT corpora atomically."""

    gsm_path = Path(gsm8k_train_path)
    expected = PINNED_INPUTS["gsm8k-train.jsonl"]
    if _sha256_file(gsm_path) != expected:
        raise Lesson14DataError("pinned GSM8K train hash mismatch")
    evaluation = load_evaluation_records(evaluation_path)
    frozen_families = {
        str(record["family_id"]) for record in evaluation
    }
    examples, example_report = build_reasoning_examples(
        arithmetic=arithmetic,
        physics=physics,
        gsm8k_rows=_load_raw_jsonl(gsm_path),
    )
    eligible_256, selection_256 = select_paired_examples(
        examples,
        context_limit=256,
        excluded_families=frozen_families,
    )
    eligible_512, selection_512 = select_paired_examples(
        examples,
        context_limit=512,
        excluded_families=frozen_families,
    )
    destination = Path(output_dir)
    if destination.exists():
        raise Lesson14DataError(
            f"output destination already exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )
    try:
        corpora = {
            "direct_256": build_mode_corpus(
                eligible_256,
                temporary / "direct_256",
                registry_path=registry_path,
                split_seed=LESSON14_SPLIT_SEED,
                modes=(DIRECT_MODE,),
                context_limit=256,
            ),
            "cot_256": build_mode_corpus(
                eligible_256,
                temporary / "cot_256",
                registry_path=registry_path,
                split_seed=LESSON14_SPLIT_SEED,
                modes=(THINK_MODE,),
                context_limit=256,
            ),
            "hybrid_256": build_mode_corpus(
                eligible_256,
                temporary / "hybrid_256",
                registry_path=registry_path,
                split_seed=LESSON14_SPLIT_SEED,
                modes=(DIRECT_MODE, THINK_MODE),
                context_limit=256,
            ),
            "hybrid_512": build_mode_corpus(
                eligible_512,
                temporary / "hybrid_512",
                registry_path=registry_path,
                split_seed=LESSON14_SPLIT_SEED,
                modes=(DIRECT_MODE, THINK_MODE),
                context_limit=512,
            ),
        }
        manifest: dict[str, object] = {
            "corpora": {
                name: {
                    "families": corpus["families"],
                    "manifest_sha256": _sha256_file(
                        temporary / name / "manifest.json"
                    ),
                    "modes": corpus["modes"],
                    "records": corpus["records"],
                    "tokens": corpus["tokens"],
                }
                for name, corpus in sorted(corpora.items())
            },
            "examples": example_report,
            "frozen_evaluation": {
                "families": len(frozen_families),
                "sha256": _sha256_file(Path(evaluation_path)),
                "training_eligible": False,
            },
            "gsm8k_train_sha256": expected,
            "schema_version": LESSON14_SCHEMA_VERSION,
            "selection": {
                "context_256": selection_256,
                "context_512": selection_512,
            },
        }
        _write_bytes(
            temporary / "manifest.json",
            _stable_json_bytes(manifest),
        )
        if destination.exists():
            raise Lesson14DataError(
                f"output destination appeared during build: {destination}"
            )
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsm8k-train", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--registry-path", type=Path, required=True)
    parser.add_argument("--arithmetic-count", type=int, default=12_000)
    parser.add_argument("--physics-count", type=int, default=4_000)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser


def main(argv: list[str] | None = None) -> int:
    from nanogpt_nspire.lesson12_curriculum import (
        generate_physics_examples,
    )
    from nanogpt_nspire.math_curriculum import (
        generate_arithmetic_examples,
    )

    arguments = _build_parser().parse_args(argv)
    manifest = build_lesson14_data(
        gsm8k_train_path=arguments.gsm8k_train,
        evaluation_path=arguments.evaluation,
        output_dir=arguments.output_dir,
        registry_path=arguments.registry_path,
        arithmetic=generate_arithmetic_examples(
            count=arguments.arithmetic_count,
            seed=arguments.seed,
        ),
        physics=generate_physics_examples(
            count=arguments.physics_count,
            seed=arguments.seed,
        ),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

