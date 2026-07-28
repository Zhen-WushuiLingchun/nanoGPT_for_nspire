"""Deterministic Lesson 12 public selection and staged-corpus construction."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, Mapping, Sequence

from nanogpt_nspire.base_corpus import (
    CorpusRecord,
    build_corpus,
    stable_family_split,
)
from nanogpt_nspire.base_train import load_packed_dataset
from nanogpt_nspire.byte_tokenizer import (
    TOKENIZER_SCHEMA_VERSION,
    VOCAB_SIZE,
    ConversationTurn,
)
from nanogpt_nspire.lesson12_curriculum import (
    CurriculumError,
    GSM8KExample,
    OASSTPair,
    PhysicsExample,
    canonical_jsonl_bytes,
    generate_physics_examples,
)
from nanogpt_nspire.math_curriculum import (
    ArithmeticExample,
    generate_arithmetic_examples,
)


COMPOSITE_SCHEMA_VERSION = 1
LESSON12_SPLIT_SEED = "lesson12-domain-v1"
LESSON12_OASST_SEED = "lesson12-oasst-v1"
GSM8K_REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
OASST1_REVISION = "fdf72ae0827c1cda404aff25b6603abec9e3399b"
PINNED_INPUTS = {
    "gsm8k-test.jsonl": (
        "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
    ),
    "gsm8k-train.jsonl": (
        "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465"
    ),
    "oasst1/data/train-00000-of-00001-b42a775f407cee45.parquet": (
        "bbfadf5ed1278ba2208c837fdcad865adf65f5df55d80abadab2745db13fcb5e"
    ),
    "oasst1/data/validation-00000-of-00001-134b8fd0c89408b6.parquet": (
        "24002597bb13a7edd42d92f773762f25e285f72c31a70449393d0ded1dc7b416"
    ),
}


class Lesson12DataError(ValueError):
    """Raised when staged data is incomplete, unstable, or contaminated."""


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lesson12DataError(f"{name} must be a positive integer")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def load_gsm8k_jsonl(
    path: str | Path,
    *,
    source_split: str,
) -> tuple[tuple[GSM8KExample, ...], dict[str, object]]:
    """Parse a pinned official GSM8K JSONL file and report every rejection."""

    source = Path(path)
    try:
        payload = source.read_bytes()
        text = payload.decode("utf-8", errors="strict")
    except (OSError, UnicodeDecodeError) as error:
        raise Lesson12DataError(f"invalid GSM8K file: {source}") from error
    examples: list[GSM8KExample] = []
    rejection_reasons: Counter[str] = Counter()
    row_count = 0
    for row_index, line in enumerate(text.splitlines()):
        if not line:
            continue
        row_count += 1
        try:
            raw = json.loads(line)
            if not isinstance(raw, Mapping):
                raise CurriculumError("GSM8K row must be a JSON object")
            example = GSM8KExample.from_record(
                raw,
                source_split=source_split,
                row_index=row_index,
            )
        except (json.JSONDecodeError, CurriculumError) as error:
            rejection_reasons[str(error)] += 1
            continue
        examples.append(example)
    if not examples:
        raise Lesson12DataError("GSM8K file produced no eligible examples")
    return tuple(examples), {
        "accepted": len(examples),
        "bytes": len(payload),
        "rejected": sum(rejection_reasons.values()),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "rows": row_count,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "source_split": source_split,
    }


def _label_value(message: Mapping[str, object], name: str) -> float | None:
    labels = message.get("labels")
    if not isinstance(labels, Mapping):
        return None
    names = labels.get("name")
    values = labels.get("value")
    if (
        not isinstance(names, Sequence)
        or isinstance(names, (str, bytes, bytearray))
        or not isinstance(values, Sequence)
        or isinstance(values, (str, bytes, bytearray))
        or len(names) != len(values)
    ):
        return None
    for label_name, value in zip(names, values, strict=True):
        if label_name == name and isinstance(value, (int, float)) and not isinstance(
            value,
            bool,
        ):
            return float(value)
    return None


def select_oasst_pairs(
    rows: Iterable[Mapping[str, object]],
    *,
    seed: str,
    max_pairs: int,
) -> tuple[tuple[OASSTPair, ...], dict[str, object]]:
    """Select hash-ranked, reviewed, short English OASST1 root/reply pairs."""

    if not isinstance(seed, str) or not seed:
        raise Lesson12DataError("seed must be a non-empty string")
    max_pairs = _positive_integer(max_pairs, "max_pairs")
    materialized = tuple(rows)
    if not materialized:
        raise Lesson12DataError("OASST1 selection requires rows")
    if any(not isinstance(row, Mapping) for row in materialized):
        raise Lesson12DataError("every OASST1 row must be a mapping")
    by_message_id: dict[str, Mapping[str, object]] = {}
    for row in materialized:
        message_id = row.get("message_id")
        if not isinstance(message_id, str) or not message_id:
            continue
        previous = by_message_id.get(message_id)
        if previous is not None and dict(previous) != dict(row):
            raise Lesson12DataError(
                f"conflicting OASST1 message_id {message_id}"
            )
        by_message_id[message_id] = row

    pairs: list[OASSTPair] = []
    rejection_reasons: Counter[str] = Counter()
    for row in materialized:
        if row.get("role") != "assistant":
            continue
        parent_id = row.get("parent_id")
        prompt = (
            by_message_id.get(parent_id)
            if isinstance(parent_id, str)
            else None
        )
        if prompt is None:
            rejection_reasons["assistant parent is unavailable"] += 1
            continue
        if (
            row.get("review_result") is not True
            or prompt.get("review_result") is not True
        ):
            rejection_reasons["pair is not positively reviewed"] += 1
            continue
        if (
            row.get("tree_state") != "ready_for_export"
            or prompt.get("tree_state") != "ready_for_export"
        ):
            rejection_reasons["tree is not ready_for_export"] += 1
            continue
        if row.get("synthetic") is not False or prompt.get("synthetic") is not False:
            rejection_reasons["synthetic pair is excluded"] += 1
            continue
        quality = _label_value(row, "quality")
        toxicity = _label_value(row, "toxicity")
        if quality is None or quality < 0.5:
            rejection_reasons["assistant quality is below 0.5"] += 1
            continue
        if toxicity is not None and toxicity > 0.5:
            rejection_reasons["assistant toxicity exceeds 0.5"] += 1
            continue
        try:
            pair = OASSTPair.from_messages(prompt, row)
        except CurriculumError as error:
            rejection_reasons[str(error)] += 1
            continue
        pairs.append(pair)
    unique_by_record = {pair.record_id: pair for pair in pairs}
    ranked = sorted(
        unique_by_record.values(),
        key=lambda pair: (
            hashlib.sha256(
                f"{seed}:{pair.record_id}".encode("utf-8")
            ).digest(),
            pair.record_id,
        ),
    )
    selected = tuple(ranked[:max_pairs])
    if not selected:
        raise Lesson12DataError("OASST1 selection produced no eligible pairs")
    return selected, {
        "accepted_before_limit": len(unique_by_record),
        "input_rows": len(materialized),
        "max_pairs": max_pairs,
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "seed": seed,
        "selected": len(selected),
    }


def load_pinned_oasst_rows(
    download_dir: str | Path,
) -> tuple[tuple[Mapping[str, object], ...], dict[str, object]]:
    """Read the two exact OASST1 Parquet files after SHA-256 verification."""

    root = Path(download_dir)
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise Lesson12DataError("OASST1 ingestion requires pyarrow") from error
    rows: list[Mapping[str, object]] = []
    files: list[dict[str, object]] = []
    for relative, expected_sha256 in sorted(PINNED_INPUTS.items()):
        if not relative.endswith(".parquet"):
            continue
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise Lesson12DataError(
                f"pinned input hash mismatch for {relative}"
            )
        table = parquet.read_table(path)
        rows.extend(table.to_pylist())
        files.append(
            {
                "bytes": path.stat().st_size,
                "path": relative,
                "rows": table.num_rows,
                "sha256": actual_sha256,
            }
        )
    if not rows:
        raise Lesson12DataError("pinned OASST1 inputs produced no rows")
    return tuple(rows), {
        "files": files,
        "repository": "OpenAssistant/oasst1",
        "revision": OASST1_REVISION,
        "rows": len(rows),
    }


def build_domain_records(
    *,
    arithmetic: Iterable[ArithmeticExample],
    physics: Iterable[PhysicsExample],
    gsm8k_train: Iterable[GSM8KExample],
    oasst_pairs: Iterable[OASSTPair],
) -> tuple[tuple[CorpusRecord, ...], tuple[CorpusRecord, ...]]:
    """Create separate base-style CPT and role-aware SFT records."""

    arithmetic_examples = tuple(arithmetic)
    physics_examples = tuple(physics)
    gsm_examples = tuple(gsm8k_train)
    assistant_pairs = tuple(oasst_pairs)
    if not all(
        (
            arithmetic_examples,
            physics_examples,
            gsm_examples,
            assistant_pairs,
        )
    ):
        raise Lesson12DataError(
            "arithmetic, physics, GSM8K, and OASST1 inputs must be nonempty"
        )
    if any(
        not isinstance(item, ArithmeticExample)
        for item in arithmetic_examples
    ):
        raise Lesson12DataError("arithmetic items must be ArithmeticExample")
    if any(not isinstance(item, PhysicsExample) for item in physics_examples):
        raise Lesson12DataError("physics items must be PhysicsExample")
    if any(not isinstance(item, GSM8KExample) for item in gsm_examples):
        raise Lesson12DataError("GSM8K items must be GSM8KExample")
    if any(item.source_split != "train" for item in gsm_examples):
        raise Lesson12DataError(
            "only the original GSM8K train split may enter training records"
        )
    if any(not isinstance(item, OASSTPair) for item in assistant_pairs):
        raise Lesson12DataError("OASST1 items must be OASSTPair")

    cpt: list[CorpusRecord] = []
    sft: list[CorpusRecord] = []
    for example in arithmetic_examples:
        cpt.append(
            CorpusRecord.base(
                record_id=f"{example.example_id}-cpt",
                family_id=example.family_id,
                text=(
                    f"Arithmetic problem: {example.question}\n"
                    f"Solution: {example.worked_answer}"
                ),
                source_id="project-arithmetic-v1",
                license_id="MIT",
            )
        )
        for style, answer in (
            ("direct", example.direct_answer),
            ("worked", example.worked_answer),
        ):
            sft.append(
                CorpusRecord.conversation(
                    record_id=f"{example.example_id}-sft-{style}",
                    family_id=example.family_id,
                    turns=(
                        ConversationTurn("user", example.question),
                        ConversationTurn("assistant", answer),
                    ),
                    source_id="project-arithmetic-v1",
                    license_id="MIT",
                )
            )
    for example in physics_examples:
        cpt.append(
            CorpusRecord.base(
                record_id=f"{example.record_id}-cpt",
                family_id=example.family_id,
                text=(
                    f"Physics problem: {example.question}\n"
                    f"Solution: {example.worked_answer}"
                ),
                source_id="project-arithmetic-v1",
                license_id="MIT",
            )
        )
        for style, answer in (
            ("direct", example.direct_answer),
            ("worked", example.worked_answer),
        ):
            sft.append(
                CorpusRecord.conversation(
                    record_id=f"{example.record_id}-sft-{style}",
                    family_id=example.family_id,
                    turns=(
                        ConversationTurn("user", example.question),
                        ConversationTurn("assistant", answer),
                    ),
                    source_id="project-arithmetic-v1",
                    license_id="MIT",
                )
            )
    for example in gsm_examples:
        cpt.append(
            CorpusRecord.base(
                record_id=f"{example.record_id}-cpt",
                family_id=example.family_id,
                text=(
                    f"Math word problem: {example.question}\n"
                    f"Answer: {example.direct_answer}"
                ),
                source_id="gsm8k",
                license_id="MIT",
            )
        )
        sft.append(
            CorpusRecord.conversation(
                record_id=f"{example.record_id}-sft",
                family_id=example.family_id,
                turns=(
                    ConversationTurn("user", example.question),
                    ConversationTurn("assistant", example.direct_answer),
                ),
                source_id="gsm8k",
                license_id="MIT",
            )
        )
    for pair in assistant_pairs:
        sft.append(
            CorpusRecord.conversation(
                record_id=f"{pair.record_id}-sft",
                family_id=pair.family_id,
                turns=(
                    ConversationTurn("user", pair.question),
                    ConversationTurn("assistant", pair.answer),
                ),
                source_id="oasst1",
                license_id="Apache-2.0",
            )
        )
    cpt.sort(key=lambda item: item.record_id)
    sft.sort(key=lambda item: item.record_id)
    if len({record.record_id for record in cpt}) != len(cpt):
        raise Lesson12DataError("CPT record IDs must be unique")
    if len({record.record_id for record in sft}) != len(sft):
        raise Lesson12DataError("SFT record IDs must be unique")
    return tuple(cpt), tuple(sft)


def compose_packed_corpora(
    components: Iterable[tuple[str, str | Path]],
    output_dir: str | Path,
) -> dict[str, object]:
    """Atomically concatenate already verified packed corpora split by split."""

    destination = Path(output_dir)
    if destination.exists():
        raise Lesson12DataError(
            f"output destination already exists: {destination}"
        )
    materialized = tuple(components)
    if not materialized:
        raise Lesson12DataError("composite corpus requires components")
    names = [name for name, _ in materialized]
    if any(not isinstance(name, str) or not name for name in names):
        raise Lesson12DataError("component names must be non-empty strings")
    if len(set(names)) != len(names):
        raise Lesson12DataError("component names must be unique")

    datasets = []
    component_metadata: list[dict[str, object]] = []
    for name, source in materialized:
        try:
            dataset = load_packed_dataset(source)
        except (OSError, ValueError) as error:
            raise Lesson12DataError(
                f"component {name} disagrees with its manifest"
            ) from error
        datasets.append((name, dataset))
        eligible_targets = {
            split_name: int(getattr(dataset, split_name).loss_mask.sum())
            for split_name in ("train", "validation", "test")
        }
        component_metadata.append(
            {
                "eligible_targets": eligible_targets,
                "manifest_sha256": _sha256_file(dataset.manifest_path),
                "name": name,
                "tokens": {
                    split_name: getattr(dataset, split_name).token_count
                    for split_name in ("train", "validation", "test")
                },
            }
        )

    file_payloads: dict[str, bytes] = {}
    token_counts: dict[str, int] = {}
    eligible_counts: dict[str, int] = {}
    for split_name in ("train", "validation", "test"):
        token_payload = b"".join(
            getattr(dataset, split_name).token_path.read_bytes()
            for _, dataset in datasets
        )
        mask_payload = b"".join(
            getattr(dataset, split_name).mask_path.read_bytes()
            for _, dataset in datasets
        )
        if len(token_payload) != 2 * len(mask_payload):
            raise Lesson12DataError(
                f"{split_name} composite token/mask length mismatch"
            )
        file_payloads[f"{split_name}.tokens.bin"] = token_payload
        file_payloads[f"{split_name}.loss.bin"] = mask_payload
        token_counts[split_name] = len(mask_payload)
        eligible_counts[split_name] = sum(mask_payload)

    replay_train_tokens = sum(
        int(metadata["tokens"]["train"])
        for metadata in component_metadata
        if metadata["name"] == "general_replay"
    )
    manifest: dict[str, object] = {
        "components": component_metadata,
        "eligible_targets": {
            **eligible_counts,
            "total": sum(eligible_counts.values()),
        },
        "files": {
            filename: _file_metadata(payload)
            for filename, payload in sorted(file_payloads.items())
        },
        "general_replay_train_fraction": (
            replay_train_tokens / token_counts["train"]
            if token_counts["train"]
            else 0.0
        ),
        "schema_version": COMPOSITE_SCHEMA_VERSION,
        "split": {
            "kind": "verified-component-concatenation-v1",
        },
        "tokenizer": {
            "schema_version": TOKENIZER_SCHEMA_VERSION,
            "vocab_size": VOCAB_SIZE,
        },
        "tokens": {
            **token_counts,
            "total": sum(token_counts.values()),
        },
    }
    manifest_payload = _stable_json_bytes(manifest)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )
    try:
        for filename, payload in file_payloads.items():
            _write_bytes(temporary / filename, payload)
        _write_bytes(temporary / "manifest.json", manifest_payload)
        if destination.exists():
            raise Lesson12DataError(
                f"output destination appeared during build: {destination}"
            )
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return manifest


def _verify_pinned_gsm8k_file(
    download_dir: Path,
    filename: str,
) -> Path:
    path = download_dir / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = PINNED_INPUTS[filename]
    if _sha256_file(path) != expected:
        raise Lesson12DataError(f"pinned input hash mismatch for {filename}")
    return path


def _evaluation_rows(
    *,
    arithmetic: Sequence[ArithmeticExample],
    physics: Sequence[PhysicsExample],
    gsm8k_test: Sequence[GSM8KExample],
    split_seed: str,
    max_per_task: int,
) -> tuple[dict[str, object], ...]:
    _positive_integer(max_per_task, "max_per_task")

    def rank(
        rows: Iterable[dict[str, object]],
        task: str,
    ) -> list[dict[str, object]]:
        return sorted(
            rows,
            key=lambda row: (
                hashlib.sha256(
                    f"lesson12-eval:{task}:{row['family_id']}".encode(
                        "utf-8"
                    )
                ).digest(),
                str(row["family_id"]),
            ),
        )[:max_per_task]

    arithmetic_rows = rank(
        (
            {
                "expected_answer": item.exact_answer,
                "expected_unit": None,
                "family_id": item.family_id,
                "prompt": item.question,
                "source_id": "project-arithmetic-v1",
                "task": "arithmetic",
            }
            for item in arithmetic
            if stable_family_split(
                item.family_id,
                split_seed=split_seed,
            )
            == "test"
        ),
        "arithmetic",
    )
    arithmetic_easy_rows = list(
        build_easy_arithmetic_evaluation(
            training=arithmetic,
            split_seed=split_seed,
            max_records=max_per_task,
        )
    )
    physics_rows = rank(
        (
            {
                "expected_answer": item.exact_answer,
                "expected_unit": item.unit,
                "family_id": item.family_id,
                "formula_id": item.formula_id,
                "prompt": item.question,
                "source_id": "project-arithmetic-v1",
                "task": "physics_numeric",
            }
            for item in physics
            if stable_family_split(
                item.family_id,
                split_seed=split_seed,
            )
            == "test"
        ),
        "physics_numeric",
    )
    official_test_rows = rank(
        (
            {
                "expected_answer": item.exact_answer,
                "expected_unit": None,
                "family_id": item.family_id,
                "prompt": item.question,
                "source_id": "gsm8k",
                "source_split": "test",
                "task": "gsm8k",
            }
            for item in gsm8k_test
        ),
        "gsm8k",
    )
    if (
        not arithmetic_rows
        or not arithmetic_easy_rows
        or not physics_rows
        or not official_test_rows
    ):
        raise Lesson12DataError(
            "evaluation requires arithmetic, easy arithmetic, physics, "
            "and GSM8K rows"
        )
    return tuple(
        arithmetic_rows
        + arithmetic_easy_rows
        + physics_rows
        + official_test_rows
    )


def build_easy_arithmetic_evaluation(
    *,
    training: Sequence[ArithmeticExample],
    split_seed: str,
    max_records: int,
) -> tuple[dict[str, object], ...]:
    """Build a 0--20 arithmetic holdout absent from all training families."""

    max_records = _positive_integer(max_records, "max_records")
    if not isinstance(split_seed, str) or not split_seed:
        raise Lesson12DataError("split_seed must be non-empty")
    if not training or any(
        not isinstance(item, ArithmeticExample) for item in training
    ):
        raise Lesson12DataError(
            "training must contain ArithmeticExample values"
        )
    training_families = {item.family_id for item in training}
    candidates: dict[str, ArithmeticExample] = {}
    for left in range(21):
        for right in range(21):
            for operator in ("+", "-", "*"):
                example = ArithmeticExample.create(
                    left=left,
                    operator=operator,
                    right=right,
                )
                candidates[example.family_id] = example
    for quotient in range(21):
        for divisor in range(1, 13):
            example = ArithmeticExample.create(
                left=quotient * divisor,
                operator="/",
                right=divisor,
            )
            candidates[example.family_id] = example
    eligible = [
        item
        for item in candidates.values()
        if item.family_id not in training_families
        and stable_family_split(
            item.family_id,
            split_seed=split_seed,
        )
        == "test"
    ]
    ranked = sorted(
        eligible,
        key=lambda item: (
            hashlib.sha256(
                f"lesson12-easy:{item.family_id}".encode("utf-8")
            ).digest(),
            item.family_id,
        ),
    )
    required_minimum = min(max_records, 32)
    if len(ranked) < required_minimum:
        raise Lesson12DataError(
            "not enough held-out easy arithmetic families"
        )
    return tuple(
        {
            "difficulty": "easy-0-to-20",
            "expected_answer": item.exact_answer,
            "expected_unit": None,
            "family_id": item.family_id,
            "prompt": item.question,
            "source_id": "project-arithmetic-v1",
            "task": "arithmetic_easy",
        }
        for item in ranked[:max_records]
    )


def build_lesson12_data(
    *,
    download_dir: str | Path,
    lesson11_data_dir: str | Path,
    output_dir: str | Path,
    registry_path: str | Path,
    arithmetic_count: int = 12_000,
    physics_count: int = 4_000,
    oasst_max_pairs: int = 512,
    evaluation_max_per_task: int = 256,
    seed: int = 20260728,
) -> dict[str, object]:
    """Build CPT, SFT, and untouched evaluation artifacts atomically."""

    arithmetic_count = _positive_integer(
        arithmetic_count,
        "arithmetic_count",
    )
    physics_count = _positive_integer(physics_count, "physics_count")
    oasst_max_pairs = _positive_integer(
        oasst_max_pairs,
        "oasst_max_pairs",
    )
    evaluation_max_per_task = _positive_integer(
        evaluation_max_per_task,
        "evaluation_max_per_task",
    )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise Lesson12DataError("seed must be an integer")
    destination = Path(output_dir)
    if destination.exists():
        raise Lesson12DataError(
            f"output destination already exists: {destination}"
        )
    downloads = Path(download_dir)
    registry = Path(registry_path)

    gsm_train_path = _verify_pinned_gsm8k_file(
        downloads,
        "gsm8k-train.jsonl",
    )
    gsm_test_path = _verify_pinned_gsm8k_file(
        downloads,
        "gsm8k-test.jsonl",
    )
    gsm_train, gsm_train_report = load_gsm8k_jsonl(
        gsm_train_path,
        source_split="train",
    )
    gsm_test, gsm_test_report = load_gsm8k_jsonl(
        gsm_test_path,
        source_split="test",
    )
    train_families = {item.family_id for item in gsm_train}
    test_families = {item.family_id for item in gsm_test}
    overlap = train_families & test_families
    if overlap:
        raise Lesson12DataError(
            "GSM8K original train/test family overlap detected"
        )
    oasst_rows, oasst_input_report = load_pinned_oasst_rows(downloads)
    oasst_pairs, oasst_selection_report = select_oasst_pairs(
        oasst_rows,
        seed=LESSON12_OASST_SEED,
        max_pairs=oasst_max_pairs,
    )
    arithmetic = generate_arithmetic_examples(
        count=arithmetic_count,
        seed=seed,
    )
    physics = generate_physics_examples(
        count=physics_count,
        seed=seed,
    )
    cpt_records, sft_records = build_domain_records(
        arithmetic=arithmetic,
        physics=physics,
        gsm8k_train=gsm_train,
        oasst_pairs=oasst_pairs,
    )
    evaluation = _evaluation_rows(
        arithmetic=arithmetic,
        physics=physics,
        gsm8k_test=gsm_test,
        split_seed=LESSON12_SPLIT_SEED,
        max_per_task=evaluation_max_per_task,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )
    try:
        domain_manifest = build_corpus(
            cpt_records,
            temporary / "domain_cpt",
            registry_path=registry,
            split_seed=LESSON12_SPLIT_SEED,
        )
        cpt_manifest = compose_packed_corpora(
            (
                ("general_replay", lesson11_data_dir),
                ("domain", temporary / "domain_cpt"),
            ),
            temporary / "cpt",
        )
        sft_manifest = build_corpus(
            sft_records,
            temporary / "sft",
            registry_path=registry,
            split_seed=LESSON12_SPLIT_SEED,
        )
        evaluation_payload = canonical_jsonl_bytes(evaluation)
        _write_bytes(temporary / "evaluation.jsonl", evaluation_payload)
        provenance = {
            "generated": {
                "arithmetic_count": len(arithmetic),
                "arithmetic_generator": "project-arithmetic-v1",
                "physics_count": len(physics),
                "physics_formulas": sorted(
                    {item.formula_id for item in physics}
                ),
                "seed": seed,
            },
            "gsm8k": {
                "repository": "openai/grade-school-math",
                "revision": GSM8K_REVISION,
                "test": gsm_test_report,
                "test_is_training_eligible": False,
                "train": gsm_train_report,
            },
            "oasst1": {
                "input": oasst_input_report,
                "selection": oasst_selection_report,
            },
            "schema_version": 1,
        }
        _write_bytes(
            temporary / "provenance.json",
            _stable_json_bytes(provenance),
        )
        files = {
            path.relative_to(temporary).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        }
        evaluation_counts = Counter(
            str(row["task"]) for row in evaluation
        )
        manifest: dict[str, object] = {
            "corpora": {
                "cpt": cpt_manifest,
                "domain_cpt": domain_manifest,
                "sft": sft_manifest,
            },
            "evaluation": {
                "counts": dict(sorted(evaluation_counts.items())),
                "families": len(
                    {str(row["family_id"]) for row in evaluation}
                ),
                "records": len(evaluation),
                "sha256": hashlib.sha256(
                    evaluation_payload
                ).hexdigest(),
            },
            "files": files,
            "provenance": provenance,
            "schema_version": 1,
            "split_seed": LESSON12_SPLIT_SEED,
        }
        _write_bytes(
            temporary / "manifest.json",
            _stable_json_bytes(manifest),
        )
        if destination.exists():
            raise Lesson12DataError(
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
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--lesson11-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--registry-path", type=Path, required=True)
    parser.add_argument("--arithmetic-count", type=int, default=12_000)
    parser.add_argument("--physics-count", type=int, default=4_000)
    parser.add_argument("--oasst-max-pairs", type=int, default=512)
    parser.add_argument(
        "--evaluation-max-per-task",
        type=int,
        default=256,
    )
    parser.add_argument("--seed", type=int, default=20260728)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    result = build_lesson12_data(
        download_dir=arguments.download_dir,
        lesson11_data_dir=arguments.lesson11_data_dir,
        output_dir=arguments.output_dir,
        registry_path=arguments.registry_path,
        arithmetic_count=arguments.arithmetic_count,
        physics_count=arguments.physics_count,
        oasst_max_pairs=arguments.oasst_max_pairs,
        evaluation_max_per_task=arguments.evaluation_max_per_task,
        seed=arguments.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
