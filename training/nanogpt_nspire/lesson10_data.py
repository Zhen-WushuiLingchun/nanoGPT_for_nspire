"""Build the deterministic Lesson 10 English arithmetic smoke corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Sequence

from nanogpt_nspire.base_corpus import (
    SPLIT_KIND,
    CorpusError,
    CorpusRecord,
    build_corpus,
)
from nanogpt_nspire.byte_tokenizer import ConversationTurn, VOCAB_SIZE
from nanogpt_nspire.math_curriculum import (
    ArithmeticError,
    generate_arithmetic_examples,
    verify_arithmetic_example,
)


DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "lesson10-public-sources.json"
)


def _verified_remove_corpus(destination: Path) -> None:
    resolved = destination.resolve()
    if resolved.parent == resolved:
        raise CorpusError("refusing to replace a filesystem root")
    if destination.is_symlink() or not destination.is_dir():
        raise CorpusError("replace target is not a verified Lesson 10 corpus")
    manifest_path = destination / "manifest.json"
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8", errors="strict")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CorpusError(
            "replace target is not a verified Lesson 10 corpus"
        ) from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or not isinstance(manifest.get("tokenizer"), dict)
        or manifest["tokenizer"].get("vocab_size") != VOCAB_SIZE
        or not isinstance(manifest.get("split"), dict)
        or manifest["split"].get("kind") != SPLIT_KIND
    ):
        raise CorpusError("replace target is not a verified Lesson 10 corpus")
    shutil.rmtree(destination)


def _smoke_records(example_count: int, seed: int) -> tuple[CorpusRecord, ...]:
    try:
        examples = generate_arithmetic_examples(
            count=example_count,
            seed=seed,
        )
    except ArithmeticError as error:
        raise CorpusError(str(error)) from error

    records: list[CorpusRecord] = []
    for example in examples:
        if not verify_arithmetic_example(example):
            raise CorpusError(
                f"generated family {example.family_id} failed exact verification"
            )
        for style, answer in (
            ("direct", example.direct_answer),
            ("worked", example.worked_answer),
        ):
            records.append(
                CorpusRecord.conversation(
                    record_id=example.variant_id(
                        style=style,
                        question=example.question,
                    ),
                    family_id=example.family_id,
                    turns=(
                        ConversationTurn("user", example.question),
                        ConversationTurn("assistant", answer),
                    ),
                    source_id="project-arithmetic-v1",
                    license_id="MIT",
                )
            )
    return tuple(records)


def run_smoke(
    output_dir: str | Path,
    *,
    seed: int,
    example_count: int,
    replace: bool = False,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, object]:
    """Generate, verify, and build one project-authored smoke corpus."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise CorpusError("seed must be an integer")
    if (
        isinstance(example_count, bool)
        or not isinstance(example_count, int)
        or example_count <= 0
    ):
        raise CorpusError("example_count must be a positive integer")
    if not isinstance(replace, bool):
        raise CorpusError("replace must be boolean")

    destination = Path(output_dir)
    if destination.exists():
        if not replace:
            raise CorpusError(f"output destination already exists: {destination}")
        _verified_remove_corpus(destination)

    records = _smoke_records(example_count, seed)
    manifest = build_corpus(
        records,
        destination,
        registry_path=registry_path,
        split_seed=f"lesson10-smoke-{seed}",
    )
    manifest_payload = (destination / "manifest.json").read_bytes()
    return {
        "families": manifest["families"],
        "files": manifest["files"],
        "generated_families": example_count,
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "output": str(destination),
        "records": manifest["records"],
        "replace": replace,
        "seed": seed,
        "tokens": manifest["tokens"],
        "vocab_size": VOCAB_SIZE,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build deterministic Lesson 10 data artifacts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser(
        "smoke",
        help="Build the exact project-authored arithmetic smoke corpus.",
    )
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--seed", type=int, default=20260728)
    smoke.add_argument("--examples", type=int, default=256)
    smoke.add_argument("--replace", action="store_true")
    smoke.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Lesson 10 data command."""

    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        summary = run_smoke(
            arguments.output,
            seed=arguments.seed,
            example_count=arguments.examples,
            replace=arguments.replace,
            registry_path=arguments.registry,
        )
    except (CorpusError, OSError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
