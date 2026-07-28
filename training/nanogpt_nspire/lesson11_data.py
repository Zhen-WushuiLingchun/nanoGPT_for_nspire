"""Build the pinned FineWeb-Edu/OpenWebMath English base pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nanogpt_nspire.public_corpus import (
    PublicSourceSnapshot,
    build_public_pilot,
    scan_parquet_snapshot,
    select_public_documents,
)


MIB = 1024 * 1024
FINEWEB_EDU_SNAPSHOT = PublicSourceSnapshot(
    source_id="fineweb-edu",
    repository="HuggingFaceFW/fineweb-edu",
    revision="92cece42bcce787ee4af4619ab449fe48d86230d",
    parquet_path="sample-10BT/train/0000.parquet",
    row_groups=(0, 181, 363, 545),
    license_id="ODC-By-1.0",
    kind="fineweb_edu",
)
OPENWEBMATH_SNAPSHOT = PublicSourceSnapshot(
    source_id="openwebmath",
    repository="open-web-math/open-web-math",
    revision="c5476cfea8186f9db20fe4b45f43fa2e231aa9ba",
    parquet_path="default/train/0000.parquet",
    row_groups=(0, 14, 28, 42),
    license_id="ODC-By-1.0",
    kind="openwebmath",
)


def run_pinned_public_pilot(
    *,
    output_dir: str | Path,
    registry_path: str | Path,
    split_seed: str = "lesson11-public-v1",
    fineweb_max_utf8_bytes: int = 6 * MIB,
    openwebmath_max_utf8_bytes: int = 2 * MIB,
    max_documents_per_source: int = 4096,
) -> dict[str, object]:
    """Range-read, filter, hash-select, and atomically shard both sources."""

    selection_limits = {
        "fineweb-edu": fineweb_max_utf8_bytes,
        "openwebmath": openwebmath_max_utf8_bytes,
    }
    selected = []
    acquisition: dict[str, object] = {
        "method": (
            "exact Parquet commit, sparse row groups, source quality gate, "
            "SHA-256 rank, normalized-text deduplication, byte cap"
        ),
        "selection": {},
    }
    for snapshot in (FINEWEB_EDU_SNAPSHOT, OPENWEBMATH_SNAPSHOT):
        scanned, scan_report = scan_parquet_snapshot(snapshot)
        source_selected = select_public_documents(
            scanned,
            seed=f"{split_seed}:{snapshot.source_id}",
            max_documents=max_documents_per_source,
            max_utf8_bytes=selection_limits[snapshot.source_id],
        )
        selected.extend(source_selected)
        selected_bytes = sum(
            len(document.text.encode("utf-8"))
            for document in source_selected
        )
        acquisition["selection"][snapshot.source_id] = {
            "max_documents": max_documents_per_source,
            "max_utf8_bytes": selection_limits[snapshot.source_id],
            "scan": scan_report,
            "selected_documents": len(source_selected),
            "selected_utf8_bytes": selected_bytes,
            "selection_seed": f"{split_seed}:{snapshot.source_id}",
        }
    return build_public_pilot(
        selected,
        output_dir,
        registry_path=registry_path,
        split_seed=split_seed,
        acquisition=acquisition,
    )


def _build_parser() -> argparse.ArgumentParser:
    repository_root = Path(__file__).parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--registry",
        type=Path,
        default=(
            repository_root
            / "experiments"
            / "lesson10-public-sources.json"
        ),
    )
    parser.add_argument("--split-seed", default="lesson11-public-v1")
    parser.add_argument(
        "--fineweb-bytes",
        type=int,
        default=6 * MIB,
    )
    parser.add_argument(
        "--openwebmath-bytes",
        type=int,
        default=2 * MIB,
    )
    parser.add_argument(
        "--max-documents-per-source",
        type=int,
        default=4096,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    manifest = run_pinned_public_pilot(
        output_dir=arguments.output,
        registry_path=arguments.registry,
        split_seed=arguments.split_seed,
        fineweb_max_utf8_bytes=arguments.fineweb_bytes,
        openwebmath_max_utf8_bytes=arguments.openwebmath_bytes,
        max_documents_per_source=arguments.max_documents_per_source,
    )
    summary = {
        "files": manifest["files"],
        "output": str(arguments.output),
        "records": manifest["records"],
        "source_snapshots": manifest["source_snapshots"],
        "tokens": manifest["corpus"]["tokens"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
