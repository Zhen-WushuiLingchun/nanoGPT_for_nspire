import hashlib
import json
from pathlib import Path

from nanogpt_nspire.base_train import load_packed_dataset
from nanogpt_nspire.byte_tokenizer import FINAL_ID, THINK_ID
from nanogpt_nspire.lesson12_curriculum import PhysicsExample
from nanogpt_nspire.lesson14_data import (
    LESSON14_SPLIT_SEED,
    GSM8KReasoningExample,
    build_mode_corpus,
    build_reasoning_examples,
    select_paired_examples,
)
from nanogpt_nspire.math_curriculum import (
    ArithmeticExample,
    generate_arithmetic_examples,
)
from nanogpt_nspire.reasoning_format import DIRECT_MODE, THINK_MODE


REGISTRY_PATH = (
    Path(__file__).parents[2] / "experiments" / "lesson10-public-sources.json"
)


def _gsm_record() -> dict[str, str]:
    return {
        "question": "A box has 12 rows of 7 bolts. How many bolts are there?",
        "answer": (
            "There are <<12*7=84>>84 bolts in the box.\n"
            "#### 84"
        ),
    }


def test_gsm8k_reasoning_parser_keeps_public_rationale_not_annotations() -> None:
    example = GSM8KReasoningExample.from_record(
        _gsm_record(),
        row_index=9,
    )

    assert example.exact_answer == "84"
    assert example.final_answer == "The answer is 84."
    assert example.reasoning == "There are 84 bolts in the box."
    assert "<<" not in example.reasoning
    assert "####" not in example.reasoning
    assert example.source_id == "gsm8k"


def test_project_reasoning_is_exact_and_explicit() -> None:
    arithmetic = ArithmeticExample.create(
        left=12,
        operator="*",
        right=7,
    )
    physics = PhysicsExample.create(
        formula_id="force",
        left=12,
        right=7,
    )
    examples, report = build_reasoning_examples(
        arithmetic=(arithmetic,),
        physics=(physics,),
        gsm8k_rows=(_gsm_record(),),
    )

    by_task = {item.task: item for item in examples}
    assert by_task["arithmetic"].reasoning == "12 * 7 = 84."
    assert "12 * 7 = 84 N" in by_task["physics_numeric"].reasoning
    assert by_task["physics_numeric"].expected_unit == "N"
    assert report["accepted"] == 3
    assert report["rejected"] == 0


def _many_examples():
    arithmetic = generate_arithmetic_examples(count=600, seed=20260728)
    physics = tuple(
        PhysicsExample.create(
            formula_id="force",
            left=index + 1,
            right=2,
        )
        for index in range(200)
    )
    examples, _ = build_reasoning_examples(
        arithmetic=arithmetic,
        physics=physics,
        gsm8k_rows=(_gsm_record(),),
    )
    return examples


def test_paired_selection_excludes_frozen_families_and_never_truncates() -> None:
    examples = _many_examples()
    excluded = {examples[0].family_id}
    eligible, report = select_paired_examples(
        examples,
        context_limit=256,
        excluded_families=excluded,
    )

    assert examples[0].family_id not in {
        item.family_id for item in eligible
    }
    assert report["excluded_frozen_families"] == 1
    assert report["eligible"] == len(eligible)
    assert report["context_limit"] == 256


def test_direct_cot_and_hybrid_corpora_share_exact_families(
    tmp_path: Path,
) -> None:
    eligible, _ = select_paired_examples(
        _many_examples(),
        context_limit=256,
        excluded_families=(),
    )
    direct_dir = tmp_path / "direct"
    cot_dir = tmp_path / "cot"
    hybrid_dir = tmp_path / "hybrid"
    direct = build_mode_corpus(
        eligible,
        direct_dir,
        registry_path=REGISTRY_PATH,
        split_seed=LESSON14_SPLIT_SEED,
        modes=(DIRECT_MODE,),
        context_limit=256,
    )
    cot = build_mode_corpus(
        eligible,
        cot_dir,
        registry_path=REGISTRY_PATH,
        split_seed=LESSON14_SPLIT_SEED,
        modes=(THINK_MODE,),
        context_limit=256,
    )
    hybrid = build_mode_corpus(
        eligible,
        hybrid_dir,
        registry_path=REGISTRY_PATH,
        split_seed=LESSON14_SPLIT_SEED,
        modes=(DIRECT_MODE, THINK_MODE),
        context_limit=256,
    )

    assert direct["families"] == cot["families"] == hybrid["families"]
    assert direct["records"]["total"] == cot["records"]["total"]
    assert hybrid["records"]["total"] == 2 * direct["records"]["total"]
    assert direct["modes"] == [DIRECT_MODE]
    assert cot["modes"] == [THINK_MODE]
    assert hybrid["modes"] == [DIRECT_MODE, THINK_MODE]
    direct_data = load_packed_dataset(direct_dir)
    cot_data = load_packed_dataset(cot_dir)
    hybrid_data = load_packed_dataset(hybrid_dir)
    assert FINAL_ID in direct_data.train.tokens
    assert THINK_ID in cot_data.train.tokens
    assert THINK_ID in hybrid_data.train.tokens
    assert FINAL_ID in hybrid_data.train.tokens


def test_mode_corpus_rebuild_is_byte_identical(tmp_path: Path) -> None:
    eligible, _ = select_paired_examples(
        _many_examples(),
        context_limit=256,
        excluded_families=(),
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = build_mode_corpus(
        eligible,
        first,
        registry_path=REGISTRY_PATH,
        split_seed=LESSON14_SPLIT_SEED,
        modes=(DIRECT_MODE, THINK_MODE),
        context_limit=256,
    )
    second_manifest = build_mode_corpus(
        reversed(eligible),
        second,
        registry_path=REGISTRY_PATH,
        split_seed=LESSON14_SPLIT_SEED,
        modes=(DIRECT_MODE, THINK_MODE),
        context_limit=256,
    )

    assert first_manifest == second_manifest
    for filename in (
        "manifest.json",
        "train.tokens.bin",
        "train.loss.bin",
        "validation.tokens.bin",
        "validation.loss.bin",
        "test.tokens.bin",
        "test.loss.bin",
    ):
        first_payload = (first / filename).read_bytes()
        second_payload = (second / filename).read_bytes()
        assert first_payload == second_payload
        assert hashlib.sha256(first_payload).digest() == hashlib.sha256(
            second_payload
        ).digest()
    parsed = json.loads((first / "manifest.json").read_text("utf-8"))
    assert parsed["split"]["seed"] == LESSON14_SPLIT_SEED

