import hashlib
import json
from pathlib import Path

import pytest

from nanogpt_nspire.base_corpus import CorpusRecord, build_corpus
from nanogpt_nspire.base_train import load_packed_dataset
from nanogpt_nspire.byte_tokenizer import ASSISTANT_ID, BOS_ID, EOS_ID, USER_ID
from nanogpt_nspire.lesson12_curriculum import (
    GSM8KExample,
    OASSTPair,
    PhysicsExample,
    generate_physics_examples,
)
from nanogpt_nspire.lesson12_data import (
    Lesson12DataError,
    build_domain_records,
    compose_packed_corpora,
    load_gsm8k_jsonl,
    select_oasst_pairs,
)
from nanogpt_nspire.math_curriculum import (
    ArithmeticExample,
    generate_arithmetic_examples,
)


REGISTRY_PATH = (
    Path(__file__).parents[2] / "experiments" / "lesson10-public-sources.json"
)


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def test_gsm8k_loader_preserves_source_split_and_reports_rejections(
    tmp_path: Path,
) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(
        path,
        [
            {"question": "What is 2 + 3?", "answer": "#### 5"},
            {"question": "x" * 250, "answer": "#### 1"},
            {"question": "Bad", "answer": "one"},
        ],
    )

    examples, report = load_gsm8k_jsonl(path, source_split="train")

    assert len(examples) == 1
    assert examples[0].source_split == "train"
    assert report["rows"] == 3
    assert report["accepted"] == 1
    assert report["rejected"] == 2
    assert report["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def _oasst_rows() -> list[dict[str, object]]:
    prompt = {
        "message_id": "p1",
        "parent_id": None,
        "message_tree_id": "t1",
        "role": "prompter",
        "lang": "en",
        "text": "What is force?",
        "deleted": False,
        "rank": None,
        "review_result": True,
        "tree_state": "ready_for_export",
        "synthetic": False,
        "labels": {
            "name": ["quality", "toxicity"],
            "value": [0.9, 0.0],
            "count": [3, 3],
        },
    }
    answer = {
        "message_id": "a1",
        "parent_id": "p1",
        "message_tree_id": "t1",
        "role": "assistant",
        "lang": "en",
        "text": "Force is mass times acceleration.",
        "deleted": False,
        "rank": 0,
        "review_result": True,
        "tree_state": "ready_for_export",
        "synthetic": False,
        "labels": {
            "name": ["quality", "toxicity"],
            "value": [0.9, 0.0],
            "count": [3, 3],
        },
    }
    lower_quality = dict(answer)
    lower_quality.update(
        {
            "message_id": "a2",
            "rank": 1,
            "text": "A lower-ranked answer.",
        }
    )
    return [lower_quality, answer, prompt]


def test_oasst_selection_is_order_independent_and_quality_gated() -> None:
    first, first_report = select_oasst_pairs(
        _oasst_rows(),
        seed="test",
        max_pairs=8,
    )
    second, second_report = select_oasst_pairs(
        reversed(_oasst_rows()),
        seed="test",
        max_pairs=8,
    )

    assert first == second
    assert first_report == second_report
    assert len(first) == 1
    assert first[0].record_id == "oasst1-a1"


def _small_domain_inputs():
    arithmetic = (
        ArithmeticExample.create(left=12, operator="*", right=7),
    )
    physics = (
        PhysicsExample.create(
            formula_id="force",
            left=12,
            right=7,
        ),
    )
    gsm = (
        GSM8KExample.from_record(
            {"question": "What is 8 plus 9?", "answer": "#### 17"},
            source_split="train",
            row_index=0,
        ),
    )
    oasst = (
        OASSTPair.from_messages(_oasst_rows()[2], _oasst_rows()[1]),
    )
    return arithmetic, physics, gsm, oasst


def test_domain_records_keep_related_cpt_and_sft_variants_in_one_family() -> None:
    arithmetic, physics, gsm, oasst = _small_domain_inputs()

    cpt, sft = build_domain_records(
        arithmetic=arithmetic,
        physics=physics,
        gsm8k_train=gsm,
        oasst_pairs=oasst,
    )

    assert {item.family_id for item in cpt} <= {
        item.family_id for item in sft
    }
    assert all(item.kind == "base" for item in cpt)
    assert all(item.kind == "conversation" for item in sft)
    arith_sft = [
        item for item in sft if item.family_id == arithmetic[0].family_id
    ]
    physics_sft = [
        item for item in sft if item.family_id == physics[0].family_id
    ]
    assert len(arith_sft) == 2
    assert len(physics_sft) == 2


def _component_corpus(
    output: Path,
    *,
    prefix: str,
    split_seed: str,
) -> Path:
    records: list[CorpusRecord] = []
    for index in range(200):
        records.append(
            CorpusRecord.base(
                record_id=f"{prefix}-{index}",
                family_id=f"{prefix}-{index}",
                text=f"{prefix} educational record {index}. " * 8,
                source_id="project-arithmetic-v1",
                license_id="MIT",
            )
        )
    build_corpus(
        records,
        output,
        registry_path=REGISTRY_PATH,
        split_seed=split_seed,
    )
    return output


def test_composite_corpus_is_hash_stable_and_tracks_component_tokens(
    tmp_path: Path,
) -> None:
    first = _component_corpus(
        tmp_path / "first",
        prefix="general",
        split_seed="component-a",
    )
    second = _component_corpus(
        tmp_path / "second",
        prefix="domain",
        split_seed="component-b",
    )

    first_output = tmp_path / "composite-one"
    second_output = tmp_path / "composite-two"
    first_manifest = compose_packed_corpora(
        (("general_replay", first), ("domain", second)),
        first_output,
    )
    second_manifest = compose_packed_corpora(
        (("general_replay", first), ("domain", second)),
        second_output,
    )

    assert first_manifest == second_manifest
    assert (first_output / "manifest.json").read_bytes() == (
        second_output / "manifest.json"
    ).read_bytes()
    dataset = load_packed_dataset(first_output)
    expected_train = sum(
        item["tokens"]["train"]
        for item in first_manifest["components"]
    )
    assert dataset.train.token_count == expected_train
    assert 0.0 < first_manifest["general_replay_train_fraction"] < 1.0


def test_composite_corpus_rejects_duplicate_names_and_corrupt_components(
    tmp_path: Path,
) -> None:
    component = _component_corpus(
        tmp_path / "component",
        prefix="general",
        split_seed="component",
    )
    with pytest.raises(Lesson12DataError, match="unique"):
        compose_packed_corpora(
            (("same", component), ("same", component)),
            tmp_path / "duplicate",
        )

    with (component / "train.tokens.bin").open("ab") as stream:
        stream.write(b"\0\0")
    with pytest.raises((Lesson12DataError, ValueError), match="manifest"):
        compose_packed_corpora(
            (("broken", component),),
            tmp_path / "broken",
        )


def test_built_sft_shard_contains_real_role_tokens_and_assistant_only_mask(
    tmp_path: Path,
) -> None:
    _, _, gsm, oasst = _small_domain_inputs()
    _, sft = build_domain_records(
        arithmetic=generate_arithmetic_examples(count=200, seed=12),
        physics=generate_physics_examples(count=100, seed=12),
        gsm8k_train=gsm,
        oasst_pairs=oasst,
    )
    output = tmp_path / "sft"
    build_corpus(
        sft,
        output,
        registry_path=REGISTRY_PATH,
        split_seed="sft-mask-test",
    )
    dataset = load_packed_dataset(output)
    tokens = dataset.train.tokens.tolist()
    mask = dataset.train.loss_mask.tolist()

    assert BOS_ID in tokens
    assert USER_ID in tokens
    assert ASSISTANT_ID in tokens
    assert EOS_ID in tokens
    assistant_position = tokens.index(ASSISTANT_ID)
    assert mask[assistant_position] == 0
    assert mask[assistant_position + 1] == 1
