import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from nanogpt_nspire.assistant_eval import (
    EvaluationError,
    SUPPORTED_ROUTES,
    encode_assistant_prompt,
    load_evaluation_model,
    load_evaluation_records,
    parse_last_decimal,
    repeated_phrase_detected,
    score_completion,
    select_evaluation_records,
)
from nanogpt_nspire.byte_tokenizer import (
    ASSISTANT_ID,
    BOS_ID,
    USER_ID,
)
from nanogpt_nspire.efficient_context import (
    ARCHITECTURE_NAME,
    GQA_LEARNED_SFT_ROUTE,
    model_state_sha256,
)
from nanogpt_nspire.models.efficient_long_context_gpt import (
    LEARNED_POSITIONS,
    EfficientLongContextConfig,
    EfficientLongContextGPT,
)
from nanogpt_nspire.training_support import sha256_file


def test_assistant_prompt_uses_real_role_tokens() -> None:
    tokens = encode_assistant_prompt("What is 2 + 3?", block_size=64)

    assert tokens[0] == BOS_ID
    assert tokens[1] == USER_ID
    assert tokens[-1] == ASSISTANT_ID
    assert bytes(tokens[2:-1]).decode("utf-8") == "What is 2 + 3?"


def test_lesson13_routes_share_the_frozen_evaluator() -> None:
    assert {
        "Verified-Sequence-SFT",
        "Local-Logit-Distilled-SFT",
        "Combined-Sequence-Logit-SFT",
        "Local-Teacher-SFT",
    } <= SUPPORTED_ROUTES


def test_lesson14_routes_share_the_strict_checkpoint_loader() -> None:
    assert {
        "Direct-Control-SFT",
        "Short-CoT-SFT",
        "Hybrid-Control-SFT",
        "Hybrid-Control-SFT-Context512",
    } <= SUPPORTED_ROUTES


def test_lesson15_routes_share_the_strict_checkpoint_loader() -> None:
    assert {
        "GQA-Learned-Hybrid-SFT-Context512",
        "GQA-ALiBi-Hybrid-SFT-Context512",
    } <= SUPPORTED_ROUTES


def test_strict_checkpoint_loader_accepts_efficient_architecture(
    tmp_path: Path,
) -> None:
    config = EfficientLongContextConfig(
        vocab_size=264,
        block_size=8,
        n_layer=1,
        n_head=4,
        n_kv_head=2,
        n_embd=16,
        mlp_ratio=2,
        dropout=0.0,
        bias=False,
        tie_embeddings=True,
        position_mode=LEARNED_POSITIONS,
    )
    model = EfficientLongContextGPT(config)
    state = model.state_dict()
    path = tmp_path / "efficient.pt"
    torch.save(
        {
            "architecture": ARCHITECTURE_NAME,
            "best_step": 1,
            "model_config": asdict(config),
            "model_state_dict": state,
            "model_state_sha256": model_state_sha256(state),
            "route": GQA_LEARNED_SFT_ROUTE,
            "schema_version": 1,
            "source_commit": "source",
            "tokenizer": {
                "kind": "byte_plus_fixed_special_tokens",
                "vocab_size": 264,
            },
        },
        path,
    )

    loaded, provenance = load_evaluation_model(
        path,
        checkpoint_sha256=sha256_file(path),
        expected_route=GQA_LEARNED_SFT_ROUTE,
        device=torch.device("cpu"),
    )

    assert isinstance(loaded, EfficientLongContextGPT)
    assert provenance["model_config"] == asdict(config)


def test_assistant_prompt_rejects_context_overflow() -> None:
    with pytest.raises(EvaluationError, match="context"):
        encode_assistant_prompt("x" * 30, block_size=16)


@pytest.mark.parametrize(
    "text,expected",
    (
        ("The answer is 84.", "84"),
        ("First 3, then 1,200.50.", "1200.5"),
        ("Use F = 12 * 7. The force is -84.0 N.", "-84"),
    ),
)
def test_last_decimal_parser_is_strict_but_allows_explanations(
    text: str,
    expected: str,
) -> None:
    assert parse_last_decimal(text) == expected


@pytest.mark.parametrize("text", ("No number.", "1/2", "NaN"))
def test_last_decimal_parser_rejects_unparseable_answer(text: str) -> None:
    with pytest.raises(EvaluationError, match="decimal"):
        parse_last_decimal(text)


def test_completion_scoring_separates_number_unit_format_and_termination() -> None:
    correct = score_completion(
        {
            "task": "physics_numeric",
            "expected_answer": "84",
            "expected_unit": "N",
        },
        text="The force is 84 N.",
        terminated=True,
        special_token_leak=False,
    )
    missing_unit = score_completion(
        {
            "task": "physics_numeric",
            "expected_answer": "84",
            "expected_unit": "N",
        },
        text="84",
        terminated=True,
        special_token_leak=False,
    )
    leaked = score_completion(
        {
            "task": "arithmetic",
            "expected_answer": "84",
            "expected_unit": None,
        },
        text="The answer is 84.",
        terminated=True,
        special_token_leak=True,
    )

    assert correct["numeric_correct"] is True
    assert correct["unit_correct"] is True
    assert correct["task_correct"] is True
    assert missing_unit["numeric_correct"] is True
    assert missing_unit["unit_correct"] is False
    assert missing_unit["task_correct"] is False
    assert leaked["task_correct"] is False


def test_repeated_phrase_detector_flags_degenerate_loops() -> None:
    assert repeated_phrase_detected(
        "the state of the state of the state of the state"
    )
    assert not repeated_phrase_detected(
        "Force is mass times acceleration."
    )


def test_evaluation_loader_and_selection_are_order_independent(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "task": task,
            "prompt": f"Prompt {task} {index}",
            "expected_answer": str(index),
            "expected_unit": "N" if task == "physics_numeric" else None,
            "family_id": f"{task}-{index}",
            "source_id": "test",
        }
        for task in (
            "arithmetic",
            "arithmetic_easy",
            "physics_numeric",
            "gsm8k",
        )
        for index in range(10)
    ]
    path = tmp_path / "evaluation.jsonl"
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    loaded = load_evaluation_records(path)
    first = select_evaluation_records(loaded, max_per_task=3)
    second = select_evaluation_records(
        reversed(loaded),
        max_per_task=3,
    )

    assert first == second
    assert len(first) == 12
    assert {
        row["task"] for row in first
    } == {
        "arithmetic",
        "arithmetic_easy",
        "physics_numeric",
        "gsm8k",
    }


def test_evaluation_loader_rejects_duplicate_families(tmp_path: Path) -> None:
    row = {
        "task": "arithmetic",
        "prompt": "Q",
        "expected_answer": "1",
        "expected_unit": None,
        "family_id": "same",
        "source_id": "test",
    }
    path = tmp_path / "evaluation.jsonl"
    path.write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(EvaluationError, match="duplicate family"):
        load_evaluation_records(path)
