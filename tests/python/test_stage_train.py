import hashlib
import json
from pathlib import Path

import pytest
import torch

from nanogpt_nspire.base_corpus import CorpusRecord, build_corpus
from nanogpt_nspire.byte_tokenizer import ConversationTurn
from nanogpt_nspire.models.direct_small_gpt import (
    DirectSmallConfig,
    DirectSmallGPT,
)
from nanogpt_nspire.stage_train import (
    StageTrainingConfig,
    load_parent_checkpoint,
    run_stage_training,
)


REGISTRY_PATH = (
    Path(__file__).parents[2] / "experiments" / "lesson10-public-sources.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tiny_model_config() -> DirectSmallConfig:
    return DirectSmallConfig(
        vocab_size=264,
        block_size=16,
        n_layer=1,
        n_head=2,
        n_embd=16,
        mlp_ratio=2,
        dropout=0.0,
        bias=False,
        tie_embeddings=True,
    )


def _parent_checkpoint(
    path: Path,
    *,
    route: str = "English-Base-Pilot",
    model_config: DirectSmallConfig | None = None,
) -> Path:
    config = model_config or _tiny_model_config()
    torch.manual_seed(7)
    model = DirectSmallGPT(config)
    torch.save(
        {
            "best_step": 1,
            "dataset_manifest_sha256": "data",
            "model_config": config.__dict__,
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "route": route,
            "schema_version": 1,
            "selected_validation_loss": 1.0,
            "source_commit": "parent",
            "tokenizer": {
                "kind": "byte_plus_fixed_special_tokens",
                "vocab_size": 264,
            },
            "training_seed": 7,
        },
        path,
    )
    return path


def _training_data(tmp_path: Path) -> Path:
    records = []
    for index in range(400):
        records.append(
            CorpusRecord.conversation(
                record_id=f"chat-{index}",
                family_id=f"family-{index}",
                turns=(
                    ConversationTurn(
                        "user",
                        f"Calculate {index} + 1.",
                    ),
                    ConversationTurn(
                        "assistant",
                        f"The answer is {index + 1}.",
                    ),
                ),
                source_id="project-arithmetic-v1",
                license_id="MIT",
            )
        )
    output = tmp_path / "data"
    build_corpus(
        records,
        output,
        registry_path=REGISTRY_PATH,
        split_seed="stage-train-test",
    )
    return output


def test_parent_checkpoint_loader_verifies_hash_route_and_exact_shapes(
    tmp_path: Path,
) -> None:
    path = _parent_checkpoint(tmp_path / "parent.pt")
    expected_config = _tiny_model_config()

    loaded = load_parent_checkpoint(
        path,
        expected_sha256=_sha256(path),
        expected_route="English-Base-Pilot",
        expected_model_config=expected_config,
    )

    assert loaded["route"] == "English-Base-Pilot"
    assert loaded["sha256"] == _sha256(path)
    assert set(loaded["model_state_dict"]) == set(
        DirectSmallGPT(expected_config).state_dict()
    )


def test_parent_checkpoint_loader_rejects_wrong_hash_route_and_architecture(
    tmp_path: Path,
) -> None:
    path = _parent_checkpoint(tmp_path / "parent.pt")
    with pytest.raises(ValueError, match="SHA-256"):
        load_parent_checkpoint(
            path,
            expected_sha256="0" * 64,
            expected_route="English-Base-Pilot",
            expected_model_config=_tiny_model_config(),
        )
    with pytest.raises(ValueError, match="route"):
        load_parent_checkpoint(
            path,
            expected_sha256=_sha256(path),
            expected_route="Math-Physics-CPT",
            expected_model_config=_tiny_model_config(),
        )
    wrong = DirectSmallConfig(
        **{
            **_tiny_model_config().__dict__,
            "n_embd": 32,
            "n_head": 4,
        }
    )
    with pytest.raises(ValueError, match="model configuration"):
        load_parent_checkpoint(
            path,
            expected_sha256=_sha256(path),
            expected_route="English-Base-Pilot",
            expected_model_config=wrong,
        )


def test_cpu_sft_smoke_starts_from_parent_and_writes_lineage(
    tmp_path: Path,
) -> None:
    data = _training_data(tmp_path)
    parent = _parent_checkpoint(
        tmp_path / "parent.pt",
        route="Math-Physics-CPT",
    )
    output = tmp_path / "run"
    config = StageTrainingConfig(
        stage="sft",
        data_dir=data,
        output_dir=output,
        parent_checkpoint=parent,
        parent_checkpoint_sha256=_sha256(parent),
        expected_parent_route="Math-Physics-CPT",
        source_commit="test",
        device="cpu",
        seed=31,
        steps=4,
        micro_batch_size=2,
        gradient_accumulation_steps=1,
        vocab_size=264,
        block_size=16,
        n_layer=1,
        n_head=2,
        n_embd=16,
        mlp_ratio=2,
        dropout=0.0,
        learning_rate=0.01,
        min_learning_rate=0.001,
        warmup_steps=1,
        eval_interval=2,
        eval_batches=2,
        log_interval=1,
        overfit_gate_steps=2,
        use_bfloat16=False,
    )

    result = run_stage_training(config)

    assert result["route"] == "Role-Aware-SFT"
    assert result["lineage"]["parent_sha256"] == _sha256(parent)
    assert result["lineage"]["parent_route"] == "Math-Physics-CPT"
    assert result["metrics"]["training_tokens"] == 4 * 2 * 1 * 16
    assert result["metrics"]["eligible_training_targets"] > 0
    assert (output / "sft_best.pt").is_file()
    stored = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert stored["artifacts"]["checkpoint"]["sha256"] == (
        result["artifacts"]["checkpoint"]["sha256"]
    )


def test_stage_and_parent_route_contract_is_not_interchangeable(
    tmp_path: Path,
) -> None:
    parent = _parent_checkpoint(
        tmp_path / "parent.pt",
        route="Math-Physics-CPT",
    )
    config = StageTrainingConfig(
        stage="cpt",
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "out",
        parent_checkpoint=parent,
        parent_checkpoint_sha256=_sha256(parent),
        expected_parent_route="Math-Physics-CPT",
        source_commit="test",
        device="cpu",
        steps=2,
        warmup_steps=1,
        block_size=16,
        n_layer=1,
        n_head=2,
        n_embd=16,
        mlp_ratio=2,
    )

    with pytest.raises(ValueError, match="CPT must start"):
        config.validate()
