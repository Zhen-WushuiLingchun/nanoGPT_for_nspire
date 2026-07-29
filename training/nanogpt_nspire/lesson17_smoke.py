"""Run one local RLVR/RLAIF/combined update with a fake AI transport."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nanogpt_nspire.assistant_eval import load_evaluation_records
from nanogpt_nspire.group_policy_train import (
    run_group_policy_training,
    smoke_policy_training_config,
)
from nanogpt_nspire.judge_cache import JudgeCache
from nanogpt_nspire.lesson17_data import (
    LESSON17_DATA_SEED,
    build_lesson17_problem_pool,
    build_prompt_schedule,
)
from nanogpt_nspire.lesson17_routes import (
    COMBINED_ROUTE,
    DIRECT_RLAIF_ROUTE,
    RLVR_ROUTE,
)
from nanogpt_nspire.preference_judge import (
    PreferenceJudgeClient,
    PreferenceJudgeConfig,
)
from nanogpt_nspire.secret_safety import assert_secret_free_tree
from nanogpt_nspire.training_support import write_json_atomic


def _fake_transport(
    _url: str,
    _headers: dict[str, str],
    body: bytes,
    _timeout_seconds: float,
) -> bytes:
    request = json.loads(body.decode("utf-8"))
    user = request["messages"][1]["content"]
    payload = json.loads(user.split("\n", 1)[1])
    identifiers = [
        item["candidate_id"] for item in payload["candidates"]
    ]
    scores = [
        {
            "candidate_id": candidate_id,
            "score": 4 if index == 0 else 0,
        }
        for index, candidate_id in enumerate(identifiers)
    ]
    content = {
        "preferred_candidate_id": identifiers[0],
        "rationale": "Deterministic fake smoke preference.",
        "scores": scores,
    }
    return json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(content),
                        "role": "assistant",
                    }
                }
            ],
            "id": "fake-smoke-judge",
            "model": request["model"],
            "usage": {
                "completion_tokens": 10,
                "prompt_tokens": 100,
                "total_tokens": 110,
            },
        }
    ).encode("utf-8")


def run_lesson17_smoke(
    *,
    start_checkpoint: Path,
    start_checkpoint_sha256: str,
    start_route: str,
    primary_evaluation: Path,
    challenge_evaluation: Path,
    output_root: Path,
    source_commit: str,
    device: str = "cuda",
) -> dict[str, object]:
    if output_root.exists():
        raise ValueError(f"output root already exists: {output_root}")
    excluded = {
        str(row["family_id"])
        for path in (primary_evaluation, challenge_evaluation)
        for row in load_evaluation_records(path)
    }
    pool = build_lesson17_problem_pool(
        count_per_task=2,
        seed=LESSON17_DATA_SEED,
        excluded_families=excluded,
    )
    schedule = build_prompt_schedule(
        pool,
        seed=20260731,
        updates=1,
        prompts_per_update=4,
    )
    fake_client = PreferenceJudgeClient(
        PreferenceJudgeConfig(
            max_requests=8,
            maximum_attempts=1,
        ),
        transport=_fake_transport,
        credential_provider=lambda: "sk-" + "s" * 32,
    )
    cache = JudgeCache(output_root / "fake-judge-cache.jsonl")
    summaries: dict[str, object] = {}
    for label, route in (
        ("rlvr", RLVR_ROUTE),
        ("rlaif", DIRECT_RLAIF_ROUTE),
        ("combined", COMBINED_ROUTE),
    ):
        config = smoke_policy_training_config(
            route=route,
            output_dir=output_root / label,
            start_checkpoint=start_checkpoint,
            start_checkpoint_sha256=start_checkpoint_sha256,
            start_route=start_route,
            source_commit=source_commit,
            seed=20260731,
            device=device,
        )
        summaries[label] = run_group_policy_training(
            config,
            schedule=schedule,
            problem_pool=pool,
            judge_client=(fake_client if route != RLVR_ROUTE else None),
            judge_cache=(cache if route != RLVR_ROUTE else None),
        )
    result = {
        "fake_ai_transport": True,
        "routes": summaries,
        "schema_version": 1,
    }
    write_json_atomic(output_root / "run.json", result)
    assert_secret_free_tree(output_root)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-checkpoint", type=Path, required=True)
    parser.add_argument("--start-checkpoint-sha256", required=True)
    parser.add_argument("--start-route", required=True)
    parser.add_argument("--primary-evaluation", type=Path, required=True)
    parser.add_argument("--challenge-evaluation", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    result = run_lesson17_smoke(
        start_checkpoint=arguments.start_checkpoint,
        start_checkpoint_sha256=arguments.start_checkpoint_sha256,
        start_route=arguments.start_route,
        primary_evaluation=arguments.primary_evaluation,
        challenge_evaluation=arguments.challenge_evaluation,
        output_root=arguments.output_root,
        source_commit=arguments.source_commit,
        device=arguments.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
