import hashlib
import json
from pathlib import Path

import pytest

from nanogpt_nspire.base_corpus import CorpusError
from nanogpt_nspire.lesson10_data import main, run_smoke


def _tree_hashes(path: Path) -> dict[str, str]:
    return {
        file.relative_to(path).as_posix(): hashlib.sha256(
            file.read_bytes()
        ).hexdigest()
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


def test_smoke_build_is_deterministic_and_contains_every_split(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_summary = run_smoke(
        first,
        seed=20260728,
        example_count=128,
    )
    second_summary = run_smoke(
        second,
        seed=20260728,
        example_count=128,
    )

    assert first_summary["seed"] == 20260728
    assert first_summary["generated_families"] == 128
    assert first_summary["records"]["total"] == 256
    assert first_summary["records"]["train"] > 0
    assert first_summary["records"]["validation"] > 0
    assert first_summary["records"]["test"] > 0
    assert first_summary["manifest_sha256"] == second_summary["manifest_sha256"]
    assert _tree_hashes(first) == _tree_hashes(second)


def test_existing_output_requires_explicit_verified_replace(tmp_path):
    output = tmp_path / "smoke"
    run_smoke(output, seed=20260728, example_count=128)

    with pytest.raises(CorpusError, match="already exists"):
        run_smoke(output, seed=20260728, example_count=128)

    original_hashes = _tree_hashes(output)
    replaced = run_smoke(
        output,
        seed=20260728,
        example_count=128,
        replace=True,
    )
    assert _tree_hashes(output) == original_hashes
    assert replaced["replace"] is True


def test_replace_refuses_unrecognized_directory(tmp_path):
    output = tmp_path / "not-a-corpus"
    output.mkdir()
    (output / "personal.txt").write_text("do not delete", encoding="utf-8")

    with pytest.raises(CorpusError, match="verified Lesson 10 corpus"):
        run_smoke(
            output,
            seed=20260728,
            example_count=128,
            replace=True,
        )

    assert (output / "personal.txt").read_text(encoding="utf-8") == "do not delete"


def test_cli_prints_bounded_json_without_environment_data(tmp_path, capsys):
    output = tmp_path / "smoke"

    status = main(
        [
            "smoke",
            "--output",
            str(output),
            "--seed",
            "20260728",
            "--examples",
            "128",
        ]
    )

    assert status == 0
    printed = capsys.readouterr().out
    summary = json.loads(printed)
    assert summary["generated_families"] == 128
    assert summary["output"] == str(output)
    assert len(printed) < 4_096
    assert "DEEPSEEK" not in printed
    assert "sk-" not in printed


@pytest.mark.parametrize(
    "seed, count, message",
    [
        (True, 128, "seed"),
        (1, 0, "example_count"),
    ],
)
def test_smoke_rejects_invalid_arguments(tmp_path, seed, count, message):
    with pytest.raises(CorpusError, match=message):
        run_smoke(
            tmp_path / "smoke",
            seed=seed,
            example_count=count,
        )
