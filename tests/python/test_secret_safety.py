from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanogpt_nspire.secret_safety import (
    CredentialSafetyError,
    MissingCredentialError,
    assert_secret_free,
    assert_secret_free_tree,
    get_deepseek_api_key,
    redact_text,
)


FAKE_SECRET = "sk-" + "x" * 32


def test_assert_secret_free_rejects_nested_secret_without_echoing_it() -> None:
    payload = {
        "safe": "metadata",
        "nested": [{"authorization": f"Bearer {FAKE_SECRET}"}],
    }

    with pytest.raises(CredentialSafetyError) as captured:
        assert_secret_free(payload, context="run metadata")

    message = str(captured.value)
    assert FAKE_SECRET not in message
    assert "run metadata" in message
    assert "credential-shaped" in message


def test_assert_secret_free_accepts_environment_variable_name_and_model() -> None:
    assert_secret_free(
        {
            "credential_source": "DEEPSEEK_API_KEY",
            "model": "deepseek-v4-pro",
            "base_url": "https://api.deepseek.com",
        },
        context="provider contract",
    )


def test_redact_text_covers_token_header_and_assignment() -> None:
    text = (
        f"token={FAKE_SECRET}; "
        f"Authorization: Bearer {FAKE_SECRET}; "
        f"DEEPSEEK_API_KEY={FAKE_SECRET}"
    )

    redacted = redact_text(text)

    assert FAKE_SECRET not in redacted
    assert redacted.count("[REDACTED]") >= 3


def test_get_deepseek_api_key_uses_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_SECRET)

    assert get_deepseek_api_key() == FAKE_SECRET


def test_get_deepseek_api_key_has_safe_missing_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(
        "nanogpt_nspire.secret_safety._windows_user_environment_value",
        lambda _: None,
    )

    with pytest.raises(MissingCredentialError) as captured:
        get_deepseek_api_key()

    assert "DEEPSEEK_API_KEY" in str(captured.value)
    assert "sk-" not in str(captured.value)


def test_tree_scan_reports_rule_and_path_but_not_secret(
    tmp_path: Path,
) -> None:
    safe = tmp_path / "safe.json"
    safe.write_text(
        json.dumps({"credential_source": "DEEPSEEK_API_KEY"}),
        encoding="utf-8",
    )
    unsafe = tmp_path / "unsafe.log"
    unsafe.write_text(f"request failed: {FAKE_SECRET}", encoding="utf-8")

    with pytest.raises(CredentialSafetyError) as captured:
        assert_secret_free_tree(tmp_path)

    message = str(captured.value)
    assert "unsafe.log" in message
    assert "api_token" in message
    assert FAKE_SECRET not in message


def test_gitignore_covers_local_credentials_and_provider_outputs() -> None:
    root = Path(__file__).resolve().parents[2]
    ignored = set(
        line.strip()
        for line in (root / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    )

    assert ".env" in ignored
    assert ".env.*" in ignored
    assert "*.credentials" in ignored
    assert "/artifacts/teacher-api/" in ignored
    assert "/logs/" in ignored
