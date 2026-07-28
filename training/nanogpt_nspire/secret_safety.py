"""Credential isolation and redaction for paid external-teacher calls."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import re
from typing import Iterator


class CredentialSafetyError(ValueError):
    """Raised when a serializable artifact contains credential-shaped data."""


class MissingCredentialError(RuntimeError):
    """Raised when a live provider call has no locally configured credential."""


_SECRET_RULES: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    (
        "environment_assignment",
        re.compile(
            rb"DEEPSEEK_API_KEY\s*=\s*sk-[A-Za-z0-9_-]{16,}",
            re.IGNORECASE,
        ),
    ),
    (
        "authorization_header",
        re.compile(
            rb"Authorization\s*:\s*Bearer\s+sk-[A-Za-z0-9_-]{16,}",
            re.IGNORECASE,
        ),
    ),
    (
        "api_token",
        re.compile(rb"sk-[A-Za-z0-9_-]{16,}"),
    ),
)

_TEXT_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"DEEPSEEK_API_KEY\s*=\s*sk-[A-Za-z0-9_-]{16,}",
            re.IGNORECASE,
        ),
        "DEEPSEEK_API_KEY=[REDACTED]",
    ),
    (
        re.compile(
            r"Authorization\s*:\s*Bearer\s+sk-[A-Za-z0-9_-]{16,}",
            re.IGNORECASE,
        ),
        "Authorization: Bearer [REDACTED]",
    ),
    (
        re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
        "[REDACTED]",
    ),
)


def _iter_payload_fragments(value: object) -> Iterator[bytes]:
    if isinstance(value, bytes):
        yield value
    elif isinstance(value, bytearray):
        yield bytes(value)
    elif isinstance(value, str):
        yield value.encode("utf-8", errors="replace")
    elif isinstance(value, Path):
        yield str(value).encode("utf-8", errors="replace")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _iter_payload_fragments(key)
            yield from _iter_payload_fragments(item)
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for item in value:
            yield from _iter_payload_fragments(item)


def _matching_rules(payload: bytes) -> tuple[str, ...]:
    return tuple(
        name for name, pattern in _SECRET_RULES if pattern.search(payload)
    )


def assert_secret_free(value: object, *, context: str) -> None:
    """Reject credential-shaped values without reflecting matched bytes."""

    rules: set[str] = set()
    for fragment in _iter_payload_fragments(value):
        rules.update(_matching_rules(fragment))
    if rules:
        raise CredentialSafetyError(
            f"{context} contains credential-shaped data "
            f"({', '.join(sorted(rules))})"
        )


def redact_text(text: str) -> str:
    """Redact credential-shaped substrings from provider-facing errors."""

    redacted = text
    for pattern, replacement in _TEXT_REDACTIONS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def _windows_user_environment_value(name: str) -> str | None:
    if os.name != "nt":
        return None
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, name)
    except FileNotFoundError:
        return None
    return value if isinstance(value, str) else None


def get_deepseek_api_key() -> str:
    """Read the key at call time without accepting it in serializable config."""

    value = os.environ.get("DEEPSEEK_API_KEY")
    if not value:
        value = _windows_user_environment_value("DEEPSEEK_API_KEY")
    if not value or not value.strip():
        raise MissingCredentialError(
            "DEEPSEEK_API_KEY is not configured in the process or Windows "
            "user environment"
        )
    return value.strip()


def assert_secret_free_tree(
    root: str | Path,
    *,
    excluded_names: frozenset[str] = frozenset(
        {".git", ".venv", "__pycache__"}
    ),
) -> None:
    """Scan a bounded directory tree and report only path plus rule name."""

    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(root_path)
    findings: list[tuple[str, str]] = []
    for path in sorted(root_path.rglob("*")):
        if not path.is_file() or any(
            part in excluded_names for part in path.relative_to(root_path).parts
        ):
            continue
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise CredentialSafetyError(
                f"could not scan {path.relative_to(root_path)}"
            ) from error
        for rule in _matching_rules(payload):
            findings.append((path.relative_to(root_path).as_posix(), rule))
    if findings:
        rendered = ", ".join(
            f"{path} [{rule}]" for path, rule in findings
        )
        raise CredentialSafetyError(
            f"credential-shaped data found in tree: {rendered}"
        )
