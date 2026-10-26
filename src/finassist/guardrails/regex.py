"""Fast deterministic checks that run before any external model call."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RegexGuardResult:
    blocked: bool
    reason: str | None = None


_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "prompt_injection",
        re.compile(
            r"(?i)\b(ignore|disregard|forget)\s+(all\s+)?"
            r"(previous|prior|above)\s+instructions\b"
        ),
    ),
    ("secret", re.compile(r"\b(?:sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,})\b")),
    ("competitor", re.compile(r"(?i)\b(?:betterment|wealthfront|robinhood)\b")),
)


def check_regex_guards(text: str) -> RegexGuardResult:
    for reason, pattern in _RULES:
        if pattern.search(text):
            return RegexGuardResult(blocked=True, reason=reason)
    return RegexGuardResult(blocked=False)
