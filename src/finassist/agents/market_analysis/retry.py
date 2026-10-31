"""Conservative transient-failure classification for market MCP calls."""

import re

import httpx

TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class MarketMCPError(RuntimeError):
    """Sanitized provider failure retaining only its retry classification."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


def transient_message(text: str) -> bool:
    # Explicit permanent failures take precedence over generic availability wording.
    if re.search(
        r"invalid (?:api[ _-]?key|symbol|interval|parameter)|unauthorized|forbidden|"
        r"api credits|quota exhausted|subscription|\b(?:401|403|404|422)\b",
        text, re.I,
    ):
        return False
    codes = re.findall(
        r'(?:\bHTTP(?:/\d(?:\.\d)?)?\s*|\b(?:status(?:_code)?|code)\b["\s:=]*)([45]\d{2})\b',
        text, re.I,
    )
    if codes:
        return all(int(code) in TRANSIENT_STATUS_CODES for code in codes)
    return bool(re.search(
        r"\b(?:429|500|502|503|504)\b|rate[ _-]?limit|too many requests|"
        r"temporar(?:y|ily)|service unavailable|bad gateway|gateway timeout|"
        r"connection (?:reset|aborted)|timed out",
        text, re.I,
    ))


def transient_exception(error: Exception) -> bool:
    if isinstance(error, MarketMCPError):
        return error.retryable
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    if status is not None:
        return status in TRANSIENT_STATUS_CODES
    if isinstance(
        error, (TimeoutError, ConnectionError, httpx.TimeoutException, httpx.NetworkError)
    ):
        return True
    return transient_message(str(error))
