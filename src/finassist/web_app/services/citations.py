"""Display-only citation labels; never rewrite source destinations or API responses."""

import re

from finassist.web_app.services.showcase import safe_source

# Match only the label and URL prefix. Leaving the rest untouched preserves URL
# parentheses, query strings, optional Markdown titles, and angle-bracket syntax.
_CITATION = re.compile(r"(?<!!)\[E\d+\](\(\s*<?)(https?://[^\s<>)]*)")


def label_citations(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        hostname = safe_source(match.group(2))
        if hostname is None:
            return match.group(0)
        return f"[{hostname}]{match.group(1)}{match.group(2)}"

    return _CITATION.sub(replace, text)


def escape_currency(text: str) -> str:
    """Keep numeric dollar amounts literal in Streamlit Markdown."""
    return re.sub(
        r"(`[^`]*`|https?://[^\s<>]+)|(?<!\\)\$(?=\d)",
        lambda match: match.group(1) or r"\$",
        text,
    )
