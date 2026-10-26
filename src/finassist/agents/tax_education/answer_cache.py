"""Only context-independent US federal tax definitions are cacheable."""

import re

_CONCEPTS = {
    "form_w2": {"form w 2", "form w2", "w 2", "w2", "form w 2 used for in us federal taxes"},
    "tax_deduction": {"tax deduction", "federal tax deduction", "us federal tax deduction"},
    "tax_credit": {"tax credit", "federal tax credit"},
    "withholding": {"tax withholding", "federal income tax withholding"},
}


def definition_scope(query: str) -> tuple[str, str] | None:
    normalized = re.sub(r"\s+", " ", query.casefold().replace("-", " ")).strip().rstrip("?.")
    match = re.fullmatch(r"(?:what (?:is|are)|define) (?:a |an |the )?(.+)", normalized)
    if match:
        for concept, aliases in _CONCEPTS.items():
            if match[1] in aliases:
                return normalized, concept
    return None
