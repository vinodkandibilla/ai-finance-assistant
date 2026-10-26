"""Finance-only eligibility for the shared semantic answer cache."""

import re

# An embedding match must also describe the same definition topic. Broader tasks
# deliberately miss rather than treating related concepts as interchangeable.
_CONCEPTS = {
    "stock": ("stock", "stocks", "share", "shares"),
    "bond": ("bond", "bonds"),
    "etf": ("etf", "etfs", "exchange traded fund", "exchange traded funds"),
    "mutual fund": ("mutual fund", "mutual funds"),
    "index fund": ("index fund", "index funds"),
    "diversification": ("diversification",),
    "concentration risk": ("concentration risk",),
    "simple interest": ("simple interest",),
    "compound interest": ("compound interest",),
    "expense ratio": ("expense ratio", "fund expense ratio"),
    "bond coupon": ("bond coupon",),
    "net asset value": ("net asset value", "nav"),
}


def definition_scope(query: str) -> tuple[str, str] | None:
    normalized = re.sub(r"\s+", " ", query.casefold().replace("-", " ")).strip().rstrip("?.")
    match = re.fullmatch(r"(?:what (?:is|are)|define) (?:a |an |the )?([a-z ]+)", normalized)
    if match:
        for concept, aliases in _CONCEPTS.items():
            if match[1] in aliases:
                return normalized, concept
    return None
