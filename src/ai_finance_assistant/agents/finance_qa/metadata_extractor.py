from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchValue


class FinanceMetadataExtractor:
    """Extract only explicit query values supported by the loaded finance corpus."""

    def __init__(self, corpus_metadata: list[dict[str, Any]]) -> None:
        vocabulary: dict[str, set[str]] = defaultdict(set)
        for metadata in corpus_metadata:
            for field, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    vocabulary[field].add(str(value))
        self._vocabulary = vocabulary

    def extract(self, query: str) -> dict[str, str]:
        extracted: dict[str, str] = {}
        for field, values in self._vocabulary.items():
            if field not in {"topic", "subtopic", "asset_class", "instrument_type", "ticker"}:
                continue
            normalized_query = query.casefold()
            candidates = [value for value in values if len(value) > 1]
            if field == "ticker":
                candidates = [value for value in candidates if value.isupper() and len(value) <= 5]
                tokens = set(re.findall(r"\b[A-Z][A-Z0-9.-]{0,4}\b", query))
                match = next((value for value in candidates if value in tokens), None)
            else:
                match = next(
                    (value for value in candidates if value.casefold() in normalized_query),
                    None,
                )
            if match is not None:
                extracted[field] = match
        return extracted

    @staticmethod
    def build_qdrant_filter(metadata: dict[str, str]) -> Filter | None:
        if not metadata:
            return None
        return Filter(
            must=[
                FieldCondition(key=f"metadata.{field}", match=MatchValue(value=value))
                for field, value in metadata.items()
            ]
        )