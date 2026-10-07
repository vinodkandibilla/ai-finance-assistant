from __future__ import annotations

import json
from types import SimpleNamespace

from ai_finance_assistant.rag.tax_education import (
    build_tax_metadata_filter,
    extract_tax_metadata,
    load_tax_documents,
)


def test_load_tax_documents_creates_langchain_documents(tmp_path) -> None:
    corpus_path = tmp_path / "sample_tax.jsonl"
    payload = {
        "content": "Married filing jointly uses a different standard deduction.",
        "tax_year": 2026,
        "filing_year": 2027,
        "topic": "filing_status",
        "subtopic": "married_filing_jointly",
        "taxpayer_type": ["married"],
        "filing_status": ["married_filing_jointly"],
        "forms": ["1040"],
        "immigration_tax_status": "general",
        "jurisdiction": "US_federal",
    }
    with corpus_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")

    documents = load_tax_documents(corpus_path)

    assert len(documents) == 1
    assert documents[0].page_content == payload["content"]
    assert documents[0].metadata["tax_year"] == 2026
    assert documents[0].metadata["filing_year"] == 2027
    assert documents[0].metadata["topic"] == "filing_status"
    assert documents[0].metadata["subtopic"] == "married_filing_jointly"
    assert documents[0].metadata["taxpayer_type"] == ["married"]
    assert documents[0].metadata["filing_status"] == ["married_filing_jointly"]
    assert documents[0].metadata["forms"] == ["1040"]
    assert documents[0].metadata["immigration_tax_status"] == "general"
    assert documents[0].metadata["jurisdiction"] == "US_federal"


def test_extract_tax_metadata_returns_only_detected_values() -> None:
    metadata = extract_tax_metadata(
        "For tax year 2026, what is the standard deduction for Married Filing Jointly?"
    )

    assert metadata["tax_year"] == 2026
    assert metadata["filing_status"] == ["married_filing_jointly"]
    assert metadata["topic"] == "standard_vs_itemized"
    assert "forms" not in metadata


def test_extract_tax_metadata_for_standard_deduction_limit_query() -> None:
    metadata = extract_tax_metadata("What is the standard deduction limit for year 2026")

    assert metadata["tax_year"] == 2026
    assert metadata["topic"] == "standard_vs_itemized"


def test_extract_tax_metadata_maps_natural_topic_and_both_years() -> None:
    metadata = extract_tax_metadata(
        "For tax year 2026 filed in filing year 2027, what is the standard deduction?"
    )

    assert metadata["tax_year"] == 2026
    assert metadata["filing_year"] == 2027
    assert metadata["topic"] == "standard_vs_itemized"


def test_build_tax_metadata_filter_is_empty_when_no_metadata() -> None:
    assert build_tax_metadata_filter({}) is None


def test_build_tax_metadata_filter_maps_detected_fields_to_payload() -> None:
    metadata_filter = build_tax_metadata_filter(
        {"tax_year": 2026, "filing_status": ["married_filing_jointly"]}
    )

    assert metadata_filter is not None
    assert {condition.key for condition in metadata_filter.must} == {
        "metadata.tax_year",
        "metadata.filing_status",
    }
    filing_status_condition = next(
        condition for condition in metadata_filter.must if condition.key == "metadata.filing_status"
    )
    assert filing_status_condition.match.any == ["married_filing_jointly", "all"]


def test_build_tax_hybrid_vector_store_reuses_existing_collection(monkeypatch) -> None:
    from ai_finance_assistant.rag import hybrid

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def collection_exists(self, _collection_name: str) -> bool:
            return True

        def count(self, collection_name: str) -> SimpleNamespace:
            return SimpleNamespace(count=5)

        def close(self) -> None:
            pass

    class FakeVectorStore:
        @classmethod
        def from_existing_collection(cls, **kwargs):
            return cls()

    monkeypatch.setattr(
        hybrid,
        "get_settings",
        lambda: SimpleNamespace(
            openai_api_key=SimpleNamespace(get_secret_value=lambda: "test-key"),
            openai_embedding_model="text-embedding-3-small",
        ),
    )
    monkeypatch.setattr(hybrid, "QdrantClient", FakeClient)
    monkeypatch.setattr(hybrid, "OpenAIEmbeddings", lambda **kwargs: object())
    monkeypatch.setattr(hybrid, "FastEmbedSparse", lambda model_name: object())
    monkeypatch.setattr(hybrid, "QdrantVectorStore", FakeVectorStore)

    from ai_finance_assistant.rag.tax_education import build_tax_hybrid_vector_store

    store = build_tax_hybrid_vector_store()

    assert isinstance(store, FakeVectorStore)
