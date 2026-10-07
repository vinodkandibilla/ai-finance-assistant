from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from ai_finance_assistant.rag.hybrid import (
    build_hybrid_vector_store as _build_hybrid_vector_store,
    get_hybrid_retriever,
)
from ai_finance_assistant.rag.retriever import QdrantRetriever

TAX_CORPUS_PATH = Path(__file__).resolve().parent / "tax-education" / "irs_tax_2026_rag_corpus_expanded.jsonl"
TAX_METADATA_FIELDS = (
    "tax_year",
    "filing_year",
    "topic",
    "subtopic",
    "taxpayer_type",
    "filing_status",
    "forms",
    "immigration_tax_status",
    "jurisdiction",
)
TAX_COLLECTION_NAME = "tax_education_hybrid"
LOCAL_QDRANT_PATH = Path(__file__).resolve().parents[3] / ".qdrant"
TAX_LIST_METADATA_FIELDS = {
    "taxpayer_type",
    "filing_status",
    "forms",
    "immigration_tax_status",
    "jurisdiction",
}
TAX_VALUE_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "filing_status": {
        "married_filing_jointly": ("married filing jointly", "jointly", "mfj"),
        "married_filing_separately": ("married filing separately", "mfs"),
        "head_of_household": ("head of household", "hoh"),
        "qualifying_surviving_spouse": ("qualifying surviving spouse", "surviving spouse"),
        "single": ("single",),
    },
    "taxpayer_type": {
        "business_owner": ("business owner", "sole proprietor", "small business"),
        "retirement_saver": ("retirement", "401(k)", "403(b)", "457", "ira"),
        "investor": ("investor", "stock", "stocks", "capital gain", "capital gains", "dividend"),
        "employee": ("employee", "wages", "withholding"),
        "student": ("student", "college", "education credit"),
        "immigrant": ("immigrant", "immigration", "visa", "h-1b", "h1b", "h-4", "h4"),
        "nonimmigrant": ("nonimmigrant", "non-immigrant", "visa", "h-1b", "h1b", "h-4", "h4"),
        "homeowner": ("homeowner", "main home", "home sale"),
        "landlord": ("landlord", "rental property", "rental real estate"),
        "self_employed": ("self-employed", "self employed", "freelancer"),
    },
    "immigration_tax_status": {
        "nonresident_alien": ("nonresident alien", "non-resident alien"),
        "resident_alien": ("resident alien",),
        "dual_status": ("dual status",),
        "h1b": ("h-1b", "h1b"),
        "h4": ("h-4", "h4"),
    },
    "jurisdiction": {
        "US_federal": ("federal", "irs", "united states", "u.s."),
    },
}
TAX_TOPIC_ALIASES = {
    "standard_vs_itemized": ("standard deduction", "itemized deduction", "itemized deductions"),
    "capital_gains": ("capital gain", "capital gains", "holding period"),
    "education_tax": ("education credit", "education credits", "college tax", "529 plan"),
    "tax_residency": ("tax residency", "tax resident", "resident alien", "nonresident alien"),
    "estimated_tax": ("estimated tax", "underpayment penalty", "quarterly tax"),
    "foreign_accounts": ("foreign account", "foreign accounts", "fbar", "form 8938"),
    "foreign_income": ("foreign income", "foreign tax credit"),
    "filing_status": ("filing status", "married filing", "head of household"),
}


def _normalize_query(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _query_contains(query: str, candidate: str) -> bool:
    normalized_query = f" {_normalize_query(query)} "
    normalized_candidate = _normalize_query(candidate)
    return bool(normalized_candidate) and f" {normalized_candidate} " in normalized_query


def extract_tax_metadata(query: str) -> dict[str, Any]:
    """Extract corpus metadata explicitly indicated by a tax question."""
    extracted: dict[str, Any] = {}
    tax_year_match = re.search(
        r"\btax\s+year\s+(20\d{2})\b|\bfor\s+(?:the\s+)?year\s+(20\d{2})\b",
        query,
        re.IGNORECASE,
    )
    filing_year_match = re.search(r"\bfiling\s+year\s+(20\d{2})\b", query, re.IGNORECASE)
    if filing_year_match:
        extracted["filing_year"] = int(filing_year_match.group(1))
    if tax_year_match:
        year = next(group for group in tax_year_match.groups() if group is not None)
        extracted["tax_year"] = int(year)
    elif not filing_year_match:
        year_match = re.search(r"\b(20\d{2})\b", query)
        if year_match:
            extracted["tax_year"] = int(year_match.group(1))

    for field, aliases in TAX_VALUE_ALIASES.items():
        matches = [value for value, phrases in aliases.items() if any(_query_contains(query, phrase) for phrase in phrases)]
        if matches:
            extracted[field] = matches if field in TAX_LIST_METADATA_FIELDS else matches[0]

    docs = load_tax_documents()
    for field in ("topic", "subtopic"):
        values = {str(document.metadata[field]) for document in docs if document.metadata.get(field)}
        matches = [value for value in values if _query_contains(query, value.replace("_", " "))]
        if field == "topic":
            matches.extend(
                topic
                for topic, aliases in TAX_TOPIC_ALIASES.items()
                if any(_query_contains(query, alias) for alias in aliases)
            )
        if matches:
            extracted[field] = max(matches, key=len)

    form_values = {str(value) for document in docs for value in (document.metadata.get("forms") or [])}
    matched_forms = [form for form in form_values if _query_contains(query, form)]
    if matched_forms:
        extracted["forms"] = matched_forms

    return extracted


def build_tax_metadata_filter(metadata: dict[str, Any]) -> Filter | None:
    """Build Qdrant payload conditions for the extracted metadata, or no filter."""
    conditions = []
    for field, value in metadata.items():
        if value is None or value == [] or value == "":
            continue
        if field in TAX_LIST_METADATA_FIELDS and isinstance(value, list):
            value = [*value, "all"]
        match = MatchAny(any=value) if isinstance(value, list) else MatchValue(value=value)
        conditions.append(FieldCondition(key=f"metadata.{field}", match=match))
    return Filter(must=conditions) if conditions else None


def chunk_tax_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks, start=1):
        source_chunk_id = chunk.metadata.get("chunk_id", "tax")
        chunk.metadata["chunk_id"] = f"{source_chunk_id}-{index:04d}"
    return chunks


def load_tax_documents(path: str | Path = TAX_CORPUS_PATH) -> list[Document]:
    """Load the tax corpus JSONL and convert each record into a LangChain Document."""
    corpus_path = Path(path)
    documents: list[Document] = []

    with corpus_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            record = json.loads(line)
            content = record.get("content") or ""
            metadata: dict[str, Any] = {}
            for field in TAX_METADATA_FIELDS:
                value = record.get(field)
                if value is not None:
                    metadata[field] = value

            metadata.setdefault("subtopic", record.get("subtopic") or record.get("topic") or "general")
            metadata.setdefault("source", str(corpus_path))
            for field in ("title", "chunk_id"):
                if record.get(field) is not None:
                    metadata[field] = record[field]
            documents.append(Document(page_content=content, metadata=metadata))

    return documents


def build_tax_hybrid_vector_store(
    documents: list[Document] | None = None,
    *,
    client: QdrantClient | None = None,
) -> QdrantVectorStore:
    document_loader = (
        lambda: chunk_tax_documents(documents)
        if documents is not None
        else chunk_tax_documents(load_tax_documents())
    )
    return _build_hybrid_vector_store(
        TAX_COLLECTION_NAME,
        LOCAL_QDRANT_PATH,
        document_loader,
        client=client,
        source_name="the tax corpus",
        require_documents=False,
    )


@lru_cache
def _get_tax_retriever_without_shared_client() -> QdrantRetriever:
    return get_hybrid_retriever(
        TAX_COLLECTION_NAME,
        LOCAL_QDRANT_PATH,
        lambda: chunk_tax_documents(load_tax_documents()),
        source_name="the tax corpus",
    )


def get_tax_retriever(client: QdrantClient | None = None) -> QdrantRetriever:
    if client is None:
        return _get_tax_retriever_without_shared_client()
    return get_hybrid_retriever(
        TAX_COLLECTION_NAME,
        LOCAL_QDRANT_PATH,
        lambda: chunk_tax_documents(load_tax_documents()),
        client=client,
        source_name="the tax corpus",
    )
