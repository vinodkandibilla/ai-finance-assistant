from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from ai_finance_assistant.agents.finance_qa.config import FinanceAgentSettings
from ai_finance_assistant.agents.finance_qa.corpus_index import load_finance_corpus
from ai_finance_assistant.agents.finance_qa.metadata_extractor import FinanceMetadataExtractor
from ai_finance_assistant.agents.finance_qa.retrieval import FinanceHybridRetriever
from ai_finance_assistant.agents.finance_qa.schemas import (
    ContextEvaluation,
    FinanceDocument,
    QueryAnalysis,
)
from ai_finance_assistant.rag.orchestration import should_replace_best as _should_replace_best
from ai_finance_assistant.rag.retriever import QdrantHybridSearch


def make_doc(doc_id: str, content: str) -> FinanceDocument:
    return FinanceDocument(doc_id=doc_id, content=content, source="handbook.pdf")


def test_metadata_filter_only_uses_corpus_supported_values() -> None:
    extractor = FinanceMetadataExtractor(
        [
            {"topic": "options", "asset_class": "equity", "ticker": "QQQ"},
            {"topic": "funds", "source": "handbook.pdf"},
        ]
    )

    assert extractor.extract("Explain QQQ options") == {
        "topic": "options",
        "ticker": "QQQ",
    }
    assert extractor.build_qdrant_filter({}) is None
    assert extractor.extract("This word is not a ticker: ALL") == {}


def test_finance_corpus_loader_preserves_nonempty_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    in_scope = writer.add_blank_page(width=300, height=300)
    in_scope.extract_text = lambda: "Exchange-traded funds and index tracking"
    out_scope = writer.add_blank_page(width=300, height=300)
    out_scope.extract_text = lambda: "Traditional IRA retirement tax filing guidance"
    target = tmp_path / "handbook.pdf"
    with target.open("wb") as handle:
        writer.write(handle)

    class FakePage:
        def __init__(self, text: str) -> None:
            self.text = text

        def extract_text(self) -> str:
            return self.text

    class FakeReader:
        pages = [
            FakePage("ETF index tracking concept. " * 30),
            FakePage("Traditional IRA retirement tax."),
            FakePage("   "),
        ]

    from ai_finance_assistant.agents.finance_qa import corpus_index

    monkeypatch.setattr(corpus_index, "PdfReader", lambda _: FakeReader())
    chunks = load_finance_corpus(target, FinanceAgentSettings(chunk_size=150, chunk_overlap=10))

    assert chunks
    assert any("Traditional IRA retirement tax." in chunk.page_content for chunk in chunks)
    assert {chunk.metadata["page_number"] for chunk in chunks} == {1, 2}
    assert all(chunk.metadata["source"] == str(target) for chunk in chunks)
    assert len({chunk.metadata["doc_id"] for chunk in chunks}) == len(chunks)


def test_handbook_preserves_compound_interest_section() -> None:
    chunks = load_finance_corpus()
    page_nine = [chunk for chunk in chunks if chunk.metadata["page_number"] == 9]

    assert any("2.2 Simple Interest versus Compound Interest" in c.page_content for c in page_nine)
    assert any(
        "Compound interest is calculated on the principal plus any interest already earned."
        in c.page_content
        for c in page_nine
    )
    assert any("tax rates" in c.page_content for c in page_nine)


class FakeClassifier:
    def __init__(self, intents: list[str]) -> None:
        self.intents = iter(intents)
        self.calls: list[str] = []

    async def classify(self, query: str) -> str:
        self.calls.append(query)
        return next(self.intents)


class FakeStructuredCall:
    def __init__(self, resolver) -> None:
        self.resolver = resolver

    async def ainvoke(self, messages: object) -> object:
        return self.resolver()


class FakeLLM:
    def __init__(self, analyses: list[QueryAnalysis], evaluations: list[ContextEvaluation]) -> None:
        self.analyses = list(analyses)
        self.evaluations = list(evaluations)
        self.answer_calls = 0
        self.answer_messages: list[object] = []

    def with_structured_output(self, schema: type):
        if schema is QueryAnalysis:
            return FakeStructuredCall(lambda: self.analyses.pop(0))
        return FakeStructuredCall(lambda: self.evaluations.pop(0))

    async def ainvoke(self, messages: object) -> str:
        self.answer_calls += 1
        self.answer_messages.append(messages)
        return "A grounded answer with a handbook citation."


class FakeRetriever:
    def __init__(
        self, lists: list[list[FinanceDocument]], sparse: list[FinanceDocument] | None = None
    ) -> None:
        self.lists = list(lists)
        self.sparse = sparse or []
        self.calls: list[tuple[list[str], list[str]]] = []

    async def retrieve(
        self, dense_queries: list[str], sparse_keywords: list[str], *, metadata_filter=None
    ):
        self.calls.append((dense_queries, sparse_keywords))
        lists = self.lists.pop(0)
        return [*lists, *self.sparse], []


class FakeReranker:
    async def rerank(self, query: str, documents: list[FinanceDocument], *, limit: int):
        return documents[:limit]


def test_best_context_selection_uses_only_sufficient_flag() -> None:
    high = ContextEvaluation(sufficient=True, reason="covers all parts")
    worse = ContextEvaluation(sufficient=True, reason="slightly lower")
    better = ContextEvaluation(sufficient=True, reason="covers all parts")
    insufficient = ContextEvaluation(sufficient=False, reason="missing concept")

    assert not _should_replace_best(high, worse)
    assert not _should_replace_best(high, better)
    assert not _should_replace_best(high, insufficient)
    assert _should_replace_best(insufficient, high)
    assert _should_replace_best(None, insufficient)


@pytest.mark.asyncio
async def test_hybrid_retrieval_returns_qdrant_fused_results(monkeypatch) -> None:
    from types import SimpleNamespace

    settings = FinanceAgentSettings()
    retriever = object.__new__(FinanceHybridRetriever)
    retriever._settings = settings
    retriever._semaphore = __import__("asyncio").Semaphore(settings.max_parallel_searches)
    retriever._index_ready = True
    retriever._collection_name = settings.finance_collection
    retriever._qdrant_client = SimpleNamespace(
        query_points=lambda **kwargs: SimpleNamespace(
            points=[
                SimpleNamespace(
                    id="qdrant-id",
                    score=0.75,
                    payload={
                        "page_content": "Delta describes price sensitivity",
                        "metadata": {
                            "doc_id": "delta",
                            "source": "handbook.pdf",
                        },
                    },
                )
            ]
        )
    )

    async def embed_query(query: str) -> list[float]:
        return [1.0, 0.0]

    class SparseVectorStub:
        indices = [1]
        values = [1.0]

    retriever._hybrid_search = QdrantHybridSearch(
        retriever._qdrant_client,
        retriever._collection_name,
        SimpleNamespace(aembed_query=embed_query),
        SimpleNamespace(embed_query=lambda query: SparseVectorStub()),
    )

    fused, errors = await retriever.retrieve(["Delta"], ["Delta"])

    assert errors == []
    assert len(fused) == 1
    assert fused[0].doc_id == "delta"
    assert fused[0].content == "Delta describes price sensitivity"
    assert fused[0].score == 0.75
    assert fused[0].rank == 1


@pytest.mark.asyncio
async def test_hybrid_retrieval_reports_qdrant_fusion_failure() -> None:
    settings = FinanceAgentSettings()
    retriever = object.__new__(FinanceHybridRetriever)
    retriever._settings = settings
    retriever._semaphore = __import__("asyncio").Semaphore(settings.max_parallel_searches)
    retriever._index_ready = True
    retriever._collection_name = settings.finance_collection
    retriever._qdrant_client = __import__("types").SimpleNamespace(
        query_points=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("Qdrant unavailable"))
    )

    async def embed_query(query: str) -> list[float]:
        return [1.0, 0.0]

    retriever._hybrid_search = QdrantHybridSearch(
        retriever._qdrant_client,
        retriever._collection_name,
        __import__("types").SimpleNamespace(aembed_query=embed_query),
        __import__("types").SimpleNamespace(
            embed_query=lambda query: __import__("types").SimpleNamespace(indices=[1], values=[1.0])
        ),
    )

    candidates, errors = await retriever.retrieve(["Delta"], ["Delta"])

    assert candidates == []
    assert errors == ["hybrid retrieval failed: RuntimeError"]


@pytest.mark.asyncio
async def test_finance_index_creation_reuses_client_and_supports_search(monkeypatch) -> None:
    from langchain_core.embeddings import Embeddings
    from qdrant_client import QdrantClient

    class LocalEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0, 0.0] for _ in texts]

        def embed_query(self, text: str) -> list[float]:
            return [1.0, 0.0, 0.0]

    monkeypatch.setattr(
        "ai_finance_assistant.agents.finance_qa.retrieval.OpenAIEmbeddings",
        lambda **kwargs: LocalEmbeddings(),
    )
    from langchain_qdrant.sparse_embeddings import SparseEmbeddings, SparseVector

    class LocalSparseEmbeddings(SparseEmbeddings):
        def embed_documents(self, texts: list[str]) -> list[SparseVector]:
            return [self.embed_query(text) for text in texts]

        def embed_query(self, text: str) -> SparseVector:
            return SparseVector(indices=[1], values=[1.0])

    monkeypatch.setattr(
        "ai_finance_assistant.agents.finance_qa.retrieval.FastEmbedSparse",
        lambda **kwargs: LocalSparseEmbeddings(),
    )
    documents = [
        Document(
            page_content="ETF net asset value and market price explained.",
            metadata={"doc_id": "etf-nav", "source": "handbook.pdf", "page_number": 1},
        )
    ]
    settings = FinanceAgentSettings()
    client = QdrantClient(location=":memory:")
    try:
        retriever = FinanceHybridRetriever.create(
            documents,
            qdrant_client=client,
            api_key="test-key",
            embedding_model="test-model",
            settings=settings,
        )
        assert client.count(settings.finance_collection).count == 1
        matches, errors = await retriever.retrieve(["ETF NAV"], ["ETF", "NAV"])
        assert errors == []
        assert matches[0].doc_id == "etf-nav"
        assert matches[0].rank == 1
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        excluded = Filter(
            must=[FieldCondition(key="metadata.page_number", match=MatchValue(value=2))]
        )
        filtered_matches, errors = await retriever.retrieve(
            ["ETF NAV"], ["ETF"], metadata_filter=excluded
        )
        assert errors == []
        assert filtered_matches == []
        FinanceHybridRetriever.create(
            documents,
            qdrant_client=client,
            api_key="test-key",
            embedding_model="test-model",
            settings=settings,
        )
        assert client.count(settings.finance_collection).count == 1
        # An interrupted first ingestion can leave a valid but empty collection.
        client.delete(
            settings.finance_collection,
            points_selector=Filter(),
        )
        assert client.count(settings.finance_collection).count == 0
        recovered = FinanceHybridRetriever.create(
            documents,
            qdrant_client=client,
            api_key="test-key",
            embedding_model="test-model",
            settings=settings,
        )
        assert client.count(settings.finance_collection).count == 1
        recovered_matches, errors = await recovered.retrieve(["ETF NAV"], ["ETF"])
        assert errors == []
        assert recovered_matches[0].doc_id == "etf-nav"
        client.create_collection(
            collection_name="legacy-dense", vectors_config={"": {"size": 3, "distance": "Cosine"}}
        )
        with pytest.raises(ValueError, match="must have sparse vectors configured"):
            FinanceHybridRetriever.create(
                documents,
                qdrant_client=client,
                api_key="test-key",
                embedding_model="test-model",
                settings=FinanceAgentSettings(finance_collection="legacy-dense"),
            )
        assert client.collection_exists("legacy-dense")
        assert client.count("legacy-dense").count == 0
    finally:
        client.close()


@pytest.mark.asyncio
async def test_finance_retriever_returns_empty_results_when_no_documents(monkeypatch) -> None:
    from qdrant_client import QdrantClient

    monkeypatch.setattr(
        "ai_finance_assistant.agents.finance_qa.retrieval.OpenAIEmbeddings",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "ai_finance_assistant.agents.finance_qa.retrieval.FastEmbedSparse",
        lambda **kwargs: object(),
    )
    settings = FinanceAgentSettings()
    client = QdrantClient(location=":memory:")
    try:
        retriever = FinanceHybridRetriever.create(
            [],
            qdrant_client=client,
            api_key="test-key",
            embedding_model="test-model",
            settings=settings,
        )

        assert not client.collection_exists(settings.finance_collection)
        assert await retriever.retrieve(["ETF"], ["ETF"]) == ([], [])
    finally:
        client.close()


async def test_etf_glossary_supports_definition_but_not_comparison() -> None:
    from ai_finance_assistant.agents.finance_qa.evaluator import FinanceContextEvaluator

    llm = FakeLLM(
        [],
        [
            ContextEvaluation(
                sufficient=False,
                missing_concepts=["mutual fund"],
                reason="No comparison evidence",
            )
        ],
    )
    evaluator = FinanceContextEvaluator(llm)
    glossary = make_doc(
        "etf-definition",
        "European-style option\nAn option exercisable only at expiration.\n"
        "Exchange-traded fund (ETF)\n"
        "An investment fund that trades on an exchange throughout the day "
        "and typically holds a portfolio of securities.\nExercise\n",
    )
    result = await evaluator.evaluate("What is ETF?", [glossary])
    assert result.sufficient
    assert len(llm.evaluations) == 1
    comparison = await evaluator.evaluate("What is ETF versus a mutual fund?", [glossary])
    assert not comparison.sufficient
    assert not llm.evaluations
    empty = await evaluator.evaluate("What is ETF?", [])
    assert not empty.sufficient


def test_public_citations_and_generation_context_omit_internal_ids() -> None:
    from ai_finance_assistant.agents.finance_qa.graph import _citations, _format_context

    documents = [
        FinanceDocument(
            doc_id="internal-chunk-a",
            content="ETF evidence.",
            source="/private/handbook.pdf",
            metadata={"page_number": 50},
        ),
        FinanceDocument(
            doc_id="internal-chunk-b",
            content="More ETF evidence.",
            source="/private/handbook.pdf",
            metadata={"page_number": 50},
        ),
        FinanceDocument(
            doc_id="internal-chunk-c",
            content="Risk evidence.",
            source="/private/handbook.pdf",
            metadata={"page_number": 35},
        ),
    ]
    assert _citations(documents) == [
        "Financial Education Handbook, page 50",
        "Financial Education Handbook, page 35",
    ]
    context = _format_context(documents)
    assert "page 50" in context and "ETF evidence." in context
    assert "internal-chunk" not in context
    assert "/private/handbook.pdf" not in context
    assert documents[0].doc_id == "internal-chunk-a"
