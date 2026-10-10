from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from ai_finance_assistant.agents.finance_qa.config import FinanceAgentSettings
from ai_finance_assistant.agents.finance_qa.corpus_index import load_finance_corpus
from ai_finance_assistant.agents.finance_qa.graph import FinanceQaAgent, _should_replace_best
from ai_finance_assistant.agents.finance_qa.metadata_extractor import FinanceMetadataExtractor
from ai_finance_assistant.agents.finance_qa.rrf import reciprocal_rank_fusion
from ai_finance_assistant.agents.finance_qa.schemas import (
    ContextEvaluation,
    FinanceDocument,
    QueryAnalysis,
)
from ai_finance_assistant.agents.finance_qa.retrieval import FinanceHybridRetriever


def make_doc(doc_id: str, content: str) -> FinanceDocument:
    return FinanceDocument(doc_id=doc_id, content=content, source="handbook.pdf")


def test_rrf_uses_one_based_ranks_accumulates_duplicates_and_deduplicates_output() -> None:
    first = [make_doc("a", "A"), make_doc("b", "B")]
    second = [make_doc("a", "A"), make_doc("c", "C")]
    third = [make_doc("a", "A")]
    sparse = [make_doc("a", "A"), make_doc("b", "B")]

    fused = reciprocal_rank_fusion([first, second, third, sparse], k=60, limit=20)

    assert len(fused) == 3
    assert fused[0].doc_id == "a"
    assert fused[0].score == pytest.approx(4 / 61)
    assert fused[0].rank == 1
    assert fused[0].provenance == [
        "ranked_list_1:rank_1",
        "ranked_list_2:rank_1",
        "ranked_list_3:rank_1",
        "ranked_list_4:rank_1",
    ]


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


def test_finance_corpus_loader_excludes_out_of_scope_pages(tmp_path: Path) -> None:
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
        pages = [FakePage("ETF index tracking concept. " * 30), FakePage("Traditional IRA retirement tax.")]

    from ai_finance_assistant.agents.finance_qa import corpus_index

    monkeypatch_reader = pytest.MonkeyPatch()
    monkeypatch_reader.setattr(corpus_index, "PdfReader", lambda _: FakeReader())
    chunks = load_finance_corpus(target, FinanceAgentSettings(chunk_size=150, chunk_overlap=10))
    monkeypatch_reader.undo()

    assert chunks
    assert all("traditional ira" not in chunk.page_content.casefold() for chunk in chunks)
    assert all(chunk.metadata["source"] == str(target) for chunk in chunks)
    assert len({chunk.metadata["doc_id"] for chunk in chunks}) == len(chunks)


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
    def __init__(self, lists: list[list[FinanceDocument]], sparse: list[FinanceDocument] | None = None) -> None:
        self.lists = list(lists)
        self.sparse = sparse or []
        self.calls: list[tuple[list[str], list[str]]] = []

    async def retrieve(self, dense_queries: list[str], sparse_keywords: list[str], *, metadata_filter=None):
        self.calls.append((dense_queries, sparse_keywords))
        lists = self.lists.pop(0)
        return [lists], self.sparse, []


class FakeReranker:
    async def rerank(self, query: str, documents: list[FinanceDocument], *, limit: int):
        return documents[:limit]


def make_agent(llm: FakeLLM, retriever: FakeRetriever, classifier: FakeClassifier) -> FinanceQaAgent:
    return FinanceQaAgent(
        llm=llm,
        retriever=retriever,
        metadata_extractor=FinanceMetadataExtractor([]),
        classifier=classifier,
        reranker=FakeReranker(),
        settings=FinanceAgentSettings(rerank_top_n=5),
        checkpointer=MemorySaver(),
    )


@pytest.mark.asyncio
async def test_out_of_scope_question_returns_handoff_without_retrieval() -> None:
    llm = FakeLLM([], [])
    retriever = FakeRetriever([])
    agent = make_agent(llm, retriever, FakeClassifier([]))

    result = await agent.ainvoke({"user_query": "Compare Roth IRA and Traditional IRA"}, {})

    assert result["final_result"]["status"] == "handoff"
    assert "tax" in result["final_result"]["handoff_intents"]
    assert retriever.calls == []


@pytest.mark.asyncio
async def test_query_clarification_interrupt_resumes_through_parent_classifier() -> None:
    llm = FakeLLM(
        [QueryAnalysis(needs_clarification=True, clarification_question="Which two concepts?")],
        [],
    )
    classifier = FakeClassifier(["market"])
    agent = make_agent(llm, FakeRetriever([]), classifier)
    graph = agent.graph
    config = {"configurable": {"thread_id": "finance-clarification-test"}}

    paused = await graph.ainvoke({"user_query": "Which is better?"}, config=config)
    snapshot = await graph.aget_state(config)
    assert snapshot.tasks[0].interrupts[0].value["question"] == "Which two concepts?"
    resumed = await graph.ainvoke(Command(resume="What is today's AAPL price?"), config=config)

    assert resumed["final_result"]["status"] == "handoff"
    assert "market" in resumed["final_result"]["handoff_intents"]
    assert resumed["intent"] == "market"
    assert classifier.calls == ["What is today's AAPL price?"]


def test_best_context_selection_prefers_sufficient_coverage_and_avoids_small_score_swaps() -> None:
    high = ContextEvaluation(sufficient=True, sufficiency_score=0.8, reason="covers all parts")
    worse = ContextEvaluation(sufficient=True, sufficiency_score=0.75, reason="slightly lower")
    better = ContextEvaluation(sufficient=True, sufficiency_score=0.95, reason="covers all parts")
    insufficient = ContextEvaluation(sufficient=False, sufficiency_score=0.99, reason="missing concept")

    assert not _should_replace_best(high, worse, tie_tolerance=0.08)
    assert _should_replace_best(high, better, tie_tolerance=0.08)
    assert not _should_replace_best(high, insufficient, tie_tolerance=0.08)


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_channel", ["dense", "sparse"])
async def test_hybrid_retrieval_returns_surviving_channel(failed_channel: str) -> None:
    settings = FinanceAgentSettings()
    retriever = object.__new__(FinanceHybridRetriever)
    retriever._settings = settings
    retriever._semaphore = __import__("asyncio").Semaphore(settings.max_parallel_searches)
    doc = make_doc("one", "Delta describes price sensitivity")

    async def dense_ok(query: str, metadata_filter: object | None):
        if failed_channel == "dense":
            raise RuntimeError("dense down")
        return [doc]

    def sparse_ok(keywords: list[str], metadata_filter: object | None):
        if failed_channel == "sparse":
            raise RuntimeError("sparse down")
        return [doc]

    retriever._dense_search = dense_ok
    retriever._sparse_search = sparse_ok
    dense, sparse, errors = await retriever.retrieve(["Delta"], ["Delta"])

    assert errors
    if failed_channel == "dense":
        assert dense == [] and sparse == [doc]
    else:
        assert dense == [[doc]] and sparse == []


@pytest.mark.asyncio
async def test_hybrid_retrieval_reports_both_channels_failed() -> None:
    settings = FinanceAgentSettings()
    retriever = object.__new__(FinanceHybridRetriever)
    retriever._settings = settings
    retriever._semaphore = __import__("asyncio").Semaphore(settings.max_parallel_searches)

    async def dense_fail(query: str, metadata_filter: object | None):
        raise RuntimeError("dense down")

    def sparse_fail(keywords: list[str], metadata_filter: object | None):
        raise RuntimeError("sparse down")

    retriever._dense_search = dense_fail
    retriever._sparse_search = sparse_fail
    dense, sparse, errors = await retriever.retrieve(["Delta"], ["Delta"])

    assert dense == [] and sparse == []
    assert len(errors) == 2


@pytest.mark.asyncio
async def test_agent_retries_once_and_keeps_better_first_context() -> None:
    first_doc = make_doc("delta", "Delta describes call price sensitivity.")
    second_doc = make_doc("noise", "Unrelated index information.")
    analysis = QueryAnalysis(
        needs_clarification=False,
        detected_intents=["finance_qa"],
        dense_sub_queries=["call option Delta and Theta"],
        sparse_keywords=["Delta", "Theta"],
    )
    llm = FakeLLM(
        [analysis],
        [
            ContextEvaluation(
                sufficient=False,
                sufficiency_score=0.80,
                missing_concepts=["Theta"],
                suggested_queries=["Theta call option time decay"],
                reason="Initial passages are missing Theta.",
            ),
            ContextEvaluation(
                sufficient=False,
                sufficiency_score=0.60,
                missing_concepts=["Theta"],
                reason="Retry result is less complete.",
            ),
        ],
    )
    retriever = FakeRetriever([[first_doc], [second_doc]])
    agent = make_agent(llm, retriever, FakeClassifier([]))

    result = await agent.ainvoke({"user_query": "Explain Delta and Theta."}, {})

    assert len(retriever.calls) == 2
    assert result["final_result"]["status"] == "insufficient", result
    assert "Delta describes" in result["final_result"]["answer"] or result["final_result"]["citations"]
    assert llm.answer_calls == 0

def test_finance_index_creation_reuses_client_and_supports_search(monkeypatch) -> None:
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
        matches = retriever._dense_store.similarity_search("ETF NAV", k=1)
        assert matches[0].metadata["doc_id"] == "etf-nav"
        FinanceHybridRetriever.create(
            documents,
            qdrant_client=client,
            api_key="test-key",
            embedding_model="test-model",
            settings=settings,
        )
        assert client.count(settings.finance_collection).count == 1
    finally:
        client.close()


async def test_parent_passes_current_question_to_finance_subgraph() -> None:
    from ai_finance_assistant.graph.builder import build_graph

    seen_queries: list[str] = []

    class RecordingLLM(FakeLLM):
        def with_structured_output(self, schema: type):
            if schema is not QueryAnalysis:
                return super().with_structured_output(schema)

            class Analyze:
                async def ainvoke(self, messages):
                    query = messages[-1].content
                    seen_queries.append(query)
                    if query == "What is Delta?":
                        return QueryAnalysis(
                            needs_clarification=True,
                            clarification_question="Which Delta context?",
                        )
                    assert query == "What is ETF?"
                    return QueryAnalysis(
                        needs_clarification=False,
                        detected_intents=["finance_qa"],
                        dense_sub_queries=[query],
                        sparse_keywords=["ETF"],
                    )

            return Analyze()

    llm = RecordingLLM(
        [],
        [ContextEvaluation(sufficient=True, sufficiency_score=0.95, reason="ETF covered")],
    )
    retriever = FakeRetriever([[make_doc("etf", "An ETF is an exchange-traded fund.")]])
    classifier = FakeClassifier(["finance_qa", "finance_qa"])
    agent = FinanceQaAgent(
        llm=llm,
        retriever=retriever,
        metadata_extractor=FinanceMetadataExtractor([]),
        classifier=classifier,
        reranker=FakeReranker(),
        settings=FinanceAgentSettings(),
    )
    graph = build_graph(classifier, finance_agent=agent, checkpointer=MemorySaver())
    delta = await graph.ainvoke(
        {"query": "What is Delta?"},
        config={"configurable": {"thread_id": "delta-request"}},
    )
    assert delta["__interrupt__"]
    etf = await graph.ainvoke(
        {"query": "What is ETF?"},
        config={"configurable": {"thread_id": "etf-request"}},
    )
    assert seen_queries == ["What is Delta?", "What is ETF?"]
    assert etf["status"] == "answered"
    assert etf["answer"]
    assert not etf.get("clarification_question")
    assert retriever.calls[0][0] == ["What is ETF?"]
    assert "What is ETF?" in llm.answer_messages[0][-1].content


async def test_etf_glossary_supports_definition_but_not_comparison() -> None:
    from ai_finance_assistant.agents.finance_qa.evaluator import FinanceContextEvaluator

    llm = FakeLLM([], [ContextEvaluation(
        sufficient=False, sufficiency_score=0.2,
        missing_concepts=["mutual fund"], reason="No comparison evidence",
    )])
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
