import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from ai_finance_assistant.api import query as query_api
from ai_finance_assistant.api.query import get_query_graph
from ai_finance_assistant.core.config import Settings
from ai_finance_assistant.graph.builder import (
    CLARIFY_RESPONSE,
    FINANCE_QA_RESPONSE,
    GOALS_RESPONSE,
    MARKET_QUOTE_UNAVAILABLE_RESPONSE,
    NEWS_RESPONSE,
    PORTFOLIO_ANALYSIS_RESPONSE,
    TAX_RESPONSE,
    build_graph,
)
from ai_finance_assistant.main import create_app


class FakeClassifier:
    def __init__(self, intent: str = "finance_qa") -> None:
        self.intent = intent

    async def classify(self, query: str) -> str:
        return self.intent


def test_query_endpoint_returns_routed_finance_answer() -> None:
    app = create_app()
    app.dependency_overrides[get_query_graph] = lambda: build_graph(FakeClassifier())
    client = TestClient(app)

    response = client.post("/query", json={"query": "What is an ETF?"})

    assert response.status_code == 200
    assert response.json() == {"intent": "finance_qa", "answer": FINANCE_QA_RESPONSE}


def test_query_endpoint_rejects_blank_query_without_llm_call() -> None:
    app = create_app()
    app.dependency_overrides[get_query_graph] = lambda: build_graph(FakeClassifier())
    client = TestClient(app)

    response = client.post("/query", json={"query": "  "})

    assert response.status_code == 422


def test_query_endpoint_routes_portfolio_intent() -> None:
    app = create_app()
    app.dependency_overrides[get_query_graph] = lambda: build_graph(FakeClassifier("portfolio"))
    client = TestClient(app)

    response = client.post(
        "/query",
        json={"query": "Analyze my portfolio allocation and concentration."},
    )

    assert response.status_code == 200
    assert response.json() == {
        "intent": "portfolio",
        "answer": PORTFOLIO_ANALYSIS_RESPONSE,
    }


def test_query_endpoint_routes_market_intent() -> None:
    app = create_app()
    app.dependency_overrides[get_query_graph] = lambda: build_graph(FakeClassifier("market"))
    client = TestClient(app)

    response = client.post("/query", json={"query": "What is the current price of AAPL?"})

    assert response.status_code == 200
    assert response.json() == {"intent": "market", "answer": MARKET_QUOTE_UNAVAILABLE_RESPONSE}


def test_query_graph_requires_openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        query_api,
        "get_settings",
        lambda: Settings(_env_file=None, openai_api_key=None),
    )
    get_query_graph.cache_clear()

    try:
        with pytest.raises(HTTPException) as error:
            get_query_graph()
        assert get_query_graph.cache_info().currsize == 0
    finally:
        get_query_graph.cache_clear()

    assert error.value.status_code == 503


def test_query_graph_does_not_cache_when_llm_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        query_api,
        "get_settings",
        lambda: Settings(_env_file=None, openai_api_key="test-key"),
    )
    monkeypatch.setattr(query_api, "create_intent_classifier", lambda settings: FakeClassifier())
    calls = 0

    def fail_to_create_llm(**kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("LLM initialization failed")

    monkeypatch.setattr(query_api, "ChatOpenAI", fail_to_create_llm)
    get_query_graph.cache_clear()

    try:
        for _ in range(2):
            with pytest.raises(RuntimeError, match="LLM initialization failed"):
                get_query_graph()
        assert calls == 2
        assert get_query_graph.cache_info().currsize == 0
    finally:
        get_query_graph.cache_clear()


def test_query_graph_does_not_cache_when_retriever_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        query_api,
        "get_settings",
        lambda: Settings(_env_file=None, openai_api_key="test-key"),
    )
    monkeypatch.setattr(query_api, "create_intent_classifier", lambda settings: FakeClassifier())
    monkeypatch.setattr(query_api, "ChatOpenAI", lambda **kwargs: object())
    calls = 0

    def fail_to_create_retriever() -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("retriever initialization failed")

    monkeypatch.setattr(query_api, "get_finance_retriever", fail_to_create_retriever)
    get_query_graph.cache_clear()

    try:
        for _ in range(2):
            with pytest.raises(RuntimeError, match="retriever initialization failed"):
                get_query_graph()
        assert calls == 2
        assert get_query_graph.cache_info().currsize == 0
    finally:
        get_query_graph.cache_clear()


def test_settings_read_router_prompt_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENT_ROUTER_SYSTEM_PROMPT", "Use the configured test prompt.")

    settings = Settings(_env_file=None)

    assert settings.intent_router_system_prompt == "Use the configured test prompt."


def test_query_endpoint_returns_fixed_goals_response() -> None:
    app = create_app()
    app.dependency_overrides[get_query_graph] = lambda: build_graph(FakeClassifier("goals"))
    client = TestClient(app)

    response = client.post(
        "/query",
        json={"query": "How much should I save each month?"},
    )

    assert response.status_code == 200
    assert response.json() == {"intent": "goals", "answer": GOALS_RESPONSE}


def test_query_endpoint_routes_news_intent() -> None:
    app = create_app()
    app.dependency_overrides[get_query_graph] = lambda: build_graph(FakeClassifier("news"))
    client = TestClient(app)

    response = client.post(
        "/query",
        json={"query": "Give me a concise summary of today's top AI news."},
    )

    assert response.status_code == 200
    assert response.json() == {"intent": "news", "answer": NEWS_RESPONSE}


def test_query_endpoint_routes_tax_intent() -> None:
    app = create_app()
    app.dependency_overrides[get_query_graph] = lambda: build_graph(FakeClassifier("tax"))
    client = TestClient(app)

    response = client.post(
        "/query",
        json={
            "query": "What is the standard deduction for Married Filing Jointly with two qualifying children for the year 2026?"
        },
    )

    assert response.status_code == 200
    assert response.json() == {"intent": "tax", "answer": TAX_RESPONSE}


def test_query_endpoint_routes_clarify_intent_for_incomplete_details() -> None:
    app = create_app()
    app.dependency_overrides[get_query_graph] = lambda: build_graph(FakeClassifier("clarify"))
    client = TestClient(app)

    response = client.post(
        "/query",
        json={"query": "What is the stock price?"},
    )

    assert response.status_code == 200
    assert response.json() == {"intent": "clarify", "answer": CLARIFY_RESPONSE}


def test_query_endpoint_routes_generic_market_query_to_clarify() -> None:
    app = create_app()
    app.dependency_overrides[get_query_graph] = lambda: build_graph(FakeClassifier("market"))
    client = TestClient(app)

    response = client.post(
        "/query",
        json={"query": "What is the stock price?"},
    )

    assert response.status_code == 200
    assert response.json() == {"intent": "market", "answer": CLARIFY_RESPONSE}


def test_query_endpoint_routes_generic_news_query_to_clarify() -> None:
    app = create_app()
    app.dependency_overrides[get_query_graph] = lambda: build_graph(FakeClassifier("news"))
    client = TestClient(app)

    response = client.post(
        "/query",
        json={"query": "top news"},
    )

    assert response.status_code == 200
    assert response.json() == {"intent": "news", "answer": CLARIFY_RESPONSE}


def test_query_endpoint_routes_generic_tax_query_to_clarify() -> None:
    app = create_app()
    app.dependency_overrides[get_query_graph] = lambda: build_graph(FakeClassifier("tax"))
    client = TestClient(app)

    response = client.post(
        "/query",
        json={"query": "tax details"},
    )

    assert response.status_code == 200
    assert response.json() == {"intent": "tax", "answer": CLARIFY_RESPONSE}


def test_build_hybrid_vector_store_rebuilds_empty_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from langchain_core.documents import Document

    from ai_finance_assistant.rag import finance_education as ingest
    from ai_finance_assistant.rag import hybrid

    class FakeSettings:
        openai_api_key = SimpleNamespace(get_secret_value=lambda: "test-key")

    class FakeEmbeddings:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class FakeSparseEmbeddings:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def collection_exists(self, name: str) -> bool:
            return True

        def count(self, collection_name: str) -> SimpleNamespace:
            return SimpleNamespace(count=0)

        def delete_collection(self, name: str) -> None:
            self.deleted = name

        def close(self) -> None:
            return None

    called: dict[str, bool] = {"from_documents": False, "from_existing_collection": False}

    class FakeVectorStore:
        def __init__(self) -> None:
            self.client = SimpleNamespace(
                count=lambda collection_name: SimpleNamespace(count=1),
            )

    def fake_from_documents(*args: object, **kwargs: object) -> FakeVectorStore:
        called["from_documents"] = True
        return FakeVectorStore()

    def fake_from_existing_collection(*args: object, **kwargs: object) -> FakeVectorStore:
        called["from_existing_collection"] = True
        return FakeVectorStore()

    monkeypatch.setattr(hybrid, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(hybrid, "OpenAIEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(hybrid, "FastEmbedSparse", FakeSparseEmbeddings)
    monkeypatch.setattr(hybrid, "QdrantClient", FakeClient)
    monkeypatch.setattr(hybrid.QdrantVectorStore, "from_documents", staticmethod(fake_from_documents))
    monkeypatch.setattr(
        hybrid.QdrantVectorStore,
        "from_existing_collection",
        staticmethod(fake_from_existing_collection),
    )

    docs = [Document(page_content="Emergency fund basics", metadata={"page_number": 1})]
    store = ingest.build_finance_hybrid_vector_store(docs)

    assert isinstance(store, FakeVectorStore)
    assert called["from_documents"] is True
    assert called["from_existing_collection"] is False


def test_build_hybrid_vector_store_closes_inspection_client_before_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from ai_finance_assistant.rag import finance_education as ingest
    from ai_finance_assistant.rag import hybrid

    class FakeSettings:
        openai_api_key = SimpleNamespace(get_secret_value=lambda: "test-key")

    class FakeEmbeddings:
        def __init__(self, **kwargs: object) -> None:
            pass

    class FakeSparseEmbeddings:
        def __init__(self, **kwargs: object) -> None:
            pass

    class FakeClient:
        is_open = False

        def __init__(self, **kwargs: object) -> None:
            assert not FakeClient.is_open
            FakeClient.is_open = True

        def collection_exists(self, name: str) -> bool:
            return True

        def count(self, collection_name: str) -> SimpleNamespace:
            return SimpleNamespace(count=1)

        def close(self) -> None:
            FakeClient.is_open = False

    class FakeVectorStore:
        pass

    def fake_from_existing_collection(**kwargs: object) -> FakeVectorStore:
        assert not FakeClient.is_open
        return FakeVectorStore()

    monkeypatch.setattr(hybrid, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(hybrid, "OpenAIEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(hybrid, "FastEmbedSparse", FakeSparseEmbeddings)
    monkeypatch.setattr(hybrid, "QdrantClient", FakeClient)
    monkeypatch.setattr(
        hybrid.QdrantVectorStore,
        "from_existing_collection",
        staticmethod(fake_from_existing_collection),
    )

    store = ingest.build_finance_hybrid_vector_store()

    assert isinstance(store, FakeVectorStore)
