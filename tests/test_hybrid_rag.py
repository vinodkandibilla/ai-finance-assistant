from __future__ import annotations

from types import SimpleNamespace

from ai_finance_assistant.rag import hybrid


def test_shared_client_reuses_collection_without_loading_documents(monkeypatch, tmp_path) -> None:
    class FakeClient:
        closed = False

        def collection_exists(self, collection_name: str) -> bool:
            return collection_name == "tax_collection"

        def count(self, collection_name: str) -> SimpleNamespace:
            return SimpleNamespace(count=7)

        def close(self) -> None:
            self.closed = True

    class FakeVectorStore:
        def __init__(self, **kwargs) -> None:
            self.client = kwargs["client"]

    shared_client = FakeClient()
    loader_called = False

    def load_documents():
        nonlocal loader_called
        loader_called = True
        return []

    monkeypatch.setattr(
        hybrid,
        "get_settings",
        lambda: SimpleNamespace(
            openai_api_key=SimpleNamespace(get_secret_value=lambda: "test-key"),
            openai_embedding_model="text-embedding-3-small",
        ),
    )
    monkeypatch.setattr(hybrid, "OpenAIEmbeddings", lambda **kwargs: object())
    monkeypatch.setattr(hybrid, "FastEmbedSparse", lambda **kwargs: object())
    monkeypatch.setattr(hybrid, "QdrantVectorStore", FakeVectorStore)

    vector_store = hybrid.build_hybrid_vector_store(
        "tax_collection",
        tmp_path,
        load_documents,
        client=shared_client,
    )

    assert vector_store.client is shared_client
    assert shared_client.closed is False
    assert loader_called is False