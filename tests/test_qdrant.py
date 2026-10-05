from ai_finance_assistant.core.config import Settings
from ai_finance_assistant.rag.qdrant import create_qdrant_client


def test_qdrant_defaults_to_local_disk_mode() -> None:
    settings = Settings(_env_file=None)

    assert settings.qdrant_mode == "local"


def test_local_qdrant_client_can_create_a_collection() -> None:
    client = create_qdrant_client(Settings(_env_file=None, qdrant_mode="local"))
    collection_name = "test_finance_knowledge"

    try:
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)

        client.create_collection(
            collection_name=collection_name,
            vectors_config={"size": 3, "distance": "Cosine"},
        )

        assert client.collection_exists(collection_name)
    finally:
        client.close()
