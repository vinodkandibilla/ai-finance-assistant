from functools import lru_cache
from pathlib import Path

from qdrant_client import QdrantClient

from ai_finance_assistant.core.config import Settings, get_settings

LOCAL_QDRANT_PATH = Path(__file__).resolve().parents[3] / ".qdrant"


def create_qdrant_client(settings: Settings) -> QdrantClient:
    if settings.qdrant_mode == "local":
        LOCAL_QDRANT_PATH.mkdir(parents=True, exist_ok=True)
        # Qdrant local persistence requires the `path` argument; `location` is treated
        # as a remote URL unless it is exactly ':memory:'.
        return QdrantClient(path=str(LOCAL_QDRANT_PATH))

    api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    return QdrantClient(url=settings.qdrant_url, api_key=api_key)


@lru_cache
def get_qdrant_client() -> QdrantClient:
    return create_qdrant_client(get_settings())
