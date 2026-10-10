from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from ai_finance_assistant.api.health import router as health_router
from ai_finance_assistant.api.query import get_query_graph
from ai_finance_assistant.api.query import router as query_router
from ai_finance_assistant.core.config import get_settings
from ai_finance_assistant.core.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    checkpoint_path = Path(get_settings().finance_checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path.cwd() / checkpoint_path
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
        await saver.setup()
        app.state.query_checkpointer = saver
        try:
            yield
        finally:
            get_query_graph.cache_clear()
            del app.state.query_checkpointer


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        lifespan=lifespan,
        title="AI Finance Assistant",
        version="0.1.0",
        description="A modular API foundation for an AI-powered finance assistant.",
    )
    app.include_router(health_router)
    app.include_router(query_router)
    return app


app = create_app()
