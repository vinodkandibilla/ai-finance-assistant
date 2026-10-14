from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from ai_finance_assistant.agents.portfolio_analysis.authorization import UserContext
from ai_finance_assistant.agents.portfolio_analysis.store import PortfolioStore
from ai_finance_assistant.api.health import router as health_router
from ai_finance_assistant.api.query import get_query_graph
from ai_finance_assistant.api.query import router as query_router
from ai_finance_assistant.core.config import get_settings
from ai_finance_assistant.core.logging import configure_logging
from ai_finance_assistant.graph.classifier import (
    ClassificationOutputError,
    ClassificationProviderError,
    ClassificationTimeoutError,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    user_id = getattr(settings, "portfolio_simulated_user_id", None)
    role = getattr(settings, "portfolio_simulated_user_role", None)
    if bool(user_id) != bool(role):
        raise ValueError("Set both portfolio simulated user ID and role, or neither.")
    app.state.portfolio_user_context = UserContext(user_id=user_id, role=role) if user_id else None
    checkpoint_path = Path(get_settings().finance_checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path.cwd() / checkpoint_path
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
        await saver.setup()
        app.state.query_checkpointer = saver
        app.state.portfolio_store = PortfolioStore()
        try:
            yield
        finally:
            get_query_graph.cache_clear()
            app.state.portfolio_store.close()
            del app.state.portfolio_store
            del app.state.portfolio_user_context
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

    @app.exception_handler(ClassificationTimeoutError)
    async def classification_timeout_handler(request: Request, error: ClassificationTimeoutError):
        return JSONResponse(
            status_code=504,
            content={"detail": "Intent classification timed out. Please try again."},
        )

    @app.exception_handler(ClassificationProviderError)
    async def classification_provider_handler(request: Request, error: ClassificationProviderError):
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Intent classification provider is unavailable. Please try again later."
            },
        )

    @app.exception_handler(ClassificationOutputError)
    async def classification_output_handler(request: Request, error: ClassificationOutputError):
        return JSONResponse(
            status_code=502,
            content={
                "detail": "Intent classification returned an invalid response. Please try again."
            },
        )

    app.include_router(health_router)
    app.include_router(query_router)
    return app


app = create_app()
