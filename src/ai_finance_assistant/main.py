from fastapi import FastAPI

from ai_finance_assistant.api.health import router as health_router
from ai_finance_assistant.api.query import router as query_router
from ai_finance_assistant.core.config import get_settings
from ai_finance_assistant.core.logging import configure_logging


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="AI Finance Assistant",
        version="0.1.0",
        description="A modular API foundation for an AI-powered finance assistant.",
    )
    app.include_router(health_router)
    app.include_router(query_router)
    return app


app = create_app()
