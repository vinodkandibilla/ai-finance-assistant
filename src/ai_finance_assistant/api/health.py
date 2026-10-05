from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from ai_finance_assistant import __version__

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)
