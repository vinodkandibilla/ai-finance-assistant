"""Typed synchronous HTTP adapter. No agent imports, redirects, or automatic retries."""

from types import TracebackType
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from finassist.web_app.config import WebSettings
from finassist.web_app.services.models import (
    HealthResponse,
    PortfolioManagerDirectory,
    QueryRequest,
    QueryResponse,
    ResumeRequest,
)

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class ApiError(Exception):
    """Safe UI-facing failure; never includes server bodies, URLs, or request data."""

    def __init__(self, message: str, *, code: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class BackendUnavailable(ApiError):
    """The backend could not be reached."""


class FinanceApiClient:
    def __init__(
        self,
        settings: WebSettings,
        *,
        transport: httpx.BaseTransport | None = None,
        portfolio_manager_id: str | None = None,
    ) -> None:
        self._settings = settings
        self._client = httpx.Client(
            base_url=settings.api_base_url + "/",
            headers={
                "X-FinPilot-Source": "web-chat",
                **({"X-Portfolio-Manager": portfolio_manager_id} if portfolio_manager_id else {}),
            },
            timeout=httpx.Timeout(
                connect=settings.connect_timeout,
                read=settings.read_timeout,
                write=settings.write_timeout,
                pool=settings.pool_timeout,
            ),
            transport=transport,
            follow_redirects=False,
        )

    def __enter__(self) -> "FinanceApiClient":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def health(self) -> HealthResponse:
        return self._request("GET", "health", HealthResponse, timeout=self._settings.health_timeout)

    def portfolio_managers(self) -> PortfolioManagerDirectory:
        return self._request("GET", "portfolio/managers", PortfolioManagerDirectory)

    def query(self, request: QueryRequest) -> QueryResponse:
        return self._request("POST", "query", QueryResponse, request=request)

    def resume(self, request: ResumeRequest) -> QueryResponse:
        return self._request("POST", "query/resume", QueryResponse, request=request)

    def _request(
        self,
        method: str,
        path: str,
        model: type[ResponseModel],
        *,
        request: BaseModel | None = None,
        timeout: float | None = None,
    ) -> ResponseModel:
        try:
            if timeout is None:
                response = self._client.request(
                    method, path, json=request.model_dump(exclude_none=True) if request else None
                )
            else:
                response = self._client.request(method, path, timeout=timeout)
        except httpx.TimeoutException:
            raise ApiError(
                "The API request timed out. Completion is unknown; retry only when ready.",
                code="timeout",
            ) from None
        except httpx.RequestError:
            raise BackendUnavailable(
                "Backend unavailable. Check that FastAPI is running.", code="offline"
            ) from None
        if not response.is_success:
            messages = {
                422: "The API rejected the input. Check the question and supplied fields.",
                403: "This query thread is not accessible in the current session.",
                404: "The requested API resource was not found. Check the API configuration.",
                429: "The API is busy. Wait before trying again.",
                502: "The API received an invalid provider response. Try again later.",
                503: "The API is unavailable or not configured. Try again later.",
                504: "The API timed out. Completion is unknown; retry only when ready.",
            }
            raise ApiError(
                messages.get(response.status_code, "The API request failed. Try again later."),
                code="http_error",
                status_code=response.status_code,
            )
        try:
            return model.model_validate(response.json())
        except (ValueError, ValidationError):
            raise ApiError(
                "The API returned an unexpected response. Try again later.", code="invalid_response"
            ) from None


def check_health(settings: WebSettings) -> HealthResponse:
    """Short-lived convenience adapter for the sidebar connection check."""
    with FinanceApiClient(settings) as client:
        return client.health()
