import asyncio
import logging
import time
from typing import Protocol

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from openai import APIError, APITimeoutError
from pydantic import BaseModel, ValidationError

from finassist.core.config import Settings, get_settings
from finassist.workflow.state import FinanceIntent

logger = logging.getLogger(__name__)


class IntentClassifier(Protocol):
    async def classify(self, query: str) -> FinanceIntent: ...


class IntentClassification(BaseModel):
    intent: FinanceIntent


class ClassificationTimeoutError(Exception):
    """The router could not complete within its configured deadline."""


class ClassificationProviderError(Exception):
    """The routing provider rejected or could not serve the request."""


class ClassificationOutputError(Exception):
    """The routing response did not satisfy the supported intent contract."""


class LLMIntentClassifier:
    def __init__(
        self, model: BaseChatModel, system_prompt: str, *, timeout_seconds: float = 10.0
    ) -> None:
        self._classifier = model.with_structured_output(IntentClassification)
        if timeout_seconds <= 0:
            raise ValueError("Classification timeout must be positive")
        self._system_prompt = system_prompt
        self._timeout_seconds = timeout_seconds

    async def classify(self, query: str) -> FinanceIntent:
        started = time.monotonic()
        try:
            async with asyncio.timeout(self._timeout_seconds):
                result = await self._classifier.ainvoke(
                    [SystemMessage(content=self._system_prompt), HumanMessage(content=query)]
                )
                if isinstance(result, IntentClassification):
                    intent = result.intent
                else:
                    intent = IntentClassification.model_validate(result).intent
        except (TimeoutError, APITimeoutError):
            self._log_failure("timeout", started)
            raise ClassificationTimeoutError("Intent classification timed out") from None
        except APIError:
            self._log_failure("provider_unavailable", started)
            raise ClassificationProviderError(
                "Intent classification provider unavailable"
            ) from None
        except (ValidationError, OutputParserException):
            self._log_failure("invalid_output", started)
            raise ClassificationOutputError("Invalid intent classification response") from None
        logger.info("LLM determined intent: %s", intent)
        return intent

    def _log_failure(self, category: str, started: float) -> None:
        elapsed_ms = (time.monotonic() - started) * 1000
        timeout_ms = self._timeout_seconds * 1000
        logger.warning(
            "router_classification_failed category=%s elapsed_ms=%.3f timeout_ms=%.3f",
            category,
            elapsed_ms,
            timeout_ms,
            extra={
                "failure_category": category,
                "elapsed_ms": elapsed_ms,
                "timeout_ms": timeout_ms,
            },
        )


def create_intent_classifier(
    settings: Settings | None = None,
    *,
    system_prompt: str | None = None,
) -> IntentClassifier:
    from langchain_openai import ChatOpenAI

    resolved_settings = settings or get_settings()
    api_key = (
        resolved_settings.openai_api_key.get_secret_value()
        if resolved_settings.openai_api_key
        else None
    )
    model = ChatOpenAI(
        model=resolved_settings.openai_model,
        api_key=api_key,
        timeout=resolved_settings.router_timeout_seconds,
        max_retries=1,
    )
    prompt = system_prompt or resolved_settings.intent_router_system_prompt
    return LLMIntentClassifier(
        model, prompt, timeout_seconds=resolved_settings.router_timeout_seconds
    )
