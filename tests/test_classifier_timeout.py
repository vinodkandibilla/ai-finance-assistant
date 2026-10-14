import asyncio

import pytest
from fastapi.testclient import TestClient

from finassist.api.query import get_query_graph
from finassist.main import create_app
from finassist.workflow.builder import build_graph
from finassist.workflow.classifier import ClassificationTimeoutError, LLMIntentClassifier


class SlowModel:
    cancelled = False

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, messages):
        try:
            await asyncio.sleep(60)
        finally:
            self.cancelled = True


async def test_router_deadline_cancels_pending_provider_call():
    model = SlowModel()
    classifier = LLMIntentClassifier(model, "Classify", timeout_seconds=0.01)
    with pytest.raises(ClassificationTimeoutError):
        await classifier.classify("What is compound interest?")
    assert model.cancelled


def test_api_returns_504_without_dispatching_on_classifier_timeout():
    app = create_app()
    classifier = LLMIntentClassifier(SlowModel(), "Classify", timeout_seconds=0.01)
    app.dependency_overrides[get_query_graph] = lambda: build_graph(classifier)
    response = TestClient(app).post("/query", json={"query": "Explain ETFs"})
    assert response.status_code == 504
    assert response.json() == {"detail": "Intent classification timed out. Please try again."}


async def test_client_cancellation_is_not_misreported_as_router_timeout(caplog):
    started = asyncio.Event()

    class Model(SlowModel):
        async def ainvoke(self, messages):
            started.set()
            return await super().ainvoke(messages)

    classifier = LLMIntentClassifier(Model(), "Classify", timeout_seconds=10)
    task = asyncio.create_task(classifier.classify("Explain ETFs"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert "router_classification_failed" not in caplog.text


def test_factory_passes_configured_deadline_to_provider_and_classifier(monkeypatch):
    from finassist.core.config import Settings
    from finassist.workflow.classifier import create_intent_classifier

    options = {}

    def model(**kwargs):
        options.update(kwargs)
        return SlowModel()

    monkeypatch.setattr("langchain_openai.ChatOpenAI", model)
    classifier = create_intent_classifier(
        Settings(
            _env_file=None,
            openai_api_key="test-key",
            router_timeout_seconds=1,
        )
    )
    assert options["timeout"] == 1
    assert options["max_retries"] == 1
    assert classifier._timeout_seconds == 1


@pytest.mark.parametrize("timeout", [0, -1, 121])
def test_invalid_configured_deadline_is_rejected(timeout):
    from pydantic import ValidationError

    from finassist.core.config import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None, router_timeout_seconds=timeout)
