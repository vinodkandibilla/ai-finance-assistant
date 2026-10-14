import asyncio
import logging

import httpx
import pytest
from langchain_core.exceptions import OutputParserException
from openai import APIError

from ai_finance_assistant.workflow.classifier import (
    ClassificationOutputError,
    ClassificationProviderError,
    ClassificationTimeoutError,
    LLMIntentClassifier,
)

SECRET = "sk-secret-provider-payload"


class Model:
    def __init__(self, error=None, result=None, slow=False):
        self.error, self.result, self.slow = error, result, slow

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, messages):
        if self.slow:
            await asyncio.sleep(60)
        if self.error:
            raise self.error
        return self.result


@pytest.mark.parametrize(
    "model,category,exception",
    [
        (
            Model(
                error=APIError(
                    SECRET,
                    request=httpx.Request("POST", "https://example.test"),
                    body={"key": SECRET},
                )
            ),
            "provider_unavailable",
            ClassificationProviderError,
        ),
        (Model(result={"intent": SECRET}), "invalid_output", ClassificationOutputError),
        (Model(error=OutputParserException(SECRET)), "invalid_output", ClassificationOutputError),
        (Model(slow=True), "timeout", ClassificationTimeoutError),
    ],
)
async def test_failure_logs_only_safe_category_and_elapsed_time(caplog, model, category, exception):
    classifier = LLMIntentClassifier(model, SECRET, timeout_seconds=0.01)
    with caplog.at_level(logging.WARNING, logger="ai_finance_assistant.workflow.classifier"):
        with pytest.raises(exception) as raised:
            await classifier.classify(SECRET)
    records = [r for r in caplog.records if r.name == "ai_finance_assistant.workflow.classifier"]
    assert len(records) == 1
    record = records[0]
    assert record.failure_category == category
    assert record.elapsed_ms >= 0
    assert record.timeout_ms == 10
    assert f"category={category}" in record.getMessage()
    assert SECRET not in caplog.text
    assert SECRET not in repr(record.__dict__)
    assert record.exc_info is None
    assert raised.value.__suppress_context__
