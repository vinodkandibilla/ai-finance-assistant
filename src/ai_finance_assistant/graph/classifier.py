from typing import Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from ai_finance_assistant.core.config import Settings, get_settings
from ai_finance_assistant.graph.state import FinanceIntent


class IntentClassifier(Protocol):
    async def classify(self, query: str) -> FinanceIntent: ...


class IntentClassification(BaseModel):
    intent: FinanceIntent


class LLMIntentClassifier:
    def __init__(self, model: BaseChatModel, system_prompt: str) -> None:
        self._classifier = model.with_structured_output(IntentClassification)
        self._system_prompt = system_prompt

    async def classify(self, query: str) -> FinanceIntent:
        result = await self._classifier.ainvoke(
            [SystemMessage(content=self._system_prompt), HumanMessage(content=query)]
        )
        if isinstance(result, IntentClassification):
            return result.intent
        return IntentClassification.model_validate(result).intent


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
    model = ChatOpenAI(model=resolved_settings.openai_model, api_key=api_key)
    prompt = system_prompt or resolved_settings.intent_router_system_prompt
    return LLMIntentClassifier(model, prompt)
