"""OpenAI moderation guard for redacted user input."""

from dataclasses import dataclass

from openai import AsyncOpenAI


@dataclass(frozen=True)
class ModerationResult:
    flagged: bool
    category: str | None = None


async def moderate_text(text: str, *, api_key: str, model: str) -> ModerationResult:
    client = AsyncOpenAI(api_key=api_key)
    response = await client.moderations.create(model=model, input=text)
    result = response.results[0]
    categories = result.categories.model_dump()
    category = next((name for name, value in categories.items() if value), None)
    return ModerationResult(flagged=bool(result.flagged), category=category)
