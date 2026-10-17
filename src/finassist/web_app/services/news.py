"""HTTP-only news discovery and validated article metadata."""

from urllib.parse import urlsplit

from pydantic import BaseModel, ValidationError, field_validator

from finassist.web_app.config import WebSettings
from finassist.web_app.services.api_client import FinanceApiClient
from finassist.web_app.services.models import QueryRequest, QueryResponse


class NewsArticle(BaseModel):
    title: str
    url: str
    publisher: str | None = None
    published_at: str | None = None
    summary: str

    @field_validator("url")
    @classmethod
    def safe_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username:
            raise ValueError("Invalid article URL")
        return value


def articles_from_response(response: QueryResponse) -> list[NewsArticle]:
    if response.intent != "news" or response.status in {"error", "clarification", "out_of_scope"}:
        return []
    articles: dict[str, NewsArticle] = {}
    for calculation in response.calculations or []:
        if calculation.get("type") != "news_articles":
            continue
        rows = calculation.get("articles")
        if not isinstance(rows, list):
            continue
        for row in rows:
            try:
                article = NewsArticle.model_validate(row)
            except ValidationError:
                continue
            articles.setdefault(article.url, article)
    return list(articles.values())


def fetch_news(topic: str, period: str) -> QueryResponse:
    with FinanceApiClient(WebSettings.from_environment()) as client:
        return client.query(
            QueryRequest(
                query=(
                    f"Find financial news about {topic}. Requested period: {period}. "
                    "Return concise supported reported facts with original article sources and "
                    "publication dates when supplied. Do not invent metadata or imply "
                    "missing dates "
                    "are verified. Focus on reporting, not investment recommendations."
                )
            )
        )
