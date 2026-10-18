"""Keep navigation tests independent of automatic market provider requests."""

import pytest

from finassist.web_app.services.models import QueryResponse


@pytest.fixture(autouse=True)
def market_dashboard_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "finassist.web_app.components.showcase.fetch_market_snapshot",
        lambda symbol, interval: QueryResponse(
            intent="market", answer="No provider data in UI smoke tests."
        ),
    )


@pytest.fixture(autouse=True)
def portfolio_dashboard_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "finassist.web_app.components.portfolio.fetch_current_portfolios",
        lambda: QueryResponse(intent="portfolio", answer="No portfolio data in UI smoke tests."),
    )
