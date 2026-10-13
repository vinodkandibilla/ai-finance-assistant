import pytest


@pytest.fixture(autouse=True)
def clear_market_cache():
    from ai_finance_assistant.agents.market_analysis.cache import market_cache

    market_cache.entries.clear()
    yield
    market_cache.entries.clear()
