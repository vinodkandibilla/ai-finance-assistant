from __future__ import annotations

import asyncio
from types import SimpleNamespace

from ai_finance_assistant.core.config import Settings
from ai_finance_assistant.providers import market_data


def test_extract_supported_symbol_maps_company_names_to_tickers() -> None:
    assert market_data.extract_supported_symbol("What is Tesla stock price?") == "TSLA"
    assert market_data.extract_supported_symbol("What is Google stock price?") in {"GOOG", "GOOGL"}


def test_disk_quote_cache_persists_price_and_fetched_at(tmp_path) -> None:
    cache_path = tmp_path / "quotes.json"
    first_cache = market_data.DiskCache(cache_path)
    first_cache.set("AAPL", "213.45", fetched_at=1_000_000)

    second_cache = market_data.DiskCache(cache_path)

    assert second_cache.get("AAPL", now=1_000_001) == "213.45"


def test_disk_quote_cache_expires_after_24_hours(tmp_path) -> None:
    cache = market_data.DiskCache(tmp_path / "quotes.json")
    cache.set("NVDA", "140.25", fetched_at=1_000_000)

    assert (
        cache.get(
            "NVDA",
            now=1_000_000 + market_data.QUOTE_CACHE_TTL_SECONDS - 1,
        )
        == "140.25"
    )
    assert (
        cache.get(
            "NVDA",
            now=1_000_000 + market_data.QUOTE_CACHE_TTL_SECONDS,
        )
        is None
    )


async def test_stock_price_fetches_global_quote_and_reuses_disk_cache(
    monkeypatch,
    tmp_path,
) -> None:
    cache_path = tmp_path / "quotes.json"
    monkeypatch.setattr(market_data, "QUOTE_CACHE_PATH", cache_path)
    monkeypatch.setattr(
        market_data,
        "get_settings",
        lambda: Settings(_env_file=None, alpha_vantage_api_key="test-key"),
    )
    request_params: list[dict[str, str]] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, dict[str, str]]:
            return {"Global Quote": {"05. price": "213.45"}}

    class FakeAsyncClient:
        def __init__(self, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def get(self, url: str, *, params: dict[str, str]) -> FakeResponse:
            request_params.append(params)
            return FakeResponse()

    monkeypatch.setattr(market_data.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(market_data, "_symbol_locks", {})

    first_price = await market_data.get_stock_price.ainvoke({"symbol": "AAPL"})
    second_price = await market_data.get_stock_price.ainvoke({"symbol": "AAPL"})

    assert first_price == "213.45"
    assert second_price == "213.45"
    assert request_params == [
        {
            "function": "GLOBAL_QUOTE",
            "symbol": "AAPL",
            "apikey": "test-key",
        }
    ]
    assert cache_path.exists()


async def test_concurrent_cache_misses_make_one_alpha_vantage_request(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(market_data, "QUOTE_CACHE_PATH", tmp_path / "quotes.json")
    monkeypatch.setattr(
        market_data,
        "get_settings",
        lambda: Settings(_env_file=None, alpha_vantage_api_key="test-key"),
    )
    monkeypatch.setattr(market_data, "_symbol_locks", {})
    request_count = 0

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, dict[str, str]]:
            return {"Global Quote": {"05. price": "213.45"}}

    class FakeAsyncClient:
        def __init__(self, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def get(self, url: str, *, params: dict[str, str]) -> FakeResponse:
            nonlocal request_count
            request_count += 1
            await asyncio.sleep(0)
            return FakeResponse()

    monkeypatch.setattr(market_data.httpx, "AsyncClient", FakeAsyncClient)

    prices = await asyncio.gather(
        market_data.get_stock_price.ainvoke({"symbol": "NVDA"}),
        market_data.get_stock_price.ainvoke({"symbol": "NVDA"}),
    )

    assert prices == ["213.45", "213.45"]
    assert request_count == 1


async def test_stock_price_rejects_unsupported_symbol_without_request(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(market_data, "QUOTE_CACHE_PATH", tmp_path / "quotes.json")
    monkeypatch.setattr(market_data, "_symbol_locks", {})

    async def unexpected_request(*args: object, **kwargs: object) -> None:
        raise AssertionError("Unsupported symbols must not trigger a request")

    monkeypatch.setattr(
        market_data.httpx,
        "AsyncClient",
        lambda **kwargs: SimpleNamespace(get=unexpected_request),
    )

    result = await market_data.get_stock_price.ainvoke({"symbol": "IBM"})

    assert "not a supported MAG7 ticker" in result


async def test_stock_price_reports_missing_api_key_without_request(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(market_data, "QUOTE_CACHE_PATH", tmp_path / "quotes.json")
    monkeypatch.setattr(
        market_data,
        "get_settings",
        lambda: Settings(_env_file=None, alpha_vantage_api_key=None),
    )
    monkeypatch.setattr(market_data, "_symbol_locks", {})

    class UnexpectedAsyncClient:
        def __init__(self, **kwargs: object) -> None:
            raise AssertionError("Missing credentials must not trigger a request")

    monkeypatch.setattr(market_data.httpx, "AsyncClient", UnexpectedAsyncClient)

    result = await market_data.get_stock_price.ainvoke({"symbol": "AAPL"})

    assert "API key is not configured" in result


def test_settings_load_alpha_vantage_key_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test-key")

    settings = Settings(_env_file=None)

    assert settings.alpha_vantage_api_key is not None
    assert settings.alpha_vantage_api_key.get_secret_value() == "test-key"
