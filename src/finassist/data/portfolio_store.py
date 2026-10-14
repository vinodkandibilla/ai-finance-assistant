"""Deterministic synthetic portfolios; no live prices or persistent storage.

Each store owns one SQLite :memory: connection. Use it as a context manager or
call close(). Customer risk profiles are pre-established seed data, never inferred.
Names, emails, holdings and costs are fictional. Prices are frozen historical
reference closes from October 9, 2026, embedded for testing; no provider calls.
"""

import sqlite3
from collections.abc import Sequence
from decimal import Decimal
from types import TracebackType
from typing import Any

PRICE_AS_OF_DATE = "2026-10-09"
PRICE_BASIS = "frozen_historical_close_usd"
RISK_PROFILE_LAST_UPDATED = "2025-12-01"
DATA_SOURCE = "synthetic_test_data"

# symbol, asset type, sector, frozen October 9 close in USD.
# Source URLs and observed reference values: data/portfolio/price_snapshot_2026-10-09.json
INSTRUMENTS = (
    ("AAPL", "stock", "Technology", 336.64),
    ("MSFT", "stock", "Technology", 535.07),
    ("NVDA", "stock", "Technology", 229.28),
    ("JPM", "stock", "Financials", 332.99),
    ("JNJ", "stock", "Healthcare", 261.44),
    ("XOM", "stock", "Energy", 168.94),
    ("PG", "stock", "Consumer Staples", 151.23),
    ("VTI", "broad_market_etf", "Broad Market", 381.94),
    ("VXUS", "broad_market_etf", "International Broad Market", 84.82),
    ("XLK", "sector_etf", "Technology", 198.78),
    ("XLF", "sector_etf", "Financials", 54.73),
    ("XLV", "sector_etf", "Healthcare", 170.81),
    ("BND", "bond_etf", "Fixed Income", 70.22),
    ("AGG", "bond_etf", "Fixed Income", 94.59),
    ("SHY", "bond_etf", "Fixed Income", 81.2),
)

# Original design prices define target allocations and cost ratios for each case.
# Seed quantities/costs are rescaled to the reference closes, preserving the deliberately
# constructed scenarios. Fractional shares and costs are rounded to eight decimal places.
DESIGN_PRICES = {
    "AAPL": 200,
    "MSFT": 400,
    "NVDA": 120,
    "JPM": 200,
    "JNJ": 150,
    "XOM": 100,
    "PG": 160,
    "VTI": 250,
    "VXUS": 60,
    "XLK": 200,
    "XLF": 50,
    "XLV": 140,
    "BND": 75,
    "AGG": 100,
    "SHY": 80,
}

# Explicit allocations (symbol, quantity, average cost), not random generation.
# Each manager receives all five contrasting cases at different portfolio sizes.
PORTFOLIO_CASES = (
    (
        "Diversified moderate",
        "moderate",
        10,
        "Long-term wealth",
        "low",
        (
            ("VTI", 80, 200),
            ("VXUS", 200, 70),
            ("BND", 200, 72),
            ("AGG", 100, 105),
            ("JPM", 20, 160),
            ("JNJ", 30, 170),
            ("XOM", 40, 80),
        ),
    ),
    (
        "Single-stock concentration / gains",
        "aggressive",
        15,
        "Capital growth",
        "low",
        (("AAPL", 400, 100), ("VTI", 20, 180), ("VXUS", 30, 45), ("BND", 20, 70), ("JPM", 10, 120)),
    ),
    (
        "Technology concentration / mixed",
        "aggressive",
        12,
        "Capital growth",
        "medium",
        (
            ("AAPL", 100, 150),
            ("MSFT", 100, 450),
            ("NVDA", 200, 60),
            ("XLK", 100, 230),
            ("VTI", 20, 240),
            ("BND", 20, 80),
        ),
    ),
    (
        "Conservative bond-heavy / losses",
        "conservative",
        3,
        "Capital preservation",
        "high",
        (
            ("BND", 500, 85),
            ("AGG", 300, 115),
            ("SHY", 250, 84),
            ("VTI", 20, 280),
            ("PG", 20, 180),
            ("JNJ", 20, 170),
        ),
    ),
    (
        "Aggressive diversified equity / gains",
        "aggressive",
        20,
        "Long-term growth",
        "low",
        (
            ("VTI", 100, 175),
            ("VXUS", 250, 40),
            ("JPM", 50, 130),
            ("JNJ", 50, 110),
            ("XOM", 100, 65),
            ("PG", 50, 120),
            ("XLF", 100, 30),
            ("XLV", 50, 100),
        ),
    ),
)

SCHEMA = """
CREATE TABLE portfolio_managers (
    manager_id TEXT PRIMARY KEY,
    manager_name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL CHECK (status IN ('active', 'inactive'))
);
CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY,
    customer_name TEXT NOT NULL,
    manager_id TEXT NOT NULL REFERENCES portfolio_managers(manager_id),
    risk_profile TEXT NOT NULL CHECK (risk_profile IN ('conservative', 'moderate', 'aggressive')),
    investment_horizon_years INTEGER NOT NULL CHECK (investment_horizon_years > 0),
    investment_goal TEXT NOT NULL,
    liquidity_need TEXT NOT NULL CHECK (liquidity_need IN ('low', 'medium', 'high')),
    risk_profile_last_updated TEXT NOT NULL
);
CREATE TABLE market_prices (
    symbol TEXT PRIMARY KEY,
    current_price REAL NOT NULL CHECK (current_price > 0),
    as_of_date TEXT NOT NULL
);
CREATE TABLE portfolio_holdings (
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    symbol TEXT NOT NULL REFERENCES market_prices(symbol),
    asset_type TEXT NOT NULL CHECK (
        asset_type IN ('stock', 'broad_market_etf', 'sector_etf', 'bond_etf')
    ),
    quantity REAL NOT NULL CHECK (quantity > 0),
    average_cost REAL NOT NULL CHECK (average_cost > 0),
    sector TEXT NOT NULL,
    PRIMARY KEY (customer_id, symbol)
);
CREATE INDEX customers_manager_id ON customers(manager_id);
"""


class PortfolioStore:
    """Small read-only retrieval API over an owned, seeded in-memory database.

    Construct and use the store on the same thread (SQLite's default contract).
    Missing identifiers return None or an empty list; callers handle clarification.
    """

    data_source = DATA_SOURCE

    def __init__(self) -> None:
        self._connection = sqlite3.connect(":memory:")
        self._connection.row_factory = sqlite3.Row
        try:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.executescript(SCHEMA)
            self._seed()
            # Seed is the only write phase; analysis cannot update official profiles.
            self._connection.execute("PRAGMA query_only = ON")
        except Exception:
            self._connection.close()
            raise

    def _seed(self) -> None:
        instruments = {row[0]: row for row in INSTRUMENTS}
        with self._connection:
            self._connection.executemany(
                "INSERT INTO market_prices VALUES (?, ?, ?)",
                [(symbol, price, PRICE_AS_OF_DATE) for symbol, _, _, price in INSTRUMENTS],
            )
            for manager_number in range(1, 6):
                manager_id = f"PM{manager_number:03d}"
                self._connection.execute(
                    "INSERT INTO portfolio_managers VALUES (?, ?, ?, ?)",
                    (
                        manager_id,
                        f"Synthetic Manager {manager_number}",
                        f"manager{manager_number}@example.com",
                        "active",
                    ),
                )
                for case_number, case in enumerate(PORTFOLIO_CASES, start=1):
                    label, risk, horizon, goal, liquidity, holdings = case
                    customer_number = (manager_number - 1) * 5 + case_number
                    customer_id = f"C{customer_number:03d}"
                    self._connection.execute(
                        "INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            customer_id,
                            f"Synthetic Customer {customer_number}: {label}",
                            manager_id,
                            risk,
                            horizon,
                            goal,
                            liquidity,
                            RISK_PROFILE_LAST_UPDATED,
                        ),
                    )
                    self._connection.executemany(
                        "INSERT INTO portfolio_holdings VALUES (?, ?, ?, ?, ?, ?)",
                        [
                            (
                                customer_id,
                                symbol,
                                instruments[symbol][1],
                                float(
                                    round(
                                        Decimal(quantity * manager_number)
                                        * Decimal(DESIGN_PRICES[symbol])
                                        / Decimal(str(instruments[symbol][3])),
                                        8,
                                    )
                                ),
                                float(
                                    round(
                                        Decimal(cost)
                                        * Decimal(str(instruments[symbol][3]))
                                        / Decimal(DESIGN_PRICES[symbol]),
                                        8,
                                    )
                                ),
                                instruments[symbol][2],
                            )
                            for symbol, quantity, cost in holdings
                        ],
                    )

    def _rows(self, sql: str, parameters: Sequence[object] = ()) -> list[dict[str, Any]]:
        return [dict(row) for row in self._connection.execute(sql, parameters).fetchall()]

    def get_managers(self) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM portfolio_managers ORDER BY manager_id")

    def get_customers(self, manager_id: str | None = None) -> list[dict[str, Any]]:
        if manager_id is None:
            return self._rows("SELECT * FROM customers ORDER BY customer_id")
        return self._rows(
            "SELECT * FROM customers WHERE manager_id = ? ORDER BY customer_id", (manager_id,)
        )

    def get_customer(self, customer_id: str) -> dict[str, Any] | None:
        rows = self._rows("SELECT * FROM customers WHERE customer_id = ?", (customer_id,))
        return rows[0] if rows else None

    def get_holdings(self, customer_id: str) -> list[dict[str, Any]]:
        return self._rows(
            "SELECT * FROM portfolio_holdings WHERE customer_id = ? ORDER BY symbol",
            (customer_id,),
        )

    def get_market_prices(self, symbols: Sequence[str] | None = None) -> list[dict[str, Any]]:
        if symbols is None:
            return self._rows("SELECT * FROM market_prices ORDER BY symbol")
        if not symbols:
            return []
        placeholders = ",".join("?" for _ in symbols)
        return self._rows(
            f"SELECT * FROM market_prices WHERE symbol IN ({placeholders}) ORDER BY symbol",
            symbols,
        )

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "PortfolioStore":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
