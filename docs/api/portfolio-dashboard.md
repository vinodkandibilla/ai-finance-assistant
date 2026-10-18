# Portfolio dashboard data contract

The Streamlit dashboard reads only `/query` and `/query/resume` responses. It never
opens the SQLite store or imports portfolio agents/tools. No portfolio GET route is
assumed. Optional `user_role`, `customer_id`, and `manager_id` are request scope,
not authenticated identity. Access decisions stay in the backend.

The dashboard uses the latest portfolio response in the current session. A snapshot
must appear in `calculation`, `calculations`, or a successful calculation's
`portfolios` array. Each snapshot must contain:

- `status: success`, `customer_id`, and positive finite `portfolio_value`.
- `data_source: synthetic_test_data` and `price_basis: frozen_historical_close_usd`.
- Nonempty `holdings`, each with unique `symbol`, nonnegative finite `market_value`,
  and `holding_allocation_pct` between 0 and 100.

Holding values must sum to the portfolio valuation within rounding tolerance, and
weights must match values. Inadequate/error/clarification responses do not create
charts. Unknown provenance is rejected. Unsupported snapshots within comparisons
are omitted; the UI labels comparisons as the adequate returned subset.

Optional holding `sector`, `asset_type`, and `as_of_date` supply chart grouping and
price-date labels. Missing labels appear as Unspecified; dates are not invented.
Asset/sector charts group holding market values without ETF look-through. Holding
HHI, effective holding count, and largest holding derive from those same values,
with explanatory labels; the UI invents no concentration thresholds or risk score.

`performance_complete`, positive `cost_basis`, `unrealized_gain_loss`, and optional
`unrealized_gain_loss_pct` enable unrealized performance metrics. Values must
reconcile to valuation and cost basis. Missing costs produce an unavailable state.
Unrealized gains are not historical/time-weighted returns. A historical performance
chart requires a dated valuation/return series and cash-flow methodology, which the
current contract does not supply.

Synthetic sample mode is explicitly selected and uses fictional UI-only instruments.
It does not copy, represent, or query the backend SQLite customer dataset and is
never presented as live data. Switching to API response mode does not refresh data;
submit a question below for a new result. New Chat clears session snapshots.
