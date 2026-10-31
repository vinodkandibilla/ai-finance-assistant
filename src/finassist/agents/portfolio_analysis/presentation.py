"""Consistent routine portfolio snapshots from validated tool results."""

import re


def is_snapshot_request(query: str) -> bool:
    if re.search(
        r"\b(?:analy[sz]e|compare|risk|alignment|recommend|explain|brief|summary)\b",
        query, re.I,
    ):
        return False
    return bool(re.fullmatch(
        r"\s*(?:show|display)\s+(?:me\s+)?(?:the\s+)?portfolio\s+"
        r"(?:of|for)\s+[\w'-]+(?:\s+[\w'-]+){0,2}[?.!]?\s*",
        query, re.I,
    ))


def format_snapshot(portfolio: dict) -> str:
    def money(value):
        return f"${value:,.2f}" if isinstance(value, (int, float)) else "Unavailable"

    def pct(value):
        return f"{value:.2f}%" if isinstance(value, (int, float)) else "Unavailable"

    lines = [
        f"### {portfolio.get('customer_name') or portfolio['customer_id']}",
        f"Customer: {portfolio['customer_id']}",
        "### Portfolio overview",
        f"- Total value: {money(portfolio['portfolio_value'])}",
        f"- Cost basis: {money(portfolio.get('cost_basis'))}",
        f"- Unrealized gain/loss: {money(portfolio.get('unrealized_gain_loss'))} "
        f"({pct(portfolio.get('unrealized_gain_loss_pct'))})",
        "### Holdings",
        "| Symbol | Quantity | Reference price | Value | Weight | Unrealized gain/loss |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in portfolio['holdings']:
        lines.append(
            f"| {row['symbol']} | {row['quantity']:.2f} | {money(row.get('current_price'))} "
            f"| {money(row['market_value'])} | {pct(row['holding_allocation_pct'])} "
            f"| {money(row.get('unrealized_gain_loss'))} |"
        )
    for heading, key in [('Asset allocation', 'asset_class_allocation'),
                         ('Sector allocation', 'sector_allocation')]:
        lines.append(f"\n### {heading}")
        lines.extend(f"- {r['name']}: {pct(r['allocation_pct'])}" for r in portfolio.get(key, []))
    lines.append("\n### Concentration")
    largest = portfolio.get('largest_holding') or {}
    lines.append(f"- Largest holding: {largest.get('symbol', 'Unavailable')} "
                 f"({pct(largest.get('holding_allocation_pct'))})")
    hhi = portfolio.get('diversification', {}).get('holding_hhi')
    if isinstance(hhi, (int, float)):
        lines.append(f"- Holding HHI: {hhi:.4f} (holding-weight concentration, not a risk score)")
    for heading, key in [('Biggest gainers', 'biggest_gainers'),
                         ('Biggest losers', 'biggest_losers')]:
        lines.append(f"\n### {heading}")
        rows = portfolio.get(key, [])
        lines.extend(f"- {r['symbol']}: {money(r['unrealized_gain_loss'])}" for r in rows)
        if not rows:
            lines.append("None reported." if portfolio.get('performance_complete')
                         else "Unavailable: incomplete cost basis.")
    dates = sorted({str(r['as_of_date']) for r in portfolio['holdings'] if r.get('as_of_date')})
    lines.append("\nSynthetic demonstration data; frozen reference prices as of "
                 + (', '.join(dates) or 'an unavailable date')
                 + ". Gains/losses are unrealized, not historical returns.")
    return '\n\n'.join(lines[:8]) + '\n' + '\n'.join(lines[8:])
