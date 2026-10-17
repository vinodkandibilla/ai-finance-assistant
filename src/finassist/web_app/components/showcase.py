"""Market, goals, and news presentations using only session state and HTTP chat."""

import math
import re

import plotly.graph_objects as go
import streamlit as st

from finassist.web_app.components.chat import ChatState
from finassist.web_app.components.layout import card, metric_card
from finassist.web_app.services.api_client import ApiError
from finassist.web_app.services.citations import label_citations
from finassist.web_app.services.market import (
    comparison_series,
    fetch_market_snapshot,
    market_chart_data,
)
from finassist.web_app.services.models import QueryResponse
from finassist.web_app.services.showcase import project_goal, safe_source


def latest_response(intent: str) -> QueryResponse | None:
    state: ChatState | None = st.session_state.get("conversation")
    if state:
        for message in reversed(state.messages):
            if message.response and message.response.intent == intent:
                return message.response
    return None


def request_blocked() -> bool:
    state: ChatState | None = st.session_state.get("conversation")
    return bool(state and (state.thread_id or state.failed_request))


def ask_question(question: str) -> None:
    st.session_state["queued_question"] = question


def market_showcase() -> None:
    st.subheader("Market overview")
    st.caption("U.S. NASDAQ · USD · Provider quotes may be delayed or cached")
    mode = st.radio(
        "Market view",
        ["Individual stock", "Compare stocks"],
        horizontal=True,
        label_visibility="collapsed",
        key="market_view",
    )
    if mode == "Compare stocks":
        market_comparison()
        return
    with st.container(key="market_search_toolbar", border=True):
        stock, frequency = st.columns([4, 2], vertical_alignment="center")
        with stock:
            symbol = st.text_input(
                "Stock symbol",
                value="AAPL",
                max_chars=20,
                placeholder="e.g. AAPL",
                key="market_chart_symbol",
                label_visibility="collapsed",
                help="Enter a NASDAQ symbol and press Enter to search.",
                icon=":material/search:",
                disabled=request_blocked(),
            )
        with frequency:
            interval = st.radio(
                "Interval", ["Daily", "Weekly"], horizontal=True, label_visibility="collapsed"
            )
    symbol = symbol.strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9.:-]{0,19}", symbol):
        st.warning("Enter a valid NASDAQ stock symbol.")
        return
    response = None
    if not request_blocked():
        try:
            with st.spinner(f"Loading {symbol} {interval.lower()} market data…"):
                response = fetch_market_snapshot(symbol, interval)
        except (ApiError, ValueError):
            pass
    window_label = "One month" if interval == "Daily" else "Three months"
    st.subheader(f"{symbol} · {interval} · {window_label}")
    if response is None:
        st.error("Market data is unavailable. Reload the page to retry.")
    else:
        bars, quote = market_chart_data(response, symbol, interval)
        if quote:
            st.caption("Latest available quote · USD")
            with st.container(horizontal=True, key="market_quote_details"):
                for label, field in [
                    ("Latest price", "close"),
                    ("Open", "open"),
                    ("Day high", "high"),
                    ("Day low", "low"),
                    ("Volume", "volume"),
                    ("Previous close", "previous_close"),
                ]:
                    if quote.get(field) is not None:
                        st.metric(label, str(quote[field]))
            observed = quote.get("datetime", quote.get("timestamp", "Not supplied"))
            st.caption(f"Quote observation: {observed}")
        else:
            st.caption("Latest quote unavailable; historical observations are shown when supplied.")
        if bars:
            figure = go.Figure(
                go.Candlestick(
                    x=[bar.datetime for bar in bars],
                    open=[bar.open for bar in bars],
                    high=[bar.high for bar in bars],
                    low=[bar.low for bar in bars],
                    close=[bar.close for bar in bars],
                    name=symbol,
                    increasing_line_color="#0F766E",
                    decreasing_line_color="#B91C1C",
                )
            )
            latest = bars[-1]
            figure.update_layout(
                height=360,
                yaxis_title="USD",
                xaxis_rangeslider_visible=False,
                margin={"l": 10, "r": 10, "t": 65, "b": 15},
                annotations=[
                    dict(
                        text=f"Latest bar · Open {latest.open:,.2f} · High {latest.high:,.2f} · "
                        f"Low {latest.low:,.2f} · Close {latest.close:,.2f}",
                        x=0,
                        y=1.15,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        xanchor="left",
                    )
                ],
            )
            try:
                quote_price = float(quote.get("close", 0))
            except (TypeError, ValueError):
                quote_price = 0
            if math.isfinite(quote_price) and quote_price > 0:
                figure.add_hline(
                    y=quote_price,
                    line_dash="dot",
                    line_color="#153E75",
                    annotation_text=f"Latest quote ${quote_price:,.2f}",
                )
            st.plotly_chart(figure, width="stretch", key="market_ohlc")
            st.caption(
                f"Twelve Data · Last bar: {latest.datetime} · Current bar may be incomplete."
            )
        else:
            st.info("No validated OHLC series returned for this stock and interval.")
        if not bars or response.status == "error":
            st.markdown(label_citations(response.answer), unsafe_allow_html=False)
        st.caption("Provider data, not guaranteed live. Search by symbol; no automatic polling.")
    st.divider()
    st.subheader("Watchlist")
    symbol = st.text_input("Watchlist symbol", max_chars=20, placeholder="AAPL")
    if st.button("Add to watchlist"):
        normalized = symbol.strip().upper()
        watchlist = list(st.session_state.get("market_watchlist", []))
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9.^:-]{0,19}", normalized):
            st.warning("Enter a ticker symbol using letters, digits, dots, colons, or hyphens.")
        elif len(watchlist) >= 20 and normalized not in watchlist:
            st.warning("The session watchlist supports up to 20 symbols.")
        else:
            st.session_state["market_watchlist"] = list(dict.fromkeys([*watchlist, normalized]))
            st.session_state["market_selected_symbol"] = normalized
            if normalized in watchlist:
                st.info(f"{normalized} is already in your watchlist.")
            else:
                st.success(f"Added {normalized} to your watchlist.")
    watchlist = st.session_state.get("market_watchlist", [])
    if watchlist:
        st.caption(f"Saved symbols ({len(watchlist)}): {', '.join(watchlist)}")
        selected = st.selectbox(
            "Saved symbols · This session only", watchlist, key="market_selected_symbol"
        )
        with st.container(horizontal=True, gap="small"):
            if st.button("Ask for latest available quote", disabled=request_blocked()):
                ask_question(
                    f"What is the latest available quote for {selected}? "
                    "Include provider and observation time if available."
                )
            if st.button("Remove selected symbol"):
                st.session_state["market_watchlist"] = [s for s in watchlist if s != selected]
                st.rerun()
    else:
        st.info("Your watchlist is empty. Add a symbol to begin.")


def market_comparison() -> None:
    """Fetch two independent HTTP snapshots; never derive prices from chat prose."""
    with st.container(border=True, key="market_compare_toolbar"):
        first, second, frequency = st.columns([2, 2, 2], vertical_alignment="center")
        with first:
            first_symbol = (
                st.text_input(
                    "First stock",
                    value="AAPL",
                    max_chars=20,
                    key="compare_first",
                    icon=":material/search:",
                    label_visibility="collapsed",
                    help="First NASDAQ symbol · press Enter to search",
                    disabled=request_blocked(),
                )
                .strip()
                .upper()
            )
        with second:
            second_symbol = (
                st.text_input(
                    "Second stock",
                    value="MSFT",
                    max_chars=20,
                    key="compare_second",
                    icon=":material/search:",
                    label_visibility="collapsed",
                    help="Second NASDAQ symbol · press Enter to search",
                    disabled=request_blocked(),
                )
                .strip()
                .upper()
            )
        with frequency:
            interval = st.radio(
                "Comparison interval",
                ["Daily", "Weekly"],
                horizontal=True,
                label_visibility="collapsed",
                key="compare_interval",
            )
    symbols = [first_symbol, second_symbol]
    if any(not re.fullmatch(r"[A-Z][A-Z0-9.:-]{0,19}", symbol) for symbol in symbols):
        st.warning("Enter two valid NASDAQ stock symbols.")
        return
    if first_symbol == second_symbol:
        st.info("Choose two different symbols to compare.")
        return
    if request_blocked():
        st.info("Complete or clear the pending chat request before fetching a comparison.")
        return
    snapshots = []
    with st.spinner("Loading stock comparison…"):
        for symbol in symbols:
            try:
                response = fetch_market_snapshot(symbol, interval)
                bars, _ = market_chart_data(response, symbol, interval)
            except (ApiError, ValueError):
                bars = []
            snapshots.append(bars)
    missing = [symbol for symbol, bars in zip(symbols, snapshots, strict=True) if not bars]
    if missing:
        st.info(
            f"No validated {interval.lower()} history available for {', '.join(missing)}. "
            "Change the symbols or interval to retry."
        )
        return
    dates, first_values, second_values = comparison_series(*snapshots)
    if not dates:
        st.info("At least two shared observation dates are required to compare these stocks.")
        return
    window = "One month" if interval == "Daily" else "Three months"
    st.subheader(f"{first_symbol} vs {second_symbol} · {interval} · {window}")
    st.caption(
        "Actual closing prices in USD. Switch to Start = 100 to compare relative changes. "
        "Only shared observation dates are plotted."
    )
    actual_prices = [
        [prices[day] for day in dates]
        for prices in [{bar.datetime[:10]: bar.close for bar in bars} for bars in snapshots]
    ]
    figure = go.Figure()
    for symbol, prices, indices, color, dash, marker in zip(
        symbols,
        actual_prices,
        [first_values, second_values],
        ["#1D4ED8", "#C2410C"],
        ["solid", "dash"],
        ["circle", "diamond"],
        strict=True,
    ):
        figure.add_trace(
            go.Scatter(
                x=dates,
                y=prices,
                customdata=list(zip(prices, indices, strict=True)),
                name=symbol,
                mode="lines+markers",
                line={"color": color, "width": 2.5, "dash": dash},
                marker={"symbol": marker, "size": 6},
                hovertemplate="%{x}<br>Close: $%{customdata[0]:,.2f}"
                "<br>Index: %{customdata[1]:.2f}<extra>%{fullData.name}</extra>",
            )
        )
    figure.update_layout(
        height=380,
        yaxis_title="Closing price (USD)",
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0,
                "y": 1.28,
                "buttons": [
                    {
                        "label": "Actual price (USD)",
                        "method": "update",
                        "args": [{"y": actual_prices}, {"yaxis.title.text": "Closing price (USD)"}],
                    },
                    {
                        "label": "Start = 100",
                        "method": "update",
                        "args": [
                            {"y": [first_values, second_values]},
                            {"yaxis.title.text": "Closing price index (start = 100)"},
                        ],
                    },
                ],
            }
        ],
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12},
        margin={"l": 10, "r": 10, "t": 90, "b": 15},
    )
    st.plotly_chart(figure, width="stretch", key="market_comparison_chart")
    with st.expander("Closing prices by date · USD"):
        st.dataframe(
            {"Date": dates, first_symbol: actual_prices[0], second_symbol: actual_prices[1]},
            hide_index=True,
            width="stretch",
        )
    with st.container(horizontal=True):
        for symbol, values in zip(symbols, [first_values, second_values], strict=True):
            st.metric(f"{symbol} · price change", f"{values[-1] - 100:+.2f}%")
    st.caption(
        f"Shared observations: {len(dates)} · {dates[0]} to {dates[-1]}. "
        "Provider history, not guaranteed live. Price change excludes dividends; "
        "split/adjustment treatment depends on provider data. No automatic polling."
    )


def goals_showcase() -> None:
    st.subheader("Explore a savings scenario")
    st.info("LOCAL HYPOTHETICAL PROJECTION · Calculator output, not a forecast or backend result.")
    target = st.number_input(
        "Target amount (USD)", min_value=0.0, max_value=1e9, value=10000.0, step=500.0
    )
    savings = st.number_input(
        "Current savings (USD)", min_value=0.0, max_value=1e9, value=1000.0, step=100.0
    )
    contribution = st.number_input(
        "Monthly contribution (USD)", min_value=0.0, max_value=1e7, value=200.0, step=50.0
    )
    months = st.slider("Time horizon (months)", 1, 600, 36)
    rate = st.number_input(
        "Assumed nominal annual return (%)", min_value=0.0, max_value=20.0, value=0.0, step=0.5
    )
    result = project_goal(target, savings, contribution, rate, months)
    with st.container(horizontal=True, gap="small"):
        metric_card(
            "Projected ending balance",
            f"${result.balances[-1]:,.2f}",
            "Hypothetical constant return",
        )
        metric_card(
            "Required monthly saving",
            f"${result.required_monthly:,.2f}",
            "At these assumptions, rounded up to cents",
        )
        metric_card(
            "Projected shortfall",
            f"${result.shortfall:,.2f}",
            "Relative to target; not a guarantee",
        )
    figure = go.Figure()
    figure.add_scatter(x=list(range(months + 1)), y=result.balances, name="Hypothetical balance")
    figure.add_scatter(
        x=list(range(months + 1)), y=result.contributed, name="Savings + contributions"
    )
    figure.add_scatter(x=[0, months], y=[target, target], name="Target", line={"dash": "dash"})
    figure.update_layout(
        xaxis_title="Month",
        yaxis_title="USD",
        height=320,
        legend={"orientation": "h", "y": -0.25},
        margin={"b": 100},
    )
    st.plotly_chart(figure, width="stretch", key="goal_projection")
    st.caption(
        "Assumptions: contributions at month end; nominal annual return divided by 12; "
        "constant monthly compounding; excludes taxes, fees, inflation, and withdrawals. "
        "Returns vary; losses are possible. This scenario does not model downside risk."
    )
    if st.button("Ask AI about this scenario", disabled=request_blocked()):
        ask_question(
            f"Evaluate a savings goal with target ${target}, current savings ${savings}, "
            f"monthly contribution ${contribution}, horizon {months} months "
            f"({months / 12} years), and assumed nominal annual return {rate}%. "
            "Show calculations and assumptions. Do not treat the return as guaranteed."
        )


def news_showcase() -> None:
    st.subheader("News with source context")
    topic = st.text_input(
        "News topic", max_chars=200, placeholder="Financial markets, banking, technology…"
    )
    period = st.selectbox(
        "Requested news window", ["Latest available", "Past 24 hours", "Past week"]
    )
    if st.button("Get news summary", disabled=request_blocked()):
        if topic.strip():
            ask_question(
                f"Summarize financial news about {topic.strip()}. Requested window: {period}. "
                "Include sources and publication dates when available; identify missing dates."
            )
        else:
            st.warning("Enter a topic for the news summary.")
    st.caption(
        "The API returns a summary and citations, not typed article/publication metadata. "
        "Dates and freshness are unverified unless stated in the answer; none are inferred."
    )
    response = latest_response("news")
    if response:
        with card("Latest returned news summary"):
            st.markdown(
                label_citations(response.answer or "No summary text was supplied."),
                unsafe_allow_html=False,
            )
            if response.status:
                st.text(f"Response status: {response.status}")
            st.caption(
                "Publication dates are not structured fields. See attributed dates in the summary."
            )
            for index, citation in enumerate(response.citations or [], 1):
                host = safe_source(citation)
                if host:
                    st.link_button(citation, citation)
                else:
                    st.text(f"Source {index}: {citation}")
            if not response.citations:
                st.info("No sources supplied. The response contains no linked evidence.")
    else:
        st.info("No news summary in this session. Enter a topic above to request one.")
    with st.expander("View a synthetic news example"):
        st.caption("SYNTHETIC EXAMPLE · Not a real article or current event")
        st.write("Fictional Demo Bank announces a financial-literacy workshop.")
        st.text("Source: Fictional Demo Bulletin · Publication date: 2026-10-01 (invented)")
        st.caption("No external link: this example was invented solely to demonstrate the layout.")
