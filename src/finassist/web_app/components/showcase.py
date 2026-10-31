"""Market, goals, and news presentations using only session state and HTTP chat."""

import math
import re

import plotly.graph_objects as go
import streamlit as st

from finassist.web_app.components.chat import ChatState
from finassist.web_app.components.layout import metric_card
from finassist.web_app.services.api_client import ApiError
from finassist.web_app.services.citations import label_citations
from finassist.web_app.services.market import (
    comparison_series,
    fetch_market_quote,
    fetch_market_snapshot,
    market_chart_data,
    resolved_market_symbol,
)
from finassist.web_app.services.models import QueryResponse
from finassist.web_app.services.showcase import project_goal


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
    resolved = st.session_state.pop("market_resolved_snapshot", None)
    if resolved:
        st.session_state["market_chart_symbol"] = resolved[0]
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
        st.divider()
        market_watchlist()
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
                response = (
                    resolved[2]
                    if resolved and resolved[:2] == (symbol, interval)
                    else fetch_market_snapshot(symbol, interval)
                )
        except (ApiError, ValueError):
            pass
    if response is not None:
        canonical = resolved_market_symbol(response)
        if canonical and canonical != symbol:
            st.session_state["market_resolved_snapshot"] = (canonical, interval, response)
            st.rerun()
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
    market_watchlist()


def _remove_watchlist_symbol(symbol: str, name: str) -> None:
    lists = st.session_state["market_watchlists"]
    lists[name] = [item for item in lists[name] if item != symbol]
    if name == st.session_state.get("market_active_watchlist"):
        st.session_state["market_watchlist"] = list(lists[name])


def _create_watchlist() -> None:
    name = st.session_state.get("new_watchlist_name", "").strip()
    lists = st.session_state["market_watchlists"]
    if not name or len(name) > 40:
        st.session_state["watchlist_notice"] = "Enter a watchlist name between 1 and 40 characters."
    elif any(existing.casefold() == name.casefold() for existing in lists):
        st.session_state["watchlist_notice"] = "A watchlist with that name already exists."
    else:
        st.session_state["market_watchlists"] = {name: [], **lists}
        st.session_state["watchlist_card_revision"] = (
            st.session_state.get("watchlist_card_revision", 0) + 1
        )
        st.session_state["market_active_watchlist"] = name
        st.session_state["new_watchlist_name"] = ""
        st.session_state["market_watchlist"] = []
        st.session_state.pop("watchlist_quote_result", None)


@st.fragment
def market_watchlist() -> None:
    """Local list editing reruns independently of provider-backed charts."""
    if not st.session_state.get("market_watchlists"):
        st.session_state["market_watchlists"] = {
            "My watchlist": list(st.session_state.get("market_watchlist", []))
        }
    lists = st.session_state["market_watchlists"]
    active = st.session_state.get("market_active_watchlist")
    if active not in lists:
        active = next(iter(lists))
        st.session_state.pop("watchlist_quote_result", None)
    st.session_state["market_active_watchlist"] = active
    st.session_state["market_watchlist"] = list(lists[active])
    with st.container(key="market_watchlist_panel", gap="small", border=True):
        with st.container(key="watchlist_section_header"):
            heading, create = st.columns([5, 1], vertical_alignment="center")
            with heading:
                st.subheader("Watchlist")
            with create:
                with st.container(
                    horizontal=True, horizontal_alignment="right", key="watchlist_create_control"
                ):
                    with st.popover("", icon=":material/add:", help="Create a watchlist"):
                        st.text_input(
                            "Watchlist name",
                            max_chars=40,
                            key="new_watchlist_name",
                            placeholder="e.g. Tech stocks",
                        )
                        st.button("Create watchlist", on_click=_create_watchlist, type="primary")
                        notice = st.session_state.pop("watchlist_notice", None)
                        if notice:
                            st.warning(notice)
        st.caption("NASDAQ · This session only · Quotes on request")
        revision = st.session_state.get("watchlist_card_revision", 0)
        with st.container(horizontal=True, gap="medium", vertical_alignment="top"):
            for index, (name, symbols) in enumerate(lists.items()):
                with st.container(key=f"watchlist_card_shell_{index}", width=360, gap=None):
                    with st.expander(
                        name,
                        expanded=index < 2,
                        key=f"watchlist_card_{revision}_{index}",
                    ):
                        with st.container(horizontal=True, horizontal_alignment="right"):
                            _watchlist_add_symbol(name, symbols)
                        _watchlist_contents(name, symbols)


def _submit_watchlist_symbol(name: str, key: str) -> None:
    symbols = st.session_state["market_watchlists"][name]
    normalized = st.session_state.get(key, "").strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9.:-]{0,19}", normalized):
        message = "Enter a valid symbol."
    elif normalized in symbols:
        message = f"{normalized} is already in your watchlist."
    elif len(symbols) >= 20:
        message = "Your watchlist supports up to 20 symbols."
    else:
        symbols.append(normalized)
        if name == st.session_state["market_active_watchlist"]:
            st.session_state["market_watchlist"] = list(symbols)
        st.session_state[key] = ""
        st.session_state[f"watchlist_editing_{name}"] = False
        message = f"Added {normalized}."
    st.session_state[f"watchlist_symbol_notice_{name}"] = message


def _show_watchlist_symbol_input(name: str) -> None:
    st.session_state[f"watchlist_editing_{name}"] = True


def _watchlist_add_symbol(name: str, symbols: list[str]) -> None:
    suffix = "" if name == "My watchlist" else f"_{name}"
    key = f"watchlist_symbol_input{suffix}"
    if not st.session_state.get(f"watchlist_editing_{name}", False):
        st.button(
            "",
            icon=":material/add:",
            key=f"watchlist_show_input{suffix}",
            help=f"Add a symbol to {name}",
            type="tertiary",
            on_click=_show_watchlist_symbol_input,
            args=(name,),
        )
        return
    st.text_input(
        "Watchlist symbol",
        max_chars=20,
        placeholder="Symbol",
        key=key,
        label_visibility="collapsed",
        help=f"Add a NASDAQ symbol to {name} · press Enter",
        on_change=_submit_watchlist_symbol,
        args=(name, key),
        width=80,
    )


def _watchlist_contents(name: str, symbols: list[str]) -> None:
    suffix = "" if name == "My watchlist" else f"_{name}"
    notice = st.session_state.pop(f"watchlist_symbol_notice_{name}", None)
    if notice:
        st.caption(notice)
    if not symbols:
        st.info("Your watchlist is empty. Add a symbol to follow it here.")
    requested_symbol = None
    with st.container(horizontal=True, gap="xsmall", key=f"watchlist_tickers{suffix}"):
        for symbol in symbols:
            with st.container(key=f"watchlist_row_{symbol}{suffix}", width="content"):
                with st.container(horizontal=True, gap="xsmall", vertical_alignment="center"):
                    if st.button(
                        symbol,
                        key=f"watchlist_quote_{symbol}{suffix}",
                        type="tertiary",
                        help=f"Retrieve {symbol} quote",
                        disabled=request_blocked(),
                    ):
                        requested_symbol = symbol
                    st.button(
                        "",
                        icon=":material/close:",
                        key=f"watchlist_remove_{symbol}{suffix}",
                        help=f"Remove {symbol} from {name}",
                        type="tertiary",
                        on_click=_remove_watchlist_symbol,
                        args=(symbol, name),
                    )
    if requested_symbol:
        st.session_state.pop("watchlist_quote_result", None)
        try:
            with st.spinner(f"Loading {requested_symbol} quote…"):
                response = fetch_market_quote(requested_symbol)
            st.session_state["watchlist_quote_list"] = name
            st.session_state["watchlist_quote_result"] = (requested_symbol, response)
        except (ApiError, ValueError):
            st.error(f"Could not retrieve {requested_symbol}. Select the symbol to retry.")
    result = st.session_state.get("watchlist_quote_result")
    if result and result[0] in symbols and st.session_state.get("watchlist_quote_list") == name:
        watchlist_quote_card(*result)


def _dismiss_watchlist_quote() -> None:
    st.session_state.pop("watchlist_quote_result", None)
    st.session_state.pop("watchlist_quote_list", None)


def watchlist_quote_card(symbol: str, response: QueryResponse) -> None:
    # The callback clears the result before the watchlist fragment renders again.
    with st.container(border=True, key="watchlist_quote_card", gap="small"):
        heading, close = st.columns([5, 1], vertical_alignment="center")
        with heading:
            st.subheader(f"{symbol} · Latest available quote")
        with close, st.container(horizontal=True, horizontal_alignment="right"):
            st.button(
                "",
                icon=":material/close:",
                key="dismiss_watchlist_quote",
                help="Close quote",
                type="tertiary",
                on_click=_dismiss_watchlist_quote,
            )
        _, quote = market_chart_data(response, symbol, "Daily")
        if quote:
            with st.container(horizontal=True, gap="medium"):
                for label, field in [
                    ("Latest price", "close"),
                    ("Open", "open"),
                    ("Day high", "high"),
                    ("Day low", "low"),
                    ("Previous close", "previous_close"),
                    ("Volume", "volume"),
                ]:
                    try:
                        value = float(quote.get(field, ""))
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(value) or value < 0:
                        continue
                    display = f"{value:,.0f}" if field == "volume" else f"${value:,.2f}"
                    st.metric(label, display)
            observed = quote.get("datetime", quote.get("timestamp", "Not supplied"))
            st.caption(f"NASDAQ · USD · Observation: {observed}")
        else:
            st.markdown(label_citations(response.answer), unsafe_allow_html=False)
            st.caption("No structured quote fields supplied.")
        if response.intent != "market":
            st.caption(f"Returned intent: {response.intent}")
        if response.status in {"error", "insufficient"}:
            st.warning("Quote unavailable. Select the symbol to retry.")
        st.caption("Provider data, not guaranteed live. Select the symbol to refresh.")


def market_comparison() -> None:
    """Fetch both stocks in one request; never derive prices from chat prose."""
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
        try:
            response = fetch_market_snapshot(symbols, interval)
        except (ApiError, ValueError):
            response = None
        for symbol in symbols:
            bars, _ = market_chart_data(response, symbol, interval) if response else ([], {})
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
    from finassist.web_app.components.news import news_browser

    news_browser()
