"""Reusable metrics, holdings, and Plotly charts for validated portfolio snapshots."""

import plotly.graph_objects as go
import streamlit as st

from finassist.web_app.components.chat import ChatState
from finassist.web_app.components.layout import card, metric_card
from finassist.web_app.services.models import QueryResponse
from finassist.web_app.services.portfolio import PortfolioSnapshot, sample_snapshot, snapshots

COLORS = ["#153E75", "#0F766E", "#2563EB", "#7C3AED", "#B45309"]


def latest_portfolio_response() -> QueryResponse | None:
    state: ChatState | None = st.session_state.get("conversation")
    if state:
        for message in reversed(state.messages):
            if message.response and message.response.intent == "portfolio":
                return message.response
    return None


def holdings_table(snapshot: PortfolioSnapshot) -> None:
    st.dataframe([h.model_dump() for h in snapshot.holdings], hide_index=True, width="stretch")


def portfolio_metrics(snapshot: PortfolioSnapshot) -> None:
    with st.container(horizontal=True, gap="small"):
        metric_card(
            "Portfolio valuation (USD)", f"${snapshot.portfolio_value:,.2f}", "Snapshot, not live"
        )
        metric_card(
            "Holdings", str(len(snapshot.holdings)), "Instrument count; no ETF look-through"
        )
        largest = max(snapshot.holdings, key=lambda h: h.market_value)
        metric_card("Largest holding", f"{largest.holding_allocation_pct:.2f}%", largest.symbol)


def allocation_chart(snapshot: PortfolioSnapshot, dimension: str) -> go.Figure:
    grouped: dict[str, float] = {}
    for holding in snapshot.holdings:
        label = getattr(holding, dimension) or "Unspecified"
        grouped[label] = grouped.get(label, 0) + holding.market_value
    figure = go.Figure(
        go.Pie(
            labels=list(grouped),
            values=list(grouped.values()),
            hole=0.6,
            marker={"colors": COLORS},
            textinfo="label+percent",
        )
    )
    figure.update_layout(height=320, margin=dict(l=15, r=15, t=20, b=20), showlegend=False)
    return figure


def render_snapshot(snapshot: PortfolioSnapshot) -> None:
    st.text(f"Customer: {snapshot.customer_id} · Manager: {snapshot.manager_id or 'Not supplied'}")
    st.text(f"Source: {snapshot.data_source} · Price basis: {snapshot.price_basis}")
    dates = sorted({h.as_of_date for h in snapshot.holdings if h.as_of_date})
    st.caption("Price dates: " + (", ".join(dates) if dates else "Not supplied; not live"))
    if snapshot.data_source == "synthetic_test_data":
        st.info("Backend synthetic SQLite dataset · Frozen reference prices; not live.")
    portfolio_metrics(snapshot)
    with st.container(horizontal=True, gap="small"):
        for field, title in [("asset_type", "Asset allocation"), ("sector", "Sector allocation")]:
            with card(title, width=520):
                st.plotly_chart(
                    allocation_chart(snapshot, field),
                    width="stretch",
                    config={"displayModeBar": False},
                    key=f"allocation_{field}",
                )
    with card("Holdings"):
        holdings_table(snapshot)
    with card("Diversification & concentration"):
        weights = [h.market_value / snapshot.portfolio_value for h in snapshot.holdings]
        hhi = sum(w * w for w in weights)
        st.write(f"Holding HHI: **{hhi:.4f}** · Effective holding count: **{1 / hhi:.2f}**")
        st.caption(
            "Calculated from returned holding values. HHI measures concentration, not a risk score."
        )
        figure = go.Figure(
            go.Bar(
                x=[h.symbol for h in snapshot.holdings],
                y=[h.holding_allocation_pct for h in snapshot.holdings],
                marker_color=COLORS[0],
            )
        )
        figure.update_layout(
            yaxis_title="Allocation (%)", height=300, margin=dict(l=15, r=15, t=20, b=60)
        )
        st.plotly_chart(figure, width="stretch", key="holding_weights")
        st.caption("No concentration thresholds or risk recommendations are inferred by this UI.")
    with card("Performance context"):
        if (
            snapshot.performance_complete
            and snapshot.cost_basis is not None
            and snapshot.unrealized_gain_loss is not None
        ):
            st.metric("Unrealized gain / loss (USD)", f"${snapshot.unrealized_gain_loss:,.2f}")
            st.write(f"Cost basis: **${snapshot.cost_basis:,.2f}**")
            if snapshot.unrealized_gain_loss_pct is not None:
                st.write(f"Unrealized gain / loss: **{snapshot.unrealized_gain_loss_pct:.2f}%**")
        else:
            st.info("Complete cost-basis and gain data are unavailable. Performance is not shown.")
        st.caption(
            "Unrealized gains are not historical returns. No performance timeline is available."
        )


def portfolio_dashboard() -> None:
    st.subheader("Portfolio analytics")
    source = st.radio("Dashboard data", ["API response", "Synthetic sample"], horizontal=True)
    response = latest_portfolio_response()
    portfolios = snapshots(response) if source == "API response" else [sample_snapshot()]
    if source == "Synthetic sample":
        st.warning("UI SAMPLE ONLY · Invented values, separate from backend SQLite data; not live.")
    elif response is not None:
        st.caption("Latest portfolio response in this session · Not automatically refreshed.")
    if not portfolios:
        st.info(
            "No adequate structured portfolio snapshot is available. Ask below for a customer "
            "portfolio or manager comparison, or select Synthetic sample to explore the layout."
        )
        with st.expander("Data needed for these charts"):
            st.write(
                "The API has no portfolio GET route. A successful /query calculation needs "
                "customer_id, data_source, price_basis, portfolio_value, and holdings with symbol, "
                "market_value and holding_allocation_pct. Values must reconcile. Manager "
                "comparisons may return these snapshots inside portfolios. Cost basis and "
                "unrealized gains are optional. Historical returns need a dated series, which "
                "is not currently supplied. No charts are inferred from answer text."
            )
        return
    if len(portfolios) > 1:
        st.subheader("Returned customer comparison")
        st.caption(
            "Only adequate returned snapshots are compared; not a complete customer directory."
        )
        figure = go.Figure(
            go.Bar(
                x=[p.customer_id for p in portfolios],
                y=[p.portfolio_value for p in portfolios],
                marker_color=COLORS[0],
            )
        )
        figure.update_layout(
            yaxis_title="Snapshot value (USD)", height=300, margin=dict(l=15, r=15, t=20, b=50)
        )
        st.plotly_chart(figure, width="stretch", key="customer_comparison")
        st.dataframe(
            [
                {
                    "Customer": p.customer_id,
                    "Manager": p.manager_id,
                    "Value (USD)": p.portfolio_value,
                    "Source": p.data_source,
                    "Price basis": p.price_basis,
                }
                for p in portfolios
            ],
            hide_index=True,
        )
    selected = st.selectbox("Customer snapshot", [p.customer_id for p in portfolios])
    render_snapshot(next(p for p in portfolios if p.customer_id == selected))
