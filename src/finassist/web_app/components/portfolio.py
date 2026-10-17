"""Reusable metrics, holdings, and Plotly charts for validated portfolio snapshots."""

import plotly.graph_objects as go
import streamlit as st

from finassist.web_app.components.layout import card
from finassist.web_app.services.api_client import ApiError
from finassist.web_app.services.portfolio import (
    PortfolioSnapshot,
    fetch_current_portfolios,
    performance_label,
    snapshots,
)

COLORS = ["#153E75", "#0F766E", "#2563EB", "#7C3AED", "#B45309"]


def holdings_table(snapshot: PortfolioSnapshot) -> None:
    st.dataframe(
        [h.model_dump() for h in snapshot.holdings],
        hide_index=True,
        width="stretch",
        column_order=[
            "symbol",
            "asset_type",
            "sector",
            "market_value",
            "holding_allocation_pct",
            "unrealized_gain_loss",
            "as_of_date",
        ],
        column_config={
            "symbol": st.column_config.TextColumn("Symbol"),
            "asset_type": st.column_config.TextColumn("Asset Type"),
            "sector": st.column_config.TextColumn("Sector"),
            "market_value": st.column_config.NumberColumn("Market Value (USD)", format="$%.2f"),
            "holding_allocation_pct": st.column_config.NumberColumn(
                "Portfolio Weight (%)",
                format="%.2f%%",
            ),
            "unrealized_gain_loss": st.column_config.NumberColumn(
                "Unrealized Gain/Loss (USD)",
                format="$%.2f",
            ),
            "as_of_date": st.column_config.TextColumn("Price As Of"),
        },
    )


def portfolio_metrics(snapshot: PortfolioSnapshot) -> None:
    with st.container(horizontal=True, gap="small", key="portfolio_summary_metrics"):
        largest = max(snapshot.holdings, key=lambda h: h.market_value)
        for label, value in [
            ("Portfolio value (USD)", f"${snapshot.portfolio_value:,.2f}"),
            ("Holdings", str(len(snapshot.holdings))),
            (f"Largest holding · {largest.symbol}", f"{largest.holding_allocation_pct:.2f}%"),
        ]:
            with st.container(border=True, width=240):
                st.metric(label, value)


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
    dates = sorted({h.as_of_date for h in snapshot.holdings if h.as_of_date})
    st.caption("Price dates: " + (", ".join(dates) if dates else "Not supplied; not live"))
    summary, positions, allocation = st.tabs(["Summary", "Positions", "Allocation"])
    with summary:
        portfolio_metrics(snapshot)
        st.subheader("Performance context")
        if (
            snapshot.performance_complete
            and snapshot.cost_basis is not None
            and snapshot.unrealized_gain_loss is not None
        ):
            outcome = (
                "loss"
                if snapshot.unrealized_gain_loss < 0
                else ("gain" if snapshot.unrealized_gain_loss > 0 else "neutral")
            )
            with st.container(horizontal=True, gap="small", key="portfolio_performance_metrics"):
                with st.container(border=True, width=240):
                    st.metric("Cost basis (USD)", f"${snapshot.cost_basis:,.2f}")
                with st.container(border=True, width=240, key=f"portfolio_performance_{outcome}"):
                    st.metric(
                        "Unrealized gain / loss (USD)", f"${snapshot.unrealized_gain_loss:,.2f}"
                    )
                if snapshot.unrealized_gain_loss_pct is not None:
                    with st.container(
                        border=True, width=240, key=f"portfolio_performance_pct_{outcome}"
                    ):
                        st.metric(
                            "Unrealized gain / loss (%)",
                            f"{snapshot.unrealized_gain_loss_pct:+.2f}%",
                        )
                if snapshot.concentration.get("is_concentrated") is True:
                    with st.container(border=True, width=240, key="portfolio_concentration_flag"):
                        st.metric("Concentration", "Flagged")
        else:
            st.info("Complete cost-basis and gain data are unavailable. Performance is not shown.")
    with positions:
        with card("Holdings"):
            holdings_table(snapshot)
    with allocation:
        with st.container(horizontal=True, gap="small"):
            for field, title in [
                ("asset_type", "Asset allocation"),
                ("sector", "Sector allocation"),
            ]:
                with card(title, width=520):
                    st.plotly_chart(
                        allocation_chart(snapshot, field),
                        width="stretch",
                        config={"displayModeBar": False},
                        key=f"allocation_{field}",
                    )
        with card("Diversification & concentration"):
            weights = [h.market_value / snapshot.portfolio_value for h in snapshot.holdings]
            hhi = sum(w * w for w in weights)
            st.write(f"Holding HHI: **{hhi:.4f}** · Effective holding count: **{1 / hhi:.2f}**")
            st.caption(
                "Calculated from returned holding values. "
                "HHI measures concentration, not a risk score."
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
            st.caption(
                "No concentration thresholds or risk recommendations are inferred by this UI."
            )


def _select_customer(customer_id: str) -> None:
    st.session_state["portfolio_selected_customer"] = customer_id


def portfolio_dashboard() -> None:
    st.subheader("Portfolio analytics")
    with st.expander("Portfolio Data", expanded=True, key="portfolio_data_details"):
        st.write(
            "Portfolios use illustrative holdings and frozen reference prices. "
            "Application identity is simulated; the backend determines available customers."
        )
    if "portfolio_landing_response" not in st.session_state:
        st.session_state.pop("portfolio_landing_response", None)
        try:
            with st.spinner("Loading your portfolio workspace…"):
                st.session_state["portfolio_landing_response"] = fetch_current_portfolios()
        except (ApiError, ValueError):
            st.error(
                "Portfolio data is unavailable. "
                "Check the API connection and reopen the page to retry."
            )
    response = st.session_state.get("portfolio_landing_response")
    portfolios = snapshots(response)
    if not portfolios:
        st.info(
            "No adequate structured portfolio snapshot is available for the current application "
            "identity."
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
        if response is not None:
            st.write(response.answer)
            if response.clarification_question:
                st.info(response.clarification_question)
        return
    items = ([response.calculation] if response and response.calculation else []) + (
        response.calculations or [] if response else []
    )
    manager_view = any("portfolios" in item for item in items)
    if manager_view:
        st.subheader("Your customers")
        for item in items:
            if item.get("excluded_portfolios"):
                st.warning("Some assigned portfolios could not be displayed.")
                with st.expander("Unavailable portfolio details"):
                    st.json(item["excluded_portfolios"])
    else:
        st.subheader("Your portfolio")
    if manager_view:
        with st.container(horizontal=True, gap="medium", key="portfolio_customer_totals"):
            st.metric("Customers shown", str(len(portfolios)))
            st.metric(
                "Combined portfolio value (USD)",
                f"${sum(p.portfolio_value for p in portfolios):,.2f}",
            )
    search = (
        st.text_input(
            "Find a customer",
            placeholder="Search by name or customer ID",
            key="portfolio_customer_search",
        )
        if manager_view
        else ""
    )
    visible = [
        p
        for p in portfolios
        if search.strip().casefold() in f"{p.customer_name or ''} {p.customer_id}".casefold()
    ]
    if not visible:
        st.info("No customers match your search.")
        return
    identifiers = [p.customer_id for p in visible]
    selected = st.session_state.get("portfolio_selected_customer")
    if selected not in identifiers:
        selected = identifiers[0]
        st.session_state["portfolio_selected_customer"] = selected
    if manager_view:
        with st.container(key="portfolio_customer_directory", gap="small"):
            widths = [0.8, 1.4, 2.0, 1.3, 1.1, 1.2]
            with st.container(key="portfolio_directory_header"):
                for column, label in zip(
                    st.columns(widths),
                    ["ID", "Name", "Scenario", "Value (USD)", "Positions", "Performance"],
                    strict=True,
                ):
                    column.markdown(f"**{label}**")
            for row_index, customer in enumerate(visible):
                with st.container(
                    key=f"portfolio_directory_row_{row_index % 2}_{customer.customer_id}"
                ):
                    identifier, name, scenario, value, positions, performance = st.columns(
                        widths, vertical_alignment="center"
                    )
                    with identifier:
                        if customer.customer_id == selected:
                            st.text(customer.customer_id)
                        else:
                            st.button(
                                customer.customer_id,
                                key=f"portfolio_customer_{customer.customer_id}",
                                type="tertiary",
                                help=f"View {customer.customer_id} portfolio",
                                on_click=_select_customer,
                                args=(customer.customer_id,),
                            )
                    name.write(
                        (customer.customer_name or customer.customer_id).partition(":")[0].strip()
                    )
                    scenario.write(
                        (customer.customer_name or "").partition(":")[2].split("/")[0].strip()
                        or "—"
                    )
                    value.write(f"${customer.portfolio_value:,.2f}")
                    positions.write(str(len(customer.holdings)))
                    performance.write(performance_label(customer))
    snapshot = next(p for p in visible if p.customer_id == selected)
    with st.container(key="portfolio_customer_heading"):
        title = (
            f"{snapshot.customer_id} - {snapshot.customer_name}"
            if snapshot.customer_name
            else snapshot.customer_id
        )
        st.subheader(title)
    render_snapshot(snapshot)
