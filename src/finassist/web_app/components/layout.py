"""Native, responsive presentation primitives shared by every FinPilot page."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

import streamlit as st

DISCLAIMER = (
    "For educational use only. FinPilot does not provide personalized investment, "
    "financial, or tax advice. Verify information before making financial decisions."
)


def page_header(title: str, description: str) -> None:
    with st.container(key="finpilot_header", gap="small"):
        st.title(title)
        with st.container(key="header_description"):
            st.caption(description)


@contextmanager
def card(title: str, *, width: int | Literal["stretch"] = "stretch") -> Iterator[None]:
    """Content-sized card; fixed preferred widths shrink to fit small screens."""
    with st.container(border=True, width=width):
        st.subheader(title)
        yield


def metric_card(label: str, value: str, context: str) -> None:
    with st.container(border=True, width=240):
        st.metric(label, value)
        st.caption(context)


def status_indicator(label: str, state: Literal["info", "success", "warning", "error"]) -> None:
    """Always include text so status never depends on color alone."""
    icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}
    getattr(st, state)(label, icon=icons[state])


def empty_state(title: str, description: str) -> None:
    with card(title):
        st.write(description)
        st.caption("Choose another workspace from the navigation to keep exploring.")


def foundation_notice() -> None:
    empty_state(
        "Your workspace is ready",
        "Interactive questions and results will arrive in a later update. "
        "There are no results to display yet.",
    )


def error_state(message: str) -> None:
    status_indicator(message, "error")
    st.caption("Check that FastAPI is running, then select Check API connection to retry.")


@contextmanager
def loading_state(message: str) -> Iterator[None]:
    with st.spinner(message):
        yield


def educational_disclaimer() -> None:
    st.divider()
    st.markdown(f"**Educational use** · {DISCLAIMER}")


def compact_spacing() -> None:
    """Consistent page hierarchy, with denser spacing limited to chat content."""
    st.html("""
    <style>
    header.stAppHeader,
    header[data-testid="stHeader"] { display: none; }
    .stMainBlockContainer { padding: 0 1rem 1rem; }
    /* Empty style/script blocks must not create a gap above the app header. */
    .stMainBlockContainer > [data-testid="stVerticalBlock"] { gap: 0; }
    :is(.st-key-market_search_toolbar, .st-key-market_compare_toolbar) {
        background: #F8FAFC;
        border-color: #DCE5EF;
        padding: 0.875rem 1rem;
        border-radius: 0.875rem;
    }
    :is(.st-key-market_search_toolbar, .st-key-market_compare_toolbar)
    [data-testid="stWidgetLabel"] p {
        color: #475569;
        font-size: 0.8125rem;
        font-weight: 600;
    }
    :is(.st-key-market_search_toolbar, .st-key-market_compare_toolbar) input {
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    :is(.st-key-market_search_toolbar, .st-key-market_compare_toolbar) [role="radiogroup"] {
        min-height: 2.5rem;
        align-items: center;
        justify-content: flex-end;
        gap: 0.375rem;
    }
    :is(.st-key-market_search_toolbar, .st-key-market_compare_toolbar) [role="radiogroup"] label {
        margin: 0;
        padding: 0.375rem 0.625rem;
        border-radius: 0.625rem;
    }
    :is(.st-key-market_search_toolbar, .st-key-market_compare_toolbar)
    [role="radiogroup"] label:has(input:checked) {
        background: #DDF1ED;
        color: #115E59;
    }
    .st-key-market_quote_details [data-testid="stMetricValue"] {
        font-size: 1.125rem;
        line-height: 1.4;
    }
    .st-key-watchlist_quote_card [data-testid="stMetricValue"] {
        font-size: 1.25rem;
    }
    .st-key-market_quote_details [data-testid="stMetricLabel"] p {
        font-size: 0.8125rem;
    }
    .st-key-market_watchlist_panel [class*="st-key-watchlist_row_"] {
        padding-block: 0;
    }
    .st-key-watchlist_section_header {
        background: #F1F5F9;
        border-radius: 0.5rem;
        padding: 0.25rem 0.75rem;
    }
    .st-key-portfolio_customer_heading {
        background: #F1F5F9;
        padding: 0.5rem 0.875rem;
        border-left: 3px solid #0F766E;
        border-radius: 0.375rem;
    }
    .st-key-portfolio_customer_totals [data-testid="stMetricValue"] {
        font-size: 1.25rem;
        line-height: 1.4;
    }
    .st-key-portfolio_summary_metrics [data-testid="stMetricValue"] {
        font-size: 1.25rem;
        line-height: 1.4;
    }
    .st-key-portfolio_performance_metrics [data-testid="stMetricValue"] {
        font-size: 1.25rem;
        line-height: 1.4;
    }
    .st-key-portfolio_performance_metrics > [data-testid="stLayoutWrapper"] >
    [data-testid="stVerticalBlock"] { padding: 0.5rem 0.75rem; }
    :is(.st-key-portfolio_performance_loss, .st-key-portfolio_performance_pct_loss)
    [data-testid="stMetricValue"] { color: #B91C1C; }
    :is(.st-key-portfolio_performance_gain, .st-key-portfolio_performance_pct_gain)
    [data-testid="stMetricValue"] { color: #047857; }
    .st-key-portfolio_concentration_flag [data-testid="stMetricValue"] { color: #92400E; }
    .st-key-portfolio_summary_metrics > [data-testid="stLayoutWrapper"] >
    [data-testid="stVerticalBlock"] {
        padding: 0.5rem 0.75rem;
    }
    .st-key-portfolio_customer_totals [data-testid="stMetricLabel"] p {
        font-size: 0.8125rem;
    }
    .st-key-portfolio_customer_directory [class*="st-key-portfolio_customer_"] button {
        color: #1D4ED8;
        text-decoration: underline;
        justify-content: flex-start;
        padding: 0;
        min-height: 0;
        line-height: 1.5;
    }
    .st-key-portfolio_customer_directory [class*="st-key-portfolio_directory_row_"] {
        border-bottom: 1px solid #E2E8F0;
        padding: 0.375rem 0.5rem;
    }
    .st-key-portfolio_directory_header {
        background: #153E75;
        color: #FFFFFF;
        padding: 0.5rem;
        border-radius: 0.375rem 0.375rem 0 0;
    }
    .st-key-portfolio_directory_header p { color: #FFFFFF; }
    .st-key-portfolio_customer_directory [class*="st-key-portfolio_directory_row_0_"] {
        background: #F1F5F9;
    }
    .st-key-portfolio_customer_directory [class*="st-key-portfolio_directory_row_1_"] {
        background: #FFFFFF;
    }
    .st-key-market_watchlist_panel [class*="st-key-watchlist_card_shell_"] {
        position: relative;
    }
    .st-key-watchlist_create_control [data-testid="stIconMaterial"] {
        font-size: 0.875rem;
    }
    /* Hide only the popover's trailing chevron, preserving its leading + icon
       and the separate expander arrow on the card heading. */
    .st-key-market_watchlist_panel [data-testid="stPopoverButton"] > div >
    [aria-hidden="true"] { display: none; }
    .st-key-market_watchlist_panel [class*="st-key-watchlist_tickers"] {
        row-gap: 0.125rem;
        column-gap: 0.5rem;
        align-content: flex-start;
    }
    .st-key-market_watchlist_panel [class*="st-key-watchlist_compact_card_"] {
        padding: 0.625rem 0.75rem;
        gap: 0.25rem;
    }
    .st-key-market_watchlist_panel [class*="st-key-watchlist_compact_card_"] p {
        margin-block: 0;
    }
    .st-key-market_watchlist_panel [data-testid="stExpanderDetails"] {
        padding: 0.25rem;
    }
    .st-key-market_watchlist_panel [data-testid="stExpanderDetails"]
    [data-testid="stTextInputRootElement"] {
        min-height: 1.75rem;
        height: 1.75rem;
    }
    .st-key-market_watchlist_panel [data-testid="stExpanderDetails"]
    [data-testid="stTextInputRootElement"] input {
        min-height: 0;
        height: 100%;
        padding-block: 0.125rem;
        font-size: 0.8125rem;
    }
    .st-key-market_watchlist_panel [data-testid="stExpanderDetails"] >
    [data-testid="stVerticalBlock"] { gap: 0.25rem; }
    .st-key-market_watchlist_panel [class*="st-key-watchlist_quote_"] button {
        justify-content: flex-start;
        font-weight: 600;
        color: #153E75;
    }
    .st-key-market_watchlist_panel [class*="st-key-watchlist_remove_"] button {
        color: #64748B;
    }
    /* Scrollable layout wrappers can become focused on click, independently of
       stMain. Suppress their outlines, not those of widgets or ARIA controls. */
    .stApp :is(div, section, main):focus:not([role]):not([contenteditable="true"]) {
        outline: none !important;
        box-shadow: none !important;
    }
    /* The page scroller receives focus on wheel/keyboard scrolling. Its browser
       outline crosses the header; interactive controls keep their own rings. */
    #stMain,
    [data-testid="stMain"],
    [data-testid="stAppScrollToBottomContainer"] {
        outline: none !important;
        border: 0 !important;
        box-shadow: none !important;
    }
    /* Independently scrolling panes also receive browser focus on scroll. */
    .st-key-workspace_content,
    .st-key-chat_history,
    .st-key-workspace_body,
    .st-key-workspace_content:focus-visible,
    .st-key-chat_history:focus-visible {
        outline: none !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebarHeader"] { display: none; }
    [data-testid="stSidebarContent"] { padding-top: 0; }
    [data-testid="stSidebarUserContent"] { padding-top: 0; }
    [data-testid="stSidebar"] { border-right: 0; }
    @media (min-width: 901px) {
        [data-testid="stSidebar"] {
            width: clamp(220px, 16vw, 250px) !important;
            min-width: 220px;
            max-width: 250px;
        }
        [data-testid="stSidebarUserContent"] {
            padding-inline: 0.875rem;
        }
    }
    /* Paint the sidebar band edge to edge, outside its padded content. */
    [data-testid="stSidebar"]::before {
        content: "";
        position: absolute;
        inset: 0 0 auto;
        height: 80px;
        background: linear-gradient(180deg, #E8EFF7 0%, #F1F5F9 100%);
        box-sizing: border-box;
        pointer-events: none;
    }
    .st-key-sidebar_brand {
        position: sticky;
        top: 0;
        z-index: 20;
        background: linear-gradient(180deg, #E8EFF7 0%, #F1F5F9 100%);
        min-height: 80px;
        padding: 0.75rem 0;
        box-sizing: border-box;
        margin-bottom: 0.5rem;
    }
    .st-key-sidebar_brand strong { color: #153E75; font-size: 1.25rem; }
    .st-key-sidebar_brand p { margin: 0; }
    .st-key-sidebar_brand [data-testid="stCaptionContainer"] { color: #475569; }
    [data-testid="stSidebarNav"] { padding-top: 0; }
    .st-key-workspace_header {
        position: sticky;
        top: 0;
        z-index: 20;
        background: linear-gradient(180deg, #E8EFF7 0%, #F1F5F9 100%);
        border-radius: 0;
        padding: 0.75rem 1rem;
        min-height: 80px;
        box-sizing: border-box;
        width: 100%;
    }
    /* Include Streamlit's immediate header wrapper in the sticky region. */
    .st-key-finpilot_page > :has(.st-key-workspace_header) {
        position: sticky;
        top: 0;
        z-index: 20;
        background: transparent;
        margin-inline: -1rem;
        width: calc(100% + 2rem);
        max-width: none;
    }
    .st-key-workspace_body {
        border: 1px solid #CBD5E1;
        border-radius: 0.75rem;
        padding: 1rem;
        box-sizing: border-box;
    }
    .st-key-workspace_content {
        overflow-y: scroll !important;
        overflow-x: hidden;
        height: var(--pane-height, 500px) !important;
        max-height: var(--pane-height, 500px) !important;
        min-height: 0;
        overscroll-behavior: contain;
        scrollbar-gutter: stable;
        scrollbar-width: thin;
        scrollbar-color: #94A3B8 #F1F5F9;
        padding-right: 0.5rem;
    }
    .st-key-workspace_content::-webkit-scrollbar { width: 8px; }
    .st-key-workspace_content::-webkit-scrollbar-track { background: #F1F5F9; }
    .st-key-workspace_content::-webkit-scrollbar-thumb {
        background: #94A3B8;
        border-radius: 4px;
    }
    .st-key-finpilot_chat_panel {
        overflow: hidden;
        min-height: 0;
        background: #F8FAFC;
        border-radius: 0.75rem;
    }
    .st-key-finpilot_chat_header {
        background: linear-gradient(115deg, #153E75, #115E59);
        padding: 0.75rem 0.875rem;
        border-radius: 0.75rem 0.75rem 0 0;
    }
    .st-key-finpilot_chat .st-key-finpilot_chat_header h3,
    .st-key-finpilot_chat_header [data-testid="stCaptionContainer"],
    .st-key-finpilot_chat_header [data-testid="stCaptionContainer"] p,
    .st-key-finpilot_chat_header button { color: #FFFFFF; }
    .st-key-finpilot_chat_header button:hover { background: #FFFFFF20; }
    .st-key-chat_history { background: #F8FAFC; padding-inline: 0.5rem; }
    .st-key-chat_history [data-testid="stChatMessage"] {
        background: #FFFFFF;
        border-radius: 0.625rem;
        padding: 0.75rem;
    }
    .st-key-chat_history [data-testid="stChatMessage"]:has(
        [data-testid="stChatMessageAvatarUser"]
    ) { background: #E5F2F0; }
    .st-key-chat_composer {
        background: #EAF0F6;
        padding: 0.5rem;
        border-radius: 0 0 0.75rem 0.75rem;
    }
    .st-key-chat_composer [data-testid="stChatInput"] { background: #FFFFFF; }
    .st-key-chat_scope { padding-inline: 0.5rem; }
    @media (min-width: 901px) {
        [data-testid="stMain"] { overflow: hidden; }
        .st-key-workspace_body { overflow: hidden; }
        .st-key-finpilot_chat_panel { height: var(--pane-height) !important; }
    }
    @media (max-width: 900px) {
        .st-key-workspace_content { overflow: visible; padding-right: 0; }
    }

    .st-key-finpilot_header h1 {
        color: #153E75;
        font-size: 1.5rem;
        line-height: 1.2;
        margin: 0;
        padding-block: 0.15rem;
    }
    .st-key-finpilot_header p { margin-block: 0; }
    @media (max-width: 900px) {
        .st-key-header_description { display: none; }
        .st-key-workspace_header { min-height: 80px; }
    }
    .st-key-finpilot_page h2,
    .st-key-finpilot_page h3 {
        color: #153E75;
        font-size: 1.25rem;
        line-height: 1.4;
        margin-block: 0;
        padding-block: 0.6rem;
    }
    .st-key-finpilot_chat h1,
    .st-key-finpilot_chat h2,
    .st-key-finpilot_chat h3 {
        font-size: 1.2rem;
        margin-block: 0;
        padding-block: 0.2rem 0.35rem;
    }
    .st-key-finpilot_chat p { margin-block: 0 0.35rem; }
    .st-key-finpilot_chat hr { margin-block: 0.5rem; }
    </style>
    """)
    st.html(
        """
        <script>
        (() => {
            window.finpilotPageFocusCleanup?.();
            const controller = new AbortController();
            const releasePageFocus = () => {
                requestAnimationFrame(() => {
                    const active = document.activeElement;
                    // Mouse scrolling should not focus the whole workspace. Keep
                    // keyboard navigation and focus inside actual widgets intact.
                    if (active?.id === 'stMain') active.blur();
                });
            };
            document.addEventListener('pointerdown', releasePageFocus,
                {capture: true, passive: true, signal: controller.signal});
            document.addEventListener('wheel', releasePageFocus,
                {capture: true, passive: true, signal: controller.signal});
            window.finpilotPageFocusCleanup = () => controller.abort();
        })();
        </script>
        """,
        unsafe_allow_javascript=True,
    )
