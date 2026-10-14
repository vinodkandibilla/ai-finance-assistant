"""HTTP-to-router-to-agent-to-authorized SQLite/tools acceptance tests.

Model decisions are scripted; actual retrieval, arithmetic and authorization run.
No external providers are contacted and no model accuracy claims are made.
"""

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from ai_finance_assistant.agents.portfolio_analysis.agent import PortfolioAnalysisResult
from ai_finance_assistant.agents.portfolio_analysis.authorization import UserContext
from ai_finance_assistant.api.query import get_query_graph
from ai_finance_assistant.data.portfolio_store import PortfolioStore
from ai_finance_assistant.main import create_app
from ai_finance_assistant.workflow.builder import build_graph
from test_finance_autonomous_agent import call
from test_portfolio_agent import make_agent


async def request_analysis(store, identity, query, responses, final=None):
    class Classifier:
        async def classify(self, query):
            return "portfolio"

    kwargs = {"final": final} if final is not None else {}
    agent, model = make_agent(store, responses, user_context=None, **kwargs)
    graph = build_graph(Classifier(), portfolio_agent=agent)
    app = create_app()
    app.state.portfolio_user_context = identity
    app.dependency_overrides[get_query_graph] = lambda: graph
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/query", json={"query": query})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == body["classified_intent"] == "portfolio"
    assert body["executed_handler"] == "portfolio_analysis_node"
    return body, model


@pytest.fixture
def store():
    with PortfolioStore() as database:
        yield database


@pytest.mark.parametrize(
    "query",
    [
        "Show me my current portfolio value and allocation.",
        "Analyze my diversification and concentration across sectors and asset classes.",
    ],
)
async def test_customer_value_allocation_and_diversification(store, query):
    body, model = await request_analysis(
        store,
        UserContext(user_id="CUST001", role="CUSTOMER"),
        query,
        [call("get_customer_portfolio", {}, "p"), AIMessage(content="Done")],
    )
    metrics = body["calculation"]
    assert metrics["portfolio_value"] == 69500
    assert metrics["cost_basis"] == 66400
    assert metrics["unrealized_gain_loss"] == 3100
    assert metrics["concentration"]["is_concentrated"] is False
    assert {r["name"]: r["market_value"] for r in metrics["asset_class_allocation"]} == {
        "stock": 12500,
        "bond_etf": 25000,
        "broad_market_etf": 32000,
    }
    assert {r["name"]: r["market_value"] for r in metrics["sector_allocation"]} == {
        "Broad Market": 20000,
        "International Broad Market": 12000,
        "Fixed Income": 25000,
        "Financials": 4000,
        "Healthcare": 4500,
        "Energy": 4000,
    }
    vti = next(h for h in metrics["holdings"] if h["symbol"] == "VTI")
    assert vti["holding_allocation_pct"] == 28.78
    assert vti["current_price"] == 381.94
    assert len(model.seen) == 2  # One retrieval and completion; no forced extra tools.
    assert metrics["data_source"] == "synthetic_test_data"


async def test_customer_performance_and_sector_concentration(store):
    body, _ = await request_analysis(
        store,
        UserContext(user_id="C003", role="CUSTOMER"),
        "How is my portfolio performing? Show gainers, losers and overall unrealized gain.",
        [call("get_customer_portfolio", {}, "p"), AIMessage(content="Done")],
    )
    metrics = body["calculation"]
    assert metrics["portfolio_value"] == 110500
    assert metrics["cost_basis"] == 101400
    assert metrics["unrealized_gain_loss"] == 9100
    assert metrics["unrealized_gain_loss_pct"] == 8.97
    assert metrics["biggest_gainers"][0]["symbol"] == "NVDA"
    assert metrics["biggest_gainers"][0]["unrealized_gain_loss"] == 12000
    assert metrics["biggest_losers"][0]["symbol"] == "MSFT"
    assert metrics["biggest_losers"][0]["unrealized_gain_loss"] == -5000
    assert metrics["concentration"]["holdings_over_threshold"] == ["MSFT"]
    assert metrics["concentration"]["sectors_over_threshold"] == ["Technology"]
    assert metrics["largest_sector"]["allocation_pct"] == 94.12


@pytest.mark.parametrize(
    "identity,target,allowed",
    [
        (UserContext(user_id="C001", role="CUSTOMER"), "C002", False),
        (UserContext(user_id="PM001", role="PORTFOLIO_MANAGER"), "C001", True),
        (UserContext(user_id="PM001", role="PORTFOLIO_MANAGER"), "C020", False),
    ],
)
async def test_http_customer_and_manager_authorization(store, identity, target, allowed):
    body, _ = await request_analysis(
        store,
        identity,
        f"Analyze {target}",
        [call("get_customer_portfolio", {"customer_id": target}, "p"), AIMessage(content="Done")],
    )
    if allowed:
        assert body["calculation"]["customer_id"] == target
    else:
        assert body["status"] == "error"
        assert "calculation" not in body
        assert body["answer"] == "You do not have access to the requested portfolio information."


async def test_manager_screen_and_rankings_end_to_end(store):
    responses = [call("screen_manager_portfolios", {}, "s")]
    responses += [
        call("compare_manager_portfolios", {"rank_by": rank}, rank)
        for rank in ("concentration", "diversification", "unrealized_loss")
    ]
    body, _ = await request_analysis(
        store,
        UserContext(user_id="PM001", role="PORTFOLIO_MANAGER"),
        "Screen concentration and compare highest concentration, "
        "lowest diversification and largest losses.",
        [*responses, AIMessage(content="Done")],
    )
    screened, concentration, diversification, losses = body["calculations"]
    assert {r["customer_id"] for r in screened["portfolios"]} == {"C002", "C003", "C004"}
    single = next(r for r in screened["portfolios"] if r["customer_id"] == "C002")
    assert single["largest_holding"]["holding_allocation_pct"] == 88.59
    assert single["concentration"]["holdings_over_threshold"] == ["AAPL"]
    assert concentration["portfolios"][0]["customer_id"] == "C002"
    assert diversification["portfolios"][-1]["customer_id"] == "C002"
    assert losses["portfolios"][0]["customer_id"] == "C004"
    assert losses["portfolios"][0]["unrealized_gain_loss"] == -11900
    assert all(r["manager_id"] == "PM001" for r in concentration["portfolios"])


async def test_profile_is_pre_established_and_unchanged_end_to_end(store):
    before = store.get_customer("C001")
    final = PortfolioAnalysisResult(
        status="answered",
        sufficient=True,
        aligned=True,
        answer="Interpretation only, not an official suitability determination.",
        profile_alignment_customer_ids=["C001"],
    )
    body, _ = await request_analysis(
        store,
        UserContext(user_id="C001", role="CUSTOMER"),
        "Is my portfolio aligned with my stored profile and horizon?",
        [
            call("get_customer_portfolio", {}, "p"),
            call("get_customer_investment_profile", {}, "profile"),
            AIMessage(content="Done"),
        ],
        final,
    )
    assert store.get_customer("C001") == before
    assert body["calculations"][1]["risk_profile_source"] == "pre_established"
    assert body["calculations"][1]["profile"]["investment_horizon_years"] == 10


@pytest.mark.parametrize("role", ["ADMIN", "ADVISOR", "portfolio_manager", "individual_customer"])
def test_only_two_identity_roles(role):
    with pytest.raises(ValidationError):
        UserContext(user_id="PM001", role=role)


async def test_name_lookup_alone_is_not_financial_evidence(store):
    body, _ = await request_analysis(
        store,
        UserContext(user_id="PM001", role="PORTFOLIO_MANAGER"),
        "Tell me the value of Customer 1",
        [
            call("resolve_portfolio_customer", {"name": "Customer 1:"}, "name"),
            AIMessage(content="An unsupported value"),
        ],
    )
    assert body["status"] == "insufficient"
    assert body["answer"] != "Synthetic test portfolio analysis."
