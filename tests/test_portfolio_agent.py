import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from finassist.agents.portfolio_analysis.agent import (
    MAX_ITERATIONS,
    PortfolioAnalysisAgent,
    PortfolioAnalysisResult,
)
from finassist.agents.portfolio_analysis.authorization import UserContext
from finassist.data.portfolio_store import PortfolioStore
from finassist.workflow.builder import build_graph
from test_finance_autonomous_agent import ScriptedModel, call


class PortfolioModel(ScriptedModel):
    final: PortfolioAnalysisResult


@pytest.fixture
def store():
    with PortfolioStore() as database:
        yield database


def make_agent(store, responses, **kwargs):
    final = kwargs.pop(
        "final",
        PortfolioAnalysisResult(
            status="answered",
            answer="Synthetic test portfolio analysis.",
            sufficient=True,
            aligned=True,
        ),
    )
    model = PortfolioModel(responses=responses, final=final)
    kwargs.setdefault("user_context", UserContext(user_id="PM001", role="PORTFOLIO_MANAGER"))
    return PortfolioAnalysisAgent(llm=model, store=store, **kwargs), model


@pytest.mark.parametrize(
    "query",
    [
        "Show me my current portfolio value and allocation.",
        "Analyze diversification and concentration across holdings, sectors and asset classes.",
        "Show biggest gainers and losers and overall unrealized gain or loss.",
    ],
)
async def test_customer_scenarios(store, query):
    agent, model = make_agent(
        store,
        [call("get_customer_portfolio", {"customer_id": "C001"}, "p"), AIMessage(content="Done")],
    )
    result = await agent.ainvoke(
        {"query": query, "customer_id": "C001", "user_role": "individual_customer"}
    )
    assert result["status"] == "answered"
    assert result["calculation"]["portfolio_value"] == 69500
    assert result["calculation"]["biggest_losers"]
    assert '"customer_id": "C001"' in str(model.seen[0])


async def test_manager_screen_and_three_rankings_preserved(store):
    responses = [call("screen_manager_portfolios", {"manager_id": "PM001"}, "s")]
    responses += [
        call("compare_manager_portfolios", {"manager_id": "PM001", "rank_by": rank}, rank)
        for rank in ("concentration", "diversification", "unrealized_loss")
    ]
    agent, _ = make_agent(store, [*responses, AIMessage(content="Done")])
    result = await agent.ainvoke({"query": "Screen concentration and compare all three rankings."})
    assert result["status"] == "answered"
    assert len(result["calculations"]) == 4
    assert [r["portfolios"][0]["customer_id"] for r in result["calculations"][1:]] == [
        "C002",
        "C005",
        "C004",
    ]


async def test_profile_context_requires_both_tools(store):
    final = PortfolioAnalysisResult(
        status="answered",
        answer="Interpretation, not suitability.",
        sufficient=True,
        aligned=True,
        profile_alignment_customer_ids=["C001"],
    )
    responses = [
        call("get_customer_portfolio", {"customer_id": "C001"}, "p"),
        call("get_customer_investment_profile", {"customer_id": "C001"}, "profile"),
        AIMessage(content="Done"),
    ]
    before = store.get_customer("C001")
    agent, _ = make_agent(store, responses, final=final)
    assert (await agent.ainvoke({"query": "Aligned with my stored profile and horizon?"}))[
        "status"
    ] == "answered"
    assert store.get_customer("C001") == before
    agent, _ = make_agent(store, [responses[0], responses[2]], final=final)
    assert (await agent.ainvoke({"query": "Profile alignment?"}))["status"] == "insufficient"


async def test_parent_clarification_resumes_without_router(store):
    class Classifier:
        calls = 0

        async def classify(self, query):
            self.calls += 1
            return "portfolio"

    classifier = Classifier()
    agent, _ = make_agent(
        store,
        [
            call("ask_portfolio_clarification", {"question": "Which customer identifier?"}, "ask"),
            call("get_customer_portfolio", {"customer_id": "C001"}, "p"),
            AIMessage(content="Done"),
        ],
    )
    graph = build_graph(classifier, portfolio_agent=agent, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "portfolio-resume"}}
    paused = await graph.ainvoke({"query": "My allocation?"}, config)
    assert paused["__interrupt__"][0].value["question"] == "Which customer identifier?"
    result = await graph.ainvoke(Command(resume="C001"), config)
    assert result["executed_handler"] == "portfolio_analysis_node"
    assert result["calculation"]["portfolio_value"] == 69500
    assert classifier.calls == 1


async def test_no_evidence_and_invalid_identifier_cannot_answer(store):
    for responses in (
        [AIMessage(content="Invented")],
        [
            call("get_customer_portfolio", {"customer_id": "unknown"}, "p"),
            AIMessage(content="Done"),
        ],
    ):
        agent, _ = make_agent(store, responses)
        assert (await agent.ainvoke({"query": "Portfolio?"}))["status"] == (
            "insufficient" if len(responses) == 1 else "error"
        )


async def test_iteration_budget(store):
    responses = [
        call("get_customer_portfolio", {"customer_id": "C001"}, str(i))
        for i in range(MAX_ITERATIONS + 1)
    ]
    final = PortfolioAnalysisResult(
        status="insufficient",
        answer="Incomplete",
        sufficient=False,
        aligned=True,
        next_action="Need more information",
    )
    agent, model = make_agent(store, responses, final=final)
    result = await agent.ainvoke({"query": "Analyze portfolio"})
    assert result["errors"] == ["portfolio_iteration_limit"]
    assert len(model.seen) == MAX_ITERATIONS


async def test_api_context_and_metrics_on_same_thread(store):
    from httpx import ASGITransport, AsyncClient

    from finassist.api.query import get_query_graph
    from finassist.main import create_app

    class Classifier:
        async def classify(self, query):
            return "portfolio"

    agent, model = make_agent(
        store,
        [call("get_customer_portfolio", {"customer_id": "C002"}, "p"), AIMessage(content="Done")],
    )
    app = create_app()
    app.dependency_overrides[get_query_graph] = lambda: build_graph(
        Classifier(), portfolio_agent=agent
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/query",
            json={"query": "My value?", "customer_id": "C002", "user_role": "individual_customer"},
        )
    assert response.status_code == 200
    assert response.json()["calculation"]["portfolio_value"] == 90300
    assert response.json()["executed_handler"] == "portfolio_analysis_node"
    assert '"customer_id": "C002"' in str(model.seen[0])


async def test_out_of_scope_drops_metrics(store):
    final = PortfolioAnalysisResult(
        status="out_of_scope", answer="Trading is outside scope.", sufficient=True, aligned=True
    )
    agent, _ = make_agent(
        store,
        [call("get_customer_portfolio", {"customer_id": "C001"}, "p"), AIMessage(content="Done")],
        final=final,
    )
    result = await agent.ainvoke({"query": "Execute trades"})
    assert result["status"] == "out_of_scope"
    assert "calculation" not in result


async def test_timeout_returns_controlled_error(store):
    import asyncio

    from langchain_core.runnables import RunnableLambda

    agent, _ = make_agent(store, [AIMessage(content="Done")], timeout_seconds=0.01)

    async def slow(state):
        await asyncio.sleep(0.1)
        return state

    agent.agent = RunnableLambda(slow)
    result = await agent.ainvoke({"query": "My portfolio?"})
    assert result["status"] == "error"
    assert "calculation" not in result
