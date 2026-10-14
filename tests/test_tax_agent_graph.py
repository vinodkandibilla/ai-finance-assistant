from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from ai_finance_assistant.agents.tax_education.config import TaxAgentSettings
from ai_finance_assistant.agents.tax_education.graph import TaxEducationAgent
from ai_finance_assistant.agents.tax_education.schemas import TaxAgentResult
from ai_finance_assistant.rag.models import RetrievedDocument
from ai_finance_assistant.workflow.builder import build_graph
from test_finance_autonomous_agent import ScriptedModel, call


class TaxModel(ScriptedModel):
    final: TaxAgentResult


class Retriever:
    def __init__(self, year=2026):
        self.calls = []
        self.year = year

    async def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return [
            RetrievedDocument(
                content="Supported bracket evidence",
                source="tax.jsonl",
                metadata={"title": "MFJ brackets", "tax_year": self.year},
            )
        ]


class Reranker:
    async def rerank(self, query, documents, limit):
        return documents[:limit]


class Classifier:
    async def classify(self, query):
        return "tax"


def make_agent(responses, final, year=2026, **settings):
    model = TaxModel(responses=responses, final=final)
    retriever = Retriever(year)
    agent = TaxEducationAgent(
        llm=model, retriever=retriever, reranker=Reranker(), settings=TaxAgentSettings(**settings)
    )
    return agent, retriever, model


async def test_model_directs_searches_without_changing_original_year():
    agent, retriever, _ = make_agent(
        [
            call(
                "tax_rag_query",
                {"query": "2025 deduction", "dense_queries": ["eligibility"]},
                "one",
            ),
            call("tax_rag_query", {"query": "More tax evidence"}, "two"),
            AIMessage(content="Done"),
        ],
        TaxAgentResult(status="answered", answer="Supported answer", citations=["invented"]),
    )
    graph = build_graph(Classifier(), tax_agent=agent)
    result = await graph.ainvoke({"query": "Explain deductions for tax year 2026"})
    assert len(retriever.calls) == 2
    assert retriever.calls[0][0] == "2025 deduction"
    for _, kwargs in retriever.calls:
        conditions = {c.key: c.match for c in kwargs["metadata_filter"].must}
        assert conditions["metadata.tax_year"].value == 2026
    assert result["intent"] == "tax"
    assert result["status"] == "answered"
    assert result["citations"] == ["MFJ brackets"]


async def test_answer_without_search_is_rejected():
    agent, retriever, _ = make_agent(
        [AIMessage(content="Unsupported")], TaxAgentResult(status="answered", answer="Unsupported")
    )
    result = await agent.graph.ainvoke({"query": "What is a deduction?"})
    assert result["status"] == "insufficient"
    assert not retriever.calls


async def test_mismatched_year_evidence_cannot_support_answer():
    agent, _, _ = make_agent(
        [call("tax_rag_query", {"query": "deductions"}, "one"), AIMessage(content="Done")],
        TaxAgentResult(status="answered", answer="Wrong year"),
        year=2025,
    )
    result = await agent.graph.ainvoke({"query": "Deductions for tax year 2026"})
    assert result["status"] == "insufficient"
    assert not result["citations"]


async def test_budget_caps_actual_searches():
    agent, retriever, _ = make_agent(
        [
            call("tax_rag_query", {"query": "deductions"}, "one"),
            call("tax_rag_query", {"query": "eligibility"}, "two"),
            AIMessage(content="Done"),
        ],
        TaxAgentResult(status="answered", answer="Supported"),
        agent_max_searches=1,
    )
    await agent.graph.ainvoke({"query": "Explain deductions"})
    assert len(retriever.calls) == 1


async def test_clarification_resumes_and_updates_user_constraints():
    agent, retriever, model = make_agent(
        [
            call(
                "ask_tax_clarification", {"question": "What income and filing status?"}, "clarify"
            ),
            call("tax_rag_query", {"query": "MFJ brackets"}, "search"),
            AIMessage(content="Done"),
        ],
        TaxAgentResult(status="answered", answer="Supported bracket answer"),
    )
    graph = build_graph(Classifier(), tax_agent=agent, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "tax-autonomous"}}
    paused = await graph.ainvoke({"query": "What will be my tax bracket?"}, config)
    assert paused["classified_intent"] == "tax"
    assert paused["executed_handler"] == "tax_education_node"
    assert paused["__interrupt__"][0].value["question"] == "What income and filing status?"
    assert not retriever.calls
    result = await graph.ainvoke(
        Command(resume="$250K married filing jointly tax year 2026"), config
    )
    assert result["status"] == "answered"
    assert result["classified_intent"] == "tax"
    assert result["executed_handler"] == "tax_education_node"
    conditions = {c.key: c.match for c in retriever.calls[0][1]["metadata_filter"].must}
    assert conditions["metadata.tax_year"].value == 2026
    assert "mfj" in conditions["metadata.filing_status"].any
    assert any(
        isinstance(m, ToolMessage) and m.name == "ask_tax_clarification" for m in model.seen[-1]
    )


async def test_domain_change_on_resume_returns_out_of_scope_without_dispatch():
    agent, retriever, _ = make_agent(
        [
            call("ask_tax_clarification", {"question": "Which tax topic?"}, "clarify"),
            AIMessage(content="Outside tax scope"),
        ],
        TaxAgentResult(
            status="out_of_scope", answer="An unsupported response", citations=["invalid"]
        ),
    )

    class MarketAgent:
        async def ainvoke(self, input):
            raise AssertionError("Resume must not invoke another domain")

    graph = build_graph(
        Classifier(), tax_agent=agent, market_agent=MarketAgent(), checkpointer=MemorySaver()
    )
    config = {"configurable": {"thread_id": "tax-scope-change"}}
    await graph.ainvoke({"query": "tax help"}, config)
    result = await graph.ainvoke(Command(resume="What is today's AAPL price?"), config)
    assert result["intent"] == "tax"
    assert result["status"] == "out_of_scope"
    assert "new question" in result["answer"]
    assert not result["citations"]
    assert not retriever.calls


async def test_reranker_failure_produces_controlled_error():
    agent, _, _ = make_agent(
        [call("tax_rag_query", {"query": "deductions"}, "one"), AIMessage(content="Done")],
        TaxAgentResult(status="answered", answer="Unsupported"),
    )

    # The reranker dependency is captured by the tool at construction time.
    class Broken:
        async def rerank(self, *args, **kwargs):
            raise RuntimeError("private details")

    model = TaxModel(
        responses=[
            call("tax_rag_query", {"query": "deductions"}, "one"),
            AIMessage(content="Failed"),
        ],
        final=TaxAgentResult(status="answered", answer="Unsupported"),
    )
    agent = TaxEducationAgent(llm=model, retriever=Retriever(), reranker=Broken())
    result = await agent.graph.ainvoke({"query": "Explain deductions"})
    assert result["status"] == "error"
    assert "private details" not in result["answer"]


async def test_year_confirmation_alone_does_not_interrupt():
    agent, retriever, _ = make_agent(
        [
            call("ask_tax_clarification", {"question": "Do you mean tax year 2026?"}, "clarify"),
            call("tax_rag_query", {"query": "MFJ tax brackets"}, "search"),
            AIMessage(content="Done"),
        ],
        TaxAgentResult(status="answered", answer="Supported bracket answer"),
    )
    graph = build_graph(Classifier(), tax_agent=agent, checkpointer=MemorySaver())
    result = await graph.ainvoke(
        {"query": "My tax bracket for income $250K married filing jointly"},
        {"configurable": {"thread_id": "no-year-followup"}},
    )
    assert not result.get("__interrupt__")
    assert result["status"] == "answered"
    assert len(retriever.calls) == 1


async def test_supplied_filing_status_is_not_requested_again():
    agent, retriever, _ = make_agent(
        [
            call("ask_tax_clarification", {"question": "What is your filing status?"}, "clarify"),
            call("tax_rag_query", {"query": "MFJ brackets $250K"}, "search"),
            AIMessage(content="Done"),
        ],
        TaxAgentResult(status="answered", answer="Supported bracket answer"),
    )
    graph = build_graph(Classifier(), tax_agent=agent, checkpointer=MemorySaver())
    result = await graph.ainvoke(
        {
            "query": "What will be my tax bracket for the income $250K "
            "with filing status Married filing jointly"
        },
        {"configurable": {"thread_id": "supplied-filing-status"}},
    )
    assert not result.get("__interrupt__")
    assert result["status"] == "answered"
    conditions = {c.key: c.match for c in retriever.calls[0][1]["metadata_filter"].must}
    assert "mfj" in conditions["metadata.filing_status"].any


async def test_out_of_scope_instruction_does_not_create_another_interrupt():
    agent, _, _ = make_agent(
        [
            call("ask_tax_clarification", {"question": "What is your filing status?"}, "first"),
            call(
                "ask_tax_clarification",
                {"question": "Please submit a new question about stocks."},
                "second",
            ),
            AIMessage(content="Outside tax scope"),
        ],
        TaxAgentResult(status="out_of_scope"),
    )
    graph = build_graph(Classifier(), tax_agent=agent, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "scope-completion"}}
    await graph.ainvoke({"query": "My tax bracket for $100K"}, config)
    result = await graph.ainvoke(
        Command(resume="Actually, get Apple's current stock price"), config
    )
    assert result["status"] == "out_of_scope"
    assert not result.get("__interrupt__")
    assert result["citations"] == []
