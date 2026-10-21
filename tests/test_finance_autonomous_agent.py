import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import Field

from finassist.agents.finance_qa.config import FinanceAgentSettings
from finassist.agents.finance_qa.graph import FinanceQaAgent
from finassist.agents.finance_qa.metadata_extractor import FinanceMetadataExtractor
from finassist.agents.finance_qa.schemas import FinanceAgentResult, FinanceDocument
from finassist.workflow.builder import build_graph


class ScriptedModel(BaseChatModel):
    responses: list[AIMessage]
    final: FinanceAgentResult
    seen: list = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "scripted-tool-model"

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(lambda messages: self.final)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(messages)
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])


class Retriever:
    def __init__(self, documents):
        self.documents = documents
        self.calls = []

    async def retrieve(self, dense_queries, sparse_keywords, *, metadata_filter):
        self.calls.append((dense_queries, sparse_keywords))
        return self.documents, []


class Reranker:
    async def rerank(self, query, documents, limit):
        return documents[:limit]


class Classifier:
    async def classify(self, query):
        return "finance_qa"


def call(name, args, identifier):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": identifier}])


def make_agent(responses, final, documents=(), **settings):
    model = ScriptedModel(responses=responses, final=final)
    retriever = Retriever(list(documents))
    agent = FinanceQaAgent(
        llm=model,
        retriever=retriever,
        reranker=Reranker(),
        classifier=Classifier(),
        metadata_extractor=FinanceMetadataExtractor([]),
        settings=FinanceAgentSettings(**settings),
    )
    return agent, retriever, model


async def test_model_selects_searches_and_follow_up_terms():
    agent, retriever, model = make_agent(
        [
            call(
                "finance_rag_query",
                {
                    "query": "Delta",
                    "dense_queries": ["Delta exposure"],
                    "sparse_keywords": ["Delta"],
                },
                "one",
            ),
            call(
                "finance_targeted_search",
                {"query": "Theta", "missing_aspect": "Theta", "sparse_keywords": ["time decay"]},
                "two",
            ),
            AIMessage(content="Both concepts are supported."),
        ],
        FinanceAgentResult(
            status="answered",
            answer="Delta measures sensitivity; Theta time decay.",
            citations=["invented citation"],
        ),
        [
            FinanceDocument(
                content="Delta sensitivity and Theta time decay.",
                source="handbook",
                metadata={"page_number": 9},
            )
        ],
    )
    graph = build_graph(Classifier(), finance_agent=agent)
    result = await graph.ainvoke({"query": "Explain Delta and Theta"})
    assert retriever.calls == [(["Delta exposure"], ["Delta"]), (["Theta"], ["time decay"])]
    assert any(isinstance(message, ToolMessage) for message in model.seen[-1])
    assert result["status"] == "answered"
    assert result["citations"] == ["Financial Education Handbook, page 9"]


async def test_answer_without_search_is_rejected():
    agent, retriever, _ = make_agent(
        [AIMessage(content="Unsupported answer")],
        FinanceAgentResult(status="answered", answer="Unsupported answer"),
    )
    result = await agent.graph.ainvoke({"query": "What is compound interest?"})
    assert result["status"] == "insufficient"
    assert retriever.calls == []


async def test_model_can_report_out_of_scope_without_searching():
    agent, retriever, _ = make_agent(
        [AIMessage(content="This is a tax question.")],
        FinanceAgentResult(status="out_of_scope"),
    )
    result = await agent.graph.ainvoke({"query": "Explain Roth IRA rules"})
    assert result["status"] == "out_of_scope"
    assert not retriever.calls


async def test_search_budget_caps_retriever_calls():
    agent, retriever, _ = make_agent(
        [
            call("finance_rag_query", {"query": "ETF"}, "one"),
            call("finance_rag_query", {"query": "ETF definition"}, "two"),
            AIMessage(content="No evidence"),
        ],
        FinanceAgentResult(status="answered", answer="Unsupported answer"),
        agent_max_searches=1,
    )
    result = await agent.graph.ainvoke({"query": "What is ETF?"})
    assert len(retriever.calls) == 1
    assert result["status"] == "error"


async def test_parent_clarification_interrupt_resumes_inside_agent():
    agent, retriever, model = make_agent(
        [
            call("ask_finance_clarification", {"question": "Which funds?"}, "clarify"),
            call("finance_rag_query", {"query": "ETFs versus mutual funds"}, "search"),
            AIMessage(content="Evidence-supported comparison"),
        ],
        FinanceAgentResult(status="answered", answer="Comparison"),
        [
            FinanceDocument(
                content="Fund comparison", source="handbook", metadata={"page_number": 34}
            )
        ],
    )
    graph = build_graph(Classifier(), finance_agent=agent, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "autonomous-clarification"}}
    paused = await graph.ainvoke({"query": "Which is better?"}, config)
    assert paused["__interrupt__"][0].value["question"] == "Which funds?"
    assert not retriever.calls  # Missing details must pause before searching.
    assert not paused.get("calculation")
    result = await graph.ainvoke(Command(resume="ETFs and mutual funds"), config)
    assert result["status"] == "answered"
    assert retriever.calls[0][0] == ["ETFs versus mutual funds"]
    assert any(
        isinstance(m, ToolMessage) and m.content == "ETFs and mutual funds" for m in model.seen[-1]
    )


async def test_targeted_evidence_is_collected_by_shared_runtime():
    agent, retriever, model = make_agent(
        [
            call("finance_rag_query", {"query": "ETF"}, "initial"),
            call(
                "finance_targeted_search",
                {
                    "query": "ETF fees",
                    "missing_aspect": "expense ratios",
                },
                "targeted",
            ),
            AIMessage(content="Supported explanation."),
        ],
        FinanceAgentResult(status="answered", answer="Supported explanation."),
    )

    async def retrieve(dense_queries, sparse_keywords, **kwargs):
        retriever.calls.append((dense_queries, sparse_keywords))
        if dense_queries == ["ETF"]:
            return [], []
        return [
            FinanceDocument(
                content="ETF fees evidence",
                source="handbook",
                metadata={"page_number": 12},
            )
        ], []

    retriever.retrieve = retrieve
    result = await agent.graph.ainvoke({"query": "Explain ETF fees"})
    assert result["status"] == "answered"
    assert result["citations"] == ["Financial Education Handbook, page 12"]
    assert len(retriever.calls) == 2
    assert any(
        isinstance(message, ToolMessage) and message.name == "finance_targeted_search"
        for message in model.seen[-1]
    )


async def test_new_question_guidance_does_not_open_another_clarification():
    agent, retriever, model = make_agent(
        [
            call(
                "ask_finance_clarification",
                {"question": "Please submit a new question about current prices."},
                "redirect",
            ),
            AIMessage(content="Outside scope"),
        ],
        FinanceAgentResult(status="out_of_scope"),
    )
    result = await agent.graph.ainvoke({"query": "What is Apple's current stock price?"})
    assert not result.get("__interrupt__")
    assert result["status"] == "out_of_scope"
    assert "new question" in result["answer"]
    assert not retriever.calls
    assert not result.get("clarification_question")
    replies = [message for message in model.seen[-1] if isinstance(message, ToolMessage)]
    assert replies[0].artifact["kind"] == "policy_guidance"


async def test_changed_task_after_general_clarification_stops_without_search():
    agent, retriever, model = make_agent(
        [
            call("ask_finance_clarification", {"question": "Which funds?"}, "clarify"),
            AIMessage(content="The new request is outside scope"),
        ],
        FinanceAgentResult(status="out_of_scope"),
    )
    graph = build_graph(Classifier(), finance_agent=agent, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "finance-scope-change"}}
    paused = await graph.ainvoke({"query": "Compare these funds"}, config)
    assert paused.get("__interrupt__")
    result = await graph.ainvoke(Command(resume="Instead, give me the weather"), config)
    assert result["status"] == "out_of_scope"
    assert not result.get("__interrupt__")
    assert not result.get("clarification_question")
    assert not retriever.calls
    assert any(
        isinstance(message, ToolMessage) and message.content == "Instead, give me the weather"
        for message in model.seen[-1]
    )


async def test_empty_search_cannot_be_presented_as_a_supported_answer():
    agent, retriever, _ = make_agent(
        [
            call("finance_rag_query", {"query": "ETF"}, "search"),
            AIMessage(content="Invented answer"),
        ],
        FinanceAgentResult(status="answered", answer="Invented answer", citations=["invented"]),
    )
    result = await agent.graph.ainvoke({"query": "What is an ETF?"})
    assert len(retriever.calls) == 1
    assert result["status"] == "insufficient"
    assert not result["citations"]
    assert "Invented answer" not in result["answer"]


@pytest.mark.parametrize("exception", [RuntimeError, TimeoutError])
async def test_failed_search_returns_error_without_fabricated_answer(exception, monkeypatch):
    agent, retriever, _ = make_agent(
        [
            call("finance_rag_query", {"query": "ETF"}, "search"),
            AIMessage(content="Invented answer"),
        ],
        FinanceAgentResult(status="answered", answer="Invented answer", citations=["invented"]),
    )

    async def fail(*args, **kwargs):
        raise exception("private provider payload")

    monkeypatch.setattr(retriever, "retrieve", fail)
    result = await agent.graph.ainvoke({"query": "What is an ETF?"})
    assert result["status"] == "error"
    assert result["errors"]
    assert not result["citations"]
    assert "Invented answer" not in result["answer"]
    assert "private provider payload" not in str(result)
