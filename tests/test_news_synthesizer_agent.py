import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from finassist.agents.news_synthesizer import NewsSynthesizerAgent
from finassist.agents.news_synthesizer.schemas import (
    EvidenceAssessment,
    NewsAgentResult,
    NewsClaim,
)
from test_finance_autonomous_agent import ScriptedModel, call


class NewsModel(ScriptedModel):
    final: NewsAgentResult


def make_agent(responses, final=None, batches=None, assessments=None, **kwargs):
    calls = []
    batches = list(batches or [])

    @tool("search_financial_news", response_format="content_and_artifact")
    def search(query: str) -> tuple[dict, dict]:
        """Retrieve mocked evidence."""
        calls.append(query)
        payload = batches.pop(0)
        return payload, payload

    model = NewsModel(
        responses=responses,
        final=final
        or NewsAgentResult(
            status="answered", claims=[NewsClaim(text="Reported development.", evidence_ids=["E1"])]
        ),
    )
    agent = NewsSynthesizerAgent(llm=model, search_tool=search, **kwargs)
    assessments = list(assessments or [EvidenceAssessment(sufficient=True)])

    def assess(messages):
        model.seen.append(messages)
        return assessments.pop(0)

    agent._assessor = RunnableLambda(assess)
    return agent, model, calls


def evidence(identifier="E1", url="https://example.com/story?b=2&a=1"):
    return {
        "status": "success",
        "evidence": [
            {
                "evidence_id": identifier,
                "url": url,
                "title": "Report",
                "source": "Example",
                "published_at": None,
                "content": "Reported development.",
            }
        ],
        "errors": [],
    }


async def test_model_selects_focused_search_and_synthesizes_grounded_claims():
    final = NewsAgentResult(
        status="answered",
        answer="Untrusted extra prose",
        claims=[
            NewsClaim(text="Reported development.", evidence_ids=["E1"]),
            NewsClaim(
                text="This may affect the sector.", evidence_ids=["E2"], kind="interpretation"
            ),
        ],
    )
    agent, model, calls = make_agent(
        [
            call("search_financial_news", {"query": "bank earnings reports"}, "one"),
            call("search_financial_news", {"query": "bank earnings sector outlook"}, "two"),
            AIMessage(content="Done"),
        ],
        final,
        [evidence(), evidence("E2", "https://other.example/report")],
        assessments=[
            EvidenceAssessment(
                sufficient=False,
                missing_aspects=["outlook"],
                next_search_query="bank earnings sector outlook",
            ),
            EvidenceAssessment(sufficient=True),
        ],
    )
    result = await agent.ainvoke({"query": "Summarize bank earnings and implications."})
    assert calls == ["bank earnings reports", "bank earnings sector outlook"]
    assert result["status"] == "answered"
    assert result["citations"] == [
        "https://example.com/story?b=2&a=1",
        "https://other.example/report",
    ]
    assert "Interpretation: " in result["answer"]
    assert "Untrusted extra prose" not in result["answer"]
    assert any(isinstance(m, ToolMessage) for m in model.seen[-1])


async def test_clarification_preserves_original_request():
    agent, model, calls = make_agent(
        [
            call("ask_news_clarification", {"question": "Which sector?"}, "ask"),
            call("search_financial_news", {"query": "banking news"}, "search"),
            AIMessage(content="Done"),
        ],
        batches=[evidence()],
        checkpointer=MemorySaver(),
    )
    config = {"configurable": {"thread_id": "news-resume"}}
    original = "Summarize recent sector news and implications."
    paused = await agent.ainvoke({"query": original}, config)
    assert paused["__interrupt__"][0].value["question"] == "Which sector?"
    assert calls == []
    result = await agent.ainvoke(Command(resume="Banking"), config)
    assert result["status"] == "answered"
    assert any(m.content == original for m in model.seen[-1])
    assert any(isinstance(m, ToolMessage) and m.content == "Banking" for m in model.seen[-1])


@pytest.mark.parametrize(
    "batches,ids,expected",
    [
        ([], ["E1"], "insufficient"),
        ([evidence()], ["E99"], "insufficient"),
        ([{"status": "error", "evidence": [], "errors": ["news_provider_error"]}], ["E1"], "error"),
        (
            [{"status": "insufficient", "evidence": [], "errors": ["no_news_results"]}],
            ["E1"],
            "insufficient",
        ),
    ],
)
async def test_answer_requires_retrieved_evidence_and_valid_ids(batches, ids, expected):
    responses = [call("search_financial_news", {"query": "news"}, "search")] if batches else []
    responses.append(AIMessage(content="Done"))
    final = NewsAgentResult(status="answered", claims=[NewsClaim(text="Claim", evidence_ids=ids)])
    agent, _, _ = make_agent(responses, final, batches)
    result = await agent.ainvoke({"query": "News summary"})
    assert result["status"] == expected
    assert result["citations"] == []
    assert result["answer"] != "Claim"


async def test_model_can_report_missing_coverage_instead_of_answering():
    agent, _, _ = make_agent(
        [call("search_financial_news", {"query": "news"}, "search"), AIMessage(content="Limited")],
        NewsAgentResult(
            status="insufficient", answer="The articles do not cover the requested event."
        ),
        [evidence()],
    )
    result = await agent.ainvoke({"query": "Explain the specified event"})
    assert result["status"] == "insufficient"
    assert "do not cover" in result["answer"]


async def test_sufficient_first_search_stops_without_consuming_remaining_budget(caplog):
    caplog.set_level("INFO", logger="finassist.agents.news_synthesizer.agent")
    agent, model, calls = make_agent(
        [
            call("search_financial_news", {"query": "first query"}, "first"),
            call("search_financial_news", {"query": "must not run"}, "blocked"),
        ],
        batches=[evidence()],
        assessments=[EvidenceAssessment(sufficient=True)],
    )
    result = await agent.ainvoke({"query": "Summarize the event"})
    assert result["status"] == "answered"
    assert calls == ["first query"]
    assert "Starting search 1/3" in caplog.text
    assert "Sufficient evidence after 1/3 searches" in caplog.text
    assert "Budget exhausted" not in caplog.text
    assert "first query" not in caplog.text
    assert len(model.responses) == 1


async def test_three_search_budget_returns_grounded_partial_summary_and_missing_aspects(caplog):
    caplog.set_level("INFO", logger="finassist.agents.news_synthesizer.agent")
    assessments = [
        EvidenceAssessment(
            sufficient=False,
            missing_aspects=["confirmed outlook"],
            next_search_query=f"focused query {i}",
        )
        for i in range(3)
    ]
    agent, model, calls = make_agent(
        [
            call("search_financial_news", {"query": "initial"}, "first"),
            call("search_financial_news", {"query": "fourth blocked"}, "blocked"),
        ],
        batches=[
            evidence(),
            evidence("E2", "https://example.com/two"),
            evidence("E3", "https://example.com/three"),
        ],
        assessments=assessments,
    )
    seen = []
    original_assessor = agent._assessor

    async def record(messages):
        seen.append(messages)
        return await original_assessor.ainvoke(messages)

    agent._assessor = RunnableLambda(record)
    result = await agent.ainvoke({"query": "Summarize the event and confirmed outlook"})
    assert calls == ["initial", "focused query 0", "focused query 1"]
    assert len(model.responses) == 1
    assert len(seen) == 3
    assert "Budget exhausted (3/3) | hard stop" in caplog.text
    assert "focused search next | remaining=2" in caplog.text
    assert "Starting search 4/3" not in caplog.text
    assert "confirmed outlook" not in caplog.text
    assert [sum(isinstance(m, ToolMessage) for m in messages) for messages in seen] == [1, 2, 3]
    assert result["status"] == "insufficient"
    assert "Reported development." in result["answer"]
    assert "Could not establish: confirmed outlook" in result["answer"]
    assert result["citations"] == ["https://example.com/story?b=2&a=1"]


async def test_incomplete_evidence_without_focused_query_stops_early():
    agent, _, calls = make_agent(
        [call("search_financial_news", {"query": "initial"}, "first")],
        batches=[evidence()],
        assessments=[EvidenceAssessment(sufficient=False, missing_aspects=["future announcement"])],
    )
    result = await agent.ainvoke({"query": "What has been reported and what is not known?"})
    assert calls == ["initial"]
    assert result["status"] == "insufficient"
    assert "future announcement" in result["answer"]


@pytest.mark.parametrize(
    "bad_text,bad_ids",
    [
        ("Invented report https://invented.example/story", ["E1"]),
        ("Invented report [source](https://invented.example)", ["E1"]),
        ("Invented claim [E99]", ["E1"]),
        ("Unknown evidence reference", ["E99"]),
    ],
)
async def test_invalid_claim_is_omitted_but_valid_claim_and_trusted_url_remain(bad_text, bad_ids):
    final = NewsAgentResult(
        status="answered",
        citations=["https://invented.example"],
        claims=[
            NewsClaim(text="Supported reported development.", evidence_ids=["E1"]),
            NewsClaim(text=bad_text, evidence_ids=bad_ids),
        ],
    )
    agent, _, _ = make_agent(
        [call("search_financial_news", {"query": "news"}, "search")], final, [evidence()]
    )
    result = await agent.ainvoke({"query": "Summarize reported developments"})
    assert result["status"] == "insufficient"
    assert "Supported reported development." in result["answer"]
    assert bad_text not in result["answer"]
    assert "invented.example" not in str(result)
    assert "citations could not be verified" in result["answer"]
    assert result["citations"] == ["https://example.com/story?b=2&a=1"]


async def test_synthesis_receives_evidence_blocks_without_source_urls():
    agent, _, _ = make_agent(
        [call("search_financial_news", {"query": "news"}, "search")], batches=[evidence()]
    )
    formatter = agent._formatter
    seen = []

    async def record(messages):
        seen.extend(messages)
        return await formatter.ainvoke(messages)

    agent._formatter = RunnableLambda(record)
    result = await agent.ainvoke({"query": "Summarize the event"})
    text = "\n".join(str(message.content) for message in seen)
    assert "[E1]\nTitle: Report\nSource: Example\nPublished: Unavailable\nEvidence: " in text
    assert "https://example.com/story?b=2&a=1" not in text
    assert result["citations"] == ["https://example.com/story?b=2&a=1"]


async def test_structured_assessment_stops_after_second_sufficient_search():
    agent, model, calls = make_agent(
        [
            call("search_financial_news", {"query": "company report"}, "one"),
            call("search_financial_news", {"query": "must not run"}, "blocked"),
        ],
        batches=[evidence(), evidence("E2", "https://example.com/outlook")],
    )
    assessments = [
        {
            "sufficient": False,
            "missing_aspects": ["earnings outlook"],
            "next_search_query": "company earnings outlook",
        },
        {"sufficient": True, "missing_aspects": [], "next_search_query": "unused query"},
    ]
    agent._assessor = RunnableLambda(lambda _: assessments.pop(0))
    result = await agent.ainvoke({"query": "Summarize the company report and outlook."})
    assert result["status"] == "answered"
    assert calls == ["company report", "company earnings outlook"]
    assert len(model.responses) == 1
    assert not assessments


async def test_malformed_assessment_returns_safe_error_without_more_searches():
    agent, _, calls = make_agent(
        [call("search_financial_news", {"query": "news"}, "one")], batches=[evidence()]
    )
    agent._assessor = RunnableLambda(lambda _: {"missing_aspects": ["outlook"]})
    result = await agent.ainvoke({"query": "Summarize the news"})
    assert calls == ["news"]
    assert result["status"] == "error"
    assert result["errors"] == ["ValidationError"]
    assert result["citations"] == []


async def test_conflicting_id_url_mapping_is_rejected():
    agent, _, _ = make_agent(
        [call("search_financial_news", {"query": "news"}, "one")],
        batches=[evidence(), evidence("E1", "https://different.example/story")],
        assessments=[
            EvidenceAssessment(
                sufficient=False, missing_aspects=["outlook"], next_search_query="outlook"
            ),
            EvidenceAssessment(sufficient=True),
        ],
    )
    result = await agent.ainvoke({"query": "Summarize the event and outlook"})
    assert result["status"] == "error"
    assert result["citations"] == []


def test_material_claim_requires_citation_ids():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        NewsClaim(text="Material current-news claim", evidence_ids=[])


async def test_focused_search_preserves_requested_calendar_dates():
    agent, _, _ = make_agent(
        [
            call(
                "search_financial_news",
                {"query": "initial", "start_date": "2026-09-01", "end_date": "2026-09-30"},
                "first",
            )
        ],
        assessments=[
            EvidenceAssessment(
                sufficient=False, missing_aspects=["outlook"], next_search_query="focused outlook"
            ),
            EvidenceAssessment(sufficient=True),
        ],
    )
    seen = []

    @tool("search_financial_news", response_format="content_and_artifact")
    def search(query: str, start_date: str, end_date: str) -> tuple[dict, dict]:
        """Mock timeframe-aware search."""
        seen.append((query, start_date, end_date))
        payload = evidence()
        return payload, payload

    from langgraph.prebuilt import ToolNode

    agent._tool_node = ToolNode([search], handle_tool_errors=True)
    result = await agent.ainvoke({"query": "Summarize last calendar month news and outlook"})
    assert result["status"] == "answered"
    assert seen == [
        ("initial", "2026-09-01", "2026-09-30"),
        ("focused outlook", "2026-09-01", "2026-09-30"),
    ]


async def test_news_runtime_logs_safe_exception_category(caplog):
    from unittest.mock import AsyncMock

    agent, _, _ = make_agent([])
    agent.agent = AsyncMock()
    agent.agent.ainvoke.side_effect = RuntimeError("private-provider-error-sentinel")
    with caplog.at_level("ERROR", logger="finassist.agents.news_synthesizer.agent"):
        result = await agent.ainvoke({"query": "Recent technology stocks and interest rates news"})
    assert result["status"] == "error"
    assert "category=RuntimeError" in caplog.text
    assert "private-provider-error-sentinel" not in caplog.text


async def test_article_cards_expose_only_grounded_reporting_metadata():
    final = NewsAgentResult(
        status="answered",
        claims=[
            NewsClaim(text="Reported development.", evidence_ids=["E1"]),
            NewsClaim(text="A possible implication.", evidence_ids=["E1"], kind="interpretation"),
        ],
    )
    agent, _, _ = make_agent(
        [call("search_financial_news", {"query": "earnings news"}, "search")],
        final=final,
        batches=[evidence()],
    )
    result = await agent.ainvoke({"query": "Recent earnings news"})
    articles = result["calculations"][0]["articles"]
    assert len(articles) == 1
    assert articles[0] == {
        "title": "Report",
        "url": "https://example.com/story?b=2&a=1",
        "publisher": "Example",
        "published_at": None,
        "summary": "Reported development.",
    }


def test_news_structured_results_use_function_calling(monkeypatch):
    methods = []
    original = NewsModel.with_structured_output

    def structured(model, schema, **kwargs):
        methods.append((schema, kwargs.get("method")))
        return original(model, schema, **kwargs)

    monkeypatch.setattr(NewsModel, "with_structured_output", structured)
    make_agent([])
    assert methods == [
        (EvidenceAssessment, "function_calling"),
        (NewsAgentResult, "function_calling"),
    ]
