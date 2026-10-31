from langchain_core.callbacks import AsyncCallbackHandler

from finassist.api import query as api
from finassist.core.config import Settings
from finassist.workflow.builder import build_graph


class Classifier:
    async def classify(self, query):
        return "finance_qa"


class Tags(AsyncCallbackHandler):
    def __init__(self):
        self.seen = []

    async def on_chain_start(self, serialized, inputs, *, tags=None, **kwargs):
        self.seen.append(set(tags or []))


def test_source_and_configuration(monkeypatch):
    monkeypatch.setattr(api, "get_settings", lambda: Settings(
        _env_file=None, trace_app_tag="demo", trace_environment="showcase"
    ))
    assert api._trace_config("web-chat")["tags"] == [
        "demo", "env:showcase", "source:web-chat"
    ]
    assert api._trace_config("untrusted-arbitrary-tag")["tags"][-1] == "source:api"
    monkeypatch.setattr(api, "get_settings", lambda: Settings(
        _env_file=None, trace_tags_enabled=False
    ))
    assert api._trace_config("web-chat") == {}


async def test_agent_tags_are_scoped_and_opt_in():
    graph = build_graph(Classifier())
    callback = Tags()
    await graph.ainvoke({"query": "What is a stock?"}, config={
        "tags": ["finpilot", "source:api"],
        "metadata": {"finpilot_agent_tags": True}, "callbacks": [callback],
    })
    for agent in ("router", "finance"):
        assert any({f"agent:{agent}", "finpilot", "source:api"} <= t for t in callback.seen)
    assert not any({"agent:router", "agent:finance"} <= t for t in callback.seen)
    callback = Tags()
    await graph.ainvoke({"query": "What is a stock?"}, config={"callbacks": [callback]})
    assert not any(tag.startswith("agent:") for tags in callback.seen for tag in tags)


async def test_request_trace_groups_graph_and_output_check(monkeypatch):
    from langsmith import get_current_run_tree, tracing_context

    from finassist.guardrails.pii import RedactionResult

    parents = []

    class Graph:
        async def ainvoke(self, state, config):
            parents.append(get_current_run_tree().id)
            return {"intent": "finance_qa", "answer": "An ETF is a fund."}

    async def no_interrupt(*args):
        return None

    monkeypatch.setattr(api, "redact_pii", lambda text: RedactionResult(text, ()))
    monkeypatch.setattr(api, "_get_interrupt_payload", no_interrupt)
    monkeypatch.setattr(api, "_trace_usage_summary", lambda result: None)
    monkeypatch.setattr(
        api, "_trace_guardrail", lambda *args, **kwargs: parents.append(get_current_run_tree().id)
    )
    with tracing_context(enabled="local"):
        response = await api.query_assistant(
            api.QueryRequest(query="What is an ETF?"), Graph(), None, "api"
        )
    assert response.answer == "An ETF is a fund."
    assert len(parents) == 2
    assert parents[0] == parents[1]


def test_request_preview_contains_only_redacted_user_text(monkeypatch):
    from finassist.guardrails.pii import RedactionResult

    monkeypatch.setattr(
        api,
        "redact_pii",
        lambda text: RedactionResult("[REDACTED_EMAIL_ADDRESS]", ("EMAIL_ADDRESS",)),
    )
    for request, field in [
        (api.QueryRequest(query="alice@example.com"), "query"),
        (api.ResumeRequest(thread_id="private-thread", reply="alice@example.com"), "reply"),
    ]:
        assert api._trace_request_inputs(
            {"request": request, "graph": object(), "user_context": "private-context"}
        ) == {field: "[REDACTED_EMAIL_ADDRESS]"}


def test_response_preview_excludes_structured_payloads(monkeypatch):
    from finassist.guardrails.pii import RedactionResult

    monkeypatch.setattr(
        api, "redact_pii", lambda text: RedactionResult("Safe answer", ())
    )
    response = api.QueryResponse(
        intent="portfolio",
        answer="Sensitive answer",
        status="answered",
        calculation={"private": "account data"},
        thread_id="private-thread",
    )
    assert api._trace_response_outputs(response) == {
        "answer": "Safe answer", "intent": "portfolio", "status": "answered"
    }
