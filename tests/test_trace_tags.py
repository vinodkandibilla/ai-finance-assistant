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
