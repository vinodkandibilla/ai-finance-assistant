"""Provider-backed smoke evaluation using isolated Qdrant and real corpus.

Run: .venv/bin/python scripts/evaluate_tax_live.py
Requires OPENAI_API_KEY; incurs embedding/model calls and may download models.
Reports workflow checks, not independent factual or tax-law validation.
"""

import asyncio
import json
import time

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from qdrant_client import QdrantClient

from finassist.agents.finance_qa.reranker import FinanceCrossEncoderReranker
from finassist.agents.tax_education.graph import TaxEducationAgent
from finassist.core.config import get_settings
from finassist.rag.tax_education import get_tax_retriever


async def main():
    settings = get_settings()
    if settings.openai_api_key is None:
        raise SystemExit("OPENAI_API_KEY is required; no live evaluation performed.")
    llm = ChatOpenAI(
        model=settings.openai_model, api_key=settings.openai_api_key.get_secret_value()
    )
    client = QdrantClient(location=":memory:")
    report = []
    try:
        retriever = get_tax_retriever(client, llm)
        agent = TaxEducationAgent(
            checkpointer=MemorySaver(),
            retriever=retriever,
            llm=llm,
            reranker=FinanceCrossEncoderReranker(settings.finance_reranker_model),
        )
        for query in ("What is a standard deduction?", "Explain deductions for tax year 1900."):
            started = time.monotonic()
            result = await agent.graph.ainvoke(
                {"query": query}, config={"configurable": {"thread_id": query}}
            )
            interrupted = bool(result.get("__interrupt__"))
            report.append(
                {
                    "query": query,
                    "status": "clarification" if interrupted else result.get("status"),
                    "retry_count": result.get("retry_count", 0),
                    "evaluation": result.get("best_evaluation", {}),
                    "citations": result.get("citations", []),
                    "latency_seconds": round(time.monotonic() - started, 2),
                    "bounded_retry": result.get("retry_count", 0) <= 1,
                    "answer": result.get("answer", ""),
                }
            )
        print(json.dumps(report, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    asyncio.run(main())
