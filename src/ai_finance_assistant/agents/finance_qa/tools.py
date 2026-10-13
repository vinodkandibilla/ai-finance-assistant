"""Tools for querying the indexed finance handbook."""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ai_finance_assistant.agents.common.rag_tools import finance_citations, rerank_evidence
from ai_finance_assistant.agents.common.schemas import EvidenceResult
from ai_finance_assistant.agents.finance_qa.metadata_extractor import FinanceMetadataExtractor
from ai_finance_assistant.agents.finance_qa.retrieval import FinanceHybridRetriever


class FinanceRagQueryInput(BaseModel):
    query: str = Field(min_length=1, description="Question used to extract metadata constraints.")
    dense_queries: list[str] = Field(
        default_factory=list, max_length=3, description="Semantic search queries."
    )
    sparse_keywords: list[str] | None = Field(default=None, description="Lexical search terms.")


def create_finance_rag_query_tool(
    retriever: FinanceHybridRetriever,
    metadata_extractor: FinanceMetadataExtractor,
    *,
    reranker,
    rerank_top_n: int = 5,
) -> StructuredTool:
    """Bind a retrieval tool to the application's configured index dependencies."""

    async def finance_rag_query(
        query: str,
        dense_queries: list[str] | None = None,
        sparse_keywords: list[str] | None = None,
    ) -> dict:
        metadata = metadata_extractor.extract(query)
        try:
            documents, errors = await retriever.retrieve(
                dense_queries or [query],
                sparse_keywords if sparse_keywords is not None else [query],
                metadata_filter=metadata_extractor.build_qdrant_filter(metadata),
            )
            return await rerank_evidence(
                query=query,
                documents=documents,
                metadata=metadata,
                errors=errors,
                reranker=reranker,
                limit=rerank_top_n,
                citations=finance_citations,
            )
        except Exception as error:
            return EvidenceResult(
                metadata_filters=metadata, errors=[f"Finance search failed: {type(error).__name__}"]
            ).model_dump()

    return StructuredTool.from_function(
        coroutine=finance_rag_query,
        name="finance_rag_query",
        description=(
            "Search and rerank the indexed Financial Education Handbook using semantic and lexical "
            "queries with corpus-supported metadata filters. Returns evidence passages, "
            "citation metadata, and retrieval errors for subsequent evaluation."
        ),
        args_schema=FinanceRagQueryInput,
    )
