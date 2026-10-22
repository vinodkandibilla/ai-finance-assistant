"""Tools for querying the indexed finance handbook."""

from langchain_core.tools import StructuredTool
from langsmith import trace
from pydantic import BaseModel, Field

from finassist.agents.common.rag_tools import finance_citations, rerank_evidence
from finassist.agents.common.schemas import EvidenceResult
from finassist.agents.finance_qa.metadata_extractor import FinanceMetadataExtractor
from finassist.agents.finance_qa.retrieval import FinanceHybridRetriever


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
        with trace(
            "Extract finance metadata", run_type="chain", inputs={"query_length": len(query)}
        ) as span:
            metadata = metadata_extractor.extract(query)
            metadata_filter = metadata_extractor.build_qdrant_filter(metadata)
            span.add_outputs({"filter_fields": sorted(metadata)})
        try:
            with trace(
                "Retrieve finance candidates",
                run_type="retriever",
                inputs={"dense_query_count": len(dense_queries or [query])},
            ):
                documents, errors = await retriever.retrieve(
                    dense_queries or [query],
                    sparse_keywords if sparse_keywords is not None else [query],
                    metadata_filter=metadata_filter,
                )
            with trace(
                "Rerank and prepare finance evidence"
                if reranker is not None
                else "Select and prepare finance evidence",
                run_type="retriever",
                inputs={"candidate_count": len(documents), "limit": rerank_top_n},
            ):
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
