"""Tools for querying the indexed tax education corpus."""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from finassist.agents.common.rag_tools import rerank_evidence, tax_citations
from finassist.agents.common.schemas import EvidenceResult
from finassist.rag.tax_education import build_tax_metadata_filter, extract_tax_metadata


class TaxRagQueryInput(BaseModel):
    query: str = Field(
        min_length=1,
        description="Original question, including user clarification, for constraints.",
    )
    search_query: str | None = Field(
        default=None, min_length=1, description="Optional reformulated retrieval question."
    )
    dense_queries: list[str] = Field(default_factory=list, max_length=3)
    sparse_keywords: list[str] | None = Field(default=None)


def create_tax_rag_query_tool(retriever, *, reranker, rerank_top_n: int = 5) -> StructuredTool:
    """Bind tax search while preserving constraints from the original question."""

    async def tax_rag_query(
        query: str,
        search_query: str | None = None,
        dense_queries: list[str] | None = None,
        sparse_keywords: list[str] | None = None,
    ) -> dict:
        metadata = extract_tax_metadata(query)
        effective_query = search_query or query
        try:
            documents = await retriever.search(
                effective_query,
                limit=20,
                metadata_filter=build_tax_metadata_filter(metadata),
                dense_queries=dense_queries or [effective_query],
                sparse_query=" ".join(
                    sparse_keywords if sparse_keywords is not None else [effective_query]
                ),
            )
            # Enforce explicit requested years before reranking or exposing evidence.
            documents = [
                doc
                for doc in documents
                if all(
                    str(doc.metadata.get(field)) == str(metadata[field])
                    for field in ("tax_year", "filing_year")
                    if field in metadata
                )
            ]
            return await rerank_evidence(
                query=query,
                documents=documents,
                metadata=metadata,
                errors=[],
                reranker=reranker,
                limit=rerank_top_n,
                citations=tax_citations,
            )
        except Exception as error:
            return EvidenceResult(
                metadata_filters=metadata, errors=[type(error).__name__]
            ).model_dump()

    return StructuredTool.from_function(
        coroutine=tax_rag_query,
        name="tax_rag_query",
        description=(
            "Search the US tax education corpus using semantic queries "
            "and lexical keywords. "
            "Always supply the original user question as query; use search_query for retries. "
            "Tax-year, filing-year, filing-status, and other supported metadata constraints "
            "come from the original question and are preserved across reformulations. "
            "Returns evidence passages with source metadata and retrieval errors."
        ),
        args_schema=TaxRagQueryInput,
    )
