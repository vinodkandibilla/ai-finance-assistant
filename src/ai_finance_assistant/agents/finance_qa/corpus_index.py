from __future__ import annotations

import hashlib
import re
from functools import lru_cache

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from ai_finance_assistant.agents.finance_qa.config import FINANCE_PDF_PATH, FinanceAgentSettings

OUT_OF_SCOPE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:roth|traditional|sep|simple)\s+ira\b",
        r"\b401\s*\(?k\)?\b",
        r"\bretirement\b",
        r"\b(?:tax|taxation|irs|deduction|filing status|tax credit)\b",
        r"\b(?:immigration|visa|resident alien|nonresident alien)\b",
    )
)


def load_finance_corpus(
    pdf_path=FINANCE_PDF_PATH,
    settings: FinanceAgentSettings | None = None,
) -> list[Document]:
    """Extract in-scope pages, chunk them deterministically, and retain citations."""
    resolved_settings = settings or FinanceAgentSettings()
    reader = PdfReader(str(pdf_path))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=resolved_settings.chunk_size,
        chunk_overlap=resolved_settings.chunk_overlap,
    )
    chunks: list[Document] = []
    source = str(pdf_path)
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text or any(pattern.search(text) for pattern in OUT_OF_SCOPE_PATTERNS):
            continue
        page_document = Document(
            page_content=text,
            metadata={
                "source": source,
                "document_title": "Financial Education Handbook",
                "page_number": page_number,
            },
        )
        for chunk_index, chunk in enumerate(splitter.split_documents([page_document]), start=1):
            chapter_match = re.search(r"(?im)^\s*chapter\s+([\w.-]+)", chunk.page_content)
            section_match = re.search(r"(?im)^\s*(\d+\.\d+(?:\.\d+)*)\b", chunk.page_content)
            chunk.metadata.update(
                chapter=chapter_match.group(1) if chapter_match else "",
                section=section_match.group(1) if section_match else "",
                chunk_id=f"page-{page_number:04d}-chunk-{chunk_index:03d}",
            )
            identity = f"{source}:{page_number}:{chunk_index}:{chunk.page_content}"
            chunk.metadata["doc_id"] = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            chunks.append(chunk)
    return chunks


@lru_cache(maxsize=1)
def get_finance_corpus() -> tuple[Document, ...]:
    return tuple(load_finance_corpus())