"""
Session-only vector store for report-level follow-up context.

The Streamlit app stores each generated report in an in-memory Chroma
collection so short follow-up questions can retrieve the latest analysis
summary without reprocessing the original log files.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from chromadb.config import Settings

from llm_factory import get_embeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


EMBEDDING_SAFE_MAX_CHARS = 6000
EMBEDDING_CHUNK_OVERLAP_CHARS = 400


class ChatVectorStore:
    """
    In-memory Chroma store for analysis summaries and reports.

    Args:
        collection_name: Chroma collection name
    """

    def __init__(
        self,
        collection_name: str = "iam_log_analysis_context",
    ) -> None:
        self.collection_name = collection_name
        self.embeddings = get_embeddings()
        self.store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            client_settings=Settings(anonymized_telemetry=False, is_persistent=False),
        )

    def _split_text_for_embedding(self, text: str) -> list[str]:
        """
        Split text into conservative character-bounded chunks for embedding.

        Args:
            text: Input text to split

        Returns:
            Ordered text chunks
        """
        stripped = text.strip()
        if not stripped:
            return []

        if len(stripped) <= EMBEDDING_SAFE_MAX_CHARS:
            return [stripped]

        chunks: list[str] = []
        step = EMBEDDING_SAFE_MAX_CHARS - EMBEDDING_CHUNK_OVERLAP_CHARS
        start = 0

        while start < len(stripped):
            end = min(start + EMBEDDING_SAFE_MAX_CHARS, len(stripped))
            chunks.append(stripped[start:end])
            if end >= len(stripped):
                break
            start += step

        return chunks

    def add_report(
        self,
        query_text: str,
        report_text: str,
        metadata: dict[str, Any],
    ) -> str:
        """
        Add an analysis report as retrievable context.

        Args:
            query_text: Original user query text
            report_text: Final report text
            metadata: Additional metadata (category, file path, etc.)

        Returns:
            Analysis record identifier
        """
        timestamp = datetime.utcnow().isoformat()
        analysis_id = f"analysis_{timestamp}"

        report_chunks = self._split_text_for_embedding(report_text)
        if not report_chunks:
            report_chunks = ["(empty report)"]

        docs: list[Document] = []
        total_chunks = len(report_chunks)

        for chunk_index, chunk_text in enumerate(report_chunks):
            payload = (
                f"Query: {query_text}\n"
                f"Timestamp: {timestamp}\n"
                f"Chunk: {chunk_index + 1}/{total_chunks}\n"
                f"Report:\n{chunk_text}"
            )
            docs.append(
                Document(
                    page_content=payload,
                    metadata={
                        "analysis_id": analysis_id,
                        "query_text": query_text,
                        "created_at": timestamp,
                        "chunk_index": chunk_index,
                        "chunk_total": total_chunks,
                        **metadata,
                    },
                )
            )

        self.store.add_documents(docs)
        return analysis_id

    def retrieve_context(
        self,
        query_text: str,
        k: int = 3,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Retrieve similar past analyses for chat context.

        Args:
            query_text: User query
            k: Number of contexts
            metadata_filter: Optional metadata filter

        Returns:
            Retrieved document list
        """
        if metadata_filter:
            return self.store.similarity_search(query_text, k=k, filter=metadata_filter)
        return self.store.similarity_search(query_text, k=k)
