from langchain_core.documents import Document

import pipeline.evidence as e


def test_select_evidence_chunks_budget(sample_schema, mock_embeddings):
    docs = [
        Document(
            page_content="x" * 1000,
            metadata={
                "source_file": "a.log",
                "line_ranges": [i],
                "anomaly_score": float(i),
                "primary_key": str(i),
            },
        )
        for i in range(10)
    ]
    evidence_text, row_ids, refs = e.select_evidence_chunks(
        docs, top_n=5, max_total_chars=2000
    )
    assert len(refs) <= 5
