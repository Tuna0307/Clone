from langchain_core.documents import Document

import pipeline.dedup as d


def test_build_metadata_rows_from_docs():
    docs = [
        Document(page_content="error", metadata={"line_ranges": [1, 2], "source_file": "a.log"}),
        Document(page_content="ok", metadata={"line_ranges": [3], "source_file": "a.log"}),
    ]
    rows = d.build_metadata_rows_from_docs(docs)
    assert len(rows) == 2
    assert "line_ranges" in rows[0]