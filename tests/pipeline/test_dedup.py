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


def test_deduplicate_chunks_safe():
    schema = {"timestamp_re": None, "timestamp_fmt": None}
    docs = [
        Document(page_content="same content", metadata={"source_file": "a.log", "line_ranges": [1]}),
        Document(page_content="same content", metadata={"source_file": "a.log", "line_ranges": [2]}),
        Document(page_content="different", metadata={"source_file": "a.log", "line_ranges": [3]}),
    ]
    result = d.deduplicate_chunks_safe(docs, schema)
    assert len(result) <= 2


def test_filter_chunks_by_signal():
    docs = [
        Document(page_content="ERROR: fail", metadata={"source_file": "a.log", "line_ranges": [1]}),
        Document(page_content="INFO: ok", metadata={"source_file": "a.log", "line_ranges": [2]}),
    ]
    result = d.filter_chunks_by_signal(docs)
    assert len(result) >= 1
