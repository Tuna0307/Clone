"""Unit tests for followup.sources module."""
import tempfile

import followup.sources as fs
from followup.context import AnalysisContext, ArtifactEntry


def test_parse_debug_evidence_file():
    content = "[REF_001] Some evidence\n---\n[REF_002] More evidence"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        path = f.name
    import os
    try:
        result = fs.parse_debug_evidence_file(path)
        assert "raw" in result
        assert "REF_001" in result["raw"]
        assert "Some evidence" in result["raw"]
    finally:
        os.unlink(path)


def test_extract_ref_ids():
    ids = fs._extract_ref_ids("See [REF_001] and [REF_002] for details")
    assert ids == ["REF_001", "REF_002"]


class StreamingOnlyFile:
    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def __iter__(self):
        return iter(self._lines)

    def readlines(self):
        raise AssertionError("_raw_log_candidates should stream source logs instead of reading all lines")


def test_raw_log_candidates_stream_source_file(monkeypatch, tmp_path):
    source_path = tmp_path / "large.log"
    source_path.write_text("placeholder", encoding="utf-8")
    context = AnalysisContext(
        query_text="analyze",
        log_path=str(source_path),
        start_time="",
        end_time="",
        report_text="report",
        entries=[
            ArtifactEntry(
                file_name="large.log",
                source_path=str(source_path),
                faiss_index_dir="",
                debug_evidence_file="",
                metadata_rows=[],
                selected_row_ids_for_reduce=[],
                category="server_monitoring",
                subcategory="",
            )
        ],
        created_at="2024-01-15T00:00:00",
    )

    def streaming_open(*args, **kwargs):
        return StreamingOnlyFile([
            "INFO start\n",
            "ERROR auth failed\n",
            "INFO after\n",
        ])

    monkeypatch.setattr(fs, "open", streaming_open, raising=False)

    candidates = fs._raw_log_candidates(context, ["auth"])

    assert candidates
    assert "ERROR auth failed" in candidates[0].raw_text
