"""End-to-end smoke tests verifying the thin shims expose the original API."""

import os
import tempfile


def test_cli_smoke(mock_llm, mock_embeddings, sample_log_file):
    """Simulate CLI entry point via run_pipeline."""
    from iam_log_intelligence_agent_hybridChunking2 import run_pipeline
    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        try:
            report = run_pipeline([sample_log_file])
            assert isinstance(report, str)
            assert len(report) > 0
        finally:
            os.chdir(original_cwd)


def test_followup_query_smoke():
    """Simulate a follow-up query with fake artifacts."""
    from followup_retrieval import (
        ArtifactEntry, AnalysisContext, answer_analysis_results_query,
    )
    entry = ArtifactEntry(
        file_name="test.log",
        source_path="/tmp/test.log",
        faiss_index_dir="/tmp/faiss",
        debug_evidence_file="/tmp/debug.txt",
        metadata_rows=[],
        selected_row_ids_for_reduce=[],
        category="api_request",
        subcategory="unknown_error",
    )
    ctx = AnalysisContext(
        query_text="analyze test.log",
        log_path="/tmp/test.log",
        start_time="",
        end_time="",
        report_text="Mock report text",
        entries=[entry],
        created_at="2024-01-15T00:00:00",
    )
    answer = answer_analysis_results_query(ctx, "summary of issues", chat_history=None)
    assert isinstance(answer, str)
    assert len(answer) > 0


def test_app_py_imports_unchanged():
    """Verify app.py can still import its two public symbols."""
    from iam_log_intelligence_agent_hybridChunking2 import build_query_context, run_pipeline
    assert callable(build_query_context)
    assert callable(run_pipeline)
