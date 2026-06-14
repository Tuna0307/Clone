"""Unit tests for followup.context module."""
from datetime import datetime

import followup.context as fc


def test_artifact_entry():
    entry = fc.ArtifactEntry(
        file_name="test.log",
        source_path="/tmp/test.log",
        faiss_index_dir="/tmp/faiss",
        debug_evidence_file="/tmp/debug.txt",
        metadata_rows=[],
        selected_row_ids_for_reduce=[],
        category="",
        subcategory="",
    )
    assert entry.file_name == "test.log"


def test_build_analysis_context():
    ctx = fc.build_analysis_context(
        query_text="analyze",
        log_path="/tmp",
        start_time="",
        end_time="",
        report_text="report",
        per_file_reports=[
            {
                "file": "test.log",
                "source_path": "/tmp/test.log",
                "faiss_index_dir": "/tmp/faiss",
                "debug_evidence_file": "/tmp/debug.txt",
                "metadata_rows": [],
                "selected_row_ids_for_reduce": [],
                "evidence_profile": {"total_lines": 123},
                "category": "",
                "subcategory": "",
            },
        ],
    )
    assert len(ctx.entries) == 1
    assert ctx.entries[0].file_name == "test.log"
    assert ctx.entries[0].evidence_profile["total_lines"] == 123


def test_build_analysis_context_wires_server_monitoring_counts():
    ctx = fc.build_analysis_context(
        query_text="analyze",
        log_path="/tmp",
        start_time="",
        end_time="",
        report_text="report",
        per_file_reports=[
            {
                "file": "server.log",
                "source_path": "/tmp/server.log",
                "faiss_index_dir": None,
                "debug_evidence_file": "/tmp/debug.txt",
                "metadata_rows": [],
                "selected_row_ids_for_reduce": [],
                "category": "server_monitoring",
                "subcategory": "server_monitoring",
                "duckdb_row_count": 42,
                "log_event_row_count": 9001,
            },
        ],
    )
    entry = ctx.entries[0]
    assert entry.duckdb_row_count == 42
    assert entry.log_event_row_count == 9001
    assert "None" not in entry.faiss_index_dir


def test_try_parse_datetime():
    result = fc._try_parse_datetime("2024-01-15T09:30:00")
    assert result == datetime(2024, 1, 15, 9, 30, 0)


def test_metadata_markdown_table_separator_matches_header():
    ctx = fc.AnalysisContext(
        query_text="analyze",
        log_path="/tmp",
        start_time="",
        end_time="",
        report_text="report",
        entries=[
            fc.ArtifactEntry(
                file_name="test.log",
                source_path="/tmp/test.log",
                faiss_index_dir="/tmp/faiss",
                debug_evidence_file="/tmp/debug.txt",
                metadata_rows=[
                    {
                        "row_id": "test.log::1",
                        "anomaly_score": 3.2,
                        "primary_key": "thread-1",
                        "start_time": "2024-01-15T09:30:00",
                        "end_time": "2024-01-15T09:31:00",
                        "content": "ERROR failed auth",
                    }
                ],
                selected_row_ids_for_reduce=[],
                category="server_monitoring",
                subcategory="",
            )
        ],
        created_at="2024-01-15T00:00:00",
    )

    lines = fc.build_analysis_results_metadata_markdown(ctx).splitlines()
    header = next(line for line in lines if line.startswith("| File |"))
    separator = lines[lines.index(header) + 1]

    assert header.count("|") == separator.count("|")


def test_metadata_markdown_escapes_pipe_characters_in_cells():
    ctx = fc.AnalysisContext(
        query_text="analyze",
        log_path="/tmp",
        start_time="",
        end_time="",
        report_text="report",
        entries=[
            fc.ArtifactEntry(
                file_name="test|file.log",
                source_path="/tmp/test.log",
                faiss_index_dir="/tmp/faiss",
                debug_evidence_file="/tmp/debug.txt",
                metadata_rows=[
                    {
                        "anomaly_score": 3.2,
                        "primary_key": "thread|1",
                        "start_time": "2024-01-15T09:30:00",
                        "end_time": "2024-01-15T09:31:00",
                        "content": "ERROR auth | failed",
                    }
                ],
                selected_row_ids_for_reduce=[],
                category="server_monitoring",
                subcategory="",
            )
        ],
        created_at="2024-01-15T00:00:00",
    )

    rows = [
        line for line in fc.build_analysis_results_metadata_markdown(ctx).splitlines()
        if line.startswith("| test")
    ]

    assert "\\|" in rows[0]


def test_build_coverage_summary_table_data_counts_selected_line_buckets():
    ctx = fc.AnalysisContext(
        query_text="analyze",
        log_path="/tmp",
        start_time="",
        end_time="",
        report_text="report",
        entries=[
            fc.ArtifactEntry(
                file_name="large.log",
                source_path="/tmp/large.log",
                faiss_index_dir="/tmp/faiss",
                debug_evidence_file="/tmp/debug.txt",
                metadata_rows=[
                    {
                        "row_id": "large.log::0",
                        "start_line": 10,
                        "end_line": 12,
                        "line_ranges": "10-12",
                    },
                    {
                        "row_id": "large.log::1",
                        "start_line": 75,
                        "end_line": 80,
                        "line_ranges": "75-80",
                        "error_line_ranges": "77",
                    },
                    {
                        "row_id": "large.log::2",
                        "start_line": 900,
                        "end_line": 910,
                        "line_ranges": "900-910",
                    },
                ],
                selected_row_ids_for_reduce=["large.log::0", "large.log::1", "large.log::2"],
                category="server_monitoring",
                subcategory="",
                evidence_profile={
                    "total_lines": 1000,
                    "time_range": {
                        "start": "2025-09-19T00:00:00",
                        "end": "2025-09-19T23:59:59",
                    },
                },
            )
        ],
        created_at="2024-01-15T00:00:00",
    )

    result = fc.build_coverage_summary_table_data(ctx)

    assert result["files"][0]["total_lines_scanned"] == 1000
    assert result["files"][0]["selected_evidence_items"] == 3
    assert result["files"][0]["earliest_selected_line"] == 10
    assert result["files"][0]["latest_selected_line"] == 910
    assert result["buckets"] == [
        {"file_name": "large.log", "line_range": "1-250", "selected_evidence_items": 2},
        {"file_name": "large.log", "line_range": "251-500", "selected_evidence_items": 0},
        {"file_name": "large.log", "line_range": "501-750", "selected_evidence_items": 0},
        {"file_name": "large.log", "line_range": "751-1000", "selected_evidence_items": 1},
    ]
