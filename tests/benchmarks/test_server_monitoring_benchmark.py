"""Tests for benchmarks/server_monitoring_benchmark.py (Part A and Part B)."""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import patch

import duckdb
import pytest

from pipeline.parsing import detect_log_structure
from pipeline.server_metrics import (
    LOG_EVENTS_CREATE_SQL,
    LOG_EVENTS_FLAG_INDEX_SQL,
    LOG_EVENTS_INDEX_SQL,
    _ingest_server_log,
)

# Module under test
from benchmarks.server_monitoring_benchmark import (
    _LEGACY_LOG_EVENTS_CREATE_SQL,
    _LEGACY_LOG_EVENTS_INDEX_SQL,
    QUERY_PAIRS,
    SCHEMA_SAMPLE_SIZE,
    _PHASE_REGISTRY_KEYS,
    _ingest_both_schemas,
    _time_query,
    _warmup_query,
    run_part_a,
    run_part_b,
    _write_markdown_report,
    _write_json_report,
    main,
)


SAMPLE_LOG_LINES = [
    "2024-01-15 09:23:45.123 [main] INFO Server started on port 8080",
    "2024-01-15 09:24:01.456 [worker-1] ERROR lapse(ms)=5000 connection wait",
    "2024-01-15 09:24:02.789 [worker-1] WARN Count = 1234 rows returned",
    "2024-01-15 09:24:03.001 [worker-1] INFO - entry RoleValidator checkCredentialRole",
    "2024-01-15 09:24:03.002 [worker-1] DEBUG com.example.Service.java:142",
    "2024-01-15 09:25:10.333 [main] INFO jdbc ldap hibernate staleobject scheduled cleanupJob",
    "2024-01-15 09:26:00.000 [worker-2] WARN High memory usage: 87%",
]


@pytest.fixture
def sample_log_path(tmp_path) -> str:
    """Return a path to a temporary sample log file."""
    path = tmp_path / "sample.log"
    path.write_text("\n".join(SAMPLE_LOG_LINES) + "\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def sample_schema() -> dict:
    """Return detected schema from sample log lines."""
    return detect_log_structure(SAMPLE_LOG_LINES)


@pytest.fixture
def sample_conn(sample_log_path: str, sample_schema: dict) -> duckdb.DuckDBPyConnection:
    """Yield a DuckDB connection with both schemas ingested, then close it."""
    conn = _ingest_both_schemas(sample_log_path, sample_schema)
    try:
        yield conn
    finally:
        conn.close()


class TestSchemaConstants:
    def test_legacy_create_is_string(self) -> None:
        """Verify legacy CREATE TABLE constant is a string with expected keywords."""
        assert isinstance(_LEGACY_LOG_EVENTS_CREATE_SQL, str)
        assert "CREATE TABLE" in _LEGACY_LOG_EVENTS_CREATE_SQL
        assert "timestamp" in _LEGACY_LOG_EVENTS_CREATE_SQL
        assert "thread" in _LEGACY_LOG_EVENTS_CREATE_SQL
        assert "raw_line" in _LEGACY_LOG_EVENTS_CREATE_SQL

    def test_legacy_index_is_string(self) -> None:
        """Verify legacy CREATE INDEX constant is a string."""
        assert isinstance(_LEGACY_LOG_EVENTS_INDEX_SQL, str)
        assert "CREATE INDEX" in _LEGACY_LOG_EVENTS_INDEX_SQL


class TestQueryPairs:
    def test_query_pairs_are_non_empty_and_unique(self) -> None:
        """Ensure QUERY_PAIRS is non-empty and all query IDs are unique."""
        assert len(QUERY_PAIRS) > 0
        ids = [p[0] for p in QUERY_PAIRS]
        assert len(ids) == len(set(ids))

    def test_each_pair_is_three_tuple(self) -> None:
        """Ensure every query pair is a 3-tuple."""
        for pair in QUERY_PAIRS:
            assert isinstance(pair, tuple)
            assert len(pair) == 3

    def test_query_ids_are_non_empty_strings(self) -> None:
        """Ensure each query ID is a non-empty string."""
        for qid, _, _ in QUERY_PAIRS:
            assert isinstance(qid, str)
            assert len(qid) > 0

    def test_legacy_and_flag_sql_are_strings(self) -> None:
        """Ensure legacy and flag SQL are strings containing SELECT."""
        for qid, legacy, flag in QUERY_PAIRS:
            assert isinstance(qid, str)
            assert isinstance(legacy, str)
            assert isinstance(flag, str)
            assert "SELECT" in legacy.upper()
            assert "SELECT" in flag.upper()


class TestSchemaSampleSizeConstant:
    def test_schema_sample_size_is_positive(self) -> None:
        """Ensure SCHEMA_SAMPLE_SIZE is a positive integer."""
        assert isinstance(SCHEMA_SAMPLE_SIZE, int)
        assert SCHEMA_SAMPLE_SIZE > 0


class TestIngestBothSchemas:
    def test_creates_both_tables(self, sample_conn: duckdb.DuckDBPyConnection) -> None:
        """Verify that both legacy and current schema tables are created."""
        tables = {r[0] for r in sample_conn.execute("SHOW TABLES").fetchall()}
        assert "log_events" in tables
        assert "legacy_log_events" in tables

        cols = [r[0] for r in sample_conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'legacy_log_events'"
        ).fetchall()]
        assert set(cols) == {"timestamp", "thread", "raw_line"}

        count = sample_conn.execute("SELECT COUNT(*) FROM legacy_log_events").fetchone()[0]
        assert count > 0

    def test_ingest_both_schemas_raises_on_missing_file(self) -> None:
        """Ensure _ingest_both_schemas raises OSError for a non-existent file path."""
        with pytest.raises(OSError):
            _ingest_both_schemas("/nonexistent/path/to/file.log", {})


class TestTimingHelpers:
    def test_time_query_returns_float(self) -> None:
        """Ensure _time_query returns a non-negative float."""
        conn = duckdb.connect(":memory:")
        try:
            conn.execute("CREATE TABLE dummy (x INTEGER)")
            conn.execute("INSERT INTO dummy VALUES (1)")
            elapsed = _time_query(conn, "SELECT * FROM dummy")
            assert isinstance(elapsed, float)
            assert elapsed >= 0.0
        finally:
            conn.close()

    def test_warmup_query_runs(self) -> None:
        """Ensure _warmup_query executes without raising."""
        conn = duckdb.connect(":memory:")
        try:
            conn.execute("CREATE TABLE dummy (x INTEGER)")
            conn.execute("INSERT INTO dummy VALUES (1)")
            _warmup_query(conn, "SELECT * FROM dummy")
        finally:
            conn.close()


class TestRunPartA:
    def test_returns_results_for_all_pairs(self, sample_conn: duckdb.DuckDBPyConnection) -> None:
        """Ensure run_part_a produces a result dict for every query pair."""
        results = run_part_a(sample_conn, iterations=1)
        assert isinstance(results, list)
        assert len(results) == len(QUERY_PAIRS)

        for r in results:
            assert "query_id" in r
            assert "legacy_median_ms" in r
            assert "flag_median_ms" in r
            assert "speedup_ratio" in r
            assert isinstance(r["legacy_median_ms"], float)
            assert isinstance(r["flag_median_ms"], float)
            assert isinstance(r["speedup_ratio"], float)


class TestRunPartB:
    def test_run_part_b_skip_llm_returns_timings(self, sample_log_path: str, sample_schema: dict) -> None:
        """Verify run_part_b with skip_llm=True returns timings and positive duration."""
        timings, total_ms = run_part_b(sample_log_path, sample_schema, skip_llm=True)
        assert isinstance(timings, list)
        # This length check is only valid for the linear skip_llm=True path
        # where each phase runs exactly once. Real LLM paths may retry or branch.
        assert len(timings) == len(_PHASE_REGISTRY_KEYS)
        assert total_ms > 0
        phases = {t["phase"] for t in timings}
        assert phases == set(_PHASE_REGISTRY_KEYS)

    def test_run_part_b_no_llm_nodes_are_noop(self, sample_log_path: str, sample_schema: dict) -> None:
        """Verify skip_llm=True never invokes the LLM (get_llm patched to raise)."""
        with patch("llm_factory.get_llm", side_effect=RuntimeError("LLM called")):
            timings, total_ms = run_part_b(sample_log_path, sample_schema, skip_llm=True)
            assert isinstance(timings, list)
            assert total_ms > 0


class TestReportWriters:
    def test_markdown_report_written(self) -> None:
        """Ensure _write_markdown_report produces a readable Markdown file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.md")
            results = [
                {
                    "query_id": "test_query",
                    "legacy_median_ms": 10.0,
                    "flag_median_ms": 1.0,
                    "speedup_ratio": 10.0,
                }
            ]
            _write_markdown_report(results, path)
            assert os.path.exists(path)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "test_query" in content
            assert "10.0" in content

    def test_json_report_written(self) -> None:
        """Ensure _write_json_report produces a valid JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.json")
            results = [
                {
                    "query_id": "test_query",
                    "legacy_median_ms": 10.0,
                    "flag_median_ms": 1.0,
                    "speedup_ratio": 10.0,
                }
            ]
            _write_json_report(results, path)
            assert os.path.exists(path)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data[0]["query_id"] == "test_query"
            assert data[0]["speedup_ratio"] == 10.0


class TestMain:
    def test_argparse_parses_args(self, sample_log_path: str) -> None:
        """Verify main writes reports when given a valid log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = os.path.join(tmpdir, "out")
            test_args = [
                "prog",
                sample_log_path,
                "--output-dir", out_dir,
                "--iterations", "1",
            ]
            with patch("sys.argv", test_args):
                main()
            assert os.path.exists(out_dir)
            md_path = os.path.join(out_dir, "benchmark_report.md")
            json_path = os.path.join(out_dir, "benchmark_report.json")
            assert os.path.exists(md_path)
            assert os.path.exists(json_path)

    def test_main_exits_on_missing_file(self) -> None:
        """Verify main exits with code 1 when the log file does not exist."""
        test_args = [
            "prog",
            "/nonexistent/path/to/file.log",
            "--iterations", "1",
        ]
        with patch("sys.argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 1
