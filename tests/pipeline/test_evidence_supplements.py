"""Tests for deterministic evidence supplement queries."""

import os
import tempfile

from pipeline.server_metrics import load_server_metrics_into_duckdb
from pipeline.server_sql.evidence_supplements import (
    build_evidence_supplement_queries,
    run_evidence_supplement_queries,
)


def _make_schema():
    import re

    return {
        "timestamp_re": re.compile(r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)"),
        "timestamp_fmt": "%Y-%m-%d %H:%M:%S.%f",
        "thread_re": re.compile(r"\[(\S+)\]"),
        "session_re": None,
        "stack_trace_re": None,
        "timestamp_group": 1,
        "thread_group": 1,
        "session_group": None,
        "has_timestamp": True,
        "has_thread": True,
        "has_session": False,
    }


def test_build_evidence_supplement_queries_uses_onset_anchor():
    onset = {"degradation_start": "2026-02-25T15:51:35.286000"}
    queries = build_evidence_supplement_queries(onset, [])
    assert len(queries) == 3
    labels = {label for label, _ in queries}
    assert "onset_minute_metrics" in labels
    assert "onset_extreme_latencies" in labels
    assert "onset_backend_dependency_signals" in labels
    sql_blob = " ".join(sql for _, sql in queries)
    assert "2026-02-25 15:51:35" in sql_blob
    assert "time_bucket" in sql_blob


def test_run_evidence_supplement_queries_returns_rows():
    lines = [
        "2026-02-25 15:51:35.286000 [main] INFO Server statistics={jvm.threadCount=344, dbcp.ActiveConnections=8}",
        "2026-02-25 15:52:13.036000 [exec-1] DEBUG REST:authn/login - exit,lapse(ms)=97550",
        "2026-02-25 15:52:14.000000 [exec-2] WARN Hibernate connection wait detected",
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "server.log")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

        conn = load_server_metrics_into_duckdb(
            path, _make_schema(), db_path=os.path.join(tmpdir, "test.duckdb")
        )
        try:
            onset = {"degradation_start": "2026-02-25T15:51:35.286000"}
            results = run_evidence_supplement_queries(conn, onset, [])
            assert len(results) == 3
            latency = next(r for r in results if r[0] == "onset_extreme_latencies")
            assert latency[3] >= 1
            assert "97550" in latency[2]
        finally:
            conn.close()