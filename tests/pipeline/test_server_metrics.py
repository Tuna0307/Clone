import os
import tempfile

import duckdb
import pytest

import pandas as pd

from pipeline.server_metrics import (
    LOG_EVENTS_CREATE_SQL,
    SERVER_METRICS_CREATE_SQL,
    format_duckdb_observation_bounds,
    format_query_dataframe,
    get_duckdb_observation_bounds,
    get_sql_safety_rejection_reason,
    is_safe_select,
    load_server_metrics_into_duckdb,
    load_server_metrics_into_duckdb_with_signals,
    normalize_llm_sql,
    pre_detect_high_signal_events,
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


def test_metric_gate_skips_non_metric_lines():
    """Lines without 'Server statistics' or 'msg=' should not produce metric rows."""
    lines = [
        "2024-01-15 09:00:00.000 [main] INFO falseMetric=999",
        "2024-01-15 09:00:01.000 [main] INFO Server statistics={am.serverName=SRV1, jvm.freeMemory=1000000}",
        "2024-01-15 09:00:02.000 [main] ERROR NullPointerException in CryptoService",
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "server.log")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        schema = _make_schema()
        conn = load_server_metrics_into_duckdb(path, schema, db_path=os.path.join(tmpdir, "test.duckdb"))
        try:
            cnt = conn.execute("SELECT COUNT(*) FROM server_metrics").fetchone()[0]
            assert cnt == 1, f"Expected 1 metric row (jvm.freeMemory), got {cnt}"
        finally:
            conn.close()


def test_merged_loader_returns_signals_and_correct_db():
    """Single-scan loader must return both a populated DB and high-signal candidates."""
    lines = [
        "2024-01-15 09:00:00.000 [main] INFO Server statistics={am.serverName=SRV1, jvm.freeMemory=1000000}",
        "2024-01-15 09:00:01.000 [main] INFO Count = 6891 rows returned for listMyRequests",
        "2024-01-15 09:00:02.000 [main] ERROR RoleValidator - entry repeated 50 times",
        "2024-01-15 09:00:03.000 [main] INFO lapse(ms)=124517 on LDAP findByFilter",
        "2024-01-15 09:00:04.000 [main] INFO Healthy heartbeat",
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "server.log")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        schema = _make_schema()
        conn, signals = load_server_metrics_into_duckdb_with_signals(
            path, schema, db_path=os.path.join(tmpdir, "test.duckdb")
        )
        try:
            # DB sanity
            metric_cnt = conn.execute("SELECT COUNT(*) FROM server_metrics").fetchone()[0]
            assert metric_cnt == 1
            log_cnt = conn.execute("SELECT COUNT(*) FROM log_events").fetchone()[0]
            assert log_cnt == 5

            # Signals
            assert len(signals) == 3
            types = {s["signal_type"] for s in signals}
            assert "high_result_count" in types
            assert "authz_loop_candidate" in types
            assert "extreme_latency" in types
        finally:
            conn.close()


def test_copy_from_path_produces_correct_counts():
    """Verify that temp-CSV + COPY FROM loading produces correct row counts."""
    lines = [
        # 5 metric-bearing lines (6 numeric metrics total; am.serverName is string and skipped)
        "2024-01-15 09:00:00.000 [main] INFO Server statistics={am.serverName=SRV1, jvm.freeMemory=1000000}",
        "2024-01-15 09:00:01.000 [main] INFO Server statistics={dbcp.ActiveConnections=5, dbcp.AllConnections=10}",
        "2024-01-15 09:00:02.000 [main] INFO Server statistics={hibernate.sessionCount=3}",
        "2024-01-15 09:00:03.000 [main] INFO msg={eventManager.threadPoolActiveCount=2}",
        "2024-01-15 09:00:04.000 [main] INFO Server statistics={am.tomcat.thread.current.count=50}",
        # 5 non-metric lines
        "2024-01-15 09:00:05.000 [main] ERROR at com.foo.Bar.method(Bar.java:123)",
        "2024-01-15 09:00:06.000 [main] INFO Health check OK",
        "2024-01-15 09:00:07.000 [main] WARN something happened",
        "2024-01-15 09:00:08.000 [main] INFO routine heartbeat",
        "2024-01-15 09:00:09.000 [main] DEBUG trace message",
        # 1 line with no timestamp
        "no timestamp here at all",
        # 1 signal line
        "2024-01-15 09:00:10.000 [main] INFO Count = 6891 rows returned",
        # CSV escaping line with commas and quotes
        '2024-01-15 09:00:11.000 [main] INFO query="select * from users, items", count=5',
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "server.log")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        schema = _make_schema()
        db_file = os.path.join(tmpdir, "test.duckdb")
        conn = load_server_metrics_into_duckdb(path, schema, db_path=db_file)
        try:
            log_cnt = conn.execute("SELECT COUNT(*) FROM log_events").fetchone()[0]
            metric_cnt = conn.execute("SELECT COUNT(*) FROM server_metrics").fetchone()[0]

            # 12 timestamped lines + 1 no-timestamp line = 13 total in log_events
            assert log_cnt == 13, f"Expected 13 log_events rows, got {log_cnt}"
            # am.serverName=SRV1 is non-numeric and skipped by _extract_metric_pairs
            assert metric_cnt == 6, f"Expected 6 server_metrics rows, got {metric_cnt}"

            # Verify the no-timestamp line was inserted with NULL timestamp
            null_ts_cnt = conn.execute(
                "SELECT COUNT(*) FROM log_events WHERE timestamp IS NULL"
            ).fetchone()[0]
            assert null_ts_cnt == 1, f"Expected 1 NULL timestamp row, got {null_ts_cnt}"

            # Verify CSV escaping preserved commas and quotes in raw_line
            escaped_row = conn.execute(
                "SELECT raw_line FROM log_events WHERE raw_line LIKE '%select * from users%'"
            ).fetchone()
            assert escaped_row is not None
            assert 'select * from users, items' in escaped_row[0]
            assert 'count=5' in escaped_row[0]
        finally:
            conn.close()


def test_appender_path_produces_correct_counts():
    """Verify that DuckDB Appender ingestion produces correct row counts (functional equivalence)."""
    lines = [
        # 5 metric-bearing lines (6 numeric metrics total; am.serverName is string and skipped)
        "2024-01-15 09:00:00.000 [main] INFO Server statistics={am.serverName=SRV1, jvm.freeMemory=1000000}",
        "2024-01-15 09:00:01.000 [main] INFO Server statistics={dbcp.ActiveConnections=5, dbcp.AllConnections=10}",
        "2024-01-15 09:00:02.000 [main] INFO Server statistics={hibernate.sessionCount=3}",
        "2024-01-15 09:00:03.000 [main] INFO msg={eventManager.threadPoolActiveCount=2}",
        "2024-01-15 09:00:04.000 [main] INFO Server statistics={am.tomcat.thread.current.count=50}",
        # 5 non-metric lines
        "2024-01-15 09:00:05.000 [main] ERROR at com.foo.Bar.method(Bar.java:123)",
        "2024-01-15 09:00:06.000 [main] INFO Health check OK",
        "2024-01-15 09:00:07.000 [main] WARN something happened",
        "2024-01-15 09:00:08.000 [main] INFO routine heartbeat",
        "2024-01-15 09:00:09.000 [main] DEBUG trace message",
        # 1 line with no timestamp
        "no timestamp here at all",
        # 1 signal line
        "2024-01-15 09:00:10.000 [main] INFO Count = 6891 rows returned",
        # CSV escaping line with commas and quotes
        '2024-01-15 09:00:11.000 [main] INFO query="select * from users, items", count=5',
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "server.log")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        schema = _make_schema()
        db_file = os.path.join(tmpdir, "test.duckdb")
        conn = load_server_metrics_into_duckdb(path, schema, db_path=db_file)
        try:
            log_cnt = conn.execute("SELECT COUNT(*) FROM log_events").fetchone()[0]
            metric_cnt = conn.execute("SELECT COUNT(*) FROM server_metrics").fetchone()[0]

            # 12 timestamped lines + 1 no-timestamp line = 13 total in log_events
            assert log_cnt == 13, f"Expected 13 log_events rows, got {log_cnt}"
            # am.serverName=SRV1 is non-numeric and skipped by _extract_metric_pairs
            assert metric_cnt == 6, f"Expected 6 server_metrics rows, got {metric_cnt}"

            # Verify the no-timestamp line was inserted with NULL timestamp
            null_ts_cnt = conn.execute(
                "SELECT COUNT(*) FROM log_events WHERE timestamp IS NULL"
            ).fetchone()[0]
            assert null_ts_cnt == 1, f"Expected 1 NULL timestamp row, got {null_ts_cnt}"

            # Verify commas and quotes are preserved in raw_line via Appender
            escaped_row = conn.execute(
                "SELECT raw_line FROM log_events WHERE raw_line LIKE '%select * from users%'"
            ).fetchone()
            assert escaped_row is not None
            assert 'select * from users, items' in escaped_row[0]
            assert 'count=5' in escaped_row[0]
        finally:
            conn.close()


def test_signal_quick_reject_skips_benign_lines():
    """The quick-reject regex should prevent signal regexes from running on benign lines."""
    lines = [
        # 5 benign lines (no signal keywords)
        "2024-01-15 09:00:00.000 [main] INFO Server statistics={am.serverName=SRV1}",
        "2024-01-15 09:00:01.000 [main] ERROR at com.foo.Bar.method(Bar.java:123)",
        "2024-01-15 09:00:02.000 [main] INFO Health check OK",
        "2024-01-15 09:00:03.000 [main] WARN something happened",
        "2024-01-15 09:00:04.000 [main] INFO nothing special here",
        # 1 signal line
        "2024-01-15 09:00:05.000 [main] INFO Count = 6891 rows returned",
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "server.log")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        schema = _make_schema()
        db_file = os.path.join(tmpdir, "test.duckdb")
        conn, signals = load_server_metrics_into_duckdb_with_signals(path, schema, db_path=db_file)

        try:
            # Only the line with "Count = 6891" should produce a signal
            assert len(signals) == 1, f"Expected 1 signal, got {len(signals)}"
            assert signals[0]["signal_type"] == "high_result_count"
            assert signals[0]["captured_value"] == 6891
        finally:
            conn.close()


def test_pre_detect_uses_quick_reject_gate():
    """pre_detect_high_signal_events should use the quick-reject gate and find exactly 1 signal."""
    lines = [
        "2024-01-15 09:00:00.000 [main] INFO Server statistics={am.serverName=SRV1}",
        "2024-01-15 09:00:01.000 [main] ERROR at com.foo.Bar.method(Bar.java:123)",
        "2024-01-15 09:00:02.000 [main] INFO Health check OK",
        "2024-01-15 09:00:03.000 [main] WARN something happened",
        "2024-01-15 09:00:04.000 [main] INFO nothing special here",
        "2024-01-15 09:00:05.000 [main] INFO Count = 6891 rows returned",
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "server.log")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        schema = _make_schema()
        signals = pre_detect_high_signal_events(path, schema)
        assert len(signals) == 1, f"Expected 1 signal, got {len(signals)}"
        assert signals[0]["signal_type"] == "high_result_count"


def test_quick_reject_false_positive_yields_no_signal():
    """A line passing quick-reject but matching no pattern must not yield a signal."""
    lines = [
        "2024-01-15 09:00:00.000 [main] INFO account discount recounted",
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "server.log")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        schema = _make_schema()
        conn, signals = load_server_metrics_into_duckdb_with_signals(
            path, schema, db_path=os.path.join(tmpdir, "test.duckdb")
        )
        try:
            assert len(signals) == 0, f"Expected 0 signals, got {len(signals)}"
        finally:
            conn.close()


def test_is_safe_select_allows_hibernate_exception_literals():
    sql = """
    WITH stale AS (
      SELECT timestamp, thread, raw_line
      FROM log_events
      WHERE raw_line ILIKE '%StaleObjectStateException%'
         OR raw_line ILIKE '%Could not synchronize database state with session%'
         OR raw_line ILIKE '%Row was updated or deleted by another transaction%'
    )
    SELECT min(timestamp) AS first_seen, max(timestamp) AS last_seen, count(*) AS total
    FROM stale
    """
    assert is_safe_select(sql) is True


def test_is_safe_select_rejects_mutating_statements():
    assert is_safe_select("DELETE FROM log_events") is False
    assert is_safe_select("UPDATE log_events SET raw_line = 'x'") is False
    assert is_safe_select("INSERT INTO log_events VALUES (now(), 't', 'line')") is False


def test_is_safe_select_allows_leading_sql_comments():
    sql = """-- Q1) Timeline test around onset
WITH bounds AS (
  SELECT MIN(timestamp) AS t0 FROM log_events
)
SELECT * FROM bounds"""
    assert is_safe_select(sql) is True
    assert get_sql_safety_rejection_reason(sql) is None


def test_is_safe_select_allows_leading_block_comments():
    sql = """/* purpose: onset window */
SELECT COUNT(*) FROM log_events"""
    assert is_safe_select(sql) is True


def test_get_sql_safety_rejection_reason_for_mutating_keyword():
    reason = get_sql_safety_rejection_reason(
        "WITH doomed AS (SELECT 1) UPDATE log_events SET raw_line = 'x'"
    )
    assert reason is not None
    assert "Forbidden keyword" in reason


def test_format_query_dataframe_without_tabulate(monkeypatch):
    df = pd.DataFrame(
        {
            "timestamp": ["2026-01-07 14:12:25.632000"],
            "raw_line": ["StaleObjectStateException in session"],
        }
    )

    def _raise_tabulate(*_args, **_kwargs):
        raise ImportError("Missing optional dependency 'tabulate'")

    monkeypatch.setattr(pd.DataFrame, "to_markdown", _raise_tabulate)
    rendered = format_query_dataframe(df)
    assert "2026-01-07" in rendered
    assert "StaleObjectStateException" in rendered


def test_get_duckdb_observation_bounds():
    lines = [
        "2026-02-25 15:51:00.000 [main] INFO Server statistics={jvm.threadCount=344}",
        "2026-02-25 15:52:00.000 [exec-1] DEBUG REST:authn/login lapse(ms)=97550",
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "server.log")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        schema = _make_schema()
        conn = load_server_metrics_into_duckdb(path, schema, db_path=os.path.join(tmpdir, "test.duckdb"))
        try:
            bounds = get_duckdb_observation_bounds(conn)
            assert bounds["log_events_min"].startswith("2026-02-25")
            assert bounds["log_events_max"].startswith("2026-02-25")
            rendered = format_duckdb_observation_bounds(bounds)
            assert "2026-02-25" in rendered
            assert "authoritative" in rendered.lower()
        finally:
            conn.close()


def test_server_metrics_wide_view_created_on_load():
    lines = [
        "2024-01-15 09:00:00.000 [main] INFO Server statistics={jvm.threadCount=120, dbcp.ActiveConnections=4}",
        "2024-01-15 09:00:01.000 [main] INFO Server statistics={jvm.threadCount=130, am.auth.responseTime=900}",
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "server.log")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        schema = _make_schema()
        conn = load_server_metrics_into_duckdb(path, schema, db_path=os.path.join(tmpdir, "test.duckdb"))
        try:
            row = conn.execute(
                """
                SELECT thread_count, dbcp_active_connections, response_time_ms
                FROM server_metrics_wide
                ORDER BY timestamp
                LIMIT 1
                """
            ).fetchone()
            assert row == (120.0, 4.0, None)

            row2 = conn.execute(
                "SELECT thread_count, response_time_ms FROM server_metrics_wide ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            assert row2 == (130.0, 900.0)
        finally:
            conn.close()


def test_flag_columns_populated_after_load():
    """Pre-computed categorical flags and extracted values must be present in log_events."""
    lines = [
        "2024-01-15 09:00:00.000 [main] INFO Server statistics={jvm.freeMemory=1000000}",
        "2024-01-15 09:00:01.000 [main] INFO Count = 6891 rows returned for listMyRequests",
        "2024-01-15 09:00:02.000 [main] ERROR RoleValidator - entry repeated 50 times",
        "2024-01-15 09:00:03.000 [main] INFO lapse(ms)=124517 on LDAP findByFilter",
        "2024-01-15 09:00:04.000 [main] INFO REST:authn/login lapse(ms)=550",
        "2024-01-15 09:00:05.000 [main] INFO scheduled createIndexAllAvailableCredentialTO lapse(ms)=52000",
        "2024-01-15 09:00:06.000 [main] INFO at com.foo.Bar.method(Bar.java:123)",
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "server.log")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        schema = _make_schema()
        conn = load_server_metrics_into_duckdb(path, schema, db_path=os.path.join(tmpdir, "test.duckdb"))
        try:
            # Boolean flags
            latency_cnt = conn.execute(
                "SELECT COUNT(*) FROM log_events WHERE has_latency = TRUE"
            ).fetchone()[0]
            assert latency_cnt == 3, f"Expected 3 has_latency rows, got {latency_cnt}"

            count_cnt = conn.execute(
                "SELECT COUNT(*) FROM log_events WHERE has_count_rows = TRUE"
            ).fetchone()[0]
            assert count_cnt == 1, f"Expected 1 has_count_rows row, got {count_cnt}"

            entry_cnt = conn.execute(
                "SELECT COUNT(*) FROM log_events WHERE has_entry_authz = TRUE"
            ).fetchone()[0]
            assert entry_cnt == 1, f"Expected 1 has_entry_authz row, got {entry_cnt}"

            ldap_cnt = conn.execute(
                "SELECT COUNT(*) FROM log_events WHERE has_ldap = TRUE"
            ).fetchone()[0]
            assert ldap_cnt == 1, f"Expected 1 has_ldap row, got {ldap_cnt}"

            rest_cnt = conn.execute(
                "SELECT COUNT(*) FROM log_events WHERE has_rest = TRUE"
            ).fetchone()[0]
            assert rest_cnt == 1, f"Expected 1 has_rest row, got {rest_cnt}"

            sched_cnt = conn.execute(
                "SELECT COUNT(*) FROM log_events WHERE has_scheduled = TRUE"
            ).fetchone()[0]
            assert sched_cnt == 1, f"Expected 1 has_scheduled row, got {sched_cnt}"

            # Extracted values
            lat_row = conn.execute(
                "SELECT latency_ms FROM log_events WHERE has_latency = TRUE ORDER BY latency_ms DESC"
            ).fetchone()
            assert lat_row[0] == 124517, f"Expected latency_ms=124517, got {lat_row[0]}"

            count_row = conn.execute(
                "SELECT result_count FROM log_events WHERE has_count_rows = TRUE"
            ).fetchone()
            assert count_row[0] == 6891, f"Expected result_count=6891, got {count_row[0]}"

            sig_row = conn.execute(
                "SELECT method_sig FROM log_events WHERE method_sig IS NOT NULL"
            ).fetchone()
            assert "Bar.java:123" in sig_row[0], f"Expected method_sig with Bar.java:123, got {sig_row[0]}"

            op_row = conn.execute(
                "SELECT scheduled_op_name FROM log_events WHERE has_scheduled = TRUE"
            ).fetchone()
            assert "createIndexAllAvailableCredentialTO" in op_row[0], f"Expected scheduled_op_name, got {op_row[0]}"
        finally:
            conn.close()


def test_normalize_llm_sql_rewrites_wide_metric_queries():
    sql = """
    SELECT ts, thread_count, tomcat_busy_threads
    FROM server_metrics
    WHERE ts BETWEEN TIMESTAMP '2026-02-25 15:49:00' AND TIMESTAMP '2026-02-25 16:02:00'
    """
    normalized = normalize_llm_sql(sql)
    assert "server_metrics_wide" in normalized
    assert "server_metrics\n" not in normalized.lower() or "server_metrics_wide" in normalized


def test_normalize_llm_sql_rewrites_cast_regexp_extract_to_try_cast():
    sql = """
    SELECT CASE
      WHEN regexp_extract(raw_line, 'lapse\\(ms\\)=([0-9]+)', 1) != ''
      THEN CAST(regexp_extract(raw_line, 'lapse\\(ms\\)=([0-9]+)', 1) AS INT)
      ELSE NULL END AS lapse_ms
    FROM log_events
    """
    normalized = normalize_llm_sql(sql)
    assert "TRY_CAST(regexp_extract" in normalized
    assert "THEN CAST(regexp_extract" not in normalized


def test_normalize_llm_sql_fixes_log_events_ts_alias():
    sql = """
    SELECT le.ts, le.raw_line
    FROM log_events le
    WHERE le.ts BETWEEN TIMESTAMP '2026-02-25 15:50:30' AND TIMESTAMP '2026-02-25 15:53:30'
    """
    normalized = normalize_llm_sql(sql)
    assert "le.timestamp" in normalized
    assert "le.ts" not in normalized


def test_normalize_llm_sql_preserves_metric_name_pivot_queries():
    sql = """
    WITH pivoted AS (
        SELECT timestamp,
               MAX(CASE WHEN metric_name = 'jvm.threadCount' THEN metric_value END) AS thread_count
        FROM server_metrics
        GROUP BY timestamp
    )
    SELECT timestamp, thread_count FROM pivoted
    """
    normalized = normalize_llm_sql(sql)
    assert "server_metrics_wide" not in normalized
    assert "FROM server_metrics" in normalized


def test_normalize_llm_sql_executes_evidence_gathering_style_query():
    lines = [
        "2026-02-25 15:51:00.000 [main] INFO Server statistics={jvm.threadCount=344, dbcp.ActiveConnections=8}",
        "2026-02-25 15:51:35.286000 [main] INFO Server statistics={jvm.threadCount=344, am.auth.responseTime=1200}",
        "2026-02-25 15:52:00.000 [exec-1] DEBUG REST:authn/login lapse(ms)=97550",
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "server.log")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        schema = _make_schema()
        conn = load_server_metrics_into_duckdb(path, schema, db_path=os.path.join(tmpdir, "test.duckdb"))
        try:
            metric_sql = normalize_llm_sql(
                """
                SELECT ts, thread_count, dbcp_active_connections, response_time_ms
                FROM server_metrics sm
                WHERE sm.ts BETWEEN TIMESTAMP '2026-02-25 15:51:00' AND TIMESTAMP '2026-02-25 15:52:00'
                ORDER BY ts
                """
            )
            rows = conn.execute(metric_sql).fetchdf()
            assert len(rows) >= 2
            assert "thread_count" in rows.columns

            log_sql = normalize_llm_sql(
                """
                SELECT le.ts, le.raw_line
                FROM log_events le
                WHERE le.ts BETWEEN TIMESTAMP '2026-02-25 15:51:00' AND TIMESTAMP '2026-02-25 15:53:00'
                """
            )
            log_rows = conn.execute(log_sql).fetchdf()
            assert len(log_rows) >= 1
            assert any("lapse(ms)=97550" in str(row["raw_line"]) for _, row in log_rows.iterrows())
        finally:
            conn.close()


def test_gated_flags_skip_regex_on_benign_lines():
    """Benign lines without signal keywords must leave gated extracted columns NULL/False."""
    from pipeline.server_metrics import _compute_log_event_flags

    benign_lines = [
        "2024-01-15 09:00:00.000 [main] INFO Server statistics={am.serverName=SRV1}",
        "2024-01-15 09:00:02.000 [main] INFO Health check OK",
        "2024-01-15 09:00:03.000 [main] WARN something happened",
        "2024-01-15 09:00:04.000 [main] INFO nothing special here",
        "2024-01-15 09:00:05.000 [main] DEBUG trace message",
    ]
    for line in benign_lines:
        flags = _compute_log_event_flags(line)
        assert flags["method_sig"] is None
        assert flags["latency_ms"] is None
        assert flags["result_count"] is None
        assert flags["scheduled_op_name"] is None


def test_ingest_log_chunk_produces_expected_rows():
    """_ingest_log_chunk should return the same rows as the old inline loop."""
    from pipeline.server_metrics import _ingest_log_chunk

    lines = [
        "2024-01-15 09:00:00.000 [main] INFO Server statistics={am.serverName=SRV1, jvm.freeMemory=1000000}",
        "2024-01-15 09:00:01.000 [main] INFO Count = 6891 rows returned for listMyRequests",
        "2024-01-15 09:00:02.000 [main] ERROR RoleValidator - entry repeated 50 times",
        "2024-01-15 09:00:03.000 [main] INFO lapse(ms)=124517 on LDAP findByFilter",
        "2024-01-15 09:00:04.000 [main] INFO Healthy heartbeat",
        "2024-01-15 09:00:05.000 [main] INFO Server statistics={dbcp.ActiveConnections=5}",
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "server.log")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        schema = _make_schema()
        log_rows, metric_rows, signals = _ingest_log_chunk(
            path, schema, start_offset=0, end_offset=None,
        )
        assert len(log_rows) == 6, f"Expected 6 log rows, got {len(log_rows)}"
        assert len(metric_rows) == 2, f"Expected 2 metric rows, got {len(metric_rows)}"
        assert len(signals) == 0, f"Expected 0 signals (collect_signals=False), got {len(signals)}"

        # Verify flags are present in log_rows
        row = log_rows[1]  # Count = 6891 line
        assert row[10] is True, "has_count_rows should be True"
        assert row[16] == 6891, "result_count should be 6891"


def test_parallel_ingestion_produces_same_counts_as_serial():
    """ProcessPoolExecutor path must match single-threaded counts exactly."""
    lines = [
        "2024-01-15 09:00:00.000 [main] INFO Server statistics={am.serverName=SRV1, jvm.freeMemory=1000000}",
        "2024-01-15 09:00:01.000 [main] INFO Count = 6891 rows returned for listMyRequests",
        "2024-01-15 09:00:02.000 [main] ERROR RoleValidator - entry repeated 50 times",
        "2024-01-15 09:00:03.000 [main] INFO lapse(ms)=124517 on LDAP findByFilter",
        "2024-01-15 09:00:04.000 [main] INFO Healthy heartbeat",
        "2024-01-15 09:00:05.000 [main] INFO Server statistics={dbcp.ActiveConnections=5}",
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "server.log")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        schema = _make_schema()
        conn = duckdb.connect(os.path.join(tmpdir, "parallel.duckdb"))
        conn.execute(LOG_EVENTS_CREATE_SQL)
        conn.execute(SERVER_METRICS_CREATE_SQL)

        from pipeline.server_metrics import _ingest_parallel
        _ingest_parallel(path, schema, conn, max_workers=2)

        log_cnt = conn.execute("SELECT COUNT(*) FROM log_events").fetchone()[0]
        metric_cnt = conn.execute("SELECT COUNT(*) FROM server_metrics").fetchone()[0]
        assert log_cnt == 6, f"Expected 6 log_events, got {log_cnt}"
        assert metric_cnt == 2, f"Expected 2 metric rows, got {metric_cnt}"
        conn.close()
