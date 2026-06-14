"""Tests for balanced deterministic diagnostics."""

import duckdb

from pipeline.server_sql.deterministic_diagnostics import (
    classification_from_prescores,
    run_broad_diagnostic_queries,
    score_archetype_candidates,
)


def _seed_test_db(conn) -> None:
    conn.execute("""
        CREATE TABLE server_metrics (
            timestamp TIMESTAMP, thread VARCHAR, metric_name VARCHAR,
            metric_value DOUBLE, category VARCHAR, raw_line VARCHAR
        )
    """)
    conn.execute("""
        CREATE TABLE log_events (timestamp TIMESTAMP, thread VARCHAR, raw_line VARCHAR)
    """)
    conn.execute("""
        INSERT INTO server_metrics VALUES
        ('2024-01-01 10:00:00', 't1', 'jvm.threadCount', 50, 'System Information', 'line1'),
        ('2024-01-01 10:01:00', 't1', 'jvm.threadCount', 120, 'System Information', 'line2'),
        ('2024-01-01 10:02:00', 't1', 'dbcp.ActiveConnections', 25, 'DBCP', 'line3'),
        ('2024-01-01 10:03:00', 't1', 'am.auth.responseTime', 800, 'System Information', 'line4')
    """)
    conn.execute("""
        INSERT INTO log_events VALUES
        ('2024-01-01 10:00:00', 't1', 'RoleValidator - entry Count=5000'),
        ('2024-01-01 10:00:05', 't1', 'RoleValidator - entry Count=5001'),
        ('2024-01-01 10:05:00', 't1', 'SomeOther.java:10 - entry'),
        ('2024-01-01 10:10:00', 't1', 'After long gap line')
    """)


def test_run_broad_diagnostic_queries_returns_signals():
    conn = duckdb.connect(":memory:")
    _seed_test_db(conn)
    signals = run_broad_diagnostic_queries(conn)
    assert isinstance(signals, list)
    conn.close()


def test_score_archetype_candidates_high_volume():
    structural = [
        {"signal_family": "high_volume_indicator", "strength": 0.9, "signal_id": "s1"},
    ]
    pre_scan = [{"signal_type": "authz_loop_candidate"}]
    scores = score_archetype_candidates(structural, pre_scan)
    assert scores["high_volume_cardinality"] > 0.5


def test_classification_from_prescores_fallback():
    pre_scores = {
        "global_runtime_stall": 0.7,
        "high_volume_cardinality": 0.3,
        "thread_pool_pressure": 0.1,
        "db_connection_pressure": 0.2,
        "mixed_compound": 0.15,
    }
    structural = [{"signal_id": "s1", "signal_family": "log_gap", "strength": 0.7}]
    result = classification_from_prescores(pre_scores, structural)
    assert result["primary"]["archetype"] == "global_runtime_stall"
    assert result["classification_method"] == "deterministic_only"