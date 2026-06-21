"""Deterministic evidence supplements run after LLM evidence gathering."""

from __future__ import annotations

from typing import Any

from pipeline.constants import SERVER_LOG_EVENTS_TABLE
from pipeline.server_metrics import format_query_dataframe, is_safe_select


def _resolve_onset_timestamp(
    onset_analysis: dict[str, Any] | None,
    critical_windows: list[dict[str, Any]],
) -> str | None:
    if onset_analysis:
        raw = onset_analysis.get("degradation_start")
        if raw:
            text = str(raw).strip().replace("T", " ")
            if len(text) >= 19:
                return text[:26]
    for window in critical_windows[:3]:
        start = window.get("start_time")
        if start:
            text = str(start).strip().replace("T", " ")
            if len(text) >= 19:
                return text[:26]
    return None


def _sql_literal_ts(ts: str) -> str:
    return ts.replace("'", "''")


def build_evidence_supplement_queries(
    onset_analysis: dict[str, Any] | None,
    critical_windows: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    """Return (label, sql) pairs for mandatory post-LLM evidence supplements."""
    onset_ts = _resolve_onset_timestamp(onset_analysis, critical_windows)
    if not onset_ts:
        return [
            (
                "observation_bounds",
                f"""
                SELECT MIN(timestamp) AS log_min, MAX(timestamp) AS log_max
                FROM {SERVER_LOG_EVENTS_TABLE}
                """,
            ),
            (
                "top_extreme_latencies",
                f"""
                SELECT timestamp, raw_line
                FROM {SERVER_LOG_EVENTS_TABLE}
                WHERE has_latency = TRUE
                ORDER BY latency_ms DESC NULLS LAST
                LIMIT 10
                """,
            ),
            (
                "backend_dependency_signals",
                f"""
                SELECT timestamp, raw_line
                FROM {SERVER_LOG_EVENTS_TABLE}
                WHERE has_jdbc = TRUE
                   OR has_ldap = TRUE
                   OR has_hibernate = TRUE
                   OR has_connection_wait = TRUE
                   OR has_staleobject = TRUE
                ORDER BY timestamp
                LIMIT 15
                """,
            ),
        ]

    lit = _sql_literal_ts(onset_ts)
    return [
        (
            "onset_minute_metrics",
            f"""
            SELECT
              time_bucket(INTERVAL 1 MINUTE, timestamp) AS minute,
              MAX(thread_count) AS max_thread_count,
              MAX(dbcp_active_connections) AS max_dbcp_active,
              MAX(hibernate_sessions) AS max_hibernate_sessions,
              MAX(response_time_ms) AS max_response_time_ms
            FROM server_metrics_wide
            WHERE timestamp BETWEEN TIMESTAMP '{lit}' - INTERVAL 3 MINUTE
                                AND TIMESTAMP '{lit}' + INTERVAL 5 MINUTE
              AND thread_count IS NOT NULL
            GROUP BY 1
            ORDER BY 1
            """,
        ),
        (
            "onset_extreme_latencies",
            f"""
            SELECT timestamp, raw_line
            FROM {SERVER_LOG_EVENTS_TABLE}
            WHERE timestamp BETWEEN TIMESTAMP '{lit}' - INTERVAL 3 MINUTE
                                AND TIMESTAMP '{lit}' + INTERVAL 10 MINUTE
              AND (has_latency = TRUE OR has_rest = TRUE)
            ORDER BY latency_ms DESC NULLS LAST
            LIMIT 10
            """,
        ),
        (
            "onset_backend_dependency_signals",
            f"""
            SELECT timestamp, raw_line
            FROM {SERVER_LOG_EVENTS_TABLE}
            WHERE timestamp BETWEEN TIMESTAMP '{lit}' - INTERVAL 3 MINUTE
                                AND TIMESTAMP '{lit}' + INTERVAL 10 MINUTE
              AND (
                has_jdbc = TRUE
                OR has_ldap = TRUE
                OR has_hibernate = TRUE
                OR has_connection_wait = TRUE
                OR has_staleobject = TRUE
                OR has_sql = TRUE
              )
            ORDER BY timestamp
            LIMIT 15
            """,
        ),
    ]


def run_evidence_supplement_queries(
    conn: Any,
    onset_analysis: dict[str, Any] | None,
    critical_windows: list[dict[str, Any]],
    *,
    max_rows: int = 50,
) -> list[tuple[str, str, str, int]]:
    """Execute supplement queries. Returns (label, sql, observation, row_count)."""
    results: list[tuple[str, str, str, int]] = []
    for label, sql in build_evidence_supplement_queries(onset_analysis, critical_windows):
        cleaned = sql.strip()
        if not is_safe_select(cleaned):
            continue
        try:
            df = conn.execute(cleaned).fetchdf()
            row_count = len(df)
            if row_count == 0:
                obs = "No rows returned."
            else:
                obs = format_query_dataframe(df, max_rows=max_rows)
            results.append((label, cleaned, obs, row_count))
        except Exception as exc:
            results.append((label, cleaned, f"ERROR: {exc}", -1))
    return results