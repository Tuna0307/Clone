"""Server monitoring DuckDB loader and metric utilities.

Implements deterministic parsing of UAM5 server statistics logs into the
exact schema required by the DuckDB agentic SQL path (server_monitoring mode).

Also populates a full `log_events` table (every timestamped line) so the
agentic LLM can perform full-text search for application-level outliers
(high result counts, N+1 loops, extreme latencies, etc.) that are not
captured in the periodic numeric metric snapshots.

Reuses the existing _parse_line helper for timestamp + thread extraction.
"""

from __future__ import annotations

import multiprocessing
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Optional

import duckdb
import pandas as pd

from pipeline.files import stream_file_lines
from pipeline.progress import emit_ui_progress
from pipeline.parsing import _parse_line
from pipeline.query import _line_overlaps_query_window
from pipeline.constants import (
    SERVER_LOG_EVENTS_TABLE,
    HIGH_SIGNAL_PATTERNS,
    MAX_PRE_SCAN_CANDIDATES,
    _SIGNAL_QUICK_REJECT_RE,
)

_PARALLEL_THRESHOLD_BYTES: int = 50 * 1024 * 1024  # 50 MB

# Exact table schema from the implementation prompt (non-negotiable)
SERVER_METRICS_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS server_metrics (
    timestamp TIMESTAMP,
    thread VARCHAR,
    metric_name VARCHAR,
    metric_value DOUBLE,
    category VARCHAR,
    raw_line VARCHAR
);
"""

SERVER_METRICS_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_server_metrics_ts_metric
ON server_metrics(timestamp, metric_name);
"""

# Full log events table (every parsed line with timestamp in the window).
# This is the key addition that gives the agent visibility into application
# logic (UCM request traces, RoleValidator loops, Count=NNNN result sizes, etc.)
# that lives outside the ServerMonitoring metric dumps.
LOG_EVENTS_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS log_events (
    timestamp TIMESTAMP,
    thread VARCHAR,
    raw_line VARCHAR,
    has_latency BOOLEAN,
    has_jdbc BOOLEAN,
    has_ldap BOOLEAN,
    has_hibernate BOOLEAN,
    has_connection_wait BOOLEAN,
    has_staleobject BOOLEAN,
    has_sql BOOLEAN,
    has_count_rows BOOLEAN,
    has_entry_authz BOOLEAN,
    has_rest BOOLEAN,
    has_scheduled BOOLEAN,
    method_sig VARCHAR,
    latency_ms BIGINT,
    result_count BIGINT,
    scheduled_op_name VARCHAR
);
"""

LOG_EVENTS_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_log_events_ts
ON log_events(timestamp);
"""

LOG_EVENTS_FLAG_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_log_events_latency_ms ON log_events(latency_ms);
CREATE INDEX IF NOT EXISTS idx_log_events_result_count ON log_events(result_count);
CREATE INDEX IF NOT EXISTS idx_log_events_method_sig ON log_events(method_sig);
CREATE INDEX IF NOT EXISTS idx_log_events_scheduled_op ON log_events(scheduled_op_name);
"""

SERVER_METRICS_WIDE_VIEW_SQL = """
CREATE OR REPLACE VIEW server_metrics_wide AS
WITH pivoted AS (
    SELECT
        timestamp,
        MAX(CASE WHEN metric_name = 'jvm.threadCount' THEN metric_value END) AS thread_count,
        MAX(CASE WHEN metric_name = 'am.tomcat.thread.busy.count' THEN metric_value END) AS tomcat_busy_threads,
        MAX(CASE WHEN metric_name = 'am.tomcat.thread.current.count' THEN metric_value END) AS tomcat_current_threads,
        MAX(CASE WHEN metric_name = 'dbcp.ActiveConnections' THEN metric_value END) AS dbcp_active_connections,
        MAX(CASE WHEN metric_name = 'dbcp.IdleConnections' THEN metric_value END) AS dbcp_idle_connections,
        MAX(CASE WHEN metric_name = 'dbcp.AllConnections' THEN metric_value END) AS dbcp_all_connections,
        MAX(CASE WHEN metric_name = 'am.auth.responseTime' THEN metric_value END) AS response_time_ms,
        MAX(CASE WHEN metric_name = 'hibernate.sessionCount' THEN metric_value END) AS hibernate_sessions,
        MAX(CASE WHEN metric_name = 'eventManager.threadPoolQueueSize' THEN metric_value END) AS event_queue_size,
        MAX(CASE WHEN metric_name = 'eventManager.threadPoolActiveCount' THEN metric_value END) AS event_active_count,
        MAX(CASE WHEN metric_name = 'eventManager.threadPoolMaxSize' THEN metric_value END) AS event_pool_max_size,
        MAX(CASE WHEN metric_name = 'eventManager.threadPoolRejectedCount' THEN metric_value END) AS event_rejected_count,
        MAX(CASE WHEN metric_name = 'jvm.freeMemory' THEN metric_value END) AS jvm_free_memory,
        MAX(CASE WHEN metric_name = 'jvm.maxMemory' THEN metric_value END) AS jvm_max_memory
    FROM server_metrics
    GROUP BY timestamp
)
SELECT
    timestamp,
    timestamp AS ts,
    thread_count,
    tomcat_busy_threads,
    tomcat_current_threads,
    dbcp_active_connections,
    dbcp_idle_connections,
    dbcp_all_connections,
    CAST(NULL AS DOUBLE) AS dbcp_max_active,
    response_time_ms,
    response_time_ms AS response_time,
    hibernate_sessions,
    event_queue_size,
    event_active_count,
    event_pool_max_size,
    event_rejected_count,
    jvm_free_memory,
    jvm_max_memory,
    CASE
        WHEN tomcat_current_threads IS NOT NULL
         AND tomcat_current_threads > 0
         AND tomcat_busy_threads IS NOT NULL
        THEN tomcat_busy_threads / tomcat_current_threads
        ELSE NULL
    END AS tomcat_current_threads_busy_ratio
FROM pivoted;
"""

def _load_duckdb_schema_text() -> str:
    from pipeline.prompt_loader import load_fragment

    return load_fragment("reference.duckdb_schema")


def _load_uam5_dictionary_text() -> str:
    from pipeline.prompt_loader import load_fragment

    return load_fragment("reference.uam5_dictionary")
_WIDE_METRIC_COLUMNS: frozenset[str] = frozenset({
    "thread_count",
    "tomcat_busy_threads",
    "tomcat_current_threads",
    "tomcat_current_threads_busy_ratio",
    "dbcp_active_connections",
    "dbcp_idle_connections",
    "dbcp_max_active",
    "dbcp_all_connections",
    "response_time_ms",
    "response_time",
    "hibernate_sessions",
    "active_conns",
    "busy_threads",
    "event_queue_size",
    "event_active_count",
    "event_pool_max_size",
    "event_rejected_count",
    "jvm_free_memory",
    "jvm_max_memory",
})

# Allowed categories (exact values from prompt)
ALLOWED_CATEGORIES = {
    "System Information",
    "Tomcat",
    "Hibernate",
    "DBCP",
    "deliveryManager",
    "eventManager",
}
_LOG_EVENTS_CHUNK_COLUMNS: list[str] = [
    "timestamp", "thread", "raw_line",
    "has_latency", "has_jdbc", "has_ldap", "has_hibernate",
    "has_connection_wait", "has_staleobject", "has_sql",
    "has_count_rows", "has_entry_authz", "has_rest", "has_scheduled",
    "method_sig", "latency_ms", "result_count", "scheduled_op_name",
]



# =============================================================================
# OFFICIAL UAM5 SERVER MONITORING DATA DICTIONARY
# Source: DuckDB_Server_Monitoring_Implementation_Prompt.md
# This is injected into the LLM prompt for the server_monitoring (DuckDB) path
# so the agent knows the exact official metric names and their meanings.
# =============================================================================

UAM5_MONITORING_METRICS: dict[str, list[dict[str, str]]] = {
    "System Information": [
        {"name": "am.serverName", "type": "String", "description": "Server ID of the particular UAM Instance being monitored"},
        {"name": "am.cachedSession", "type": "Integer", "description": "Session that is cached in memory (Not Applicable in IRAS context)"},
        {"name": "am.auth.responseTime", "type": "Double", "description": "Average Authentication Response Time (measured over last 1000 samples)"},
        {"name": "am.auth.responseTime90th", "type": "Double", "description": "Authentication Response Time 90th percentile (more realistic view of degradation)"},
        {"name": "jvm.freeMemory", "type": "Long", "description": "JVM Free Memory in bytes"},
        {"name": "jvm.threadCount", "type": "Integer", "description": "Current number of live threads (daemon + non-daemon)"},
        {"name": "jvm.maxMemory", "type": "Long", "description": "Maximum amount of memory the JVM will attempt to use (-Xmx)"},
        {"name": "am.e2eeNonExpiredSessionCache", "type": "Integer", "description": "Count of end-to-end encrypted (E2EE) sessions that have not yet expired"},
        {"name": "am.serverTime", "type": "Long", "description": "Server timestamp at the moment the statistics snapshot was taken (epoch ms)"},
        {"name": "eventManager.threadPoolMaxSize", "type": "Integer", "description": "Maximum allowed number of event threads in the pool"},
        {"name": "eventManager.threadPoolQueueSize", "type": "Integer", "description": "Number of event threads currently in the queue"},
        {"name": "eventManager.threadPoolMaxQueueSize", "type": "Integer", "description": "Maximum capacity of the event manager queue"},
        {"name": "eventManager.threadPoolActiveCount", "type": "Integer", "description": "Approximate number of event threads actively executing tasks"},
        {"name": "eventManager.threadPoolRejectedCount", "type": "Long", "description": "Cumulative count of tasks rejected by the event manager thread pool"},
        {"name": "eventManager.threadPoolRejectedCountInTimeWindow", "type": "Long", "description": "Rejected event manager tasks within a rolling time window"},
    ],
    "Tomcat": [
        {"name": "am.tomcat.connector.name", "type": "String", "description": "Tomcat Connector Name (SSL connector from server.xml)"},
        {"name": "am.tomcat.thread.current.count", "type": "Integer", "description": "Total threads in the Thread Pool (bound by MaxThread setting)"},
        {"name": "am.tomcat.thread.busy.count", "type": "Integer", "description": "Tomcat threads currently busy serving XML-RPC requests"},
    ],
    "Hibernate": [
        {"name": "hibernate.sessionCount", "type": "Integer", "description": "Number of active Hibernate sessions (DB connections from UAM to DB)"},
        {"name": "hibernate.relation2.cache.hitCount", "type": "Long", "description": "Number of requested Relations found in cache"},
        {"name": "hibernate.relation2.cache.missCount", "type": "Long", "description": "Number of requested Relations not found in cache"},
        {"name": "hibernate.relation2.cache.elementInMemory", "type": "Long", "description": "Number of Relation cache elements currently in memory"},
        {"name": "hibernate.baseobject.cache.hitCount", "type": "Long", "description": "Number of requested BaseObjects found in cache"},
        {"name": "hibernate.baseobject.cache.missCount", "type": "Long", "description": "Number of requested BaseObjects not found in cache"},
        {"name": "hibernate.baseobject.cache.elementInMemory", "type": "Long", "description": "Number of BaseObject cache elements in memory"},
        {"name": "hibernate.attr.cache.hitCount", "type": "Long", "description": "Number of requested Attributes found in cache"},
        {"name": "hibernate.attr.cache.missCount", "type": "Long", "description": "Number of requested Attributes not found in cache"},
        {"name": "hibernate.attr.cache.elementInMemory", "type": "Long", "description": "Number of Attribute cache elements in memory"},
    ],
    "DBCP": [
        {"name": "dbcp.ActiveConnections", "type": "Integer", "description": "Number of database connections currently active (in use)"},
        {"name": "dbcp.AllConnections", "type": "Integer", "description": "Total number of database connections allocated in the pool"},
        {"name": "dbcp.IdleConnections", "type": "Integer", "description": "Number of database connections currently idle in the pool"},
    ],
    "deliveryManager": [
        {"name": "deliveryManager.MRQ-EMAIL-GW.threadPoolActiveCount", "type": "Integer", "description": "Active threads in the Email Gateway pool"},
        {"name": "deliveryManager.MRQ-EMAIL-GW.threadPoolMaxQueueSize", "type": "Integer", "description": "Maximum queue capacity of the Email Gateway pool"},
        {"name": "deliveryManager.MRQ-EMAIL-GW.threadPoolMaxSize", "type": "Integer", "description": "Maximum thread count of the Email Gateway pool"},
        {"name": "deliveryManager.MRQ-EMAIL-GW.threadPoolQueueSize", "type": "Integer", "description": "Current queue depth of the Email Gateway pool"},
        {"name": "deliveryManager.MRQ-EMAIL-GW.threadPoolRejectedCount", "type": "Long", "description": "Cumulative tasks rejected by the Email Gateway pool"},
        {"name": "deliveryManager.MRQ-EMAIL-GW.threadPoolRejectedCountInTimeWindow", "type": "Long", "description": "Rejected tasks within a rolling time window"},
        {"name": "deliveryManager.MRQ-SMS-GW.threadPoolActiveCount", "type": "Integer", "description": "Active threads in the SMS Gateway pool"},
        {"name": "deliveryManager.MRQ-SMS-GW.threadPoolMaxQueueSize", "type": "Integer", "description": "Maximum queue capacity of the SMS Gateway pool"},
        {"name": "deliveryManager.MRQ-SMS-GW.threadPoolMaxSize", "type": "Integer", "description": "Maximum thread count of the SMS Gateway pool"},
        {"name": "deliveryManager.MRQ-SMS-GW.threadPoolQueueSize", "type": "Integer", "description": "Current queue depth of the SMS Gateway pool"},
        {"name": "deliveryManager.MRQ-SMS-GW.threadPoolRejectedCount", "type": "Long", "description": "Cumulative tasks rejected by the SMS Gateway pool"},
        {"name": "deliveryManager.MRQ-SMS-GW.threadPoolRejectedCountInTimeWindow", "type": "Long", "description": "Rejected tasks within a rolling time window"},
    ],
}

# Robust key=value extractor (handles both "Server statistics={...}" and "msg={...}" forms)
# Captures typical UAM metric names (am.*, jvm.*, hibernate.*, dbcp.*, eventManager.*, deliveryManager.*)
_METRIC_RE = re.compile(r"([a-zA-Z][\w.]+)=([^,\s{}]+)")

# Pre-compiled regexes for computing categorical flags during ingestion.
# These replace repeated LIKE / regexp_extract scans on log_events.raw_line.
_METHOD_SIG_RE = re.compile(r"([A-Za-z][A-Za-z0-9_\.]*(?:\.java)?:\d+)")
_LATENCY_MS_RE = re.compile(r"lapse\(ms\)\s*=\s*(\d+)")
_RESULT_COUNT_RE = re.compile(r"(?i)\b(Count|rows?|returned|result\s*(count|size)?)\s*[:=]?\s*(\d{3,})")
_SCHEDULED_OP_RE = re.compile(r"([A-Za-z0-9_]+(?:TO|Job|Index|Task)[A-Za-z0-9_]*)")


def _categorize_metric(metric_name: str) -> str:
    """Map metric name prefix to one of the six allowed categories."""
    n = metric_name.lower()
    if n.startswith("am.tomcat."):
        return "Tomcat"
    if n.startswith("hibernate."):
        return "Hibernate"
    if n.startswith("dbcp."):
        return "DBCP"
    if n.startswith("deliverymanager.") or "deliverymanager" in n:
        return "deliveryManager"
    if n.startswith("eventmanager."):
        return "eventManager"
    return "System Information"


def _compute_log_event_flags(clean: str) -> dict[str, Any]:
    """Return pre-computed categorical flags and extracted values for a log line.

    Computing these once during ingestion eliminates repeated full-text
    LIKE / regexp_extract scans in downstream diagnostic SQL.
    """
    lower = clean.lower()
    flags: dict[str, Any] = {
        "has_latency": "lapse" in lower,
        "has_jdbc": "jdbc" in lower,
        "has_ldap": "ldap" in lower,
        "has_hibernate": "hibernate" in lower,
        "has_connection_wait": "connection wait" in lower,
        "has_staleobject": "staleobject" in lower,
        "has_sql": "sql" in lower,
        "has_count_rows": any(k in lower for k in ("count", "rows", "returned")),
        "has_entry_authz": (
            " - entry" in clean
            or "rolevalidator" in lower
            or "checkcredentialrole" in lower
            or "getrelationsbyobj" in lower
            or "getcredential" in lower
        ),
        "has_rest": "rest:" in lower,
        "has_scheduled": any(k in lower for k in ("scheduled", "createindex", "batch", "cron")),
        "method_sig": None,
        "latency_ms": None,
        "result_count": None,
        "scheduled_op_name": None,
    }

    if ".java:" in clean or "(" in clean:
        m = _METHOD_SIG_RE.search(clean)
        if m:
            flags["method_sig"] = m.group(1)

    if flags["has_latency"]:
        m = _LATENCY_MS_RE.search(clean)
        if m:
            flags["latency_ms"] = int(m.group(1))

    if flags["has_count_rows"]:
        m = _RESULT_COUNT_RE.search(clean)
        if m:
            flags["result_count"] = int(m.group(3))

    if flags["has_scheduled"]:
        m = _SCHEDULED_OP_RE.search(clean)
        if m:
            flags["scheduled_op_name"] = m.group(1)

    return flags


def _extract_metric_pairs(line: str) -> list[tuple[str, float]]:
    """Extract numeric key=value metric pairs from a log line.

    Non-numeric values (e.g. serverName strings) are intentionally skipped;
    they remain visible to the LLM via the raw_line column.
    """
    pairs: list[tuple[str, float]] = []
    for match in _METRIC_RE.finditer(line):
        name = match.group(1)
        val_str = match.group(2)
        try:
            value = float(val_str)
            pairs.append((name, value))
        except ValueError:
            # Non-numeric (config strings, etc.) — captured via raw_line instead
            continue
    return pairs


def _scan_line_for_signal(clean: str) -> dict[str, Any] | None:
    """Apply quick-reject gate and high-signal patterns to a cleaned log line.

    Returns None if no signal matches, otherwise a candidate dict with
    signal_type, captured_value, snippet, and raw_line.
    """
    if not _SIGNAL_QUICK_REJECT_RE.search(clean):
        return None

    for signal_type, pat in HIGH_SIGNAL_PATTERNS:
        m = pat.search(clean)
        if m:
            captured = None
            for g in reversed(m.groups()):
                if g and g.isdigit():
                    captured = int(g)
                    break
            snippet = clean[:300] + ("..." if len(clean) > 300 else "")
            return {
                "signal_type": signal_type,
                "captured_value": captured,
                "snippet": snippet,
                "raw_line": clean,
            }
    return None


def _score_signal(c: dict[str, Any]) -> int:
    """Return an interestingness score for a signal candidate.

    Prefers large captured numeric values; falls back to 1 so every
    candidate still participates in ranking.
    """
    v = c.get("captured_value") or 0
    return v if v > 0 else 1


def _ingest_log_chunk(
    file_path: str,
    schema: dict,
    start_offset: int,
    end_offset: int | None,
    query_context: Optional[dict[str, Any]] = None,
    collect_signals: bool = False,
) -> tuple[list[tuple], list[tuple], list[dict[str, Any]]]:
    """Ingest a byte-aligned chunk of the log file in-process.

    Returns (log_event_rows, metric_rows, signal_candidates).
    Each log_event_row is a tuple matching the log_events table columns.
    Each metric_row is a 6-tuple (ts, thread, name, value, category, raw_line).
    """
    log_rows: list[tuple] = []
    metric_rows: list[tuple] = []
    signal_candidates: list[dict[str, Any]] = []

    for raw_line in stream_file_lines(file_path, start_offset=start_offset, end_offset=end_offset):
        ts, thread, clean = _parse_line(raw_line, schema)

        if not _line_overlaps_query_window(ts, query_context):
            continue

        thread_val = thread or ""
        raw_for_db = clean[:2000] if len(clean) > 2000 else clean
        flags = _compute_log_event_flags(clean)
        log_rows.append((
            ts, thread_val, raw_for_db,
            flags["has_latency"], flags["has_jdbc"], flags["has_ldap"], flags["has_hibernate"],
            flags["has_connection_wait"], flags["has_staleobject"], flags["has_sql"],
            flags["has_count_rows"], flags["has_entry_authz"], flags["has_rest"], flags["has_scheduled"],
            flags["method_sig"], flags["latency_ms"], flags["result_count"], flags["scheduled_op_name"],
        ))

        # Fast-path gate: only run metric regex on lines that actually carry metrics.
        if "Server statistics" in clean or "msg=" in clean:
            metric_pairs = _extract_metric_pairs(clean)
            if metric_pairs:
                for name, value in metric_pairs:
                    category = _categorize_metric(name)
                    metric_rows.append((ts, thread_val, name, value, category, raw_for_db))

        if collect_signals:
            signal = _scan_line_for_signal(clean)
            if signal is not None:
                signal["timestamp"] = ts
                signal_candidates.append(signal)

    return log_rows, metric_rows, signal_candidates


def _compute_chunk_offsets(file_path: str, num_chunks: int) -> list[tuple[int, int]]:
    """Return [(start, end), ...] byte ranges for roughly equal chunks.

    End offsets align to newline boundaries so no line is split.
    """
    size = os.path.getsize(file_path)
    if size == 0:
        return [(0, 0)]
    stride = size // num_chunks
    offsets = [0]
    with open(file_path, "rb") as f:
        for i in range(1, num_chunks):
            pos = min(i * stride, size)
            f.seek(pos)
            # Advance to next newline
            while pos < size:
                byte = f.read(1)
                if not byte or byte == b"\n":
                    break
                pos += 1
            offsets.append(pos + 1 if byte == b"\n" else pos)
    offsets.append(size)
    return list(zip(offsets[:-1], offsets[1:]))


def _ingest_parallel(
    file_path: str,
    schema: dict,
    conn: duckdb.DuckDBPyConnection,
    query_context: Optional[dict[str, Any]] = None,
    collect_signals: bool = False,
    max_candidates: int = MAX_PRE_SCAN_CANDIDATES,
    max_workers: int | None = None,
) -> tuple[int, int, int, int, list[dict[str, Any]]]:
    """Multi-process ingestion for large logs."""
    size = os.path.getsize(file_path)
    if max_workers is not None:
        num_chunks = max_workers
    else:
        cpu_count = multiprocessing.cpu_count()
        num_chunks = min(4, max(2, cpu_count - 1))
    chunk_offsets = _compute_chunk_offsets(file_path, num_chunks)

    print(f"  [Server] Parallel ingestion: {num_chunks} chunks, {len(chunk_offsets)} ranges")

    lines_total = 0
    all_signal_candidates: list[dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=num_chunks) as exe:
        futures = [
            exe.submit(
                _ingest_log_chunk,
                file_path,
                schema,
                start,
                end,
                query_context,
                collect_signals,
            )
            for start, end in chunk_offsets
        ]
        for fut in futures:
            log_rows, metric_rows, signals = fut.result()
            if log_rows:
                df = pd.DataFrame(log_rows, columns=_LOG_EVENTS_CHUNK_COLUMNS)
                conn.from_df(df).insert_into("log_events")
            if metric_rows:
                conn.executemany(
                    "INSERT INTO server_metrics VALUES (?, ?, ?, ?, ?, ?)",
                    metric_rows,
                )
            all_signal_candidates.extend(signals)
            lines_total += len(log_rows)

    metrics_total = conn.execute("SELECT COUNT(*) FROM server_metrics").fetchone()[0]
    log_events_total = conn.execute("SELECT COUNT(*) FROM log_events").fetchone()[0]
    lines_with_metrics = metrics_total

    if collect_signals:
        all_signal_candidates.sort(key=_score_signal, reverse=True)
        return lines_total, lines_with_metrics, metrics_total, log_events_total, all_signal_candidates[:max_candidates]
    return lines_total, lines_with_metrics, metrics_total, log_events_total, []


def _ingest_server_log(
    file_path: str,
    schema: dict,
    conn: duckdb.DuckDBPyConnection,
    query_context: Optional[dict[str, Any]] = None,
    collect_signals: bool = False,
    max_candidates: int = MAX_PRE_SCAN_CANDIDATES,
) -> tuple[int, int, int, int, list[dict[str, Any]]]:
    """Shared single-pass ingestion loop.

    Returns (lines_total, lines_with_metrics, metrics_total, log_events_total, signals).
    When collect_signals=False the signals list is empty.
    """
    size = os.path.getsize(file_path)
    if size > _PARALLEL_THRESHOLD_BYTES:
        return _ingest_parallel(
            file_path, schema, conn, query_context, collect_signals, max_candidates
        )
    log_rows, metric_rows, signal_candidates = _ingest_log_chunk(
        file_path, schema, start_offset=0, end_offset=None,
        query_context=query_context, collect_signals=collect_signals,
    )
    lines_total = sum(1 for _ in stream_file_lines(file_path))
    lines_with_metrics = len({r[0] for r in metric_rows})
    metrics_total = len(metric_rows)
    log_events_total = len(log_rows)
    if log_rows:
        df = pd.DataFrame(log_rows, columns=_LOG_EVENTS_CHUNK_COLUMNS)
        conn.from_df(df).insert_into("log_events")
    if metric_rows:
        conn.executemany(
            "INSERT INTO server_metrics VALUES (?, ?, ?, ?, ?, ?)",
            metric_rows,
        )
    if collect_signals:
        signal_candidates.sort(key=_score_signal, reverse=True)
        return lines_total, lines_with_metrics, metrics_total, log_events_total, signal_candidates[:max_candidates]
    return lines_total, lines_with_metrics, metrics_total, log_events_total, []


def _run_ingestion_and_index(
    file_path: str,
    schema: dict,
    conn: duckdb.DuckDBPyConnection,
    query_context: Optional[dict[str, Any]] = None,
    collect_signals: bool = False,
    max_candidates: int = MAX_PRE_SCAN_CANDIDATES,
) -> tuple[int, int, int, int, list[dict[str, Any]]]:
    """Create tables, run the ingestion transaction, and build indexes.

    Returns the 5-tuple from `_ingest_server_log`.
    """
    conn.execute(SERVER_METRICS_CREATE_SQL)
    conn.execute(LOG_EVENTS_CREATE_SQL)

    conn.execute("BEGIN TRANSACTION")
    try:
        result = _ingest_server_log(
            file_path, schema, conn, query_context,
            collect_signals=collect_signals, max_candidates=max_candidates,
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    conn.execute(SERVER_METRICS_INDEX_SQL)
    conn.execute(LOG_EVENTS_INDEX_SQL)
    conn.execute(LOG_EVENTS_FLAG_INDEX_SQL)
    create_server_monitoring_views(conn)
    return result


def create_server_monitoring_views(conn: Any) -> None:
    """Create helper views that match common LLM column expectations."""
    conn.execute(SERVER_METRICS_WIDE_VIEW_SQL)


def get_duckdb_observation_bounds(conn: Any) -> dict[str, str | None]:
    """Return min/max timestamps available in the loaded server_monitoring tables."""
    bounds: dict[str, str | None] = {
        "log_events_min": None,
        "log_events_max": None,
        "metrics_min": None,
        "metrics_max": None,
    }
    try:
        row = conn.execute(
            f"SELECT MIN(timestamp), MAX(timestamp) FROM {SERVER_LOG_EVENTS_TABLE}"
        ).fetchone()
        if row and row[0] is not None:
            bounds["log_events_min"] = str(row[0])
            bounds["log_events_max"] = str(row[1])
    except Exception:
        pass
    try:
        row = conn.execute(
            "SELECT MIN(timestamp), MAX(timestamp) FROM server_metrics_wide"
        ).fetchone()
        if row and row[0] is not None:
            bounds["metrics_min"] = str(row[0])
            bounds["metrics_max"] = str(row[1])
    except Exception:
        pass
    return bounds


def format_duckdb_observation_bounds(bounds: dict[str, str | None]) -> str:
    """Human-readable bounds block for LLM follow-up prompts."""
    log_min = bounds.get("log_events_min")
    log_max = bounds.get("log_events_max")
    met_min = bounds.get("metrics_min")
    met_max = bounds.get("metrics_max")
    if not any((log_min, log_max, met_min, met_max)):
        return "(could not determine observation bounds from DuckDB)"
    lines = ["**Loaded DuckDB observation bounds (authoritative — use these dates):**"]
    if log_min and log_max:
        lines.append(f"- `log_events`: {log_min} → {log_max}")
    if met_min and met_max:
        lines.append(f"- `server_metrics_wide`: {met_min} → {met_max}")
    lines.append(
        "- Never invent calendar dates. When the ticket/report gives clock times only "
        "(e.g. 15:59:01), combine them with the date from these bounds."
    )
    return "\n".join(lines)


def _uses_wide_metric_columns(sql: str) -> bool:
    return any(re.search(rf"\b{re.escape(col)}\b", sql, re.IGNORECASE) for col in _WIDE_METRIC_COLUMNS)


def _should_rewrite_to_wide_view(sql: str) -> bool:
    if not _uses_wide_metric_columns(sql):
        return False
    lower = sql.lower()
    if "metric_name" in lower:
        return False
    if "case when" in lower and "metric_value" in lower:
        return False
    return True


def normalize_llm_sql(sql: str) -> str:
    """Rewrite common LLM schema mistakes before executing agentic SQL."""
    normalized = sql
    rewrite_to_wide = _should_rewrite_to_wide_view(sql)

    if rewrite_to_wide:
        normalized = re.sub(r"\bserver_metrics\b", "server_metrics_wide", normalized, flags=re.IGNORECASE)

    normalized = re.sub(
        r"\b(log_events|le)\.ts\b",
        r"\1.timestamp",
        normalized,
        flags=re.IGNORECASE,
    )

    if not rewrite_to_wide:
        normalized = re.sub(
            r"\b(server_metrics|sm)\.ts\b",
            r"\1.timestamp",
            normalized,
            flags=re.IGNORECASE,
        )

    # regexp_extract returns '' on no match; CAST(... AS INT) then fails in DuckDB.
    normalized = re.sub(
        r"\bCAST\s*\(\s*regexp_extract\b",
        "TRY_CAST(regexp_extract",
        normalized,
        flags=re.IGNORECASE,
    )

    return normalized


def format_query_dataframe(df: Any, *, max_rows: int = 50) -> str:
    """Format a DuckDB/pandas query result for LLM observations."""
    if df is None or getattr(df, "empty", True):
        return "No rows returned."

    display = df.head(max_rows)
    try:
        rendered = display.to_markdown(index=False)
    except (ImportError, ModuleNotFoundError, Exception):
        cols = [str(col) for col in display.columns]
        lines = [
            " | ".join(cols),
            " | ".join("---" for _ in cols),
        ]
        for _, row in display.iterrows():
            lines.append(" | ".join(str(row[col]) for col in display.columns))
        rendered = "\n".join(lines)
        if len(df) > max_rows:
            rendered += f"\n... ({len(df) - max_rows} more rows)"

    return rendered


def _strip_sql_string_literals(sql: str) -> str:
    """Remove single-quoted literals before keyword safety checks."""
    return re.sub(r"'(?:''|[^'])*'", "''", sql, flags=re.IGNORECASE | re.DOTALL)


def _strip_sql_comments(sql: str) -> str:
    """Remove leading SQL line/block comments so read-only guards see the real statement."""
    remainder = sql.strip()
    while remainder:
        remainder = remainder.lstrip()
        if remainder.startswith("--"):
            newline = remainder.find("\n")
            remainder = remainder[newline + 1:] if newline >= 0 else ""
            continue
        if remainder.startswith("/*"):
            end = remainder.find("*/")
            if end < 0:
                return ""
            remainder = remainder[end + 2:]
            continue
        break
    return remainder.strip()


_FORBIDDEN_SQL_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\binsert\b", "INSERT"),
    (r"\bupdate\b", "UPDATE"),
    (r"\bdelete\b", "DELETE"),
    (r"\bdrop\b", "DROP"),
    (r"\balter\b", "ALTER"),
    (r"\bcreate\s+table\b", "CREATE TABLE"),
    (r"\bcreate\s+index\b", "CREATE INDEX"),
    (r"\battach\b", "ATTACH"),
    (r"\bdetach\b", "DETACH"),
    (r"\bcopy\b", "COPY"),
    (r"\bexport\b", "EXPORT"),
)


def get_sql_safety_rejection_reason(sql: str) -> str | None:
    """Return a human-readable rejection reason, or None when SQL is safe to execute."""
    if not sql or not sql.strip():
        return "Empty SQL block."

    statement = _strip_sql_comments(sql.strip())
    if not statement:
        return "SQL block contained only comments."

    normalized = statement.lower()
    if not normalized.startswith(("select", "with")):
        return "Only read-only SELECT/WITH queries are permitted."

    check_target = _strip_sql_string_literals(normalized)
    for pattern, label in _FORBIDDEN_SQL_PATTERNS:
        if re.search(pattern, check_target):
            return f"Forbidden keyword detected: {label}."
    return None


def is_safe_select(sql: str) -> bool:
    """Very lightweight guard for the agentic SQL loop.

    Only permits read-only SELECT / WITH queries. Rejects anything that
    mutates state or performs DDL beyond the initial table creation.
    Leading ``--`` and ``/* */`` comments are ignored before the prefix check.
    """
    return get_sql_safety_rejection_reason(sql) is None


def copy_duckdb_file_to_memory(db_path: str) -> Any:
    """Copy a file-backed server_monitoring DuckDB into an in-memory database.

    Used to retain queryable tables for follow-up chat without persisting artifacts.
    """
    import duckdb

    if not db_path or db_path == ":memory:" or not os.path.exists(db_path):
        raise FileNotFoundError(f"DuckDB file not found: {db_path}")

    mem_conn = duckdb.connect(":memory:")
    escaped = db_path.replace("'", "''")
    mem_conn.execute(f"ATTACH '{escaped}' AS src_db (READ_ONLY)")
    for table in ("server_metrics", SERVER_LOG_EVENTS_TABLE):
        mem_conn.execute(f"CREATE TABLE {table} AS SELECT * FROM src_db.{table}")
    mem_conn.execute("DETACH src_db")
    create_server_monitoring_views(mem_conn)
    return mem_conn


def pre_detect_high_signal_events(
    file_path: str,
    schema: dict,
    query_context: Optional[dict[str, Any]] = None,
    max_candidates: int = MAX_PRE_SCAN_CANDIDATES,
) -> list[dict[str, Any]]:
    """Lightweight deterministic pre-scan for application-level outlier signals.

    Streams the file once (respecting query window), applies HIGH_SIGNAL_PATTERNS,
    captures large numeric values and context, and returns a small curated list
    of the most "interesting" events (largest numbers, authz loop candidates, etc.).

    These are injected into the LLM seed facts so the agent sees the smoking-gun
    lines (e.g. "Count = 6891", repeated RoleValidator entry) on turn 0 without
    having to guess the right LIKE patterns.

    The patterns and this function are fully general — no incident-specific strings.

    Returns list of dicts: {timestamp, signal_type, captured_value, snippet, raw_line}
    sorted by descending "interestingness" (captured int value or burst density).
    """
    candidates: list[dict[str, Any]] = []
    lines_scanned = 0

    for raw_line in stream_file_lines(file_path):
        lines_scanned += 1
        ts, thread, clean = _parse_line(raw_line, schema)
        if not _line_overlaps_query_window(ts, query_context):
            continue

        signal = _scan_line_for_signal(clean)
        if signal is not None:
            signal["timestamp"] = ts
            candidates.append(signal)

    # Rank: prefer lines with large captured numbers, then by recency-ish
    candidates.sort(key=_score_signal, reverse=True)
    return candidates[:max_candidates]


def load_server_metrics_into_duckdb(
    file_path: str,
    schema: dict,
    query_context: Optional[dict[str, Any]] = None,
    db_path: str = ":memory:",
) -> duckdb.DuckDBPyConnection:
    """Parse a server-monitoring log file and load metrics into DuckDB."""
    t_start = time.monotonic()
    conn = duckdb.connect(db_path)

    lines_total, lines_with_metrics, metrics_total, log_events_total, _ = _run_ingestion_and_index(
        file_path, schema, conn, query_context, collect_signals=False
    )

    elapsed = time.monotonic() - t_start
    rate = lines_total / elapsed if elapsed > 0 else 0
    print(
        f"  [Server] DuckDB load complete: {lines_total:,} lines -> "
        f"{metrics_total:,} metric rows + {log_events_total:,} log events "
        f"in {elapsed:.1f}s ({rate:,.0f} lines/s)"
    )
    return conn


def load_server_metrics_into_duckdb_with_signals(
    file_path: str,
    schema: dict,
    query_context: Optional[dict[str, Any]] = None,
    db_path: str = ":memory:",
    max_candidates: int = MAX_PRE_SCAN_CANDIDATES,
) -> tuple[duckdb.DuckDBPyConnection, list[dict[str, Any]]]:
    """Single-pass loader that returns both a populated DuckDB connection and
    high-signal event candidates discovered during the scan.

    This eliminates the need for a separate pre_detect_high_signal_events pass.
    """
    t_start = time.monotonic()
    conn = duckdb.connect(db_path)

    lines_total, lines_with_metrics, metrics_total, log_events_total, signals = _run_ingestion_and_index(
        file_path, schema, conn, query_context, collect_signals=True, max_candidates=max_candidates
    )

    elapsed = time.monotonic() - t_start
    rate = lines_total / elapsed if elapsed > 0 else 0
    load_message = (
        f"DuckDB load + signal scan complete: {lines_total:,} lines -> "
        f"{metrics_total:,} metric rows + {log_events_total:,} log events + {len(signals)} signals "
        f"in {elapsed:.1f}s ({rate:,.0f} lines/s)"
    )
    print(f"  [Server] {load_message}")
    emit_ui_progress(load_message)
    return conn, signals


__all__ = [
    "load_server_metrics_into_duckdb",
    "load_server_metrics_into_duckdb_with_signals",
    "copy_duckdb_file_to_memory",
    "create_server_monitoring_views",
    "get_duckdb_observation_bounds",
    "format_duckdb_observation_bounds",
    "normalize_llm_sql",
    "format_query_dataframe",
    "get_sql_safety_rejection_reason",
    "is_safe_select",
    "pre_detect_high_signal_events",
    "ALLOWED_CATEGORIES",
    "DUCKDB_TABLE_SCHEMA_TEXT",
    "SERVER_METRICS_WIDE_VIEW_SQL",
    "UAM5_MONITORING_METRICS",
    "UAM5_SERVER_MONITORING_DICTIONARY_TEXT",
]


def __getattr__(name: str):
    """Lazy accessors for LLM reference text loaded via pipeline.prompt_loader."""
    if name == "DUCKDB_TABLE_SCHEMA_TEXT":
        return _load_duckdb_schema_text()
    if name == "UAM5_SERVER_MONITORING_DICTIONARY_TEXT":
        return _load_uam5_dictionary_text()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
