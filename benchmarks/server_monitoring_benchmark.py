"""Benchmark module for Part A: query-level A/B comparison and Part B: end-to-end workflow phase timing.

Compares legacy LIKE/regexp_extract queries against pre-computed flag-column
queries on a shared in-memory DuckDB instance (Part A).

Times each phase of the structured server-monitoring workflow
(Part B).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import sys
import time
from typing import Any

import duckdb

# Allow imports from the project root when running this module directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.parsing import detect_log_structure
from pipeline.server_metrics import (
    LOG_EVENTS_CREATE_SQL,
    LOG_EVENTS_INDEX_SQL,
    LOG_EVENTS_FLAG_INDEX_SQL,
    SERVER_METRICS_CREATE_SQL,
    _ingest_server_log,
)
from pipeline.server_sql_graph import NODE_REGISTRY

SCHEMA_SAMPLE_SIZE = 1200

_LEGACY_LOG_EVENTS_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS legacy_log_events (
    timestamp TIMESTAMP,
    thread VARCHAR,
    raw_line VARCHAR
);
"""

_LEGACY_LOG_EVENTS_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_legacy_log_events_ts
ON legacy_log_events(timestamp);
"""

QUERY_PAIRS: list[tuple[str, str, str]] = [
    (
        "top_extreme_latencies",
        """
SELECT raw_line, TRY_CAST(regexp_extract(raw_line, 'lapse\\(ms\\)=(\\d+)', 1) AS BIGINT) AS latency
FROM legacy_log_events
WHERE raw_line LIKE '%lapse(ms)%'
ORDER BY latency DESC
LIMIT 50
        """.strip(),
        """
SELECT raw_line, latency_ms
FROM log_events
WHERE has_latency = TRUE
ORDER BY latency_ms DESC
LIMIT 50
        """.strip(),
    ),
    (
        "top_result_counts",
        """
SELECT raw_line, TRY_CAST(regexp_extract(raw_line, '(\\d{3,})', 1) AS BIGINT) AS result_count
FROM legacy_log_events
WHERE raw_line LIKE '%Count%' OR raw_line LIKE '%rows%' OR raw_line LIKE '%returned%'
ORDER BY result_count DESC
LIMIT 50
        """.strip(),
        """
SELECT raw_line, result_count
FROM log_events
WHERE has_count_rows = TRUE
ORDER BY result_count DESC
LIMIT 50
        """.strip(),
    ),
    (
        "entry_authz_bursts",
        """
SELECT
    raw_line,
    regexp_extract(raw_line, '([A-Za-z0-9_\\.]+(?:\\.java)?:\\d+)', 1) AS method_sig,
    COUNT(*) OVER (
        PARTITION BY regexp_extract(raw_line, '([A-Za-z0-9_\\.]+(?:\\.java)?:\\d+)', 1)
        ORDER BY timestamp
        RANGE BETWEEN INTERVAL 1 SECOND PRECEDING AND CURRENT ROW
    ) AS burst_count
FROM legacy_log_events
WHERE raw_line LIKE '% - entry%' OR raw_line LIKE '%RoleValidator%'
  AND regexp_extract(raw_line, '([A-Za-z0-9_\\.]+(?:\\.java)?:\\d+)', 1) IS NOT NULL
ORDER BY burst_count DESC
LIMIT 50
        """.strip(),
        """
SELECT
    raw_line,
    method_sig,
    COUNT(*) OVER (
        PARTITION BY method_sig
        ORDER BY timestamp
        RANGE BETWEEN INTERVAL 1 SECOND PRECEDING AND CURRENT ROW
    ) AS burst_count
FROM log_events
WHERE has_entry_authz = TRUE AND method_sig IS NOT NULL
ORDER BY burst_count DESC
LIMIT 50
        """.strip(),
    ),
    (
        "backend_dependency_signals",
        """
SELECT raw_line
FROM legacy_log_events
WHERE raw_line ILIKE '%jdbc%'
   OR raw_line ILIKE '%ldap%'
   OR raw_line ILIKE '%hibernate%'
   OR raw_line ILIKE '%connection%wait%'
   OR raw_line ILIKE '%staleobject%'
LIMIT 200
        """.strip(),
        """
SELECT raw_line
FROM log_events
WHERE has_jdbc = TRUE
   OR has_ldap = TRUE
   OR has_hibernate = TRUE
   OR has_connection_wait = TRUE
   OR has_staleobject = TRUE
LIMIT 200
        """.strip(),
    ),
    (
        "endpoint_breadth_late",
        """
SELECT COUNT(DISTINCT regexp_extract(raw_line, '([A-Za-z0-9_\\.]+(?:\\.java)?:\\d+)', 1)) AS endpoint_count
FROM legacy_log_events
WHERE timestamp > (SELECT MAX(timestamp) - INTERVAL '1 hour' FROM legacy_log_events)
        """.strip(),
        """
SELECT COUNT(DISTINCT method_sig) AS endpoint_count
FROM log_events
WHERE timestamp > (SELECT MAX(timestamp) - INTERVAL '1 hour' FROM log_events)
        """.strip(),
    ),
    (
        "recurring_scheduled_ops",
        """
SELECT
    regexp_extract(raw_line, '([A-Za-z0-9_]+(?:TO|Job|Index|Task)[A-Za-z0-9_]*)', 1) AS op_name,
    COUNT(*) AS op_count
FROM legacy_log_events
WHERE regexp_extract(raw_line, '([A-Za-z0-9_]+(?:TO|Job|Index|Task)[A-Za-z0-9_]*)', 1) IS NOT NULL
GROUP BY op_name
ORDER BY op_count DESC
LIMIT 50
        """.strip(),
        """
SELECT
    scheduled_op_name AS op_name,
    COUNT(*) AS op_count
FROM log_events
WHERE has_scheduled = TRUE AND scheduled_op_name IS NOT NULL
GROUP BY scheduled_op_name
ORDER BY op_count DESC
LIMIT 50
        """.strip(),
    ),
]

# Part B: phase keys that exist in the server-sql NODE_REGISTRY
_PHASE_REGISTRY_KEYS: list[str] = [k for k in list(NODE_REGISTRY.keys()) if k != "finalize"]

_PHASE_SUCCESSORS = {
    "initialize": "broad_diagnostic_and_archetype_classification",
    "broad_diagnostic_and_archetype_classification": "onset_analysis_and_symptom_discrimination",
    "onset_analysis_and_symptom_discrimination": "red_herring_filter",
    "red_herring_filter": "evidence_gathering",
    "evidence_gathering": "critic",
    "critic": "report_synthesis",
    "report_synthesis": "ticket_refinement",
    "ticket_refinement": "autonomous_summary",
    "autonomous_summary": "finalize",
}


def _ingest_both_schemas(file_path: str, schema: dict) -> duckdb.DuckDBPyConnection:
    """Create an in-memory DuckDB, ingest into current schema, then copy core columns to legacy table."""
    conn = duckdb.connect(":memory:")

    conn.execute(LOG_EVENTS_CREATE_SQL)
    conn.execute(LOG_EVENTS_INDEX_SQL)
    conn.execute(LOG_EVENTS_FLAG_INDEX_SQL)

    conn.execute(_LEGACY_LOG_EVENTS_CREATE_SQL)
    conn.execute(_LEGACY_LOG_EVENTS_INDEX_SQL)

    conn.execute(SERVER_METRICS_CREATE_SQL)

    conn.execute("BEGIN TRANSACTION")
    try:
        _ingest_server_log(file_path, schema, conn, query_context=None, collect_signals=False)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    # Copy the 3 legacy columns from the current full table into the legacy table
    conn.execute("""
        INSERT INTO legacy_log_events (timestamp, thread, raw_line)
        SELECT timestamp, thread, raw_line FROM log_events
    """)

    return conn


def _time_query(conn: duckdb.DuckDBPyConnection, sql: str) -> float:
    """Time a single query execution and return elapsed milliseconds."""
    start = time.perf_counter()
    conn.execute(sql).fetchall()
    end = time.perf_counter()
    return (end - start) * 1000.0


def _warmup_query(conn: duckdb.DuckDBPyConnection, sql: str) -> None:
    """Run a query once without timing (cache warming)."""
    conn.execute(sql).fetchall()


def run_part_a(conn: duckdb.DuckDBPyConnection, iterations: int = 3) -> list[dict[str, Any]]:
    """Run all query pairs, alternating legacy/flag, and compute median timings."""
    results: list[dict[str, Any]] = []

    for qid, legacy_sql, flag_sql in QUERY_PAIRS:
        # Warmup
        _warmup_query(conn, legacy_sql)
        _warmup_query(conn, flag_sql)

        legacy_times: list[float] = []
        flag_times: list[float] = []

        for _ in range(iterations):
            legacy_times.append(_time_query(conn, legacy_sql))
            flag_times.append(_time_query(conn, flag_sql))

        legacy_median = statistics.median(legacy_times)
        flag_median = statistics.median(flag_times)
        speedup = legacy_median / flag_median if flag_median > 0 else float("inf")

        results.append({
            "query_id": qid,
            "legacy_median_ms": round(legacy_median, 3),
            "flag_median_ms": round(flag_median, 3),
            "speedup_ratio": round(speedup, 2),
        })

    return results


def _wrap_node_with_timer(node_name: str, timings: list[dict[str, Any]]) -> None:
    """Monkey-patch NODE_REGISTRY[node_name] to record perf_counter before/after the original function.

    Appends ``{"phase": node_name, "duration_ms": ...}`` to *timings*.
    """
    original = NODE_REGISTRY[node_name]

    def _timed_node(state: Any) -> Any:
        start = time.perf_counter()
        result = original(state)
        end = time.perf_counter()
        timings.append({"phase": node_name, "duration_ms": round((end - start) * 1000.0, 3)})
        return result

    NODE_REGISTRY[node_name] = _timed_node


def run_part_b(file_path: str, schema: dict, skip_llm: bool = False) -> tuple[list[dict[str, Any]], float]:
    """Run the end-to-end structured workflow and time each phase.

    Args:
        file_path: Path to the server-monitoring log file.
        schema: Detected log schema.
        skip_llm: If *True*, replace all LLM-heavy nodes with no-ops so the
            benchmark can run without LLM credentials.

    Returns:
        ``(timings_list, total_duration_ms)`` where *timings_list* is a list of
        dicts ``{"phase": str, "duration_ms": float}``.
    """
    import pipeline.server_sql_graph as _ssg

    timings: list[dict[str, Any]] = []
    saved: dict[str, Any] = {}
    total_duration_ms = 0.0
    result_state = None

    # Save originals
    saved_has_langgraph = _ssg.HAS_LANGGRAPH
    saved_compiled_graph = _ssg._COMPILED_GRAPH
    for key in _PHASE_REGISTRY_KEYS:
        if key in NODE_REGISTRY:
            saved[key] = NODE_REGISTRY[key]

    try:
        # Force while-loop fallback so timer wrappers read from NODE_REGISTRY at runtime
        _ssg.HAS_LANGGRAPH = False
        _ssg._COMPILED_GRAPH = None

        if skip_llm:
            # Replace LLM-heavy nodes with no-ops that advance to the next phase
            for key in _PHASE_REGISTRY_KEYS:

                def _make_noop(k: str = key) -> Any:
                    def _noop_node(state: Any) -> Any:
                        state.phases_completed.add(k)
                        state.current_phase = _PHASE_SUCCESSORS.get(k, "finalize")
                        state.add_trace_step(
                            step=state.steps_taken,
                            phase=k,
                            node=k,
                            decision="skip_no_conn",
                            observations=[f"Skipped {k} due to --skip-llm"],
                        )
                        return state

                    return _noop_node

                NODE_REGISTRY[key] = _make_noop()

        # Wrap ALL nodes (including any no-ops) with timers
        for key in _PHASE_REGISTRY_KEYS:
            if key in NODE_REGISTRY:
                _wrap_node_with_timer(key, timings)

        total_start = time.perf_counter()

        dummy_llm: Any | None = None
        if skip_llm:
            class _DummyLLM:
                def invoke(self, *args, **kwargs):
                    raise RuntimeError("LLM should not be invoked in skip_llm mode")

            dummy_llm = _DummyLLM()

        result_state = _ssg.analyze_server_log_with_workflow(
            file_path,
            schema,
            ticket_text=None,
            retain_duckdb=False,
            llm=dummy_llm,
        )
        total_end = time.perf_counter()
        total_duration_ms = round((total_end - total_start) * 1000.0, 3)

        # Close any lingering DuckDB connection to prevent resource leaks
        conn = getattr(result_state, '_duckdb_conn', None)
        if conn is not None:
            conn.close()

    finally:
        # Restore original registry entries and LangGraph flags
        _ssg.HAS_LANGGRAPH = saved_has_langgraph
        _ssg._COMPILED_GRAPH = saved_compiled_graph
        for key, fn in saved.items():
            NODE_REGISTRY[key] = fn

    return timings, total_duration_ms


def _write_markdown_report(
    results: list[dict[str, Any]],
    output_path: str,
    part_b_timings: list[dict[str, Any]] | None = None,
    part_b_total_ms: float | None = None,
) -> None:
    """Write a human-readable Markdown report."""
    lines = [
        "# Server Monitoring Benchmark Report",
        "",
        "## Part A — Query-Level A/B Comparison",
        "",
        "| Query | Legacy Median (ms) | Flag Median (ms) | Speedup |",
        "|-------|-------------------:|-----------------:|--------:|",
    ]
    for r in results:
        lines.append(
            f"| {r['query_id']} | {r['legacy_median_ms']} | {r['flag_median_ms']} | {r['speedup_ratio']}x |"
        )

    if part_b_timings is not None:
        lines.append("")
        lines.append("## Part B — End-to-End Workflow Phase Timing")
        lines.append("")
        lines.append("| Phase | Duration (ms) |")
        lines.append("|-------|--------------:|")
        for t in part_b_timings:
            lines.append(f"| {t['phase']} | {t['duration_ms']} |")
        if part_b_total_ms is not None:
            lines.append(f"| **Total** | **{part_b_total_ms}** |")

    lines.append("")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_json_report(
    results: list[dict[str, Any]],
    output_path: str,
    part_b_timings: list[dict[str, Any]] | None = None,
    part_b_total_ms: float | None = None,
) -> None:
    """Write a machine-readable JSON report."""
    if part_b_timings is not None:
        payload: dict[str, Any] = {
            "part_a": results,
            "part_b": {
                "timings": part_b_timings,
                "total_duration_ms": part_b_total_ms,
            },
        }
    else:
        payload = results  # backward compat: flat list when Part B is absent
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    """CLI entry point for Part A and Part B benchmark."""
    parser = argparse.ArgumentParser(description="Server monitoring query-level benchmark (Part A) and workflow timing (Part B)")
    parser.add_argument("log_file", help="Path to server monitoring log file")
    parser.add_argument("--output-dir", default="outputs/benchmarks", help="Output directory for reports")
    parser.add_argument("--iterations", type=int, default=3, help="Number of timing iterations per query")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM analysis (Part B runs with no-op nodes)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        with open(args.log_file, "r", encoding="utf-8") as f:
            sample = list(itertools.islice(f, SCHEMA_SAMPLE_SIZE))
    except OSError as exc:
        print(f"Error: unable to read log file '{args.log_file}': {exc}")
        raise SystemExit(1)

    schema = detect_log_structure(sample)

    conn = _ingest_both_schemas(args.log_file, schema)
    try:
        part_a_results = run_part_a(conn, iterations=args.iterations)
    finally:
        conn.close()

    part_b_timings: list[dict[str, Any]] | None = None
    part_b_total_ms: float | None = None

    part_b_timings, part_b_total_ms = run_part_b(args.log_file, schema, skip_llm=args.skip_llm)

    md_path = os.path.join(args.output_dir, "benchmark_report.md")
    json_path = os.path.join(args.output_dir, "benchmark_report.json")
    _write_markdown_report(
        part_a_results,
        md_path,
        part_b_timings=part_b_timings,
        part_b_total_ms=part_b_total_ms,
    )
    _write_json_report(
        part_a_results,
        json_path,
        part_b_timings=part_b_timings,
        part_b_total_ms=part_b_total_ms,
    )

    print(f"Reports written to {args.output_dir}")


if __name__ == "__main__":
    main()
