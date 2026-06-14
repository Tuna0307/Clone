"""Balanced deterministic pre-screening for server_monitoring archetype classification."""

from __future__ import annotations

import uuid
from typing import Any

from pipeline.constants import (
    LOG_GAP_THRESHOLD_SECONDS,
    SERVER_LOG_EVENTS_TABLE,
    STRUCTURAL_SIGNAL_MIN_STRENGTH,
)
from pipeline.server_metrics import is_safe_select
from pipeline.server_sql.archetypes import ALL_ARCHETYPES, IncidentArchetype


def _new_signal_id() -> str:
    return f"sig_{uuid.uuid4().hex[:8]}"


def _run_query(conn: Any, sql: str) -> list[dict[str, Any]]:
    if not is_safe_select(sql):
        return []
    try:
        df = conn.execute(sql).fetchdf()
        if df.empty:
            return []
        return [row.to_dict() for _, row in df.iterrows()]
    except Exception:
        return []


def _make_signal(
    *,
    signal_family: str,
    summary: str,
    sql_query: str,
    observations: list[str],
    strength: float,
    timestamp_start=None,
    timestamp_end=None,
) -> dict[str, Any]:
    return {
        "signal_id": _new_signal_id(),
        "signal_family": signal_family,
        "summary": summary,
        "sql_query": sql_query,
        "observations": observations[:10],
        "strength": max(0.0, min(1.0, strength)),
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_end,
    }


def detect_log_output_gaps(conn: Any) -> dict[str, Any] | None:
    sql = f"""
    WITH ordered AS (
        SELECT timestamp,
               LAG(timestamp) OVER (ORDER BY timestamp) AS prev_ts
        FROM {SERVER_LOG_EVENTS_TABLE}
    ),
    gaps AS (
        SELECT timestamp, prev_ts,
               EXTRACT(EPOCH FROM (timestamp - prev_ts)) AS gap_seconds
        FROM ordered
        WHERE prev_ts IS NOT NULL
    )
    SELECT timestamp, prev_ts, gap_seconds
    FROM gaps
    WHERE gap_seconds >= {LOG_GAP_THRESHOLD_SECONDS}
    ORDER BY gap_seconds DESC
    LIMIT 10
    """
    rows = _run_query(conn, sql)
    if not rows:
        return None
    top = rows[0]
    gap = float(top.get("gap_seconds") or 0)
    strength = min(1.0, gap / max(LOG_GAP_THRESHOLD_SECONDS * 3, 1))
    return _make_signal(
        signal_family="log_gap",
        summary=f"Largest log output gap: {gap:.0f}s between consecutive log_events",
        sql_query=sql,
        observations=[str(r) for r in rows[:5]],
        strength=strength,
        timestamp_start=top.get("prev_ts"),
        timestamp_end=top.get("timestamp"),
    )


def detect_metric_log_divergence(conn: Any) -> dict[str, Any] | None:
    sql = f"""
    WITH metric_minutes AS (
        SELECT time_bucket(INTERVAL 1 MINUTE, timestamp) AS minute, COUNT(*) AS metric_rows
        FROM server_metrics GROUP BY minute
    ),
    log_minutes AS (
        SELECT time_bucket(INTERVAL 1 MINUTE, timestamp) AS minute, COUNT(*) AS log_rows
        FROM {SERVER_LOG_EVENTS_TABLE} GROUP BY minute
    ),
    joined AS (
        SELECT m.minute, m.metric_rows, COALESCE(l.log_rows, 0) AS log_rows
        FROM metric_minutes m
        LEFT JOIN log_minutes l ON m.minute = l.minute
    )
    SELECT minute, metric_rows, log_rows,
           CASE WHEN metric_rows > 0 THEN log_rows::DOUBLE / metric_rows ELSE 0 END AS log_to_metric_ratio
    FROM joined
    WHERE metric_rows >= 2 AND log_rows <= 1
    ORDER BY metric_rows DESC
    LIMIT 10
    """
    rows = _run_query(conn, sql)
    if not rows:
        return None
    top = rows[0]
    strength = min(1.0, float(top.get("metric_rows") or 0) / 10.0)
    return _make_signal(
        signal_family="runtime_stall_indicator",
        summary="Metric snapshots continue while log_events rate drops sharply",
        sql_query=sql,
        observations=[str(r) for r in rows[:5]],
        strength=strength,
        timestamp_start=top.get("minute"),
    )


def detect_thread_dbcp_correlation(conn: Any) -> dict[str, Any] | None:
    sql = """
    WITH pivoted AS (
        SELECT timestamp,
               MAX(CASE WHEN metric_name = 'jvm.threadCount' THEN metric_value END) AS thread_count,
               MAX(CASE WHEN metric_name = 'dbcp.ActiveConnections' THEN metric_value END) AS active_conns,
               MAX(CASE WHEN metric_name = 'hibernate.sessionCount' THEN metric_value END) AS hibernate_sessions
        FROM server_metrics
        GROUP BY timestamp
    ),
    deltas AS (
        SELECT timestamp, thread_count, active_conns, hibernate_sessions,
               thread_count - LAG(thread_count) OVER (ORDER BY timestamp) AS thread_delta,
               active_conns - LAG(active_conns) OVER (ORDER BY timestamp) AS conn_delta
        FROM pivoted
        WHERE thread_count IS NOT NULL OR active_conns IS NOT NULL
    )
    SELECT timestamp, thread_count, active_conns, hibernate_sessions, thread_delta, conn_delta
    FROM deltas
    WHERE (thread_delta > 5 AND conn_delta > 2)
       OR (thread_count > 100 AND active_conns > 20)
    ORDER BY timestamp
    LIMIT 15
    """
    rows = _run_query(conn, sql)
    if not rows:
        return None
    co_move = sum(
        1 for r in rows
        if (r.get("thread_delta") or 0) > 0 and (r.get("conn_delta") or 0) > 0
    )
    strength = min(1.0, co_move / max(len(rows), 1) + 0.2)
    return _make_signal(
        signal_family="metric_correlation",
        summary=f"Thread count and DBCP active connections co-move ({co_move} aligned deltas)",
        sql_query=sql,
        observations=[str(r) for r in rows[:5]],
        strength=strength,
        timestamp_start=rows[0].get("timestamp"),
        timestamp_end=rows[-1].get("timestamp") if rows else None,
    )


def detect_endpoint_breadth(conn: Any) -> dict[str, Any] | None:
    sql = f"""
    WITH bounds AS (
        SELECT MIN(timestamp) AS t_min, MAX(timestamp) AS t_max FROM {SERVER_LOG_EVENTS_TABLE}
    ),
    early AS (
        SELECT COUNT(DISTINCT regexp_extract(raw_line, '([A-Za-z0-9_\\.]+(?:\\.java)?:\\d+)', 1)) AS sig_count
        FROM {SERVER_LOG_EVENTS_TABLE}, bounds
        WHERE timestamp <= bounds.t_min + (bounds.t_max - bounds.t_min) / 3
    ),
    late AS (
        SELECT COUNT(DISTINCT regexp_extract(raw_line, '([A-Za-z0-9_\\.]+(?:\\.java)?:\\d+)', 1)) AS sig_count
        FROM {SERVER_LOG_EVENTS_TABLE}, bounds
        WHERE timestamp >= bounds.t_min + 2 * (bounds.t_max - bounds.t_min) / 3
    )
    SELECT early.sig_count AS early_signatures, late.sig_count AS late_signatures
    FROM early, late
    """
    rows = _run_query(conn, sql)
    if not rows:
        return None
    row = rows[0]
    early = int(row.get("early_signatures") or 0)
    late = int(row.get("late_signatures") or 0)
    ratio = late / max(early, 1)
    if ratio > 3:
        summary = f"Endpoint breadth expanded late-window ({early} → {late} distinct signatures)"
        strength = min(1.0, ratio / 10.0)
        family = "endpoint_breadth"
    elif late <= 3 and early <= 5:
        summary = f"Narrow endpoint focus ({late} late-window signatures)"
        strength = 0.6
        family = "endpoint_breadth"
    else:
        summary = f"Moderate endpoint breadth (early={early}, late={late})"
        strength = 0.35
        family = "endpoint_breadth"
    return _make_signal(
        signal_family=family,
        summary=summary,
        sql_query=sql,
        observations=[str(row)],
        strength=strength,
    )


def detect_high_volume_indicators(conn: Any) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []

    count_sql = (
        f"SELECT timestamp, raw_line FROM {SERVER_LOG_EVENTS_TABLE} "
        "WHERE raw_line LIKE '%Count%' OR raw_line LIKE '%rows%' OR raw_line LIKE '%returned%' "
        "ORDER BY regexp_extract(raw_line, '(\\d{3,})', 1) DESC NULLS LAST LIMIT 5"
    )
    count_rows = _run_query(conn, count_sql)
    if count_rows:
        signals.append(_make_signal(
            signal_family="high_volume_indicator",
            summary=f"Large result count indicators found ({len(count_rows)} hits)",
            sql_query=count_sql,
            observations=[str(r) for r in count_rows],
            strength=min(1.0, len(count_rows) / 5.0),
            timestamp_start=count_rows[-1].get("timestamp"),
            timestamp_end=count_rows[0].get("timestamp"),
        ))

    burst_sql = (
        f"WITH bursts AS ("
        f"  SELECT timestamp, "
        f"         regexp_extract(raw_line, '([A-Za-z0-9_\\.]+(?:\\.java)?:\\d+)', 1) as signature, "
        f"         COUNT(*) OVER (PARTITION BY regexp_extract(raw_line, '([A-Za-z0-9_\\.]+(?:\\.java)?:\\d+)', 1) "
        f"                        ORDER BY timestamp RANGE BETWEEN INTERVAL 30 SECOND PRECEDING AND CURRENT ROW) as burst "
        f"  FROM {SERVER_LOG_EVENTS_TABLE} "
        f"  WHERE raw_line LIKE '% - entry%' OR raw_line LIKE '%RoleValidator%' OR raw_line LIKE '%getCredential%'"
        f") "
        f"SELECT timestamp, signature, burst FROM bursts WHERE burst > 5 ORDER BY burst DESC, timestamp LIMIT 10"
    )
    burst_rows = _run_query(conn, burst_sql)
    if burst_rows:
        top_burst = int(burst_rows[0].get("burst") or 0)
        signals.append(_make_signal(
            signal_family="high_volume_indicator",
            summary=f"Method burst detected (max burst={top_burst} in 30s window)",
            sql_query=burst_sql,
            observations=[str(r) for r in burst_rows[:5]],
            strength=min(1.0, top_burst / 50.0),
            timestamp_start=burst_rows[-1].get("timestamp"),
            timestamp_end=burst_rows[0].get("timestamp"),
        ))

    rate_sql = (
        f"SELECT time_bucket(INTERVAL 1 MINUTE, timestamp) as minute, COUNT(*) as lines_per_min "
        f"FROM {SERVER_LOG_EVENTS_TABLE} GROUP BY minute ORDER BY lines_per_min DESC LIMIT 5"
    )
    rate_rows = _run_query(conn, rate_sql)
    if rate_rows:
        top_rate = int(rate_rows[0].get("lines_per_min") or 0)
        signals.append(_make_signal(
            signal_family="high_volume_indicator",
            summary=f"Log line rate spike (peak {top_rate} lines/min)",
            sql_query=rate_sql,
            observations=[str(r) for r in rate_rows],
            strength=min(1.0, top_rate / 5000.0),
            timestamp_start=rate_rows[-1].get("minute"),
            timestamp_end=rate_rows[0].get("minute"),
        ))
    return signals


def detect_runtime_stall_indicators(conn: Any) -> dict[str, Any] | None:
    sql = """
    WITH pivoted AS (
        SELECT timestamp,
               MAX(CASE WHEN metric_name = 'jvm.threadCount' THEN metric_value END) AS thread_count,
               MAX(CASE WHEN metric_name = 'am.tomcat.thread.busy.count' THEN metric_value END) AS busy_threads,
               MAX(CASE WHEN metric_name = 'am.auth.responseTime' THEN metric_value END) AS response_time
        FROM server_metrics
        GROUP BY timestamp
    )
    SELECT timestamp, thread_count, busy_threads, response_time
    FROM pivoted
    WHERE thread_count > 80
      AND (busy_threads IS NULL OR busy_threads < thread_count * 0.3)
      AND (response_time IS NULL OR response_time > 500)
    ORDER BY response_time DESC NULLS LAST
    LIMIT 10
    """
    rows = _run_query(conn, sql)
    if not rows:
        return None
    strength = min(1.0, len(rows) / 5.0 + 0.2)
    return _make_signal(
        signal_family="runtime_stall_indicator",
        summary="High JVM thread count with low Tomcat busy ratio and elevated response times",
        sql_query=sql,
        observations=[str(r) for r in rows[:5]],
        strength=strength,
        timestamp_start=rows[0].get("timestamp"),
    )


def run_broad_diagnostic_queries(conn: Any) -> list[dict[str, Any]]:
    """Run all balanced diagnostic queries and return structural signal dicts."""
    signals: list[dict[str, Any]] = []
    for detector in (
        detect_log_output_gaps,
        detect_metric_log_divergence,
        detect_thread_dbcp_correlation,
        detect_endpoint_breadth,
        detect_runtime_stall_indicators,
    ):
        result = detector(conn)
        if result and result.get("strength", 0) >= STRUCTURAL_SIGNAL_MIN_STRENGTH:
            signals.append(result)
    signals.extend(
        s for s in detect_high_volume_indicators(conn)
        if s.get("strength", 0) >= STRUCTURAL_SIGNAL_MIN_STRENGTH
    )
    return sorted(signals, key=lambda s: s.get("strength", 0), reverse=True)


def score_archetype_candidates(
    structural_signals: list[dict[str, Any]],
    pre_scan_hits: list[dict[str, Any]] | None = None,
) -> dict[IncidentArchetype, float]:
    """Lightweight deterministic pre-scores per archetype (anchors LLM synthesis)."""
    scores: dict[IncidentArchetype, float] = {a: 0.0 for a in ALL_ARCHETYPES}
    pre_scan_hits = pre_scan_hits or []

    family_weights: dict[str, dict[IncidentArchetype, float]] = {
        "log_gap": {"global_runtime_stall": 0.8, "mixed_compound": 0.3},
        "runtime_stall_indicator": {"global_runtime_stall": 0.9, "mixed_compound": 0.4},
        "metric_correlation": {
            "thread_pool_pressure": 0.5,
            "db_connection_pressure": 0.6,
            "global_runtime_stall": 0.2,
        },
        "endpoint_breadth": {"high_volume_cardinality": 0.4, "global_runtime_stall": 0.3},
        "high_volume_indicator": {"high_volume_cardinality": 0.9, "mixed_compound": 0.3},
    }

    for sig in structural_signals:
        family = sig.get("signal_family", "other")
        strength = float(sig.get("strength") or 0)
        for archetype, weight in family_weights.get(family, {}).items():
            scores[archetype] = min(1.0, scores[archetype] + strength * weight)

    for hit in pre_scan_hits:
        st = hit.get("signal_type", "")
        if st in ("high_result_count", "authz_loop_candidate", "heavy_repository_op"):
            scores["high_volume_cardinality"] = min(1.0, scores["high_volume_cardinality"] + 0.25)
        elif st == "extreme_latency":
            scores["high_volume_cardinality"] = min(1.0, scores["high_volume_cardinality"] + 0.15)
            scores["global_runtime_stall"] = min(1.0, scores["global_runtime_stall"] + 0.1)
        elif st == "slow_ldap_or_db":
            scores["db_connection_pressure"] = min(1.0, scores["db_connection_pressure"] + 0.25)

    top_two = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:2]
    if len(top_two) == 2 and top_two[0][1] > 0.4 and top_two[1][1] > 0.35:
        scores["mixed_compound"] = min(1.0, (top_two[0][1] + top_two[1][1]) / 2)

    return scores


def classification_from_prescores(
    pre_scores: dict[IncidentArchetype, float],
    structural_signals: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fallback classification when LLM synthesis fails."""
    ranked = sorted(pre_scores.items(), key=lambda x: x[1], reverse=True)
    primary_arch, primary_conf = ranked[0]
    secondary = None
    if len(ranked) > 1 and ranked[1][1] > 0.3:
        secondary = {
            "archetype": ranked[1][0],
            "confidence": round(ranked[1][1], 2),
            "supporting_signals": [s.get("signal_id", "") for s in structural_signals[:3]],
        }
    rejected = [
        {
            "archetype": arch,
            "confidence": round(conf, 2),
            "supporting_signals": [],
            "rejection_reason": "Lower deterministic pre-score than primary/secondary",
        }
        for arch, conf in ranked[2:]
        if conf < ranked[0][1] - 0.15
    ]
    return {
        "primary": {
            "archetype": primary_arch,
            "confidence": round(primary_conf, 2),
            "supporting_signals": [s.get("signal_id", "") for s in structural_signals[:5]],
        },
        "secondary": secondary,
        "rejected_hypotheses": rejected,
        "classification_method": "deterministic_only",
        "rationale": "Derived from balanced structural signal pre-scores (LLM synthesis unavailable).",
    }


def detect_recurring_operations(conn: Any) -> list[dict[str, Any]]:
    """Detect cadence-scheduled operations for red herring filtering."""
    sql = f"""
    WITH sigs AS (
        SELECT time_bucket(INTERVAL 1 HOUR, timestamp) AS hour_bucket,
               regexp_extract(raw_line, '([A-Za-z0-9_]+(?:TO|Job|Index|Task)[A-Za-z0-9_]*)', 1) AS operation,
               COUNT(*) AS occurrences,
               AVG(LENGTH(raw_line)) AS avg_line_len
        FROM {SERVER_LOG_EVENTS_TABLE}
        WHERE raw_line IS NOT NULL
        GROUP BY hour_bucket, operation
        HAVING occurrences >= 2 AND operation IS NOT NULL AND operation != ''
    )
    SELECT operation, COUNT(DISTINCT hour_bucket) AS hours_seen,
           AVG(occurrences) AS avg_occurrences, STDDEV(occurrences) AS std_occurrences
    FROM sigs
    GROUP BY operation
    HAVING hours_seen >= 2 AND COALESCE(std_occurrences, 0) < 1.5
    ORDER BY hours_seen DESC
    LIMIT 10
    """
    return _run_query(conn, sql)


def detect_metric_onset_anchors(conn: Any) -> list[dict[str, Any]]:
    """Find first significant metric threshold crossings for onset analysis."""
    sql = """
    WITH pivoted AS (
        SELECT timestamp,
               MAX(CASE WHEN metric_name = 'am.auth.responseTime' THEN metric_value END) AS response_time,
               MAX(CASE WHEN metric_name = 'jvm.threadCount' THEN metric_value END) AS thread_count,
               MAX(CASE WHEN metric_name = 'dbcp.ActiveConnections' THEN metric_value END) AS active_conns,
               MAX(CASE WHEN metric_name = 'eventManager.threadPoolQueueSize' THEN metric_value END) AS queue_size
        FROM server_metrics
        GROUP BY timestamp
    ),
    baselines AS (
        SELECT
            percentile_cont(0.5) WITHIN GROUP (ORDER BY response_time) AS rt_p50,
            percentile_cont(0.5) WITHIN GROUP (ORDER BY thread_count) AS tc_p50,
            percentile_cont(0.5) WITHIN GROUP (ORDER BY active_conns) AS ac_p50
        FROM pivoted
    )
    SELECT p.timestamp, p.response_time, p.thread_count, p.active_conns, p.queue_size,
           b.rt_p50, b.tc_p50, b.ac_p50
    FROM pivoted p, baselines b
    WHERE (p.response_time > b.rt_p50 * 2 AND p.response_time > 200)
       OR (p.thread_count > b.tc_p50 + 20)
       OR (p.active_conns > b.ac_p50 + 5)
       OR (p.queue_size > 10)
    ORDER BY p.timestamp
    LIMIT 20
    """
    return _run_query(conn, sql)