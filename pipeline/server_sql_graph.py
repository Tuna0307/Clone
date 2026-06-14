"""Structured workflow (lightweight FSM initially, upgradable to LangGraph) for the server_monitoring (DuckDB + agentic SQL) path.

This module implements the Option B architecture: an archetype-aware, hypothesis-driven
forensic engine for UAM server slowness incidents.

Core principles (non-negotiable):
- Balanced archetype classification before archetype-specific investigation.
- Onset analysis and symptom vs cause discrimination before heavy evidence gathering.
- Competing hypotheses maintained until evidence reasonably rejects them.
- First-class provenance / structured trace for full auditability.
- Reuses deterministic foundation (load_server_metrics_into_duckdb, log_events, pre_detect, is_safe_select).
- Preserves exact contracts: 3-section output, return dict shape, artifact layout, safety guards.
- Server_monitoring path only — zero impact on api_request path.

Pydantic v2 models (ServerMonitoringState + supporting classes) are the canonical typed state.
They are:
  - Fully validated on construction and assignment
  - 100% serializable (model_dump / model_dump_json / model_validate round-trips)
  - Ready for LangGraph StateGraph (add reducers via Annotated[...] when migrating)
  - The direct source of the machine-readable structured trace JSONL artifact

Start with a clean lightweight FSM (no new runtime deps beyond what's already in the project). Easy to promote to full LangGraph later.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from datetime import datetime
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

# Re-exports / thin wrappers around existing high-quality deterministic logic
from pipeline.server_metrics import (
    format_query_dataframe,
    get_sql_safety_rejection_reason,
    load_server_metrics_into_duckdb,
    load_server_metrics_into_duckdb_with_signals,
    normalize_llm_sql,
    UAM5_SERVER_MONITORING_DICTIONARY_TEXT,
)
from pipeline.constants import (
    EVIDENCE_GATHERING_MAX_QUERIES_PER_TURN,
    EVIDENCE_GATHERING_MAX_TURNS,
    SERVER_LOG_EVENTS_TABLE,
    SERVER_SQL_MAX_STEPS,
)
from pipeline.progress import emit_ui_progress
from artifact_paths import debug_evidence_path, ensure_parent_dir

# Canonical typed models (single source of truth after consolidation)
from pipeline.server_sql.state import (
    LogLineRef,
    HighSignalEvent,
    CriticalWindow,
    RedHerringRejection,
    EvidencePackage,
    TraceStep,
    ServerMonitoringState,
    ArchetypeClassification,
    ArchetypeHypothesis,
    OnsetAnalysis,
)
from pipeline.server_sql.artifacts import write_server_debug_artifacts
from pipeline.server_sql.deterministic_diagnostics import (
    run_broad_diagnostic_queries,
    score_archetype_candidates,
    classification_from_prescores,
    detect_recurring_operations,
    detect_metric_onset_anchors,
)
from pipeline.server_sql.evidence_supplements import run_evidence_supplement_queries
from pipeline.server_sql.json_parsing import extract_json_object
from pipeline.server_sql.prompts import (
    build_server_monitoring_system_prompt,
    build_archetype_classification_prompt,
    build_onset_analysis_prompt,
    build_red_herring_filter_prompt,
    build_archetype_evidence_instruction,
    build_archetype_critic_prompt,
    build_ticket_refinement_prompt,
)

# LangGraph (Phase 3)
try:
    from langgraph.graph import StateGraph, END, START
    from langgraph.graph.state import CompiledStateGraph
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    CompiledStateGraph = Any  # type: ignore

# =============================================================================
# Models consolidated into pipeline/server_sql/state.py (Phase 1).
# The duplicate class definitions that used to live here have been removed.
# All references now resolve to the canonical imported versions.
# This eliminates field drift (sql_blocks vs sql_proposed, RedHerringRejection attrs, etc.)
# and makes the LangGraph migration path clean.
# =============================================================================


# (Duplicate model definitions excised in Phase 1 consolidation.
#  All model names now come exclusively from the canonical import at the top:
#     from pipeline.server_sql.state import ServerMonitoringState, ...
#  This file is now free of duplicate class bodies and field-name drift.)

# =============================================================================
# Node function type (lightweight FSM today; ready for LangGraph StateGraph)
# =============================================================================

NodeFn = Callable[[ServerMonitoringState], ServerMonitoringState]

_NODE_UI_LABELS: dict[str, str] = {
    "broad_diagnostic_and_archetype_classification": "Archetype classification",
    "onset_analysis_and_symptom_discrimination": "Onset analysis",
    "red_herring_filter": "Red herring filter",
    "evidence_gathering": "Evidence gathering",
    "critic": "Critic review",
    "report_synthesis": "Report synthesis",
    "ticket_refinement": "Ticket refinement",
}


def _emit_node_start(node_key: str) -> None:
    label = _NODE_UI_LABELS.get(node_key)
    if label:
        emit_ui_progress(f"[Node] {label}...")


def _emit_node_complete(node_key: str) -> None:
    label = _NODE_UI_LABELS.get(node_key)
    if label:
        emit_ui_progress(f"[Node] {label} complete")


def _emit_sql_progress(
    sql: str,
    row_count: int,
    *,
    query_index: int,
    total_queries: int,
    error: str | None = None,
    rejection_reason: str | None = None,
) -> None:
    preview = sql.replace("\n", " ").strip()
    if len(preview) > 240:
        preview = preview[:240] + "..."
    emit_ui_progress(f"[SQL] Executing query {query_index}/{total_queries}:")
    emit_ui_progress(preview)
    if error:
        emit_ui_progress(f"→ SQL error: {error}")
    elif rejection_reason:
        emit_ui_progress(f"→ Rejected by SQL guard: {rejection_reason}")
    elif row_count < 0:
        emit_ui_progress("→ Rejected by SQL guard")
    else:
        emit_ui_progress(f"→ {row_count} rows")


# =============================================================================
# Example / Skeleton nodes demonstrating typed Pydantic usage
# (Real implementations will live in analysis.py or a dedicated workflow_nodes.py
#  and will call the existing deterministic helpers in server_metrics.py)
# =============================================================================

def initialize_node(state: ServerMonitoringState) -> ServerMonitoringState:
    """Entry node. Performs deterministic DuckDB load and application outlier pre-scan."""
    # Idempotency guard: if we already did the heavy load (e.g. LangGraph started
    # initialize, failed later, and the while-loop fallback re-entered this node),
    # don't repeat the expensive file scan + DB population.
    if getattr(state, "metric_row_count", 0) > 0 or "initialize" in getattr(state, "phases_completed", set()):
        return state

    # === MOST ROBUST SCHEMA HANDLING (prefers the good schema from the caller) ===
    # The caller in analysis.py already did detection and the legacy path works.
    # We strongly prefer the schema that was passed in; we only re-detect locally
    # as a fallback if the passed one is obviously broken.
    from pipeline.parsing import detect_log_structure

    schema = getattr(state, "_schema", {}) or {}

    # If the schema we were given is missing critical keys, try local detection
    # with a large sample. This is now only a fallback, not the primary path.
    if not schema or "timestamp_re" not in schema:
        print("  [Server][STRUCTURED] Passed schema missing 'timestamp_re' — attempting local detection as fallback")
        try:
            with open(state.file_path, "r", encoding="utf-8", errors="ignore") as f:
                sample = []
                for _ in range(10000):
                    line = f.readline()
                    if not line:
                        break
                    sample.append(line)
            local = detect_log_structure(sample)
            if local and "timestamp_re" in local:
                schema = local
                print("  [Server][STRUCTURED] Local fallback detection succeeded")
            else:
                print("  [Server][STRUCTURED] Local detection also did not find 'timestamp_re'")
        except Exception as e:
            print(f"  [Server][STRUCTURED] Local detection raised: {e}")

    # Final guard: if we still have nothing usable, the load/pre-detect calls
    # below will raise a clear KeyError that the outer fallback in analysis.py
    # will catch. At least the error message will be the real one.
    object.__setattr__(state, "_schema", schema)

    # === LAST-DITCH DEFENSIVE SCHEMA CHECK ===
    # If we still don't have a usable schema here, do one more aggressive
    # detection right before the calls that need it. This is intentionally
    # duplicated for maximum robustness during the transition.
    if not schema or "timestamp_re" not in schema:
        print("  [Server][STRUCTURED] WARNING: schema still bad before load — re-detecting aggressively now")
        from pipeline.parsing import detect_log_structure
        try:
            with open(state.file_path, "r", encoding="utf-8", errors="ignore") as f:
                sample = [f.readline() for _ in range(10000)]  # very large sample
            schema = detect_log_structure([ln for ln in sample if ln])
            if schema and "timestamp_re" in schema:
                object.__setattr__(state, "_schema", schema)
                print("  [Server][STRUCTURED] Schema re-detection succeeded with large sample")
            else:
                schema = getattr(state, "_schema", {}) or {}
        except Exception as e:
            print(f"  [Server][STRUCTURED] Aggressive re-detection also failed: {e}")
            schema = getattr(state, "_schema", {}) or {}

    # Use the db_path that was generated once in analyze_server_log_with_workflow.
    # Fall back to :memory: only if something went very wrong (should not happen).
    db_path = getattr(state, "db_path", None) or ":memory:"

    # Safety: if a previous attempt left behind an empty/broken file at this path,
    # remove it so DuckDB can create a fresh valid database.
    if db_path != ":memory:" and os.path.exists(db_path):
        try:
            # Quick probe — if it fails or file is tiny, it's not a valid DB.
            if os.path.getsize(db_path) < 100:
                os.unlink(db_path)
            else:
                # Try a read-only open; if it throws, the file is corrupt for our purposes.
                import duckdb as _duckdb
                probe = _duckdb.connect(db_path, read_only=True)
                probe.close()
        except Exception:
            try:
                os.unlink(db_path)
            except Exception:
                pass

    conn, pre_scan_hits = load_server_metrics_into_duckdb_with_signals(
        state.file_path,
        schema,
        query_context=state.query_context,
        db_path=db_path,
    )
    state._duckdb_conn = conn

    # Capture real row counts immediately after load (fixes the previous hardcoded-0 contract gap)
    try:
        state.metric_row_count = conn.execute("SELECT COUNT(*) FROM server_metrics").fetchone()[0]
        state.log_event_row_count = conn.execute(
            f"SELECT COUNT(*) FROM {SERVER_LOG_EVENTS_TABLE}"
        ).fetchone()[0]
        loaded_msg = (
            f"  [Server][STRUCTURED] Loaded {state.metric_row_count:,} metric rows + "
            f"{state.log_event_row_count:,} log events into DuckDB"
        )
        print(loaded_msg)
        emit_ui_progress(loaded_msg)
    except Exception as cnt_err:
        print(f"  [Server][STRUCTURED] Warning: could not query row counts after load: {cnt_err}")
        state.metric_row_count = 0
        state.log_event_row_count = 0

    # Promote pre-scan signals discovered during the single file scan
    for hit in pre_scan_hits:
        state.add_high_volume_signal(hit)

    state.current_phase = "broad_diagnostic_and_archetype_classification"
    state.phases_completed.add("initialize")

    state.add_trace_step(
        step=-1,
        phase="initialize",
        node="initialize",
        decision="continue",
        phases_completed_so_far=list(state.phases_completed),
    )
    return state


def _parse_classification_from_llm(text: str, fallback: dict) -> dict:
    parsed = extract_json_object(text)
    if not parsed or "primary" not in parsed:
        return fallback
    if parsed.get("secondary") in (None, {}, "null"):
        parsed["secondary"] = None
    parsed.setdefault("classification_method", "llm_synthesis")
    parsed.setdefault("rejected_hypotheses", [])
    return parsed


def _parse_onset_from_llm(text: str, metric_onsets: list[dict]) -> dict:
    parsed = extract_json_object(text)
    if parsed and "signal_records" in parsed:
        return parsed
    degradation_start = metric_onsets[0].get("timestamp") if metric_onsets else None
    records = []
    for row in metric_onsets[:5]:
        name = "response_time" if row.get("response_time") else "thread_count"
        records.append({
            "signal_name": name,
            "onset_time": row.get("timestamp"),
            "onset_shape": "unknown",
            "role": "ambiguous",
            "evidence": [str(row)],
        })
    return {
        "degradation_start": degradation_start,
        "onset_shape_overall": "unknown",
        "signal_records": records,
    }


def _parse_red_herrings_from_llm(text: str) -> list[dict]:
    parsed = extract_json_object(text)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict) and "rejections" in parsed:
        return parsed["rejections"]
    # Try parsing a JSON array directly
    import json as _json
    start = text.find("[")
    if start >= 0:
        try:
            arr = _json.loads(text[start:text.rfind("]") + 1])
            if isinstance(arr, list):
                return arr
        except _json.JSONDecodeError:
            pass
    return []


def broad_diagnostic_and_archetype_classification_node(
    state: ServerMonitoringState,
) -> ServerMonitoringState:
    """Balanced deterministic pre-screening + LLM archetype synthesis."""
    _emit_node_start("broad_diagnostic_and_archetype_classification")
    if state.reclassification_count > 0 and state.structural_signals:
        state.structural_signals = []
        state.phases_completed.discard("broad_diagnostic_and_archetype_classification")
        state.phases_completed.discard("onset_analysis_and_symptom_discrimination")

    conn = _get_or_reopen_duckdb_conn(state)
    pre_scan = [s.model_dump() for s in state.high_volume_signals[:10]]
    pre_scan_summary = "\n".join(
        f"- {h.get('signal_type')}: {h.get('snippet', '')[:100]}" for h in pre_scan
    ) or "(none)"

    structural: list[dict] = []
    if conn is not None:
        structural = run_broad_diagnostic_queries(conn)
        for sig in structural:
            state.add_structural_signal(sig)
            state.add_trace_step(
                step=f"struct-{sig.get('signal_id', '?')}",
                phase="broad_diagnostic_and_archetype_classification",
                node="broad_diagnostic",
                sql_blocks=[sig.get("sql_query", "")],
                observations=sig.get("observations", [])[:3],
                decision="continue",
            )

    pre_scores = score_archetype_candidates(structural, pre_scan)
    fallback = classification_from_prescores(pre_scores, structural)

    llm = _get_llm(state)
    prompt = build_archetype_classification_prompt(structural, pre_scores, pre_scan_summary)
    classification_dict = fallback
    try:
        resp = llm.invoke([{"role": "user", "content": prompt}])
        text = getattr(resp, "content", str(resp))
        classification_dict = _parse_classification_from_llm(text, fallback)
        if state.reclassification_count > 0:
            classification_dict["classification_method"] = "critic_reclassification"
    except Exception as e:
        state.add_trace_step(
            step=state.steps_taken,
            phase="broad_diagnostic_and_archetype_classification",
            node="broad_diagnostic",
            observations=[f"LLM classification error: {e}; using deterministic fallback"],
            decision="continue",
        )

    state.set_archetype_classification(classification_dict)

    competing: list[dict] = []
    secondary = classification_dict.get("secondary")
    if secondary and secondary.get("confidence", 0) > 0.3:
        competing.append(secondary)
    for rej in classification_dict.get("rejected_hypotheses", [])[:2]:
        if rej.get("confidence", 0) > 0.2:
            competing.append(rej)
    state.competing_hypotheses = [
        ArchetypeHypothesis.model_validate(h) if isinstance(h, dict) else h
        for h in competing
    ]

    state.mark_phase_complete("broad_diagnostic_and_archetype_classification")
    state.current_phase = "onset_analysis_and_symptom_discrimination"
    state.add_trace_step(
        step=state.steps_taken,
        phase="broad_diagnostic_and_archetype_classification",
        node="broad_diagnostic",
        llm_output=str(classification_dict)[:1200],
        decision="continue",
    )
    _emit_node_complete("broad_diagnostic_and_archetype_classification")
    return state


def onset_analysis_and_symptom_discrimination_node(
    state: ServerMonitoringState,
) -> ServerMonitoringState:
    """Establish degradation onset and classify signals as cause vs effect."""
    _emit_node_start("onset_analysis_and_symptom_discrimination")
    conn = _get_or_reopen_duckdb_conn(state)
    metric_onsets = detect_metric_onset_anchors(conn) if conn else []

    classification = (
        state.archetype_classification.model_dump()
        if state.archetype_classification else {}
    )
    structural = [s.model_dump() for s in state.structural_signals]

    llm = _get_llm(state)
    prompt = build_onset_analysis_prompt(classification, structural, metric_onsets)
    onset_dict = _parse_onset_from_llm("", metric_onsets)
    try:
        resp = llm.invoke([{"role": "user", "content": prompt}])
        text = getattr(resp, "content", str(resp))
        onset_dict = _parse_onset_from_llm(text, metric_onsets)
    except Exception as e:
        state.add_trace_step(
            step=state.steps_taken,
            phase="onset_analysis_and_symptom_discrimination",
            node="onset_analysis",
            observations=[f"LLM onset error: {e}; using metric anchors"],
            decision="continue",
        )

    state.update_onset_analysis(onset_dict)

    deg_start = onset_dict.get("degradation_start")
    for rec in onset_dict.get("signal_records", [])[:3]:
        onset_time = rec.get("onset_time") or deg_start
        if onset_time:
            try:
                state.add_critical_window({
                    "start_time": onset_time,
                    "end_time": onset_time,
                    "label": f"{rec.get('signal_name', 'signal')} onset ({rec.get('role', 'ambiguous')})",
                })
            except Exception:
                pass

    state.mark_phase_complete("onset_analysis_and_symptom_discrimination")
    state.current_phase = "red_herring_filter"
    state.add_trace_step(
        step=state.steps_taken,
        phase="onset_analysis_and_symptom_discrimination",
        node="onset_analysis",
        llm_output=str(onset_dict)[:1200],
        decision="continue",
    )
    _emit_node_complete("onset_analysis_and_symptom_discrimination")
    return state


def red_herring_filter_node(state: ServerMonitoringState) -> ServerMonitoringState:
    """Deterministic recurring-op detection + mandatory LLM red herring pass."""
    _emit_node_start("red_herring_filter")
    conn = _get_or_reopen_duckdb_conn(state)
    recurring = detect_recurring_operations(conn) if conn else []

    for op in recurring[:3]:
        state.add_red_herring({
            "signal_description": str(op.get("operation", "recurring operation")),
            "rejection_category": "cadence_scheduled",
            "rejection_reason": (
                f"Recurring at fixed cadence across {op.get('hours_seen', '?')} hours "
                f"with low variance (std={op.get('std_occurrences', '?')})"
            ),
            "evidence": [str(op)],
            "confidence": "STRONG",
        })

    classification = (
        state.archetype_classification.model_dump()
        if state.archetype_classification else None
    )
    onset = state.onset_analysis.model_dump() if state.onset_analysis else None
    structural = [s.model_dump() for s in state.structural_signals]

    llm = _get_llm(state)
    prompt = build_red_herring_filter_prompt(classification, onset, recurring, structural)
    try:
        resp = llm.invoke([{"role": "user", "content": prompt}])
        text = getattr(resp, "content", str(resp))
        for rej_dict in _parse_red_herrings_from_llm(text):
            try:
                state.add_red_herring(rej_dict)
            except Exception:
                pass
    except Exception as e:
        state.add_trace_step(
            step=state.steps_taken,
            phase="red_herring_filter",
            node="red_herring_filter",
            observations=[f"LLM red herring pass error: {e}"],
            decision="continue",
        )

    state.mark_phase_complete("red_herring_filter")
    state.current_phase = "evidence_gathering"
    state.add_trace_step(
        step=f"red-herring-{len(state.red_herring_rejections)}",
        phase="red_herring_filter",
        node="red_herring_filter",
        observations=[f"Total rejections: {len(state.red_herring_rejections)}"],
        decision="continue",
    )
    _emit_node_complete("red_herring_filter")
    return state


# =============================================================================
# Lightweight FSM dispatcher (the "graph" for the zero-dep path)
# =============================================================================

# =============================================================================
# Phase 2: Real LLM-driven nodes (evidence_gathering, critic, synthesis, ticket)
# =============================================================================

def _get_llm(state: ServerMonitoringState) -> Any:
    llm = getattr(state, "_llm", None) or getattr(state, "llm", None)
    if llm is None:
        from llm_factory import get_llm
        llm = get_llm()
        object.__setattr__(state, "_llm", llm)
    return llm


def _get_or_reopen_duckdb_conn(state: ServerMonitoringState) -> Any:
    """Return a live DuckDB connection for the server_monitoring workflow.

    LangGraph state handoffs (model_validate / superstep reconstruction) drop
    PrivateAttr values such as the in-memory connection created in initialize_node.
    This helper first returns the cached live object if present; otherwise it
    re-opens the file-backed DB whose path was stored in the serializable
    `state.db_path` field and caches it back into the PrivateAttr.

    Safe to call from any post-initialize node (evidence_gathering, etc.).
    The caller must still guard with is_safe_select before executing queries.
    """
    conn = getattr(state, "_duckdb_conn", None)
    if conn is not None:
        return conn

    db_path = getattr(state, "db_path", None)
    if not db_path:
        return None

    try:
        import duckdb
        conn = duckdb.connect(db_path)
        object.__setattr__(state, "_duckdb_conn", conn)
        return conn
    except Exception as e:
        # Record for provenance; caller will treat as no-conn and continue
        state.add_trace_step(
            step=state.steps_taken,
            phase=getattr(state, "current_phase", "unknown"),
            node="_get_or_reopen_duckdb_conn",
            observations=[f"Re-open failed for {db_path}: {e}"],
            decision="error",
        )
        return None


def _generate_server_monitoring_db_path() -> str:
    """Generate a unique, non-existing file path for a file-backed DuckDB.

    We deliberately do NOT create the file here (NamedTemporaryFile would leave
    an empty 0-byte file that DuckDB refuses to open as a valid database).
    DuckDB itself will create a proper database file on first connect().
    """
    base_dir = tempfile.gettempdir()
    unique = uuid.uuid4().hex
    return os.path.join(base_dir, f"servermon_{unique}.duckdb")


def _record_evidence_sql_execution(
    state: ServerMonitoringState,
    *,
    conn: Any,
    sql: str,
    node_label: str = "evidence_gathering",
    llm_output: str | None = None,
    query_index: int = 1,
    total_queries: int = 1,
) -> tuple[bool, str]:
    """Execute one evidence SQL statement and append trace/progress. Returns (success, observation)."""
    rejection_reason = get_sql_safety_rejection_reason(sql)
    if rejection_reason:
        _emit_sql_progress(
            sql,
            -1,
            query_index=query_index,
            total_queries=total_queries,
            rejection_reason=rejection_reason,
        )
        state.add_trace_step(
            step=state.steps_taken,
            phase="evidence_gathering",
            node=node_label,
            llm_output=llm_output,
            sql_blocks=[sql],
            observations=[f"REJECTED: {rejection_reason}"],
            decision="rejected_unsafe",
        )
        return False, f"REJECTED: {rejection_reason}"

    try:
        executable_sql = normalize_llm_sql(sql)
        df = conn.execute(executable_sql).fetchdf()
        row_count = len(df)
        obs = format_query_dataframe(df)
        if len(obs) > 2400:
            obs = obs[:2400] + " ... (truncated)"
        _emit_sql_progress(
            sql,
            row_count,
            query_index=query_index,
            total_queries=total_queries,
        )
        state.add_trace_step(
            step=state.steps_taken,
            phase="evidence_gathering",
            node=node_label,
            llm_output=llm_output,
            sql_blocks=[sql],
            observations=[obs],
            decision="continue",
        )
        state.queries_executed += 1
        return True, obs
    except Exception as exc:
        _emit_sql_progress(
            sql,
            -1,
            query_index=query_index,
            total_queries=total_queries,
            error=str(exc),
        )
        state.add_trace_step(
            step=state.steps_taken,
            phase="evidence_gathering",
            node=node_label,
            sql_blocks=[sql],
            observations=[f"ERROR: {exc}"],
            decision="error",
        )
        return False, f"ERROR: {exc}"


def evidence_gathering_node(state: ServerMonitoringState) -> ServerMonitoringState:
    """Archetype-aware multi-turn LLM evidence gathering plus deterministic supplements."""
    _emit_node_start("evidence_gathering")
    state.phases_completed.add("evidence_gathering")
    llm = _get_llm(state)
    conn = _get_or_reopen_duckdb_conn(state)

    if conn is None:
        state.add_trace_step(step=state.steps_taken, phase="evidence_gathering", node="evidence_gathering",
                             decision="skip_no_conn", observations=["No DuckDB conn (after re-open attempt)"])
        state.current_phase = "critic"
        _emit_node_complete("evidence_gathering")
        return state

    classification = (
        state.archetype_classification.model_dump()
        if state.archetype_classification else None
    )
    competing = [h.model_dump() for h in state.competing_hypotheses]
    onset = state.onset_analysis.model_dump() if state.onset_analysis else None
    structural = [s.model_dump() for s in state.structural_signals[:8]]
    hv = [s.model_dump() for s in state.high_volume_signals[:6]]
    wins = [w.model_dump() for w in state.critical_windows[:4]]

    primary_arch = (classification or {}).get("primary", {}).get("archetype", "other")
    pkg_id = f"pkg_{primary_arch}"
    pkg_sql: list[str] = []
    prior_observations: list[str] = []

    try:
        for turn_idx in range(EVIDENCE_GATHERING_MAX_TURNS):
            instruction = build_archetype_evidence_instruction(
                classification,
                competing,
                onset,
                structural,
                hv,
                wins,
                turn_number=turn_idx + 1,
                max_turns=EVIDENCE_GATHERING_MAX_TURNS,
                prior_observations=prior_observations,
            )
            resp = llm.invoke([{"role": "user", "content": instruction}])
            text = getattr(resp, "content", str(resp))
            ready = "READY FOR SYNTHESIS" in text.upper()
            sql_blocks = re.findall(r"```sql\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
            proposed_blocks = [
                block.strip()
                for block in sql_blocks[:EVIDENCE_GATHERING_MAX_QUERIES_PER_TURN]
                if block.strip()
            ]

            if not proposed_blocks:
                state.add_trace_step(
                    step=state.steps_taken,
                    phase="evidence_gathering",
                    node="evidence_gathering",
                    llm_output=text[:800],
                    observations=[
                        "LLM returned no fenced ```sql blocks."
                        + (" READY FOR SYNTHESIS signaled." if ready else "")
                    ],
                    decision="no_sql" if not ready else "continue",
                )
                if ready:
                    break
                if turn_idx == 0:
                    break
                continue

            for query_index, sql in enumerate(proposed_blocks, start=1):
                ok, observation = _record_evidence_sql_execution(
                    state,
                    conn=conn,
                    sql=sql,
                    llm_output=text[:800] if query_index == 1 else None,
                    query_index=query_index,
                    total_queries=len(proposed_blocks),
                )
                if ok:
                    pkg_sql.append(sql)
                prior_observations.append(
                    f"Turn {turn_idx + 1} query {query_index} ({'ok' if ok else 'failed'}):\n{observation}"
                )

            if ready:
                break

        supplement_sql: list[str] = []
        for label, sql, observation, row_count in run_evidence_supplement_queries(
            conn, onset, wins
        ):
            emit_ui_progress(f"[Evidence supplement] {label}")
            _emit_sql_progress(
                sql,
                row_count,
                query_index=1,
                total_queries=1,
            )
            state.add_trace_step(
                step=state.steps_taken,
                phase="evidence_gathering",
                node=f"evidence_supplement_{label}",
                sql_blocks=[sql],
                observations=[observation],
                decision="continue" if row_count >= 0 else "error",
            )
            if row_count >= 0:
                state.queries_executed += 1
                supplement_sql.append(sql)
                prior_observations.append(
                    f"Deterministic supplement `{label}` ({row_count} rows):\n{observation}"
                )

        all_sql = pkg_sql + supplement_sql
        if all_sql:
            existing = state.evidence_packages.get(pkg_id)
            state.upsert_evidence_package({
                "package_id": pkg_id,
                "hypothesis": f"Evidence for primary archetype: {primary_arch}",
                "category": primary_arch if primary_arch != "other" else "other",
                "sql_queries": (existing.sql_queries if existing else []) + all_sql,
                "confidence": (classification or {}).get("primary", {}).get("confidence", 0.0),
                "created_in_phase": "evidence_gathering",
            })
    except Exception as e:
        state.add_trace_step(step=state.steps_taken, phase="evidence_gathering",
                             node="evidence_gathering", observations=[f"LLM error: {e}"], decision="error")

    state.current_phase = "critic"
    _emit_node_complete("evidence_gathering")
    return state


def critic_node(state: ServerMonitoringState) -> ServerMonitoringState:
    """Archetype-aware critic with reclassification support."""
    _emit_node_start("critic")
    state.phases_completed.add("critic")
    llm = _get_llm(state)

    classification = (
        state.archetype_classification.model_dump()
        if state.archetype_classification else None
    )
    summary = (
        f"Structural signals: {len(state.structural_signals)}\n"
        f"Application outlier signals: {len(state.high_volume_signals)}\n"
        f"Critical windows: {len(state.critical_windows)}\n"
        f"Red herrings: {len(state.red_herring_rejections)}\n"
        f"Evidence packages: {len(state.evidence_packages)}\n"
        f"Queries: {state.queries_executed}\n"
        f"Reclassifications so far: {state.reclassification_count}"
    )

    prompt = build_archetype_critic_prompt(
        summary,
        state.phases_completed,
        classification,
        [h.model_dump() for h in state.competing_hypotheses],
        {k: v.model_dump() for k, v in state.evidence_packages.items()},
        state.onset_analysis.model_dump() if state.onset_analysis else None,
        len(state.red_herring_rejections),
    )

    verdict = "PASS"
    try:
        resp = llm.invoke([{"role": "user", "content": prompt}])
        t = getattr(resp, "content", "")
        t_upper = t.upper()
        if "RECLASSIFY" in t_upper:
            verdict = "RECLASSIFY"
        elif "FAIL" in t_upper:
            verdict = "FAIL"
        elif "RETRY" in t_upper:
            verdict = "RETRY"
    except Exception:
        verdict = "PASS"

    state.add_trace_step(step=state.steps_taken, phase="critic", node="critic",
                         critic_verdict=verdict, observations=[f"Verdict: {verdict}"], decision="continue")

    state.critic_feedback_history.append({
        "verdict": verdict,
        "step": state.steps_taken,
        "timestamp": datetime.utcnow().isoformat(),
    })

    if verdict == "RECLASSIFY" and state.reclassification_count < state.max_reclassifications:
        state.reclassification_count += 1
        stale_primary = (classification or {}).get("primary", {}).get("archetype", "")
        state.evidence_packages = {
            k: v for k, v in state.evidence_packages.items()
            if stale_primary not in k
        }
        state.current_phase = "broad_diagnostic_and_archetype_classification"
    elif verdict == "RETRY" and state.evidence_critic_retry_loops < state.max_evidence_critic_retry_loops:
        state.evidence_critic_retry_loops += 1
        state.current_phase = "evidence_gathering"
    elif verdict == "FAIL" and state.steps_taken < state.max_steps - 3:
        state.current_phase = "evidence_gathering"
    else:
        state.current_phase = "report_synthesis"
    _emit_node_complete("critic")
    return state


def report_synthesis_node(state: ServerMonitoringState) -> ServerMonitoringState:
    """Produce the final 3-section report (or best possible partial from typed evidence)."""
    _emit_node_start("report_synthesis")
    state.phases_completed.add("report_synthesis")
    llm = _get_llm(state)

    hv = [s.model_dump() for s in state.high_volume_signals]
    structural = [s.model_dump() for s in state.structural_signals]
    wins = [w.model_dump() for w in state.critical_windows]
    herrings = [r.model_dump() for r in state.red_herring_rejections]
    classification = (
        state.archetype_classification.model_dump()
        if state.archetype_classification else None
    )
    onset = state.onset_analysis.model_dump() if state.onset_analysis else None

    evidence = (
        f"Archetype classification:\n{json.dumps(classification, default=str, indent=2)}\n\n"
        f"Onset analysis:\n{json.dumps(onset, default=str, indent=2)}\n\n"
        f"Structural signals:\n{json.dumps(structural[:10], default=str, indent=2)}\n\n"
        f"Application outlier signals:\n{json.dumps(hv[:8], default=str, indent=2)}\n\n"
        f"Critical windows:\n{json.dumps(wins[:5], default=str, indent=2)}\n\n"
        f"Red herrings rejected:\n{json.dumps(herrings[:5], default=str, indent=2)}\n\n"
        f"Evidence packages: {list(state.evidence_packages.keys())}\n"
        f"Queries executed: {state.queries_executed} | Trace steps: {len(state.trace)}"
    )

    seed = f"File: {state.file_name}\nRows: {state.metric_row_count} metrics / {state.log_event_row_count} events\n\n{evidence}"
    system = build_server_monitoring_system_prompt(seed, state.ticket_text, classification)
    user = (
        f"Produce the final 3-section report for {state.file_name} using all collected typed evidence. "
        "Include the Incident Archetype Assessment subsection with dominant archetype, "
        "strongest rejected alternative, and why not the other archetype(s)."
    )

    try:
        resp = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
        text = getattr(resp, "content", str(resp))
        if "## 1. File-Wide Evidence Summary" in text and len(text) > 600:
            state.final_findings = text
            state.status = "success"
            decision = "finalize"
        else:
            state.final_findings = _produce_partial_report_from_typed_state(state) + "\n\n(LLM synthesis was weak — using typed diagnostic evidence)"
            state.status = "success"
            decision = "strong_partial"
    except Exception as e:
        state.final_findings = _produce_partial_report_from_typed_state(state) + f"\n\n(Synthesis error: {e})"
        state.status = "success"
        decision = "error_fallback"

    state.add_trace_step(step=state.steps_taken, phase="report_synthesis", node="report_synthesis",
                         llm_output=(state.final_findings or "")[:1000], decision=decision)

    state.current_phase = "ticket_refinement" if state.ticket_text else "finalize"
    _emit_node_complete("report_synthesis")
    return state


def ticket_refinement_node(state: ServerMonitoringState) -> ServerMonitoringState:
    """Append Ticket-Guided Root Cause Addendum when a ticket was supplied."""
    _emit_node_start("ticket_refinement")
    state.phases_completed.add("ticket_refinement")
    if not state.ticket_text or not state.final_findings:
        state.current_phase = "finalize"
        _emit_node_complete("ticket_refinement")
        return state

    llm = _get_llm(state)
    prompt = build_ticket_refinement_prompt(state.ticket_text, state.final_findings)

    try:
        resp = llm.invoke([{"role": "user", "content": prompt}])
        add = getattr(resp, "content", str(resp))
        if len(add) > 150:
            state.final_findings += "\n\n" + "="*60 + "\nTICKET-GUIDED ROOT CAUSE ADDENDUM\n" + add
            state.add_trace_step(step=f"refine-{state.steps_taken}", phase="ticket_refinement",
                                 node="ticket_refinement", llm_output=add[:1200], decision="finalize_with_ticket")
    except Exception as e:
        state.add_trace_step(step=f"refine-{state.steps_taken}", phase="ticket_refinement",
                             node="ticket_refinement", observations=[f"Refinement error: {e}"], decision="finalize")

    state.current_phase = "finalize"
    _emit_node_complete("ticket_refinement")
    return state


# Final registry with real Phase 2 nodes (still used for fallback / testing)
NODE_REGISTRY: dict[str, NodeFn] = {
    "initialize": initialize_node,
    "broad_diagnostic_and_archetype_classification": broad_diagnostic_and_archetype_classification_node,
    "onset_analysis_and_symptom_discrimination": onset_analysis_and_symptom_discrimination_node,
    "red_herring_filter": red_herring_filter_node,
    "evidence_gathering": evidence_gathering_node,
    "critic": critic_node,
    "report_synthesis": report_synthesis_node,
    "ticket_refinement": ticket_refinement_node,
    "finalize": lambda s: s,
}

# =============================================================================
# Phase 3: LangGraph StateGraph (the real compiled graph)
# =============================================================================

def _should_continue_after_critic(state: ServerMonitoringState) -> str:
    """Conditional edge after critic node."""
    if state.steps_taken >= state.max_steps - 2:
        return "report_synthesis"
    last = state.critic_feedback_history[-1] if state.critic_feedback_history else {}
    verdict = last.get("verdict")
    if verdict == "RECLASSIFY":
        return "broad_diagnostic_and_archetype_classification"
    if verdict == "FAIL":
        return "evidence_gathering"
    if getattr(state, "current_phase", None) == "broad_diagnostic_and_archetype_classification":
        return "broad_diagnostic_and_archetype_classification"
    if getattr(state, "current_phase", None) == "evidence_gathering":
        return "evidence_gathering"
    return "report_synthesis"

def _should_do_ticket_refinement(state: ServerMonitoringState) -> str:
    """Decide after report_synthesis whether to run the optional ticket-guided refinement pass."""
    if state.ticket_text and state.final_findings:
        return "ticket_refinement"
    return "finalize"

def _build_server_monitoring_graph() -> CompiledStateGraph:
    """Build and compile the LangGraph StateGraph for the structured workflow."""
    if not HAS_LANGGRAPH:
        # Fallback: return None so the old while-loop path is used
        return None

    # Compatible with both langgraph 0.2.x and 1.1.x+ (required by langchain 1.2.12)
    graph = StateGraph(state_schema=ServerMonitoringState)

    # Add all the Phase 2 nodes (they are pure functions: state -> state)
    graph.add_node("initialize", initialize_node)
    graph.add_node("broad_diagnostic_and_archetype_classification", broad_diagnostic_and_archetype_classification_node)
    graph.add_node("onset_analysis_and_symptom_discrimination", onset_analysis_and_symptom_discrimination_node)
    graph.add_node("red_herring_filter", red_herring_filter_node)
    graph.add_node("evidence_gathering", evidence_gathering_node)
    graph.add_node("critic", critic_node)
    graph.add_node("report_synthesis", report_synthesis_node)
    graph.add_node("ticket_refinement", ticket_refinement_node)

    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "broad_diagnostic_and_archetype_classification")
    graph.add_edge("broad_diagnostic_and_archetype_classification", "onset_analysis_and_symptom_discrimination")
    graph.add_edge("onset_analysis_and_symptom_discrimination", "red_herring_filter")
    graph.add_edge("red_herring_filter", "evidence_gathering")
    graph.add_edge("evidence_gathering", "critic")

    graph.add_conditional_edges(
        "critic",
        _should_continue_after_critic,
        {
            "evidence_gathering": "evidence_gathering",
            "broad_diagnostic_and_archetype_classification": "broad_diagnostic_and_archetype_classification",
            "report_synthesis": "report_synthesis",
        },
    )

    # Ticket refinement is OPTIONAL and only after synthesis.
    # Use conditional from report_synthesis so we never create a self-loop on ticket_refinement.
    graph.add_conditional_edges(
        "report_synthesis",
        _should_do_ticket_refinement,
        {
            "ticket_refinement": "ticket_refinement",
            "finalize": END,
        },
    )

    # After (optional) ticket refinement we always terminate.
    graph.add_edge("ticket_refinement", END)

    compiled = graph.compile()
    return compiled

# Compile once at import time (cheap)
_COMPILED_GRAPH: CompiledStateGraph | None = _build_server_monitoring_graph()


def run_server_monitoring_workflow(
    initial_state: ServerMonitoringState,
    max_steps: int | None = None,
) -> ServerMonitoringState:
    """Main entry point — now powered by LangGraph StateGraph (Phase 3).

    Falls back gracefully to the while-loop implementation if LangGraph is not
    available or the compiled graph fails.
    """
    state = initial_state
    if max_steps is not None:
        state.max_steps = max_steps

    if HAS_LANGGRAPH and _COMPILED_GRAPH is not None:
        try:
            # LangGraph will drive the nodes via the edges we defined.
            # Our nodes mutate the Pydantic state in place and return it.
            # Use generous recursion_limit: agentic loops (critic retries) cost ~2 supersteps each.
            # max_steps is the *agent budget* (LLM turns); graph steps are higher.
            recursion_limit = max(60, state.max_steps * 3 + 10)
            config = {"recursion_limit": recursion_limit}
            final_state: ServerMonitoringState = state
            for chunk in _COMPILED_GRAPH.stream(state, config):
                if not isinstance(chunk, dict):
                    continue
                for _node_name, node_output in chunk.items():
                    if isinstance(node_output, ServerMonitoringState):
                        final_state = node_output
                    elif isinstance(node_output, dict):
                        final_state = ServerMonitoringState.model_validate(node_output)
            return final_state
        except Exception as graph_err:
            err_str = str(graph_err)
            if "recursion" in err_str.lower() or "GRAPH_RECURSION_LIMIT" in err_str:
                print("  [Server][STRUCTURED] LangGraph hit recursion limit on a complex trace; continuing via while-loop fallback (results identical).")
            else:
                print(f"  [Server][STRUCTURED] LangGraph invocation failed, falling back to while-loop: {graph_err}")

    # --- Fallback: original while-loop (still fully functional with Phase 2 nodes) ---
    while state.current_phase != "finalize" and state.steps_taken < state.max_steps:
        node_fn = NODE_REGISTRY.get(state.current_phase)
        if node_fn is None:
            if not state.final_findings:
                state.final_findings = _produce_partial_report_from_typed_state(state)
            state.status = "success"
            state.add_trace_step(
                step=state.steps_taken,
                phase=state.current_phase,
                node="unknown_phase_fallback",
                decision="finalize_with_phase0_evidence",
                observations=[f"Unknown phase {state.current_phase} — falling back to typed diagnostic evidence"],
            )
            state.current_phase = "finalize"
            break

        state = node_fn(state)
        state.steps_taken += 1

    if state.current_phase != "finalize":
        state.status = "error"
        if not state.final_findings:
            state.final_findings = _produce_partial_report_from_typed_state(state)
        state.add_trace_step(
            step="final-error",
            phase=state.current_phase,
            decision="error",
        )

    return state


def _produce_partial_report_from_typed_state(state: ServerMonitoringState) -> str:
    """Produce a useful partial report from archetype-aware typed state."""
    lines = []
    lines.append("# 1. File-Wide Evidence Summary (Structured Workflow - Archetype-Aware Partial)\n")

    if state.archetype_classification:
        ac = state.archetype_classification
        lines.append("### Incident Archetype Assessment\n")
        lines.append(f"- Primary: {ac.primary.archetype} (confidence={ac.primary.confidence})")
        if ac.secondary:
            lines.append(f"- Secondary: {ac.secondary.archetype} (confidence={ac.secondary.confidence})")
        for rej in ac.rejected_hypotheses[:3]:
            lines.append(f"- Rejected: {rej.archetype} — {rej.rejection_reason or 'lower confidence'}")
        if ac.rationale:
            lines.append(f"- Rationale: {ac.rationale[:300]}")
        lines.append("")

    if state.onset_analysis:
        oa = state.onset_analysis
        lines.append("### Onset Analysis\n")
        lines.append(f"- Degradation start: {oa.degradation_start}")
        lines.append(f"- Overall onset shape: {oa.onset_shape_overall}")
        for rec in oa.signal_records[:6]:
            lines.append(f"- {rec.signal_name}: {rec.role} ({rec.onset_shape}) @ {rec.onset_time}")
        lines.append("")

    if state.structural_signals:
        lines.append("### Structural Signals (balanced deterministic pre-screen)\n")
        for sig in state.structural_signals[:12]:
            lines.append(f"- [{sig.signal_family}] strength={sig.strength:.2f}: {sig.summary[:180]}")
        lines.append("")

    if state.high_volume_signals:
        lines.append("### Application Outlier Signals (pre-scan)\n")
        for sig in state.high_volume_signals[:8]:
            val = f" = {sig.captured_value}" if sig.captured_value else ""
            lines.append(f"- [{sig.timestamp}] {sig.signal_type}{val}: {sig.snippet[:180]}")
        lines.append("")

    if state.critical_windows:
        lines.append("### Critical Windows\n")
        for w in state.critical_windows[:5]:
            lines.append(f"- {w.label} ({w.start_time} → {w.end_time})")
        lines.append("")

    if state.red_herring_rejections:
        lines.append("### Red Herrings Considered and Rejected\n")
        for rej in state.red_herring_rejections[:6]:
            lines.append(f"- {rej.signal_description} — {rej.rejection_category}: {rej.rejection_reason}")
        lines.append("")

    lines.append("---")
    lines.append(
        "**Note**: Partial report from the archetype-aware structured workflow. "
        "LLM synthesis did not complete; evidence above comes from balanced diagnostic SQL, "
        "classification, and onset analysis phases."
    )
    lines.append("Full step-by-step trace is in the .sql_trace.jsonl artifact.")
    return "\n".join(lines)


def _produce_partial_report_from_phase0(state: ServerMonitoringState) -> str:
    """Backward-compatible alias."""
    return _produce_partial_report_from_typed_state(state)


# =============================================================================
# Convenience entry point (target for analysis.py refactor)
# =============================================================================

def analyze_server_log_with_workflow(
    file_path: str,
    schema: dict,
    query_context: Optional[dict[str, Any]] = None,
    ticket_text: Optional[str] = None,
    llm: Any | None = None,
    retain_duckdb: bool = True,
) -> dict[str, Any]:
    """Drop-in replacement target for the current server_monitoring logic in analysis.py.

    Returns the exact same shape the rest of the pipeline (runner, reporting, followup) expects.

    llm: Optional pre-instantiated LLM (ChatOpenAI / ChatBedrockConverse). If None, one is obtained via llm_factory.
    """
    if llm is None:
        from llm_factory import get_llm
        llm = get_llm()

    initial_state = ServerMonitoringState(
        file_name=file_path.replace("\\", "/").split("/")[-1],
        file_path=file_path,
        query_context=query_context,
        ticket_text=ticket_text,
    )
    # Set the private schema and llm using Pydantic-safe private attr access
    object.__setattr__(initial_state, "_schema", schema or {})
    object.__setattr__(initial_state, "_llm", llm)

    # Generate a unique file-backed DB path *once* for this analysis run.
    # This must happen before run_server_monitoring_workflow so that both the
    # LangGraph path and any while-loop fallback see the exact same path.
    # Using a plain path string (never pre-creating the file) avoids the
    # "exists but is not a valid DuckDB database file" error.
    if not getattr(initial_state, "db_path", None):
        initial_state.db_path = _generate_server_monitoring_db_path()

    db_path_for_cleanup = initial_state.db_path

    final_state = None
    try:
        final_state = run_server_monitoring_workflow(initial_state)

        # Always write the canonical debug artifacts with the rich typed trace.
        # This ensures the user gets the Phase 0 high-volume evidence even if the
        # workflow only partially completed.
        debug_txt_path = None
        trace_jsonl_path = None
        try:
            debug_txt_path, trace_jsonl_path = write_server_debug_artifacts(
                file_name=final_state.file_name,
                state=final_state,
                findings=final_state.final_findings or "",
                ticket_text=final_state.ticket_text,
                trace_steps=[ts.model_dump(mode="json") for ts in final_state.trace],
            )
        except Exception as art_err:
            print(f"  [Server] Warning: could not write artifacts in structured path: {art_err}")
    finally:
        # Retain the temp file when requested so the UI can copy tables into an
        # in-memory DuckDB for follow-up SQL. Otherwise clean up immediately.
        if not retain_duckdb and db_path_for_cleanup and db_path_for_cleanup != ":memory:":
            try:
                if os.path.exists(db_path_for_cleanup):
                    os.unlink(db_path_for_cleanup)
            except Exception:
                pass

    # Return shape expected by the rest of the pipeline
    # (final_state may be None in extreme error cases — guard it)
    fs = final_state or initial_state
    return {
        "file": fs.file_name,
        "findings": fs.final_findings or "",
        "status": fs.status,
        "category": "server_monitoring",
        "subcategory": "server_monitoring",
        "duckdb_row_count": fs.metric_row_count,
        "log_event_row_count": fs.log_event_row_count,
        "debug_evidence_file": debug_txt_path,
        "sql_trace_file": trace_jsonl_path,
        "sql_queries_executed": fs.queries_executed,
        "agent_steps": fs.steps_taken,
        "ticket_used": bool(fs.ticket_text),
        "ticket_chars": len(fs.ticket_text or ""),
        "faiss_index_dir": None,
        "metadata_rows": [],
        "selected_row_ids_for_reduce": [],
        "evidence_profile": {},
        "source_path": fs.file_path,
        "evidence_packages": {
            pid: pkg.model_dump(mode="json") for pid, pkg in fs.evidence_packages.items()
        },
        "critical_windows": [w.model_dump(mode="json") for w in fs.critical_windows],
        "red_herring_rejections": [r.model_dump(mode="json") for r in fs.red_herring_rejections],
        "trace_len": len(fs.trace),
        "archetype_classification": (
            fs.archetype_classification.model_dump(mode="json")
            if fs.archetype_classification else None
        ),
        "onset_analysis": (
            fs.onset_analysis.model_dump(mode="json")
            if fs.onset_analysis else None
        ),
        "duckdb_temp_path": (
            db_path_for_cleanup
            if retain_duckdb and db_path_for_cleanup and db_path_for_cleanup != ":memory:"
            else None
        ),
    }


# =============================================================================
# Example usage + validation (executable documentation)
# =============================================================================

def _example_usage_and_validation() -> None:
    """Demonstrates construction, validation, serialization, and trace JSONL emission.

    Run this module directly (python -m pipeline.server_sql_graph) to execute.
    """
    print("=== Pydantic v2 Server Monitoring State — Example ===")

    # 1. Construction with nested models (full validation happens automatically)
    state = ServerMonitoringState(
        file_name="ucm.log",
        file_path="/tmp/ucm.log",
        ticket_text="High CPU on DATCKPW1/2 after listMyRequests with 6891 rows.",
    )

    # 2. Add typed evidence via convenience methods (recommended)
    sig = state.add_high_volume_signal({
        "timestamp": "2025-09-08T17:00:44",
        "signal_type": "high_result_count",
        "captured_value": 6891,
        "snippet": "Count = 6891",
        "raw_line": "17:00:44 ... Count = 6891 ... listMyRequestsCount",
        "discovery_method": "phase0_sql",
        "first_onset": True,
    })

    cw = state.add_critical_window({
        "start_time": "2025-09-08T17:00:00",
        "end_time": "2025-09-08T17:15:00",
        "label": "Count=6891 N+1 onset",
        "description": "High-cardinality result set triggering per-record RoleValidator authz loop",
        "node_or_server": "DATCKPW2",
    })

    rej = state.add_red_herring({
        "signal_description": "createIndexAllAvailableCredentialTO ~60s",
        "rejection_category": "cadence_scheduled",
        "rejection_reason": "Identical cost at 04:00 and 05:00 outside any incident window. Cadence-based false positive.",
        "evidence": ["baseline hourly runs visible in log_events"],
    })

    # 3. Record a rich TraceStep
    state.add_trace_step(
        step=3,
        phase="main_loop",
        node="evidence_gathering",
        llm_output="```sql\nSELECT ... FROM log_events WHERE ... Count = 6891 ...\n```",
        sql_blocks=["SELECT timestamp, raw_line FROM log_events WHERE raw_line LIKE '%Count = 6891%' LIMIT 50;"],
        observations=["Returned 47 rows. First onset at 17:00:44 on DATCKPW2."],
        new_critical_windows=[cw],
        decision="continue",
    )

    state.mark_phase_complete("high_volume_diagnostic")
    state.current_phase = "finalize"
    state.status = "success"
    state.final_findings = "# 1. File-Wide Evidence Summary\nRoot cause: listMyRequests Count=6891 → N+1 RoleValidator storm..."

    # 4. Validation in action (Pydantic will raise on bad data)
    try:
        bad = CriticalWindow(
            start_time=datetime(2025, 9, 8, 18, 0),
            end_time=datetime(2025, 9, 8, 17, 0),  # invalid
            label="bad window",
            description="test",
        )
    except Exception as ve:
        print(f"Validation correctly rejected bad window: {type(ve).__name__}")

    # 5. Full serialization (the key for artifacts + LangGraph checkpoints)
    safe = state.to_serializable_dict()
    print(f"Serializable keys (sample): {list(safe.keys())[:8]}...")
    print(f"Trace length: {len(safe['trace'])}")

    # 6. Exact structure for the structured trace JSONL output
    jsonl_lines = state.to_trace_jsonl_lines()
    print("\n=== Proposed .sql_trace.jsonl structure (one JSON object per line) ===")
    print("Each line = TraceStep.model_dump_json()")
    print("Example first two lines (pretty-printed for readability here):")
    import json as _json  # local import only for demo pretty-print
    for i, line in enumerate(jsonl_lines[:2]):
        print(f"Line {i}: {_json.dumps(_json.loads(line), indent=2)[:800]}...")
    print("... (remaining steps)")

    # 7. Round-trip
    restored = ServerMonitoringState.model_validate(safe)
    assert len(restored.trace) == len(state.trace)
    assert restored.high_volume_signals[0].captured_value == 6891
    print("\nRound-trip validation: SUCCESS")

    print("\n=== End of example. Models are production-ready for FSM or LangGraph. ===")


if __name__ == "__main__":
    _example_usage_and_validation()